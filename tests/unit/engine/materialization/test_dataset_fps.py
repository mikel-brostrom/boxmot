"""Dataset FPS selection keeps decoded frames, capture time, and ground truth aligned."""

from __future__ import annotations

import configparser
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.engine.dataset_variants.fps import materialize_fps_ground_truth, select_sequence_frames
from boxmot.engine.materialization import decode_source_sample
from boxmot.engine.materialization.catalog import catalog_mot_dataset, inspect_catalog_file


@pytest.fixture
def dataset(tmp_path: Path):
    """Create one second of 30-FPS images with ground truth carrying source IDs."""

    config = {
        "id": "fps-fixture",
        "layout": "mot",
        "root": "fixture",
        "default_split": "train",
        "splits": {"train": {"path": "train", "has_ground_truth": True}},
        "classes": {"person": {"id": 1, "evaluation": "target"}},
    }
    sequence = tmp_path / "fixture" / "train" / "SEQ"
    images = sequence / "img1"
    images.mkdir(parents=True)
    for frame_id in range(1, 32):
        assert cv2.imwrite(str(images / f"{frame_id:06d}.png"), np.full((4, 6, 3), frame_id, dtype=np.uint8))
    (sequence / "seqinfo.ini").write_text("[Sequence]\nframeRate=30\nseqLength=31\n", encoding="utf-8")
    (sequence / "gt").mkdir()
    (sequence / "gt" / "gt.txt").write_text(
        "".join(f"{frame_id},42,{frame_id},0,2,2,1,1,0.75\n" for frame_id in range(1, 32) if frame_id != 7),
        encoding="utf-8",
    )
    return config, tmp_path, sequence


def test_fps_catalog_decodes_only_selected_frames_with_original_timestamps(dataset) -> None:
    config, data_root, sequence = dataset
    inspected_images = []

    def inspect(path: Path, include_image_size: bool):
        if include_image_size:
            inspected_images.append(path.name)
        return inspect_catalog_file(path, include_image_size)

    catalog = catalog_mot_dataset(config, data_root=data_root, fps=5, metadata_resolver=inspect)
    original_ids = [1, 7, 13, 19, 25, 31]

    assert [sample.frame_index for sample in catalog.samples] == list(range(6))
    assert [sample.sample_id for sample in catalog.samples] == [f"train:SEQ:{index}" for index in range(6)]
    assert [sample.timestamp_s for sample in catalog.samples] == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1])
    assert [sample.source_uri for sample in catalog.samples] == [
        (sequence / "img1" / f"{index:06d}.png").as_uri() for index in original_ids
    ]
    assert inspected_images == [f"{index:06d}.png" for index in original_ids]
    assert [int(decode_source_sample(sample).image[0, 0, 0]) for sample in catalog.samples] == original_ids
    assert catalog.metadata["fps"] == 5.0
    assert catalog.metadata["frame_sampling"] == {"SEQ": original_ids}


def test_non_divisible_rate_uses_a_fixed_time_grid_without_accumulating_stride_error(dataset) -> None:
    config, data_root, _sequence = dataset
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=7)

    assert catalog.metadata["frame_sampling"] == {"SEQ": [1, 6, 10, 14, 19, 23, 27, 31]}
    assert [sample.timestamp_s for sample in catalog.samples] == pytest.approx(
        [index / 30 for index in (0, 5, 9, 13, 18, 22, 26, 30)]
    )


def test_fps_ground_truth_filters_and_renumbers_without_filling_empty_frames(dataset, tmp_path) -> None:
    config, data_root, sequence = dataset
    original_gt = (sequence / "gt" / "gt.txt").read_bytes()
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=5)
    source_root, gt_root = materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root)

    assert source_root == gt_root
    assert (gt_root / "SEQ" / "gt" / "gt.txt").read_text() == (
        "1,42,1,0,2,2,1,1,0.75\n"
        "3,42,13,0,2,2,1,1,0.75\n"
        "4,42,19,0,2,2,1,1,0.75\n"
        "5,42,25,0,2,2,1,1,0.75\n"
        "6,42,31,0,2,2,1,1,0.75\n"
    )
    assert (sequence / "gt" / "gt.txt").read_bytes() == original_gt
    assert [int(cv2.imread(str(path))[0, 0, 0]) for path in sorted((source_root / "SEQ" / "img1").iterdir())] == [
        1, 7, 13, 19, 25, 31
    ]
    parser = configparser.ConfigParser()
    parser.read(source_root / "SEQ" / "seqinfo.ini")
    assert parser.getint("Sequence", "seqLength") == 6
    assert parser.getfloat("Sequence", "frameRate") == 5
    assert (source_root / "SEQ" / "timestamps.csv").read_text() == (
        "frame_id,timestamp_s\n1,0.0\n2,0.2\n3,0.4\n4,0.6\n5,0.8\n6,1.0\n"
    )
    assert materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root) == (
        source_root, gt_root
    )


@pytest.mark.parametrize("layout", ["mot", "visdrone"])
def test_flat_annotations_preserve_obb_columns_and_selected_frame_ids(dataset, tmp_path, layout) -> None:
    config, data_root, _sequence = dataset
    config["layout"] = layout
    config["splits"]["train"]["annotations"] = "labels"
    labels = data_root / "fixture" / "labels"
    labels.mkdir()
    (labels / "SEQ.txt").write_text(
        "1,4,0,1,2,3,4,5,6,7,1,2,0\n2,4,1,2,3,4,5,6,7,8,1,2,0\n7,4,2,3,4,5,6,7,8,9,1,2,0\n"
    )
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=5)
    source_root, gt_root = materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root)

    assert gt_root != source_root
    assert (gt_root / "SEQ.txt").read_text() == "1,4,0,1,2,3,4,5,6,7,1,2,0\n2,4,2,3,4,5,6,7,8,9,1,2,0\n"


def test_obb_sequence_annotations_and_temporary_variants_are_all_aligned(dataset, tmp_path) -> None:
    config, data_root, sequence = dataset
    for name in ("gt_obb.txt", "gt_obb_raw.txt", "gt_temp.txt", "gt_obb_temp.txt", "gt_obb_raw_temp.txt"):
        (sequence / "gt" / name).write_text("7 8 0 1 2 3 4 5 6 7 1 2 0\n8 8 1 2 3 4 5 6 7 8 1 2 0\n")
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=5)
    _source_root, gt_root = materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root)

    for name in ("gt_obb.txt", "gt_obb_raw.txt", "gt_temp.txt", "gt_obb_temp.txt", "gt_obb_raw_temp.txt"):
        assert (gt_root / "SEQ" / "gt" / name).read_text() == "2 8 0 1 2 3 4 5 6 7 1 2 0\n"


def test_vfr_timeline_overrides_nominal_fps_and_is_preserved_in_sampled_tree(dataset, tmp_path) -> None:
    config, data_root, sequence = dataset
    for image in sorted((sequence / "img1").iterdir())[6:]:
        image.unlink()
    timestamps = [100, 100.04, 100.19, 100.2, 100.39, 100.4]
    (sequence / "timestamps.csv").write_text(
        "frame_id,timestamp_s\n" + "".join(f"{index},{value}\n" for index, value in enumerate(timestamps, start=1))
    )
    (sequence / "seqinfo.ini").write_text("[Sequence]\nframeRate=unavailable\n")
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=5)
    source_root, _gt_root = materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root)

    assert catalog.metadata["frame_sampling"] == {"SEQ": [1, 4, 6]}
    assert [sample.timestamp_s for sample in catalog.samples] == [100.0, 100.2, 100.4]
    assert (source_root / "SEQ" / "timestamps.csv").read_text() == (
        "frame_id,timestamp_s\n1,100.0\n2,100.2\n3,100.4\n"
    )


def test_sampled_seqinfo_ignores_stale_native_fps_when_timestamps_are_authoritative(dataset, tmp_path) -> None:
    config, data_root, sequence = dataset
    (sequence / "seqinfo.ini").write_text("[Sequence]\nframeRate=1\n")
    (sequence / "timestamps.csv").write_text(
        "frame_id,timestamp_s\n" + "".join(f"{index + 1},{index / 30}\n" for index in range(31))
    )
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=5)
    source_root, _gt_root = materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root)
    parser = configparser.ConfigParser()
    parser.read(source_root / "SEQ" / "seqinfo.ini")

    assert catalog.metadata["frame_sampling"] == {"SEQ": [1, 7, 13, 19, 25, 31]}
    assert parser.getfloat("Sequence", "frameRate") == 5


def test_rate_above_native_never_duplicates_frames_and_keeps_native_seqinfo(dataset, tmp_path) -> None:
    config, data_root, _sequence = dataset
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=60)
    source_root, _gt_root = materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root)
    parser = configparser.ConfigParser()
    parser.read(source_root / "SEQ" / "seqinfo.ini")

    assert catalog.metadata["frame_sampling"] == {"SEQ": list(range(1, 32))}
    assert len({sample.source_uri for sample in catalog.samples}) == 31
    assert parser.getfloat("Sequence", "frameRate") == 30


def test_sampled_seqinfo_accepts_case_insensitive_source_keys(dataset, tmp_path) -> None:
    config, data_root, sequence = dataset
    (sequence / "seqinfo.ini").write_text("[Sequence]\nframerate=30\nseqlength=31\n")
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=60)
    source_root, _gt_root = materialize_fps_ground_truth(config, catalog, tmp_path / "sampled", data_root=data_root)
    parser = configparser.ConfigParser()
    parser.read(source_root / "SEQ" / "seqinfo.ini")

    assert parser.getfloat("Sequence", "frameRate") == 30
    assert parser.getint("Sequence", "seqLength") == 31


@pytest.mark.parametrize("frame_rate", [None, "0", "-3", "nan", "inf", "unavailable"])
def test_fps_requires_valid_source_timing(dataset, frame_rate) -> None:
    config, data_root, sequence = dataset
    if frame_rate is None:
        (sequence / "seqinfo.ini").unlink()
    else:
        (sequence / "seqinfo.ini").write_text(f"[Sequence]\nframeRate={frame_rate}\n")

    with pytest.raises(ValueError):
        catalog_mot_dataset(config, data_root=data_root, fps=5)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), True, "bad"])
def test_invalid_requested_fps_is_rejected(dataset, fps) -> None:
    config, data_root, _sequence = dataset
    with pytest.raises(ValueError, match="finite, positive"):
        catalog_mot_dataset(config, data_root=data_root, fps=fps)


def test_catalog_identity_tracks_sampling_and_is_stable_when_source_moves(dataset, tmp_path) -> None:
    config, data_root, _sequence = dataset
    native = catalog_mot_dataset(config, data_root=data_root)
    five = catalog_mot_dataset(config, data_root=data_root, fps=5)
    six = catalog_mot_dataset(config, data_root=data_root, fps=6)
    relocated_root = tmp_path / "relocated"
    shutil.copytree(data_root / "fixture", relocated_root / "fixture")
    relocated = catalog_mot_dataset(config, data_root=relocated_root, fps=5)

    assert len({native.fingerprint, five.fingerprint, six.fingerprint}) == 3
    assert len({item.metadata["ground_truth_digest"] for item in (native, five, six)}) == 3
    assert relocated.fingerprint == five.fingerprint
    assert relocated.metadata["source_root_uri"] != five.metadata["source_root_uri"]
    assert "fps" not in native.metadata
    assert "frame_sampling" not in native.metadata


def test_empty_time_bins_do_not_synthesize_missing_frames() -> None:
    assert select_sequence_frames([0.0, 0.01, 0.4, 0.401, 0.42], fps=5, sequence_id="SEQ") == (0, 2)


@pytest.mark.parametrize("fps", [5, 7, 30, 60])
def test_epoch_timestamp_rounding_preserves_sampling_at_relative_capture_times(fps) -> None:
    timestamps = [index / 30 for index in range(301)]
    epoch_timestamps = [1_700_000_000.0 + timestamp for timestamp in timestamps]

    assert select_sequence_frames(epoch_timestamps, fps=fps, sequence_id="epoch") == select_sequence_frames(
        timestamps, fps=fps, sequence_id="relative"
    )


def test_epoch_rounding_tolerance_does_not_snap_material_vfr_offsets_to_bin_boundary() -> None:
    timestamps = [1_700_000_000.0 + value for value in (0, 0.199, 0.201, 0.399, 0.401)]

    assert select_sequence_frames(timestamps, fps=5, sequence_id="epoch-vfr") == (0, 2, 4)


def test_each_sequence_uses_its_own_native_rate(dataset) -> None:
    config, data_root, sequence = dataset
    other = sequence.parent / "SEQ-25"
    shutil.copytree(sequence, other)
    (other / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\n")
    catalog = catalog_mot_dataset(config, data_root=data_root, fps=5)

    assert catalog.metadata["frame_sampling"] == {
        "SEQ": [1, 7, 13, 19, 25, 31],
        "SEQ-25": [1, 6, 11, 16, 21, 26, 31],
    }
