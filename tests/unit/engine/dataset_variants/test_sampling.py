"""Time variants preserve source motion, time, identities, and annotations."""

from __future__ import annotations

import configparser
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.datasets.config import load_dataset_config
from boxmot.engine.dataset_variants.sampling import select_bursty_frames, timing_statistics
from boxmot.engine.dataset_variants.workflow import create_raw_variant
from boxmot.engine.materialization.catalog import catalog_mot_dataset


@pytest.fixture
def source_sequence(tmp_path: Path):
    """Include ignored GT rows and varying pixels so row/frame errors are visible."""
    sequence = tmp_path / "original" / "train" / "SEQ-10"
    (sequence / "img1").mkdir(parents=True)
    rows = []
    for frame_id in range(1, 91):
        cv2.imwrite(str(sequence / "img1" / f"{frame_id:06d}.png"), np.full((8, 10, 3), frame_id, np.uint8))
        rows.extend((f"{frame_id},42,{frame_id},2,3,4,1,1,0.85\n", f"{frame_id},99,1,2,3,4,0,8,0.00\n"))
    (sequence / "gt").mkdir()
    (sequence / "gt" / "gt.txt").write_text("".join(rows), encoding="utf-8")
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=SEQ-10\nframeRate=30\nseqLength=90\nimWidth=10\nimHeight=8\nimExt=.png\n",
        encoding="utf-8",
    )
    config = {
        "id": "original",
        "layout": "mot",
        "box_type": "aabb",
        "root": "original",
        "default_split": "train",
        "splits": {"train": {"path": "train", "has_ground_truth": True}},
        "classes": {"pedestrian": {"id": 1, "evaluation": "target"}, "distractor": {"id": 8, "evaluation": "ignore"}},
    }
    return config, catalog_mot_dataset(config, data_root=tmp_path), sequence


def test_bursty_selection_is_reproducible_preserves_duration_and_has_a_long_outage():
    times = [index / 30 for index in range(326)]
    selection = select_bursty_frames(times, seed=0)
    assert selection == select_bursty_frames(times, seed=0)
    assert selection != select_bursty_frames(times, seed=1)
    assert selection.indices[0] == 0
    assert selection.indices[-1] == 325
    assert sorted(set(selection.indices)) == list(selection.indices)
    assert all(not selection.outage_start_index <= index < selection.outage_end_index for index in selection.indices)
    stats = timing_statistics(times, selection)
    assert 150 < stats["retained_frames"] < 290
    assert stats["duration_s"] == times[-1]
    assert stats["max_dt_s"] >= 0.3
    assert stats["min_dt_s"] == pytest.approx(1 / 30)
    assert sum(item["count"] for item in stats["interval_histogram"]) == len(selection.indices) - 1


@pytest.mark.parametrize("times", [[None] * 30, [float("nan")] * 30, [0.0] * 30, [1.0, 0.0] * 15, [0.0] * 29])
def test_sampling_rejects_missing_invalid_or_short_timeline(times):
    with pytest.raises(ValueError):
        select_bursty_frames(times)


@pytest.mark.parametrize("seed", [True, -1, 1.5])
def test_sampling_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        select_bursty_frames([index / 30 for index in range(30)], seed=seed)


def test_raw_variant_preserves_images_times_classes_and_entire_selected_gt_rows(source_sequence, tmp_path):
    config, original, source = source_sequence
    original_gt = (source / "gt" / "gt.txt").read_bytes()
    raw = create_raw_variant(config, original, sequence="SEQ-10", data_root=tmp_path, parent_build_id="a" * 64)
    variant_config = load_dataset_config(raw.dataset_config)
    derived = catalog_mot_dataset(variant_config, data_root=tmp_path)
    assert variant_config["classes"] == config["classes"]
    assert variant_config["resources"] == {}
    report = json.loads(raw.report_path.read_text())
    rows = (raw.dataset_config.parent / "variable" / "SEQ-10" / "gt" / "gt.txt").read_text().splitlines()
    assert len(rows) == 2 * len(derived.samples)
    for sample, mapping, (target_gt, ignored_gt) in zip(
        derived.samples, report["frames"], zip(rows[::2], rows[1::2]), strict=True
    ):
        source_id = mapping["source_frame_id"]
        original_sample = original.samples[source_id - 1]
        assert sample.timestamp_s == original_sample.timestamp_s == mapping["timestamp_s"]
        assert sample.source_sha256 == original_sample.source_sha256
        assert raw.sample_map[sample.sample_id] == original_sample.sample_id
        assert sample.image_size == original_sample.image_size
        assert target_gt == f"{sample.frame_index + 1},42,{source_id},2,3,4,1,1,0.85"
        assert ignored_gt == f"{sample.frame_index + 1},99,1,2,3,4,0,8,0.00"
    parser = configparser.ConfigParser()
    parser.read(raw.dataset_config.parent / "variable" / "SEQ-10" / "seqinfo.ini")
    assert parser.getint("Sequence", "seqLength") == len(derived.samples)
    assert parser.getint("Sequence", "frameRate") == 30
    assert catalog_mot_dataset(config, data_root=tmp_path).fingerprint == original.fingerprint
    assert (source / "gt" / "gt.txt").read_bytes() == original_gt
    with pytest.raises(FileExistsError, match="different --name"):
        create_raw_variant(config, original, sequence="SEQ-10", data_root=tmp_path, parent_build_id="a" * 64)


def test_raw_variant_preserves_irregular_nonzero_original_timestamps(source_sequence, tmp_path):
    config, _, source = source_sequence
    times = np.cumsum(np.resize([0.033, 0.05, 0.02], 90)) + 100.0
    (source / "timestamps.csv").write_text(
        "frame_id,timestamp_s\n"
        + "".join(f"{index},{timestamp!r}\n" for index, timestamp in enumerate(times.tolist(), 1))
    )
    original = catalog_mot_dataset(config, data_root=tmp_path)
    raw = create_raw_variant(config, original, sequence="SEQ-10", data_root=tmp_path, parent_build_id="a" * 64)
    derived = catalog_mot_dataset(load_dataset_config(raw.dataset_config), data_root=tmp_path)
    assert derived.samples[0].timestamp_s == times[0]
    assert derived.samples[-1].timestamp_s == times[-1]
    assert all(sample.timestamp_s in times for sample in derived.samples)


@pytest.mark.parametrize("name", ["../outside", "UPPERCASE", "bad_name"])
def test_bad_output_name_is_rejected_without_writing(source_sequence, tmp_path, name):
    config, original, _ = source_sequence
    with pytest.raises(ValueError, match="malformed id"):
        create_raw_variant(config, original, sequence="SEQ-10", name=name, data_root=tmp_path, parent_build_id="a" * 64)
    assert not (tmp_path / "variants").exists()


@pytest.mark.parametrize("line", ["91,1,1,2,3,4,1,1,1", "1,1,nan,2,3,4,1,1,1", "1,bad"])
def test_malformed_gt_is_rejected_before_any_output(source_sequence, tmp_path, line):
    config, original, source = source_sequence
    (source / "gt" / "gt.txt").write_text(line + "\n")
    with pytest.raises(ValueError):
        create_raw_variant(config, original, sequence="SEQ-10", data_root=tmp_path, parent_build_id="a" * 64)
    assert not (tmp_path / "variants").exists()
