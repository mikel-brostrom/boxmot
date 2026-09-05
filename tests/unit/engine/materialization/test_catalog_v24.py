from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot.datasets.manifest import sha256_file
from boxmot.engine.materialization import BoundedFrameDecoder, SourceSample, decode_source_sample
from boxmot.engine.materialization.catalog import (
    catalog_local_source,
    catalog_mot_dataset,
    default_data_root,
    resolve_dataset_annotation_root,
    resolve_dataset_root,
    resolve_dataset_split_root,
)


def _save_mmot_frame(path: Path, *, height: int, width: int, value: int = 0) -> np.ndarray:
    """Write one representative eight-channel MMOT frame."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.full((height, width, 8), value, dtype=np.uint8)
    np.save(path, frame)
    return frame


def test_data_root_precedence(monkeypatch, tmp_path) -> None:
    explicit = tmp_path / "explicit"
    configured = tmp_path / "configured"
    monkeypatch.setenv("BOXMOT_DATASETS_DIR", str(configured))

    assert default_data_root(explicit) == explicit.resolve()
    assert default_data_root() == configured.resolve()
    assert resolve_dataset_root({"root": "MOT17"}, explicit) == explicit.resolve() / "MOT17"


def test_local_image_catalog_is_metadata_only_stable_and_uses_relative_references(tmp_path, monkeypatch) -> None:
    source = tmp_path / "images"
    source.mkdir()
    first = np.zeros((4, 6, 3), dtype=np.uint8)
    first[..., 2] = 255
    second = np.full((3, 5, 3), 17, dtype=np.uint8)
    assert cv2.imwrite(str(source / "b.png"), second)
    assert cv2.imwrite(str(source / "a.png"), first)
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("decoded")))

    catalog = catalog_local_source(source, split="train")
    repeated = catalog_local_source(source, split="train")

    assert catalog.fingerprint == repeated.fingerprint
    assert [sample.image_ref for sample in catalog.samples] == ["a.png", "b.png"]
    assert [sample.frame_index for sample in catalog.samples] == [0, 1]
    assert all(sample.sequence_id for sample in catalog.samples)
    assert [sample.image_size for sample in catalog.samples] == [(4, 6), (3, 5)]
    assert all(not hasattr(sample, "frame") for sample in catalog.samples)
    assert all(not isinstance(value, torch.Tensor) for sample in catalog.samples for value in vars_from_slots(sample))

    first[..., 1] = 99
    assert cv2.imwrite(str(source / "a.png"), first)
    changed = catalog_local_source(source, split="train")
    assert changed.fingerprint != catalog.fingerprint


def test_local_catalog_identity_is_stable_when_source_root_moves(tmp_path) -> None:
    catalogs = []
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    for checkout in ("checkout-a", "checkout-b"):
        source = tmp_path / checkout / "images"
        source.mkdir(parents=True)
        assert cv2.imwrite(str(source / "frame.png"), pixels)
        catalogs.append(catalog_local_source(source, split="train"))

    first, second = catalogs
    assert first.fingerprint == second.fingerprint
    assert first.samples[0].sequence_id == second.samples[0].sequence_id
    assert first.samples[0].image_ref == second.samples[0].image_ref == "frame.png"
    assert first.metadata["source_root_uri"] != second.metadata["source_root_uri"]
    assert first.samples[0].source_uri != second.samples[0].source_uri

    renamed_root = tmp_path / "renamed" / "images"
    renamed_root.mkdir(parents=True)
    assert cv2.imwrite(str(renamed_root / "other-name.png"), pixels)
    assert catalog_local_source(renamed_root, split="train").fingerprint != first.fingerprint


def test_local_catalog_discovers_nested_mmot_npy_sequences_with_stable_identity(tmp_path) -> None:
    catalogs = []
    for checkout in ("checkout-a", "checkout-b"):
        source = tmp_path / checkout / "source" / "test"
        _save_mmot_frame(source / "npy" / "SEQ-B" / "000002.npy", height=5, width=7, value=22)
        _save_mmot_frame(source / "npy" / "SEQ-A" / "000002.npy", height=4, width=6, value=12)
        _save_mmot_frame(source / "npy" / "SEQ-A" / "000001.npy", height=3, width=5, value=11)
        catalogs.append(catalog_local_source(source, split="test"))

    first, second = catalogs
    assert [sample.image_ref for sample in first.samples] == [
        "npy/SEQ-A/000001.npy",
        "npy/SEQ-A/000002.npy",
        "npy/SEQ-B/000002.npy",
    ]
    assert [sample.sample_id for sample in first.samples] == [
        "npy/SEQ-A/000001.npy",
        "npy/SEQ-A/000002.npy",
        "npy/SEQ-B/000002.npy",
    ]
    assert [sample.image_size for sample in first.samples] == [(3, 5), (4, 6), (5, 7)]
    assert [sample.frame_index for sample in first.samples] == [0, 1, 0]
    assert first.samples[0].sequence_id == first.samples[1].sequence_id
    assert first.samples[0].sequence_id != first.samples[2].sequence_id
    assert first.fingerprint == second.fingerprint
    assert [sample.sequence_id for sample in first.samples] == [sample.sequence_id for sample in second.samples]
    assert all(sample.source_frame_index is None for sample in first.samples)
    assert all(not hasattr(sample, "frame") for sample in first.samples)


def test_numpy_catalog_probe_does_not_select_or_decode_pixels(tmp_path, monkeypatch) -> None:
    path = tmp_path / "frame.npy"
    _save_mmot_frame(path, height=3, width=5)
    monkeypatch.setattr(
        np,
        "take",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("selected pixels")),
    )

    catalog = catalog_local_source(path, split="test")

    assert catalog.samples[0].image_size == (3, 5)


def test_mot_catalog_discovers_npy_frames_directly_beneath_sequence_roots(tmp_path) -> None:
    config = {
        "id": "mmot-fixture",
        "layout": "mot",
        "root": "MMOT",
        "default_split": "test",
        "splits": {"test": {"path": "test/npy", "has_ground_truth": True}},
        "classes": {"target": {"car": 1}},
    }
    split_root = tmp_path / "MMOT" / "test" / "npy"
    _save_mmot_frame(split_root / "SEQ-B" / "000001.npy", height=5, width=7, value=31)
    _save_mmot_frame(split_root / "SEQ-A" / "000002.npy", height=4, width=6, value=22)
    _save_mmot_frame(split_root / "SEQ-A" / "000001.npy", height=3, width=5, value=21)

    catalog = catalog_mot_dataset(config, split="test", data_root=tmp_path)

    assert [sample.sample_id for sample in catalog.samples] == [
        "test:SEQ-A:0",
        "test:SEQ-A:1",
        "test:SEQ-B:0",
    ]
    assert [sample.image_ref for sample in catalog.samples] == [
        "test/npy/SEQ-A/000001.npy",
        "test/npy/SEQ-A/000002.npy",
        "test/npy/SEQ-B/000001.npy",
    ]
    assert [sample.sequence_id for sample in catalog.samples] == ["SEQ-A", "SEQ-A", "SEQ-B"]
    assert [sample.frame_index for sample in catalog.samples] == [0, 1, 0]
    assert [sample.image_size for sample in catalog.samples] == [(3, 5), (4, 6), (5, 7)]
    assert [sample.source_sha256 for sample in catalog.samples] == [
        sha256_file(split_root / "SEQ-A" / "000001.npy"),
        sha256_file(split_root / "SEQ-A" / "000002.npy"),
        sha256_file(split_root / "SEQ-B" / "000001.npy"),
    ]
    assert catalog.metadata["dataset_id"] == "mmot-fixture"
    assert catalog.metadata["split"] == "test"
    assert catalog.metadata["source_count"] == 3


def test_mot_catalog_fingerprints_sibling_sequence_annotations(tmp_path) -> None:
    config = {
        "id": "mmot-fixture",
        "layout": "mot",
        "root": "data",
        "default_split": "test",
        "splits": {
            "test": {
                "path": "test/npy",
                "annotations": "test/mot",
                "has_ground_truth": True,
            }
        },
        "classes": {"target": {"car": 1}},
    }
    catalogs = []
    for checkout in ("checkout-a", "checkout-b"):
        data_root = tmp_path / checkout
        frame_path = data_root / "data" / "test" / "npy" / "SEQ-01" / "000001.npy"
        annotation_path = data_root / "data" / "test" / "mot" / "SEQ-01.txt"
        _save_mmot_frame(frame_path, height=3, width=5)
        annotation_path.parent.mkdir(parents=True)
        annotation_path.write_text("1,1,0,0,1,0,1,1,1,0,1,1,0\n", encoding="utf-8")
        catalogs.append(catalog_mot_dataset(config, split="test", data_root=data_root))

    first, relocated = catalogs
    first_ground_truth_digest = first.metadata["ground_truth_digest"]
    assert relocated.fingerprint == first.fingerprint
    assert relocated.metadata["ground_truth_digest"] == first_ground_truth_digest
    assert relocated.metadata["source_root_uri"] != first.metadata["source_root_uri"]

    annotation_path = tmp_path / "checkout-b" / "data" / "test" / "mot" / "SEQ-01.txt"
    annotation_path.write_text("1,1,0,0,2,0,2,2,2,0,2,2,0\n", encoding="utf-8")
    second = catalog_mot_dataset(config, split="test", data_root=tmp_path / "checkout-b")

    assert second.fingerprint != first.fingerprint
    assert second.metadata["ground_truth_digest"] != first_ground_truth_digest


def test_dataset_path_resolvers_honor_explicit_annotation_root(tmp_path) -> None:
    config = {
        "root": "data",
        "layout": "mot",
        "splits": {"test": {"path": "test/npy", "annotations": "labels/test"}},
    }

    assert resolve_dataset_split_root(config, "test", tmp_path) == tmp_path / "data" / "test" / "npy"
    assert resolve_dataset_annotation_root(config, "test", tmp_path) == tmp_path / "data" / "labels" / "test"


def test_mot_catalog_requires_configured_sequence_annotation(tmp_path) -> None:
    config = {
        "id": "mmot-fixture",
        "layout": "mot",
        "root": "data",
        "default_split": "test",
        "splits": {
            "test": {
                "path": "test/npy",
                "annotations": "test/mot",
                "has_ground_truth": True,
            }
        },
        "classes": {"target": {"car": 1}},
    }
    _save_mmot_frame(tmp_path / "data" / "test" / "npy" / "SEQ-01" / "000001.npy", height=3, width=5)
    (tmp_path / "data" / "test" / "mot").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="annotation for sequence 'SEQ-01'"):
        catalog_mot_dataset(config, split="test", data_root=tmp_path)


def test_visdrone_catalog_ignores_default_annotations_when_an_explicit_root_is_configured(tmp_path) -> None:
    config = {
        "id": "visdrone-fixture",
        "layout": "visdrone",
        "root": "data",
        "default_split": "test",
        "splits": {
            "test": {
                "path": "sequences",
                "annotations": "custom-labels",
                "has_ground_truth": True,
            }
        },
        "classes": {"target": {"pedestrian": 1}},
    }
    image_path = tmp_path / "data" / "sequences" / "SEQ-01" / "000001.png"
    image_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(image_path), np.zeros((3, 5, 3), dtype=np.uint8))
    configured = tmp_path / "data" / "custom-labels" / "SEQ-01.txt"
    configured.parent.mkdir(parents=True)
    configured.write_text("1,1,0,0,1,1,1,1,1\n", encoding="utf-8")
    legacy = tmp_path / "data" / "annotations" / "SEQ-01.txt"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("unconfigured-a\n", encoding="utf-8")

    first = catalog_mot_dataset(config, split="test", data_root=tmp_path)
    legacy.write_text("unconfigured-b\n", encoding="utf-8")
    second = catalog_mot_dataset(config, split="test", data_root=tmp_path)

    assert second.fingerprint == first.fingerprint
    assert second.metadata["ground_truth_digest"] == first.metadata["ground_truth_digest"]


def test_dataset_catalog_identity_uses_relative_annotation_refs_and_semantic_metadata(tmp_path) -> None:
    config = {
        "id": "fixture",
        "layout": "mot",
        "root": "Fixture",
        "default_split": "train",
        "splits": {"train": {"path": "train", "has_ground_truth": True}},
        "classes": {"person": {"id": 1, "evaluation": "target"}},
    }
    catalogs = []
    for checkout in ("checkout-a", "checkout-b"):
        sequence = tmp_path / checkout / "Fixture" / "train" / "SEQ-01"
        image_root = sequence / "img1"
        annotation_root = sequence / "gt"
        image_root.mkdir(parents=True)
        annotation_root.mkdir()
        assert cv2.imwrite(str(image_root / "000001.png"), np.zeros((3, 5, 3), dtype=np.uint8))
        (annotation_root / "gt.txt").write_text("1,1,0,0,1,1,1,1,1\n", encoding="utf-8")
        catalogs.append(catalog_mot_dataset(config, split="train", data_root=tmp_path / checkout))

    first, second = catalogs
    assert first.fingerprint == second.fingerprint
    assert first.metadata["source_root_uri"] != second.metadata["source_root_uri"]

    changed_config = {**config, "classes": {"vehicle": {"id": 1, "evaluation": "target"}}}
    changed_taxonomy = catalog_mot_dataset(
        changed_config,
        split="train",
        data_root=tmp_path / "checkout-a",
    )
    assert changed_taxonomy.fingerprint != first.fingerprint

    annotation = tmp_path / "checkout-b" / "Fixture" / "train" / "SEQ-01" / "gt" / "gt.txt"
    annotation.write_text("1,1,0,0,2,2,1,1,1\n", encoding="utf-8")
    changed_annotation = catalog_mot_dataset(
        config,
        split="train",
        data_root=tmp_path / "checkout-b",
    )
    assert changed_annotation.fingerprint != first.fingerprint


def test_dataset_root_rejects_escape(tmp_path) -> None:
    for value in ("", "../outside", str(tmp_path.resolve())):
        try:
            resolve_dataset_root({"root": value}, tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe root accepted: {json.dumps(value)}")


def test_decoder_rejects_source_changed_after_cataloging(tmp_path) -> None:
    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), np.zeros((4, 6, 3), dtype=np.uint8))
    sample = catalog_local_source(path).samples[0]
    assert cv2.imwrite(str(path), np.full((4, 6, 3), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="changed after cataloging"):
        decode_source_sample(sample)


@pytest.mark.parametrize(
    ("channel_count", "bgr_indices"),
    (
        pytest.param(8, (1, 2, 4), id="mmot-3ch-checkpoint-bgr-bands"),
        pytest.param(5, (0, 1, 2), id="other-multichannel-first-three"),
    ),
)
def test_materialization_decoder_selects_expected_numpy_bgr_channels(tmp_path, channel_count, bgr_indices) -> None:
    path = tmp_path / f"frame-{channel_count}.npy"
    source = np.empty((3, 5, channel_count), dtype=np.uint8)
    for channel_index in range(channel_count):
        source[..., channel_index] = 10 + channel_index
    np.save(path, source)
    sample = catalog_local_source(path, split="test").samples[0]

    with BoundedFrameDecoder(workers=1) as decoder:
        frame = decoder.decode((sample,))[0]

    expected_bgr = source[..., list(bgr_indices)]
    expected_rgb = torch.from_numpy(np.ascontiguousarray(expected_bgr[..., ::-1])).permute(2, 0, 1)
    assert torch.equal(frame.image, expected_rgb)
    assert frame.image.shape == (3, 3, 5)
    assert frame.image_size == sample.image_size == (3, 5)
    assert frame.sample_id == sample.sample_id
    assert frame.sequence_id == sample.sequence_id
    assert frame.frame_index == sample.frame_index
    assert frame.source_uri == path.as_uri()


@pytest.mark.parametrize(
    ("array", "reason"),
    (
        (np.zeros((3, 5, 8), dtype=np.float32), "dtype"),
        (np.zeros((3, 5), dtype=np.uint8), "shape"),
        (np.zeros((3, 5, 2), dtype=np.uint8), "channels"),
    ),
)
def test_npy_catalog_and_decoder_reject_malformed_frames(tmp_path, array, reason) -> None:
    path = tmp_path / f"malformed-{reason}.npy"
    np.save(path, array)

    with pytest.raises(ValueError):
        catalog_local_source(path, split="test")

    image_size = tuple(int(value) for value in array.shape[:2])
    sample = SourceSample(
        sample_id=path.name,
        split="test",
        sequence_id="SEQ-01",
        frame_index=0,
        timestamp_s=None,
        image_size=image_size,
        source_uri=path.as_uri(),
        source_sha256=sha256_file(path),
        image_ref=path.name,
    )
    with pytest.raises(ValueError):
        decode_source_sample(sample)


def test_bounded_decoder_hashes_each_unique_source_once(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.source as source_module

    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), np.zeros((4, 6, 3), dtype=np.uint8))
    sample = catalog_local_source(path).samples[0]
    real_sha256_file = source_module.sha256_file
    calls = []

    def recording_sha256_file(received):
        calls.append(received)
        return real_sha256_file(received)

    monkeypatch.setattr(source_module, "sha256_file", recording_sha256_file)
    with BoundedFrameDecoder(workers=2) as decoder:
        first = decoder.decode((sample,))
        second = decoder.decode((sample,))

    assert [frame.sample_id for frame in first + second] == [sample.sample_id, sample.sample_id]
    assert calls == [path]


def test_video_catalog_decode_preserves_frame_identity_and_timestamps(tmp_path) -> None:
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (8, 6),
    )
    assert writer.isOpened()
    try:
        for value in (20, 80, 140):
            writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    finally:
        writer.release()

    catalog = catalog_local_source(path, split="validation")
    with BoundedFrameDecoder(workers=2) as decoder:
        frames = decoder.decode(catalog.samples)

    assert [sample.source_frame_index for sample in catalog.samples] == [0, 1, 2]
    assert [frame.sample_id for frame in frames] == [sample.sample_id for sample in catalog.samples]
    assert [frame.sequence_id for frame in frames] == [sample.sequence_id for sample in catalog.samples]
    assert [frame.frame_index for frame in frames] == [sample.frame_index for sample in catalog.samples]
    assert [frame.timestamp_s for frame in frames] == [sample.timestamp_s for sample in catalog.samples]
    assert all(frame.source_uri == path.as_uri() for frame in frames)


def vars_from_slots(value):
    return (getattr(value, name) for name in value.__slots__)
