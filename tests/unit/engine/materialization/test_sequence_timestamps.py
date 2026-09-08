"""Sequence timestamp sidecars define a complete, content-addressed timeline."""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.engine.materialization import decode_source_sample
from boxmot.engine.materialization.catalog import catalog_mot_dataset


@pytest.fixture
def mot_sequence(tmp_path: Path):
    """Create three ordered frames with a fixed-rate fallback and unchanged GT."""

    config = {
        "id": "mot-fixture",
        "layout": "mot",
        "root": "MOT17",
        "default_split": "train",
        "splits": {"train": {"path": "train", "has_ground_truth": True}},
        "classes": {"target": {"pedestrian": 1}},
    }
    sequence = tmp_path / "MOT17" / "train" / "SEQ-01"
    images = sequence / "img1"
    images.mkdir(parents=True)
    for frame_id in range(1, 4):
        assert cv2.imwrite(str(images / f"{frame_id:06d}.png"), np.full((4, 6, 3), frame_id, dtype=np.uint8))
    (sequence / "seqinfo.ini").write_text("[Sequence]\nframeRate=25\n", encoding="utf-8")
    (sequence / "gt").mkdir()
    (sequence / "gt" / "gt.txt").write_text("1,1,0,0,2,2,1,1,1\n", encoding="utf-8")
    return config, tmp_path, sequence


def test_sidecar_overrides_nominal_fps_and_survives_frame_decode(mot_sequence) -> None:
    config, data_root, sequence = mot_sequence
    (sequence / "timestamps.csv").write_text("frame_id,timestamp_s\n1,10.0\n2,10.04\n3,10.21\n", encoding="utf-8")
    # A supplied timeline is authoritative even when nominal FPS is unavailable.
    (sequence / "seqinfo.ini").write_text("[Sequence]\nframeRate=unavailable\n", encoding="utf-8")

    catalog = catalog_mot_dataset(config, data_root=data_root)

    assert [sample.frame_index for sample in catalog.samples] == [0, 1, 2]
    assert [sample.timestamp_s for sample in catalog.samples] == [10.0, 10.04, 10.21]
    assert [sample.image_ref for sample in catalog.samples] == [
        f"train/SEQ-01/img1/{frame_id:06d}.png" for frame_id in range(1, 4)
    ]
    assert [decode_source_sample(sample).timestamp_s for sample in catalog.samples] == [10.0, 10.04, 10.21]
    assert len(catalog.metadata["timestamps_digest"]) == 64


def test_absent_sidecar_preserves_nominal_timestamps_and_existing_metadata(mot_sequence) -> None:
    config, data_root, sequence = mot_sequence
    original = catalog_mot_dataset(config, data_root=data_root)
    assert [sample.timestamp_s for sample in original.samples] == [0.0, 0.04, 0.08]
    assert "timestamps_digest" not in original.metadata

    sidecar = sequence / "timestamps.csv"
    sidecar.write_text("frame_id,timestamp_s\n1,0\n2,.04\n3,.08\n", encoding="utf-8")
    with_sidecar = catalog_mot_dataset(config, data_root=data_root)
    assert with_sidecar.fingerprint != original.fingerprint
    sidecar.unlink()
    restored = catalog_mot_dataset(config, data_root=data_root)
    assert restored.fingerprint == original.fingerprint
    assert restored.metadata == original.metadata


def test_sidecar_content_changes_invalidate_catalog_and_timestamp_metadata(mot_sequence) -> None:
    config, data_root, sequence = mot_sequence
    sidecar = sequence / "timestamps.csv"
    sidecar.write_text("frame_id,timestamp_s\n1,0\n2,.04\n3,.08\n", encoding="utf-8")
    original = catalog_mot_dataset(config, data_root=data_root)
    sidecar.write_text("frame_id,timestamp_s\n1,0.0\n2,0.04\n3,0.08\n", encoding="utf-8")
    reformatted = catalog_mot_dataset(config, data_root=data_root)
    sidecar.write_text("frame_id,timestamp_s\n1,0.0\n2,0.04\n3,0.18\n", encoding="utf-8")
    changed = catalog_mot_dataset(config, data_root=data_root)

    assert [sample.timestamp_s for sample in original.samples] == [sample.timestamp_s for sample in reformatted.samples]
    assert changed.samples[-1].timestamp_s == 0.18
    assert len({catalog.fingerprint for catalog in (original, reformatted, changed)}) == 3
    assert len({catalog.metadata["timestamps_digest"] for catalog in (original, reformatted, changed)}) == 3
    assert all(
        catalog.metadata["source_catalog_digest"] == catalog.fingerprint for catalog in (original, reformatted, changed)
    )
    assert len({catalog.metadata["ground_truth_digest"] for catalog in (original, reformatted, changed)}) == 1
    assert [sample.source_sha256 for sample in original.samples] == [sample.source_sha256 for sample in changed.samples]


def test_timestamp_identity_is_stable_when_dataset_root_moves(mot_sequence, tmp_path) -> None:
    config, data_root, sequence = mot_sequence
    (sequence / "timestamps.csv").write_text("frame_id,timestamp_s\n1,0\n2,.04\n3,.18\n", encoding="utf-8")
    original = catalog_mot_dataset(config, data_root=data_root)
    relocated_root = tmp_path / "relocated"
    shutil.copytree(data_root / "MOT17", relocated_root / "MOT17")

    relocated = catalog_mot_dataset(config, data_root=relocated_root)

    assert relocated.fingerprint == original.fingerprint
    assert relocated.metadata["timestamps_digest"] == original.metadata["timestamps_digest"]
    assert relocated.metadata["source_root_uri"] != original.metadata["source_root_uri"]


@pytest.mark.parametrize(
    "payload, detail",
    [
        ("", "columns"),
        ("timestamp_s,frame_id\n0,1\n", "columns"),
        ("frame_id,timestamp_s,path\n1,0,../../outside.png\n", "columns"),
        ("frame_id,timestamp_s\n1,0,extra\n", "two values"),
        ("frame_id,timestamp_s\n1,0\n2\n", "two values"),
        ("frame_id,timestamp_s\n1,0\n\n", "two values"),
        ("frame_id,timestamp_s\n1.0,0\n", "integer"),
        ("frame_id,timestamp_s\n0,0\n", "exactly once in order"),
        ("frame_id,timestamp_s\n1,0\n3,.08\n", "exactly once in order"),
        ("frame_id,timestamp_s\n1,0\n2,.04\n2,.08\n", "exactly once in order"),
        ("frame_id,timestamp_s\n2,.04\n1,0\n3,.08\n", "exactly once in order"),
        ("frame_id,timestamp_s\n1,0\n2,.04\n3,.08\n4,.12\n", "exactly once in order"),
        ("frame_id,timestamp_s\n1,0\n2,.04\n", "one timestamp for every image"),
        ("frame_id,timestamp_s\n1,nan\n", "finite number"),
        ("frame_id,timestamp_s\n1,inf\n", "finite number"),
        ("frame_id,timestamp_s\n1,-inf\n", "finite number"),
        ("frame_id,timestamp_s\n1,unknown\n", "finite number"),
        ("frame_id,timestamp_s\n1,0\n2,0\n3,.08\n", "increase strictly"),
        ("frame_id,timestamp_s\n1,.04\n2,0\n3,.08\n", "increase strictly"),
        ('frame_id,timestamp_s\n1,"0\n', "valid UTF-8 CSV"),
    ],
)
def test_invalid_sidecar_never_falls_back_to_nominal_fps(mot_sequence, payload, detail) -> None:
    config, data_root, sequence = mot_sequence
    (sequence / "timestamps.csv").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=detail):
        catalog_mot_dataset(config, data_root=data_root)


def test_timestamp_sidecar_rejects_non_utf8_content(mot_sequence) -> None:
    config, data_root, sequence = mot_sequence
    (sequence / "timestamps.csv").write_bytes(b"frame_id,timestamp_s\n1,\xff\n")

    with pytest.raises(ValueError, match="valid UTF-8 CSV"):
        catalog_mot_dataset(config, data_root=data_root)


@pytest.mark.parametrize("target_exists", [False, True])
def test_timestamp_sidecar_cannot_follow_a_symlink_outside_dataset(mot_sequence, target_exists) -> None:
    config, data_root, sequence = mot_sequence
    outside = data_root / "outside.csv"
    if target_exists:
        outside.write_text("frame_id,timestamp_s\n1,0\n2,.04\n3,.08\n", encoding="utf-8")
    (sequence / "timestamps.csv").symlink_to(outside)

    with pytest.raises(ValueError, match="beneath storage.root"):
        catalog_mot_dataset(config, data_root=data_root)


def test_timestamp_sidecar_must_be_a_regular_file(mot_sequence) -> None:
    config, data_root, sequence = mot_sequence
    (sequence / "timestamps.csv").mkdir()

    with pytest.raises(ValueError, match="regular file"):
        catalog_mot_dataset(config, data_root=data_root)
