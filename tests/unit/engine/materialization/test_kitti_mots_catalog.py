"""Native KITTI MOTS catalogs preserve frame identity and paired mask provenance."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from boxmot.datasets.config import load_dataset_config
from boxmot.engine.materialization.catalog import catalog_mot_dataset, inspect_catalog_file


@pytest.fixture
def kitti_config() -> dict[str, Any]:
    """Limit the official profile to one sequence for small local fixtures."""

    config = load_dataset_config("kitti-mots")
    config["splits"]["train"]["sequences"] = ["0000"]
    return config


def _write_frame(
    data_root: Path,
    config: dict[str, Any],
    *,
    sequence: str = "0000",
    frame_index: int = 0,
    split: str = "train",
    mask_size: tuple[int, int] = (4, 6),
) -> tuple[Path, Path | None]:
    """Create one RGB image and, for labeled splits, its native instance PNG."""

    split_config = config["splits"][split]
    root = data_root / config["root"]
    image_path = root / split_config["path"] / sequence / f"{frame_index:06d}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((4, 6, 3), frame_index % 256, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    mask_path = None
    if split_config["has_ground_truth"]:
        mask_path = root / split_config["annotations"] / sequence / image_path.name
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros(mask_size, dtype=np.uint16)
        mask[1:3, 2:4] = 1000
        assert cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def test_kitti_catalog_uses_headers_and_hashes_without_decoding_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kitti_config: dict[str, Any]
) -> None:
    image_path, mask_path = _write_frame(tmp_path, kitti_config)
    inspected: list[tuple[Path, bool]] = []

    def reject_decode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Catalog construction must not decode image or mask pixels.")

    def inspect(path: Path, include_image_size: bool):
        inspected.append((path, include_image_size))
        return inspect_catalog_file(path, include_image_size)

    monkeypatch.setattr(cv2, "imread", reject_decode)
    monkeypatch.setattr(cv2, "imdecode", reject_decode)
    catalog = catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path, metadata_resolver=inspect)

    assert len(catalog.samples) == 1
    sample = catalog.samples[0]
    assert sample.sample_id == "train:0000:0"
    assert sample.frame_index == 0
    assert sample.timestamp_s == 0.0
    assert sample.image_size == (4, 6)
    assert sample.image_ref == "data_tracking_image_2/training/image_02/0000/000000.png"
    assert sample.source_uri == image_path.as_uri()
    assert sample.source_frame_index is None
    assert not hasattr(sample, "frame")
    assert (image_path, True) in inspected
    assert (mask_path, True) in inspected
    assert catalog.metadata["dataset_id"] == "kitti-mots"
    assert catalog.metadata["layout"] == "kitti-mots"
    assert len(catalog.metadata["ground_truth_digest"]) == 64


def test_kitti_catalog_identity_is_portable_and_tracks_mask_content(
    tmp_path: Path, kitti_config: dict[str, Any]
) -> None:
    original_root = tmp_path / "original"
    relocated_root = tmp_path / "relocated"
    _write_frame(original_root, kitti_config)
    shutil.copytree(original_root / kitti_config["root"], relocated_root / kitti_config["root"])

    original = catalog_mot_dataset(kitti_config, split="train", data_root=original_root)
    relocated = catalog_mot_dataset(kitti_config, split="train", data_root=relocated_root)

    assert original.fingerprint == relocated.fingerprint
    assert original.metadata["ground_truth_digest"] == relocated.metadata["ground_truth_digest"]
    assert original.metadata["source_root_uri"] != relocated.metadata["source_root_uri"]
    assert original.samples[0].source_uri != relocated.samples[0].source_uri

    mask_path = relocated_root / kitti_config["root"] / "instances" / "0000" / "000000.png"
    changed_mask = np.zeros((4, 6), dtype=np.uint16)
    changed_mask[1:3, 2:5] = 1000
    assert cv2.imwrite(str(mask_path), changed_mask)
    changed = catalog_mot_dataset(kitti_config, split="train", data_root=relocated_root)

    assert changed.samples[0].source_sha256 == original.samples[0].source_sha256
    assert changed.metadata["ground_truth_digest"] != original.metadata["ground_truth_digest"]
    assert changed.fingerprint != original.fingerprint


def test_kitti_catalog_requires_a_mask_for_every_labeled_frame(tmp_path: Path, kitti_config: dict[str, Any]) -> None:
    _write_frame(tmp_path, kitti_config)
    _image, mask_path = _write_frame(tmp_path, kitti_config, frame_index=1)
    assert mask_path is not None
    mask_path.unlink()

    with pytest.raises(FileNotFoundError, match="000001.png"):
        catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path)


def test_kitti_catalog_rejects_mask_dimensions_that_differ_from_the_frame(
    tmp_path: Path, kitti_config: dict[str, Any]
) -> None:
    _write_frame(tmp_path, kitti_config, mask_size=(3, 6))

    with pytest.raises(ValueError, match="(?i)(dimension|size|shape)"):
        catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path)


def test_kitti_test_split_never_uses_training_annotations_with_the_same_sequence_id(
    tmp_path: Path, kitti_config: dict[str, Any]
) -> None:
    _write_frame(tmp_path, kitti_config)
    test_image, mask_path = _write_frame(tmp_path, kitti_config, split="test")
    assert mask_path is None
    first = catalog_mot_dataset(kitti_config, split="test", data_root=tmp_path)

    training_mask = tmp_path / kitti_config["root"] / "instances" / "0000" / "000000.png"
    training_mask.write_bytes(b"Unreadable training annotation must not affect the test split.")
    second = catalog_mot_dataset(kitti_config, split="test", data_root=tmp_path)

    assert len(first.samples) == 1
    assert first.samples[0].sample_id == "test:0000:0"
    assert first.samples[0].source_uri == test_image.as_uri()
    assert first.fingerprint == second.fingerprint


def test_kitti_catalog_respects_official_split_sequence_selection(tmp_path: Path, kitti_config: dict[str, Any]) -> None:
    _write_frame(tmp_path, kitti_config, sequence="0000")
    _image, excluded_mask = _write_frame(tmp_path, kitti_config, sequence="0002")
    assert excluded_mask is not None
    train = catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path)

    excluded_mask.write_bytes(b"Excluded validation annotations are not training inputs.")
    repeated = catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path)

    assert [sample.sequence_id for sample in train.samples] == ["0000"]
    assert train.fingerprint == repeated.fingerprint


def test_kitti_catalog_reports_missing_configured_sequences(tmp_path: Path, kitti_config: dict[str, Any]) -> None:
    _write_frame(tmp_path, kitti_config)
    kitti_config["splits"]["train"]["sequences"] = ["0000", "0001"]

    with pytest.raises(FileNotFoundError, match="0001"):
        catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path)


def test_kitti_fulltrain_discovers_all_sequences_without_a_subset(tmp_path: Path, kitti_config: dict[str, Any]) -> None:
    _write_frame(tmp_path, kitti_config, sequence="0002")
    _write_frame(tmp_path, kitti_config, sequence="0000")

    catalog = catalog_mot_dataset(kitti_config, split="fulltrain", data_root=tmp_path)

    assert [sample.sequence_id for sample in catalog.samples] == ["0000", "0002"]


def test_kitti_catalog_preserves_sparse_native_frame_numbers_and_capture_times(
    tmp_path: Path, kitti_config: dict[str, Any]
) -> None:
    for frame_index in (11, 4, 6, 5):
        _write_frame(tmp_path, kitti_config, frame_index=frame_index)

    catalog = catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path)

    assert [sample.frame_index for sample in catalog.samples] == [4, 5, 6, 11]
    assert [sample.sample_id for sample in catalog.samples] == [f"train:0000:{index}" for index in (4, 5, 6, 11)]
    assert [sample.timestamp_s for sample in catalog.samples] == pytest.approx([0.4, 0.5, 0.6, 1.1])


def test_kitti_fps_sampling_uses_capture_times_and_remaps_selected_frames(
    tmp_path: Path, kitti_config: dict[str, Any]
) -> None:
    for frame_index in (4, 5, 6, 11):
        _write_frame(tmp_path, kitti_config, frame_index=frame_index)

    catalog = catalog_mot_dataset(kitti_config, split="train", data_root=tmp_path, fps=5)

    assert [sample.frame_index for sample in catalog.samples] == [0, 1, 2]
    assert [sample.sample_id for sample in catalog.samples] == [f"train:0000:{index}" for index in range(3)]
    assert [Path(sample.image_ref).name for sample in catalog.samples] == ["000004.png", "000006.png", "000011.png"]
    assert [sample.timestamp_s for sample in catalog.samples] == pytest.approx([0.4, 0.6, 1.1])
    assert catalog.metadata["fps"] == 5.0
    assert catalog.metadata["frame_sampling"] == {"0000": [5, 7, 12]}


def test_kitti_catalog_ignores_appledouble_images_and_directories(tmp_path: Path, kitti_config: dict[str, Any]) -> None:
    image_path, _mask_path = _write_frame(tmp_path, kitti_config)
    (image_path.parent / "._000000.png").write_bytes(b"AppleDouble metadata")
    sidecar_sequence = image_path.parent.parent / "._0001"
    sidecar_sequence.mkdir()
    (sidecar_sequence / "000000.png").write_bytes(b"AppleDouble metadata")
    (image_path.parent.parent / ".DS_Store").write_bytes(b"Finder metadata")

    catalog = catalog_mot_dataset(kitti_config, split="fulltrain", data_root=tmp_path)

    assert [sample.sample_id for sample in catalog.samples] == ["fulltrain:0000:0"]
