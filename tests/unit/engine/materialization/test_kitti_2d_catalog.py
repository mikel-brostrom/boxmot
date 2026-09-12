"""KITTI image catalogs preserve native frames and text annotation provenance."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from boxmot.datasets.config import dataset_modalities, load_dataset_config
from boxmot.engine.materialization.catalog import catalog_mot_dataset, inspect_catalog_file


@pytest.fixture
def kitti_2d_config() -> dict[str, Any]:
    """Use native image/label paths with no masks, calibration, or spatial inputs."""
    config = load_dataset_config("kitti-mots")
    config["id"] = "kitti-2d"
    config["root"] = "KITTI"
    config["modalities"]["ground_truth"] = {
        "format": "kitti-tracking-labels",
        "paths": ["{partition}/label_02/{sequence}.txt"],
        "options": {},
    }
    config["splits"]["train"]["sequences"] = ["0000"]
    return config


def _write_sequence(
    data_root: Path,
    config: dict[str, Any],
    *,
    sequence: str = "0000",
    frames: tuple[int, ...] = (0, 1),
    split: str = "train",
) -> tuple[tuple[Path, ...], Path | None]:
    """Write real KITTI path shapes while leaving unavailable 3D fields unset."""
    root = data_root / config["root"]
    modalities = dataset_modalities(config, split)
    substitutions = {"partition": config["splits"][split]["partition"], "split": split, "sequence": sequence}
    image_root = root / modalities["images"]["paths"][0].format(**substitutions)
    image_root.mkdir(parents=True)
    images = tuple(image_root / f"{frame:06d}.png" for frame in frames)
    for image_path in images:
        assert cv2.imwrite(str(image_path), np.zeros((4, 6, 3), dtype=np.uint8))
    annotation = None
    if "ground_truth" in modalities:
        annotation = root / modalities["ground_truth"]["paths"][0].format(**substitutions)
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            "".join(f"{frame} 7 Car 0 0 -10 1 1 5 3 -1 -1 -1 -1000 -1000 -1000 -10\n" for frame in frames),
            encoding="utf-8",
        )
    return images, annotation


def test_kitti_2d_catalog_hashes_sequence_annotations_once_without_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kitti_2d_config: dict[str, Any]
) -> None:
    images, annotation = _write_sequence(tmp_path, kitti_2d_config)
    inspected: list[tuple[Path, bool]] = []

    def reject_decode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Catalog creation must not decode images or annotations.")

    def inspect(path: Path, include_image_size: bool):
        inspected.append((path, include_image_size))
        return inspect_catalog_file(path, include_image_size)

    monkeypatch.setattr(cv2, "imread", reject_decode)
    monkeypatch.setattr(cv2, "imdecode", reject_decode)
    catalog = catalog_mot_dataset(kitti_2d_config, split="train", data_root=tmp_path, metadata_resolver=inspect)

    assert inspected == [*((path, True) for path in images), (annotation, False)]
    assert [sample.sample_id for sample in catalog.samples] == ["train:0000:0", "train:0000:1"]
    assert catalog.samples[0].image_ref == "data_tracking_image_2/training/image_02/0000/000000.png"
    assert catalog.metadata["modalities"]["ground_truth"]["format"] == "kitti-tracking-labels"


def test_kitti_2d_catalog_identity_is_portable_and_tracks_label_content(
    tmp_path: Path, kitti_2d_config: dict[str, Any]
) -> None:
    original_root, relocated_root = tmp_path / "original", tmp_path / "relocated"
    _write_sequence(original_root, kitti_2d_config)
    shutil.copytree(original_root / kitti_2d_config["root"], relocated_root / kitti_2d_config["root"])
    original = catalog_mot_dataset(kitti_2d_config, split="train", data_root=original_root)
    relocated = catalog_mot_dataset(kitti_2d_config, split="train", data_root=relocated_root)
    assert original.fingerprint == relocated.fingerprint
    assert original.metadata["ground_truth_digest"] == relocated.metadata["ground_truth_digest"]

    annotation = relocated_root / kitti_2d_config["root"] / "training/label_02/0000.txt"
    annotation.write_text(annotation.read_text(encoding="utf-8").replace("1 1 5 3", "1 1 4 3"), encoding="utf-8")
    changed = catalog_mot_dataset(kitti_2d_config, split="train", data_root=relocated_root)
    assert changed.samples[0].source_sha256 == original.samples[0].source_sha256
    assert changed.fingerprint != original.fingerprint
    assert changed.metadata["ground_truth_digest"] != original.metadata["ground_truth_digest"]


def test_kitti_2d_catalog_preserves_native_frames_and_fps_mapping(
    tmp_path: Path, kitti_2d_config: dict[str, Any]
) -> None:
    _write_sequence(tmp_path, kitti_2d_config, frames=(4, 5, 6, 11))
    native = catalog_mot_dataset(kitti_2d_config, split="train", data_root=tmp_path)
    assert [sample.frame_index for sample in native.samples] == [4, 5, 6, 11]
    assert [sample.timestamp_s for sample in native.samples] == pytest.approx([0.4, 0.5, 0.6, 1.1])

    sampled = catalog_mot_dataset(kitti_2d_config, split="train", data_root=tmp_path, fps=5)
    assert [sample.frame_index for sample in sampled.samples] == [0, 1, 2]
    assert [Path(sample.image_ref).name for sample in sampled.samples] == ["000004.png", "000006.png", "000011.png"]
    assert sampled.metadata["frame_sampling"] == {"0000": [5, 7, 12]}
    assert sampled.metadata["ground_truth_digest"] != native.metadata["ground_truth_digest"]


def test_kitti_2d_test_split_ignores_training_annotations(tmp_path: Path, kitti_2d_config: dict[str, Any]) -> None:
    _images, training_annotation = _write_sequence(tmp_path, kitti_2d_config)
    test_images, annotation = _write_sequence(tmp_path, kitti_2d_config, split="test")
    assert annotation is None
    first = catalog_mot_dataset(kitti_2d_config, split="test", data_root=tmp_path)
    training_annotation.write_bytes(b"Unrelated invalid training annotations")
    second = catalog_mot_dataset(kitti_2d_config, split="test", data_root=tmp_path)
    assert first.fingerprint == second.fingerprint
    assert first.samples[0].source_uri == test_images[0].as_uri()
    assert set(first.metadata["modalities"]) == {"images"}


def test_kitti_2d_catalog_requires_declared_annotation_file(tmp_path: Path, kitti_2d_config: dict[str, Any]) -> None:
    _images, annotation = _write_sequence(tmp_path, kitti_2d_config)
    annotation.unlink()
    with pytest.raises(ValueError, match="ground_truth requires a file"):
        catalog_mot_dataset(kitti_2d_config, split="train", data_root=tmp_path)


def test_kitti_2d_catalog_does_not_hash_excluded_sequence_annotations(
    tmp_path: Path, kitti_2d_config: dict[str, Any]
) -> None:
    _write_sequence(tmp_path, kitti_2d_config)
    _images, excluded = _write_sequence(tmp_path, kitti_2d_config, sequence="0002")
    first = catalog_mot_dataset(kitti_2d_config, split="train", data_root=tmp_path)
    excluded.write_bytes(b"Unrelated validation annotations")
    second = catalog_mot_dataset(kitti_2d_config, split="train", data_root=tmp_path)
    assert [sample.sequence_id for sample in first.samples] == ["0000", "0000"]
    assert first.fingerprint == second.fingerprint
