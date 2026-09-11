"""Materialize configured image paths without assuming benchmark directories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from boxmot.datasets.config import ConfigurationError, load_dataset_config
from boxmot.datasets.inputs import load_dataset_inputs, resolve_dataset_inputs
from boxmot.engine.materialization.catalog import catalog_mot_dataset, resolve_dataset_split_root


def _profile(
    root: Path,
    *,
    image_path: str = "sequences/{sequence}/images",
    image_extension: str = ".png",
) -> Path:
    """Write two real frames with separate instance PNGs and an authored timeline."""

    payload: dict[str, Any] = {
        "id": "custom-camera",
        "format": {"layout": "sequence", "box_type": "aabb"},
        "storage": {"root": "."},
        "classes": {"target": {"car": 1, "pedestrian": 2}, "ignore": {"ignore": 10}},
        "fps": 12.5,
        "default_split": "validation",
        "modalities": {
            "images": {"format": "image-directory", "path": image_path},
            "ground_truth": {
                "format": "instance-png",
                "path": "labels/{partition}/{sequence}",
                "options": {"class_divisor": 1000, "background_id": 0, "ignore_ids": [10000]},
            },
        },
        "splits": {"validation": {"partition": "recordings", "sequences": ["clip-01"]}},
    }
    path = root / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    frames = root / image_path.format(partition="recordings", sequence="clip-01", split="validation")
    labels = root / "labels/recordings/clip-01"
    frames.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(2):
        assert cv2.imwrite(str(frames / f"{index:06d}{image_extension}"), np.zeros((8, 12, 3), dtype=np.uint8))
        annotation = np.zeros((8, 12), dtype=np.uint16)
        annotation[2:6, 3:8] = 1001
        assert cv2.imwrite(str(labels / f"{index:06d}.png"), annotation)
    return path


def test_sequence_placeholder_inside_directory_name_uses_the_existing_parent(tmp_path: Path) -> None:
    config = load_dataset_config(_profile(tmp_path, image_path="images/drive-{sequence}"))

    catalog = catalog_mot_dataset(config)

    assert resolve_dataset_split_root(config, "validation") == tmp_path / "images"
    assert [sample.sequence_id for sample in catalog.samples] == ["clip-01", "clip-01"]
    assert catalog.samples[0].image_ref == "images/drive-clip-01/000000.png"


def test_nested_image_directories_pair_jpeg_images_with_png_ground_truth(tmp_path: Path) -> None:
    config = load_dataset_config(_profile(tmp_path, image_extension=".jpg"))

    catalog = catalog_mot_dataset(config)

    assert [sample.sample_id for sample in catalog.samples] == ["validation:clip-01:0", "validation:clip-01:1"]
    assert [sample.image_ref for sample in catalog.samples] == [
        "sequences/clip-01/images/000000.jpg",
        "sequences/clip-01/images/000001.jpg",
    ]
    assert [sample.timestamp_s for sample in catalog.samples] == pytest.approx([0.0, 0.08])
    before = catalog.fingerprint
    labels = np.zeros((8, 12), dtype=np.uint16)
    assert cv2.imwrite(str(tmp_path / "labels/recordings/clip-01/000001.png"), labels)
    assert catalog_mot_dataset(config).fingerprint != before


def test_perception_catalog_does_not_require_unused_sensor_inputs(tmp_path: Path) -> None:
    path = _profile(tmp_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["modalities"].update(
        detections_2d={"format": "trackrcnn", "path": "missing/image/{sequence}.txt"},
        detections_3d={
            "format": "kitti-detections",
            "paths": ["missing/first/{sequence}", "missing/second/{sequence}"],
        },
        calibration={"format": "kitti-p2", "path": "missing/calibration/{sequence}.txt"},
        poses={"format": "camera-to-world-npy", "path": "missing/poses/{sequence}.npy"},
    )
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert len(catalog_mot_dataset(load_dataset_config(path)).samples) == 2
    with pytest.raises(ConfigurationError, match="requires a file"):
        load_dataset_inputs(path)


def test_authored_fps_controls_timestamps_without_reading_undeclared_sidecars(tmp_path: Path) -> None:
    config = load_dataset_config(_profile(tmp_path))
    baseline = catalog_mot_dataset(config)
    images = tmp_path / "sequences/clip-01/images"
    (images / "seqinfo.ini").write_text("[Sequence]\nframeRate=30\n", encoding="utf-8")
    (images / "timestamps.csv").write_text("frame_index,timestamp_s\n0,0\n1,0.0333333\n", encoding="utf-8")

    catalog = catalog_mot_dataset(config)

    assert [sample.timestamp_s for sample in catalog.samples] == pytest.approx([0.0, 0.08])
    assert catalog.fingerprint == baseline.fingerprint


@pytest.mark.parametrize("role", ["images", "ground_truth"])
def test_normalized_catalog_inputs_reject_multiple_paths_instead_of_ignoring_them(tmp_path: Path, role: str) -> None:
    config = load_dataset_config(_profile(tmp_path))
    config["modalities"][role]["paths"] *= 2

    with pytest.raises(ConfigurationError, match=f"{role} requires exactly one input path"):
        resolve_dataset_inputs(config)
    with pytest.raises(ConfigurationError, match=f"{role} requires exactly one input path"):
        catalog_mot_dataset(config)
