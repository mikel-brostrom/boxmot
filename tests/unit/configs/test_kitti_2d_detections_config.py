"""Saved KITTI boxes select channels independently of image-only detector experiments."""

from pathlib import Path

import pytest
import yaml

from boxmot.datasets.config import ConfigurationError, dataset_modalities, load_dataset_config


def test_saved_kitti_2d_declares_images_boxes_and_tracking_gt() -> None:
    saved = load_dataset_config("kitti-2d-detections")
    image_only = load_dataset_config("kitti-2d")
    modalities = dataset_modalities(saved, "val")

    assert saved["id"] == "kitti-2d-detections"
    assert saved["root"] == "KITTI"
    assert saved["splits"] == image_only["splits"]
    assert saved["fps"] == image_only["fps"] == 10
    assert set(modalities) == {"images", "detections_2d", "ground_truth"}
    assert modalities["detections_2d"] == {
        "format": "trackrcnn",
        "paths": ["predictions/trackrcnn/{partition}/{sequence}.txt"],
        "options": {"load_masks": False},
    }
    assert modalities["ground_truth"]["format"] == "kitti-tracking-labels"
    assert set(dataset_modalities(saved, "test")) == {"images", "detections_2d"}
    assert set(dataset_modalities(image_only, "val")) == {"images", "ground_truth"}


@pytest.mark.parametrize("load_masks", (True, False))
def test_trackrcnn_config_accepts_explicit_boolean_mask_selection(tmp_path: Path, load_masks: bool) -> None:
    source = load_dataset_config("kitti-2d-detections")["config_path"]
    payload = yaml.safe_load(source.read_text())
    payload["modalities"]["detections_2d"]["options"]["load_masks"] = load_masks
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload))

    assert dataset_modalities(load_dataset_config(path), "val")["detections_2d"]["options"] == {
        "load_masks": load_masks
    }


@pytest.mark.parametrize("load_masks", (None, 0, 1, "false", [], {}))
def test_trackrcnn_config_rejects_nonboolean_mask_selection(tmp_path: Path, load_masks: object) -> None:
    source = load_dataset_config("kitti-2d-detections")["config_path"]
    payload = yaml.safe_load(source.read_text())
    payload["modalities"]["detections_2d"]["options"]["load_masks"] = load_masks
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ConfigurationError, match="options.load_masks must be a boolean"):
        load_dataset_config(path)
