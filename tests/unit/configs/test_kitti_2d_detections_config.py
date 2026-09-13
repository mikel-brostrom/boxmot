"""Saved KITTI boxes select channels independently of image-only detector experiments."""

from pathlib import Path

import pytest
import yaml

from boxmot.datasets.config import (
    ConfigurationError,
    dataset_modalities,
    load_dataset_config,
    resolve_dataset_storage_root,
)
from boxmot.datasets.inputs import load_dataset_inputs
from boxmot.engine.config.experiments import resolve_experiment_path


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


def test_saved_kitti_multimodal_layout_selects_the_same_2d_inputs() -> None:
    config = load_dataset_config("kitti-mots-2d")
    standard = load_dataset_config("kitti-2d-detections")
    modalities = dataset_modalities(config, "val")
    standard_modalities = dataset_modalities(standard, "val")

    assert config["id"] == "kitti-mots-2d"
    assert config["root"] == "."
    assert config["splits"] == standard["splits"]
    assert config["classes"] == standard["classes"]
    assert config["fps"] == standard["fps"] == 10
    assert set(modalities) == {"images", "detections_2d", "ground_truth"}
    assert modalities["images"]["paths"] == ["sequences/{partition}/{sequence}/images"]
    assert standard_modalities["images"]["paths"] == ["{partition}/image_02/{sequence}"]
    assert modalities["detections_2d"] == standard_modalities["detections_2d"]
    assert modalities["ground_truth"] == standard_modalities["ground_truth"]
    assert set(dataset_modalities(config, "test")) == {"images", "detections_2d"}


def test_saved_kitti_multimodal_layout_uses_explicit_data_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "kitti-mots"
    images = dataset_root / "sequences/training/0002/images"
    detections = dataset_root / "predictions/trackrcnn/training/0002.txt"
    ground_truth = dataset_root / "training/label_02/0002.txt"
    images.mkdir(parents=True)
    for path in (detections, ground_truth):
        path.parent.mkdir(parents=True)
        path.touch()

    config = load_dataset_config("kitti-mots-2d")
    dataset = load_dataset_inputs("kitti-mots-2d", split="val", sequence_names=("0002",), data_root=dataset_root)

    assert resolve_dataset_storage_root(config, dataset_root) == dataset_root
    assert dataset.root == dataset_root
    assert len(dataset.sequences) == 1
    modalities = dataset.sequences[0].modalities
    assert modalities["images"].paths == (images,)
    assert modalities["detections_2d"].paths == (detections,)
    assert modalities["ground_truth"].paths == (ground_truth,)


@pytest.mark.parametrize("split", ("train", "val"))
@pytest.mark.parametrize("with_reid", (False, True))
def test_saved_kitti_2d_experiment_presets_select_existing_predictions(split: str, with_reid: bool) -> None:
    suffix = "-osnet" if with_reid else ""
    path = resolve_experiment_path(f"kitti-2d/{split}-trackrcnn{suffix}")
    experiment = yaml.safe_load(path.read_text())

    expected = {"dataset": {"ref": "kitti-mots-2d", "split": split}}
    if with_reid:
        expected["reid"] = {"ref": "osnet-x0-25-msmt17"}
    assert experiment == expected


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
