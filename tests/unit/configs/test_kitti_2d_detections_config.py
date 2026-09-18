"""Saved KITTI boxes select channels independently of the full dataset."""

from pathlib import Path

import pytest
import yaml

from boxmot.datasets.config import (
    ConfigurationError,
    dataset_modalities,
    load_dataset_config,
    resolve_dataset_storage_root,
)
from boxmot.engine.config.datasets import load_saved_2d_evaluation_inputs
from boxmot.engine.config.experiments import resolve_experiment_config, resolve_experiment_path


def test_experiments_select_views_without_mutating_the_dataset_inventory() -> None:
    inventory = load_dataset_config("kitti-mots")
    before = dataset_modalities(inventory, "val")
    saved = resolve_experiment_config("kitti-mots/2d")["dataset"]
    full = resolve_experiment_config("kitti-mots/full")["dataset"]
    modalities = dataset_modalities(saved, "val")

    assert saved["id"] == full["id"] == inventory["id"]
    assert saved["root"] == full["root"] == "."
    assert set(modalities) == {"images", "detections_2d", "ground_truth"}
    assert modalities["detections_2d"]["options"] == {"load_masks": False}
    assert modalities["ground_truth"]["format"] == "kitti-tracking-labels"
    assert dataset_modalities(full, "val") == before
    assert set(dataset_modalities(saved, "test")) == {"images", "detections_2d"}
    assert dataset_modalities(inventory, "val") == before
    assert before["detections_2d"]["options"]["load_masks"] is True
    assert "detections_3d" in before


def test_saved_kitti_multimodal_layout_uses_explicit_data_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "kitti-mots"
    images = dataset_root / "sequences/training/0002/images"
    detections = dataset_root / "predictions/trackrcnn/training/0002.txt"
    ground_truth = dataset_root / "training/label_02/0002.txt"
    images.mkdir(parents=True)
    for path in (detections, ground_truth):
        path.parent.mkdir(parents=True)
        path.touch()

    config = load_dataset_config("kitti-mots")
    dataset = load_saved_2d_evaluation_inputs(
        "kitti-mots", experiment="kitti-mots/2d", split="val", sequence_names=("0002",), data_root=dataset_root
    )

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
    suffix = "-lmbn-n-duke" if with_reid else ""
    path = resolve_experiment_path(f"kitti-mots/2d{suffix}")
    config = resolve_experiment_config(path, split=split, mode="eval")

    assert config["dataset"]["split"] == split
    assert config["detector"] is None
    assert set(dataset_modalities(config["dataset"], split)) == {"images", "detections_2d", "ground_truth"}
    if with_reid:
        assert config["reid"]["id"] == "lmbn-n-duke"
    else:
        assert config["reid"] is None


@pytest.mark.parametrize("load_masks", (True, False))
def test_trackrcnn_config_accepts_explicit_boolean_mask_selection(tmp_path: Path, load_masks: bool) -> None:
    source = load_dataset_config("kitti-mots")["config_path"]
    payload = yaml.safe_load(source.read_text())
    payload["modalities"]["detections_2d"]["options"]["load_masks"] = load_masks
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload))

    assert dataset_modalities(load_dataset_config(path), "val")["detections_2d"]["options"] == {
        "load_masks": load_masks
    }


@pytest.mark.parametrize("load_masks", (None, 0, 1, "false", [], {}))
def test_trackrcnn_config_rejects_nonboolean_mask_selection(tmp_path: Path, load_masks: object) -> None:
    source = load_dataset_config("kitti-mots")["config_path"]
    payload = yaml.safe_load(source.read_text())
    payload["modalities"]["detections_2d"]["options"]["load_masks"] = load_masks
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ConfigurationError, match="options.load_masks must be a boolean"):
        load_dataset_config(path)


@pytest.mark.parametrize(
    "selection,message",
    [
        ([], "non-empty mapping"),
        ({}, "non-empty mapping"),
        ({"unknown": {}}, "unknown modalities"),
        ({"images": "images"}, "only source and options"),
        ({"images": {"path": "another-path"}}, "only source and options"),
        ({"images": {"source": "missing"}}, "undeclared dataset source"),
        ({"images": {"source": "detections_2d"}}, "format must be"),
        ({"detections_2d": {"options": {"load_masks": 0}}}, "must be a boolean"),
    ],
)
def test_experiment_input_selection_rejects_invalid_sources_and_options(
    tmp_path: Path, selection: object, message: str
) -> None:
    """Experiments choose existing sources without redefining their locations."""
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({"dataset": {"ref": "kitti-mots", "modalities": selection}}))
    with pytest.raises(ConfigurationError, match=message):
        resolve_experiment_config(path)
