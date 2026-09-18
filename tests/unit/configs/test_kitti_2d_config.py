"""KITTI presets select full or 2D inputs from one dataset."""

import pytest

from boxmot.datasets.config import dataset_modalities
from boxmot.engine.config.experiments import resolve_experiment_config


def test_kitti_2d_exposes_images_predictions_and_tracking_box_ground_truth() -> None:
    config = resolve_experiment_config("kitti-mots/2d")["dataset"]
    assert config["layout"] == "sequence"
    assert config["box_type"] == "aabb"
    assert config["root"] == "."
    assert set(dataset_modalities(config, "val")) == {"images", "detections_2d", "ground_truth"}
    assert dataset_modalities(config, "val")["ground_truth"]["format"] == "kitti-tracking-labels"
    assert set(dataset_modalities(config, "test")) == {"images", "detections_2d"}
    train, val = (set(config["splits"][name]["sequences"]) for name in ("train", "val"))
    assert (len(train), len(val)) == (12, 9)
    assert not train & val
    assert train | val == {f"{index:04d}" for index in range(21)}


@pytest.mark.parametrize("preset", ("full", "2d", "2d-lmbn-n-duke"))
@pytest.mark.parametrize("split", ("train", "val", "fulltrain"))
def test_kitti_presets_use_dataset_default_and_accept_split_overrides(preset: str, split: str) -> None:
    config = resolve_experiment_config(f"kitti-mots/{preset}", mode="eval")
    assert config["dataset"]["split"] == "val"
    overridden = resolve_experiment_config(f"kitti-mots/{preset}", split=split, mode="eval")
    assert overridden["dataset"]["split"] == split
    assert overridden["dataset"]["id"] == "kitti-mots"
