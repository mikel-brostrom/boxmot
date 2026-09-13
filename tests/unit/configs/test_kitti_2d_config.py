"""KITTI's image-only view resolves ordinary eval and tuning experiments."""

from pathlib import Path

import pytest

from boxmot.datasets.config import dataset_modalities, load_dataset_config
from boxmot.datasets.inputs import resolve_sensor_dataset_config_path
from boxmot.engine.config.experiments import resolve_experiment_config, resolve_matching_experiment_path


def test_kitti_2d_exposes_only_images_and_tracking_box_ground_truth() -> None:
    config = load_dataset_config("kitti-2d")
    assert config["layout"] == "sequence"
    assert config["box_type"] == "aabb"
    assert config["root"] == "KITTI"
    assert set(dataset_modalities(config, "val")) == {"images", "ground_truth"}
    assert dataset_modalities(config, "val")["ground_truth"]["format"] == "kitti-tracking-labels"
    assert set(dataset_modalities(config, "test")) == {"images"}
    assert resolve_sensor_dataset_config_path("kitti-2d") is None
    train, val = (set(config["splits"][name]["sequences"]) for name in ("train", "val"))
    assert (len(train), len(val)) == (12, 9)
    assert not train & val
    assert train | val == {f"{index:04d}" for index in range(21)}


@pytest.mark.parametrize("split", ("train", "val"))
@pytest.mark.parametrize("reid", (None, "osnet-x0-25-msmt17"))
@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_direct_kitti_2d_selectors_find_one_authored_class_mapping(split: str, reid: str | None, mode: str) -> None:
    path = resolve_matching_experiment_path(dataset="kitti-2d", detector="yolo26n", reid=reid, split=split, mode=mode)
    config = resolve_experiment_config(path, mode=mode)
    assert config["dataset"]["id"] == "kitti-2d"
    assert config["dataset"]["split"] == split
    assert [(item["dataset_id"], item["detector_id"]) for item in config["evaluation"]["classes"]] == [(1, 2), (2, 0)]
    assert Path(path).parent.name == "kitti-2d"
