"""Validate the KITTI MOTS annotation and sequence-selection contract."""

from pathlib import Path

import pytest
import yaml

from boxmot.datasets.config import ConfigurationError, load_dataset_config, resolve_dataset_config_path
from boxmot.engine.experiment_config import resolve_experiment_config


def _write_profile(tmp_path: Path, *, split_changes: dict | None = None, box_type: str = "aabb") -> Path:
    """Write a modified copy of the built-in profile."""

    payload = yaml.safe_load(resolve_dataset_config_path("kitti-mots").read_text())
    payload["format"]["box_type"] = box_type
    if split_changes is not None:
        payload["splits"]["train"].update(split_changes)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


@pytest.mark.parametrize("sequences", [[], None, "0000", [0], ["../0000"], ["0000", "0000"]])
def test_dataset_rejects_ambiguous_or_unsafe_sequence_selections(tmp_path: Path, sequences: object) -> None:
    path = _write_profile(tmp_path, split_changes={"sequences": sequences})

    with pytest.raises(ConfigurationError, match="sequences"):
        load_dataset_config(path)


@pytest.mark.parametrize("split_changes", [{"annotations": None}, {"has_ground_truth": False}])
def test_kitti_mots_annotations_match_ground_truth_availability(tmp_path: Path, split_changes: dict) -> None:
    path = _write_profile(tmp_path, split_changes=split_changes)

    with pytest.raises(ConfigurationError, match="annotations exactly when has_ground_truth"):
        load_dataset_config(path)


def test_kitti_mots_rejects_oriented_box_profile(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="require box_type"):
        load_dataset_config(_write_profile(tmp_path, box_type="obb"))


def _write_experiment(tmp_path: Path) -> Path:
    """Compose a detector with KITTI's native class taxonomy."""

    path = tmp_path / "train-kitti.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"ref": "kitti-mots", "split": "train"},
                "detector": {"ref": "yolo26n", "checkpoint": "default"},
                "evaluation": {"class_map": {"car": "car", "pedestrian": "person"}},
            }
        )
    )
    return path


def test_kitti_mots_experiment_preserves_annotations_selection_and_native_class_map(tmp_path: Path) -> None:
    config = resolve_experiment_config(_write_experiment(tmp_path), mode="materialize")

    assert config["dataset"]["splits"]["train"]["annotations"] == "instances"
    assert len(config["dataset"]["splits"]["train"]["sequences"]) == 12
    assert [(entry["detector_id"], entry["dataset_id"]) for entry in config["evaluation"]["classes"]] == [
        (2, 1),
        (0, 2),
    ]


@pytest.mark.parametrize("mode", ["eval", "evaluation", "tune", "research"])
def test_kitti_mots_resolves_for_evaluation_and_tuning(tmp_path: Path, mode: str) -> None:
    config = resolve_experiment_config(_write_experiment(tmp_path), mode=mode)

    assert config["dataset"]["layout"] == "kitti-mots"
    assert config["dataset"]["has_ground_truth"] is True
