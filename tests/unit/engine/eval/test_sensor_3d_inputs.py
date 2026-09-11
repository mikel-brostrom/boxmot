"""Select sensor scoring annotations without requiring unused label payloads."""

from __future__ import annotations

import shutil
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
import yaml

from boxmot.datasets.config import load_dataset_config
from boxmot.engine.config.datasets import load_sensor_evaluation_inputs, validate_sensor_workflow_inputs
from boxmot.trackers.common.specs import TrackerSpec
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


def _declare_3d_labels(dataset_path: Path, *, create: bool = True) -> Path:
    """Add independent spatial labels to the portable sensor fixture."""
    config = yaml.safe_load(dataset_path.read_text())
    config["modalities"]["ground_truth_3d"] = {
        "format": "kitti-tracking-labels",
        "path": "sequences/{partition}/{sequence}/labels_3d.txt",
    }
    config["modalities"]["ground_truth_objects"] = {
        "format": "kitti-object-labels",
        "path": "sequences/{partition}/{sequence}/object_labels",
    }
    dataset_path.write_text(yaml.safe_dump(config))
    labels = dataset_path.parent / "sequences/training/0002/labels_3d.txt"
    if create:
        labels.write_text("0 0 Car 0 0 0 14 8 23 15 3 1 4 -5 0 20 0\n")
        objects = labels.parent / "object_labels"
        objects.mkdir()
        for image in (labels.parent / "images").glob("*.png"):
            (objects / image.with_suffix(".txt").name).write_text("Car 0.25 0 0 14 8 23 15 3 1 4 -5 0 20 0\n")
    return labels


@pytest.mark.parametrize(("eval_3d", "calibrate_kf"), ((False, False), (False, True), (True, False), (True, True)))
def test_sensor_inputs_resolve_only_consumed_annotations(tmp_path: Path, eval_3d: bool, calibrate_kf: bool) -> None:
    """Missing unused labels cannot block tracking or the selected metric."""
    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset, create=eval_3d or calibrate_kf)
    if eval_3d:
        data.ground_truth.rmdir()
    elif calibrate_kf:
        (tmp_path / "sequences/training/0002/object_labels").rmdir()

    validate_sensor_workflow_inputs(
        data.dataset, TrackerSpec("eagermot"), mode="eval", eval_3d=eval_3d, calibrate_kf=calibrate_kf
    )
    dataset = load_sensor_evaluation_inputs(data.dataset, eval_3d=eval_3d, calibrate_kf=calibrate_kf)

    expected = {"images", "detections_2d", "detections_3d", "calibration", "poses"}
    if not eval_3d:
        expected.add("ground_truth")
    if eval_3d or calibrate_kf:
        expected.add("ground_truth_3d")
    if eval_3d:
        expected.add("ground_truth_objects")
    assert set(dataset.sequences[0].modalities) == expected
    assert dataset.sequence_names == ("0002",)
    assert dataset.split == "val"


@pytest.mark.parametrize("entrypoint", ("validate", "load"))
@pytest.mark.parametrize("option", ("eval_3d", "calibrate_kf"))
def test_sensor_spatial_labels_required_only_when_consumed(tmp_path: Path, entrypoint: str, option: str) -> None:
    """Each spatial workflow explains which annotation declaration is missing."""
    data = sensor_dataset_fixture(tmp_path)
    flags = {option: True}
    message = f"--{option.replace('_', '-')} requires 3D ground truth with track IDs"
    with pytest.raises(ValueError, match=message):
        if entrypoint == "validate":
            validate_sensor_workflow_inputs(data.dataset, TrackerSpec("eagermot"), mode="eval", **flags)
        else:
            load_sensor_evaluation_inputs(data.dataset, **flags)


@pytest.mark.parametrize("entrypoint", ("validate", "load"))
def test_3d_scoring_requires_object_ground_truth_before_payload_validation(tmp_path: Path, entrypoint: str) -> None:
    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset)
    config = yaml.safe_load(data.dataset.read_text())
    del config["modalities"]["ground_truth_objects"]
    data.dataset.write_text(yaml.safe_dump(config))
    data.reader_paths["poses"].unlink()

    with pytest.raises(ValueError, match="--eval-3d requires per-image KITTI object ground truth"):
        if entrypoint == "validate":
            validate_sensor_workflow_inputs(data.dataset, TrackerSpec("eagermot"), mode="eval", eval_3d=True)
        else:
            load_sensor_evaluation_inputs(data.dataset, eval_3d=True)


@pytest.mark.parametrize("declared_ground_truth", (False, True))
def test_3d_scoring_accepts_no_mask_ground_truth_declaration(tmp_path: Path, declared_ground_truth: bool) -> None:
    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset)
    config = yaml.safe_load(data.dataset.read_text())
    del config["modalities"]["ground_truth"]
    if not declared_ground_truth:
        del config["splits"]["val"]["has_ground_truth"]
    data.dataset.write_text(yaml.safe_dump(config))
    data.ground_truth.rmdir()

    validate_sensor_workflow_inputs(data.dataset, TrackerSpec("eagermot"), mode="eval", eval_3d=True)
    dataset = load_sensor_evaluation_inputs(data.dataset, eval_3d=True)

    assert "ground_truth" not in dataset.sequences[0].modalities
    assert load_dataset_config(data.dataset)["splits"]["val"]["has_ground_truth"] is True
    with pytest.raises(ValueError, match="mask evaluation requires ground_truth"):
        load_sensor_evaluation_inputs(data.dataset)


def test_3d_scoring_does_not_apply_mots_annotation_conventions(tmp_path: Path) -> None:
    """Unused PNG encoding and ignored class IDs cannot constrain spatial scoring."""
    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset)
    config = yaml.safe_load(data.dataset.read_text())
    config["modalities"]["ground_truth"]["options"]["class_divisor"] = 100
    config["classes"]["ignore"] = {"ignore": 99}
    data.dataset.write_text(yaml.safe_dump(config))

    assert load_sensor_evaluation_inputs(data.dataset, eval_3d=True).classes["ignore"]["id"] == 99
    with pytest.raises(ValueError, match="MOTS evaluation supports only ignored class ID 10"):
        load_sensor_evaluation_inputs(data.dataset)


@pytest.mark.parametrize("targets", ({"car": 1}, {"car": 2, "pedestrian": 1}, {"bus": 1, "pedestrian": 2}))
def test_3d_scoring_requires_the_replay_class_profiles(tmp_path: Path, targets: dict[str, int]) -> None:
    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset)
    config = yaml.safe_load(data.dataset.read_text())
    config["classes"]["target"] = targets
    data.dataset.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="3D evaluation requires classes.target car: 1 and pedestrian: 2"):
        load_sensor_evaluation_inputs(data.dataset, eval_3d=True)


@pytest.mark.parametrize("eval_3d", (False, True))
def test_scoring_selection_keeps_tracking_input_requirements(tmp_path: Path, eval_3d: bool) -> None:
    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset)
    data.reader_paths["poses"].unlink()

    with pytest.raises(ValueError, match="poses.npy"):
        load_sensor_evaluation_inputs(data.dataset, eval_3d=eval_3d)


def test_cached_3d_scoring_never_reads_unused_instance_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache only selected GT while retaining masks carried by tracking detections."""
    import numpy as np
    from PIL import Image

    from boxmot.datasets import sensor_cache

    data = sensor_dataset_fixture(tmp_path)
    Image.new("RGB", (48, 24)).save(data.reader_paths["images"] / "000000.png")
    data.reader_paths["calibration"].write_text("P2: 20 0 24 0 0 20 12 0 0 0 1 0\n")
    np.save(data.reader_paths["poses"], np.eye(4)[None])
    _declare_3d_labels(data.dataset)
    shutil.rmtree(data.ground_truth)

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("3D scoring must not decode unused instance PNG annotations.")

    monkeypatch.setattr(sensor_cache, "read_instance_png", unexpected)
    dataset = load_sensor_evaluation_inputs(data.dataset, eval_3d=True)
    path = sensor_cache.prepare_sensor_sequence(dataset, "0002")
    assert sensor_cache.prepare_sensor_sequence(dataset, "0002") == path
    with closing(sensor_cache.open_sensor_sequence(path)) as sequence:
        assert sequence.ground_truth(0) is None
        labels = sequence.ground_truth_3d()
        assert labels is not None
        assert labels.track_ids.tolist() == [0]
        assert sequence.ground_truth_objects().frame_rows[0][0].split()[1] == "0.25"
        assert sequence[0].detections.masks is not None
