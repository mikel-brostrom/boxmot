"""Authored KITTI experiments select saved inputs for compatible trackers."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.datasets.config import dataset_modalities
from boxmot.engine.cli import boxmot
from boxmot.engine.config.datasets import load_sensor_evaluation_inputs
from boxmot.engine.config.experiments import resolve_experiment_config, resolve_sensor_experiment
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


@pytest.mark.parametrize("split", ("train", "val"))
def test_kitti_experiments_select_distinct_tracking_inputs(split: str) -> None:
    """Both views share frames and predictions but declare different capabilities."""
    full = resolve_experiment_config("kitti-mots/full", split=split, mode="eval")
    boxes = resolve_experiment_config("kitti-mots/2d", split=split, mode="eval")
    full_inputs = dataset_modalities(full["dataset"], split)
    box_inputs = dataset_modalities(boxes["dataset"], split)

    assert full["dataset"]["splits"][split]["sequences"] == boxes["dataset"]["splits"][split]["sequences"]
    car_predictions = "pointgnn-car-t2" if split == "val" else "pointgnn-car-t3"
    assert full_inputs["detections_3d"]["paths"][0] == f"predictions/{car_predictions}/{{partition}}/{{sequence}}"
    assert full["detector"] is boxes["detector"] is None
    assert full["reid"] is boxes["reid"] is None
    assert full_inputs["images"] == box_inputs["images"]
    assert full_inputs["detections_2d"]["paths"] == box_inputs["detections_2d"]["paths"]
    assert full_inputs["detections_2d"]["options"]["load_masks"] is True
    assert box_inputs["detections_2d"]["options"]["load_masks"] is False
    assert set(full_inputs) == {
        "images",
        "detections_2d",
        "detections_3d",
        "calibration",
        "poses",
        "ground_truth",
        "ground_truth_3d",
    }
    assert set(box_inputs) == {"images", "detections_2d", "ground_truth"}


@pytest.mark.parametrize("mode", ("materialize", "research", "inference"))
def test_sensor_experiment_rejects_perception_workflows(mode: str) -> None:
    with pytest.raises(ValueError, match="supports only eval and tune"):
        resolve_experiment_config("kitti-mots/full", mode=mode)


@pytest.mark.parametrize("component", ("detector", "segmentor", "reid", "evaluation"))
def test_sensor_experiment_rejects_unused_components(tmp_path: Path, component: str) -> None:
    experiment = tmp_path / "invalid.yaml"
    experiment.write_text(yaml.safe_dump({"dataset": {"ref": "kitti-mots"}, component: {}}))
    with pytest.raises(ValueError, match=f"must omit {component}"):
        resolve_experiment_config(experiment, mode="eval")


@pytest.mark.parametrize("mode", ("eval", "tune"))
@pytest.mark.parametrize("split_override", (False, True))
def test_sensor_experiment_dispatches_with_root_and_split_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, split_override: bool
) -> None:
    """Authored sensors bypass perception and retain their root through runtime validation."""
    import importlib
    import sys

    from boxmot.engine.eval.results import ValidationResult
    from boxmot.engine.tuning.results import TuneResult
    from tests.unit.engine.test_sensor_dataset_tune import _result

    data = sensor_dataset_fixture(tmp_path / "data")
    sequence = "0000" if split_override else "0002"
    if split_override:
        for path in list(data.root.rglob("0002")) + list(data.root.rglob("0002.txt")):
            path.rename(path.with_name(path.name.replace("0002", sequence)))
    predictions = "pointgnn-car-t3" if split_override else "pointgnn-car-t2"
    (data.root / f"predictions/{predictions}/training/{sequence}").mkdir(parents=True)
    captured = {}
    command = importlib.import_module(f"boxmot.engine.commands.{mode}")
    monkeypatch.setattr(command, "_prepare_replay_build", lambda *a, **kw: pytest.fail("Perception requested"))

    def run(args: SimpleNamespace, **kwargs: object) -> ValidationResult | TuneResult:
        captured.update(vars(args))
        inputs = load_sensor_evaluation_inputs(
            args.dataset, split=args.split, sequence_names=args.sequence_names, data_root=args.data_root
        )
        assert inputs.root == data.root
        if mode == "tune":
            return _result(args, data.project)
        return ValidationResult(inputs.id, {}, "cls_comb_cls_av", {}, exp_dir=data.project, args=args)

    if mode == "eval":
        monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    else:
        tuner = importlib.import_module("boxmot.engine.tuning.tuner")
        monkeypatch.setattr(tuner, "_run_eagermot_tuning", run)

    selected_split = "train" if split_override else "val"
    flags = ["--split", selected_split] if split_override else []
    result = CliRunner().invoke(
        boxmot,
        [
            mode,
            "--experiment",
            "kitti-mots/full",
            "--data-root",
            str(data.root),
            "--tracker",
            "eagermot",
            "--sequence",
            sequence,
            *flags,
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["split"] == selected_split
    assert captured["sequence_names"] == (sequence,)
    assert captured["experiment_id"] == "kitti-mots-full"


@pytest.mark.parametrize("tracker", ("ocsort", "bytetrack", "maf_hda"))
def test_multimodal_experiment_rejects_trackers_that_cannot_consume_its_inputs(tracker: str) -> None:
    result = CliRunner().invoke(
        boxmot, ["eval", "--experiment", "kitti-mots/full", "--tracker", tracker]
    )
    assert result.exit_code == 2
    assert "does not use inputs required" in result.output


@pytest.mark.parametrize("experiment", ("kitti-mots/full", "kitti-mots/2d"))
def test_sensor_experiment_rejects_a_conflicting_runtime_dataset(experiment: str) -> None:
    with pytest.raises(ValueError, match="does not match"):
        resolve_sensor_experiment({"experiment": experiment, "dataset": "sensor-fusion"}, mode="eval")
