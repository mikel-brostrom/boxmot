"""Exercise 3D calibration and reusable profiles through the shared commands."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.eval import eagermot_kitti as evaluation
from boxmot.engine.eval.eagermot_kitti import KITTI_CLASSES, KITTI_PROFILES, load_kitti_profiles
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture
from tests.unit.engine.eval.test_eagermot_kitti import _fixture


def _declare_3d_truth(data: Any) -> Path:
    """Add tracked camera-space boxes for all three fixture timesteps."""
    truth_path = data.root / "labels_3d.txt"
    rows = []
    for frame in range(3):
        for class_id, label, x, length in ((1, "Car", -5, 4), (2, "Pedestrian", 5, 1)):
            row = [frame, class_id, label, 0, 0, 0, 0, 0, 1, 1, 3, 1, length, x, 0, 20, 0]
            rows.append(" ".join(map(str, row)))
    truth_path.write_text("\n".join(rows) + "\n")
    config = yaml.safe_load(data.dataset.read_text())
    config["modalities"]["ground_truth_3d"] = {
        "format": "kitti-tracking-labels",
        "path": "labels_3d.txt",
    }
    data.dataset.write_text(yaml.safe_dump(config))
    return truth_path


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_missing_3d_ground_truth_explains_requirement_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Mask annotations cannot substitute for labelled spatial trajectories."""
    data = sensor_dataset_fixture(tmp_path)
    monkeypatch.setitem(sys.modules, "boxmot.engine.calibration.kalman_sensor", None)
    result = CliRunner().invoke(
        boxmot, [mode, "--dataset", str(data.dataset), "--tracker", "eagermot", "--calibrate-kf"]
    )
    assert result.exit_code == 2, (result.output, result.exception)
    assert (
        "Error: --calibrate-kf requires 3D ground truth with track IDs.\n"
        "Add ground_truth_3d with format: kitti-tracking-labels to dataset.yaml.\n"
    ) in result.output
    assert not data.project.exists()


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_cli_calibrates_once_and_reuses_class_specific_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Fit orchestration, real tracking, and real Optuna preserve all fixed settings."""
    from boxmot.engine.calibration import kalman_sensor
    from boxmot.engine.calibration.kalman import KalmanCalibrationResult

    optuna = pytest.importorskip("optuna")
    data = _fixture(tmp_path)
    _declare_3d_truth(data)
    calibrated = deepcopy(KITTI_PROFILES)
    for class_id, profile in calibrated.items():
        profile.update({name: class_id + (index + 1) / 10 for index, name in enumerate(KALMAN_NOISE_OPTIONS)})
    calibrated[2]["is_angular"] = True
    calls = []

    def calibrate(dataset: Any, profiles: Any, *, output_dir: Path, progress: Any = None) -> KalmanCalibrationResult:
        assert dataset.config_path == data.dataset.resolve()
        assert "ground_truth_3d" in dataset.sequences[0].modalities
        assert profiles == KITTI_PROFILES
        calls.append(output_dir)
        directory = output_dir / "kf-tuning"
        directory.mkdir(parents=True)
        profile_path, report_path = directory / "calibrated.yaml", directory / "calibration.json"
        evaluation.write_kitti_profiles(profile_path, calibrated)
        report_path.write_text("{}")
        return KalmanCalibrationResult(profile_path, report_path, 4, 2, ())

    monkeypatch.setattr(kalman_sensor, "calibrate_sensor_kalman", calibrate)
    real_evaluate = evaluation.evaluate_eagermot_kitti
    replay_profiles = []

    def replay(inputs: Any, profiles: Any, output: Path, **kwargs: Any) -> Any:
        replay_profiles.append(deepcopy(profiles))
        return real_evaluate(inputs, profiles, output, **kwargs)

    monkeypatch.setattr(evaluation, "evaluate_eagermot_kitti", replay)
    arguments = [mode, "--dataset", str(data.dataset), "--tracker", "eagermot", "--project", str(data.project)]
    if mode == "tune":
        arguments += ["--n-trials", "3"]
    result = CliRunner().invoke(boxmot, [*arguments, "--calibrate-kf"])

    assert result.exit_code == 0, (result.output, result.exception)
    output = data.project / "val"
    assert calls == [output]
    assert len(replay_profiles) == (3 if mode == "tune" else 1)
    for profiles in replay_profiles:
        for class_id in calibrated:
            for name in (*KALMAN_NOISE_OPTIONS, "is_angular"):
                assert profiles[class_id][name] == calibrated[class_id][name]
    report = json.loads((output / "kf-tuning/calibration.json").read_text())
    assert report["final_summary"]["HOTA"] == pytest.approx(100)
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["kf_calibration"] == {
        "config_path": str(output / "kf-tuning/calibrated.yaml"),
        "report_path": str(output / "kf-tuning/calibration.json"),
    }
    profile_path = output / ("best.yaml" if mode == "tune" else "kf-tuning/calibrated.yaml")
    if mode == "tune":
        assert manifest["baseline_profiles"] == {str(key): value for key, value in calibrated.items()}
        assert set((*KALMAN_NOISE_OPTIONS, "is_angular")) <= set(manifest["fixed_parameters"])
        study = optuna.load_study(study_name=None, storage=f"sqlite:///{(output / 'study.sqlite3').as_uri()}?uri=true")
        for trial in study.trials:
            assert not any(name.rpartition(".")[2] in (*KALMAN_NOISE_OPTIONS, "is_angular") for name in trial.params)

    # A second run consumes the exported profiles without invoking calibration.
    replay_profiles.clear()
    reused = CliRunner().invoke(boxmot, [*arguments, "--class-config", str(profile_path)])
    assert reused.exit_code == 0, (reused.output, reused.exception)
    assert calls == [output]
    for profiles in replay_profiles:
        for class_id in calibrated:
            for name in (*KALMAN_NOISE_OPTIONS, "is_angular"):
                assert profiles[class_id][name] == calibrated[class_id][name]
    exported = load_kitti_profiles(profile_path)
    assert set(exported) == set(KITTI_CLASSES)


@pytest.mark.parametrize("mode", ("eval", "tune"))
def test_real_3d_calibration_runs_before_sensor_replay(tmp_path: Path, mode: str) -> None:
    """Exercise the annotation reader, world transform, fit, artifact, and tracker together."""
    data = _fixture(tmp_path)
    truth_path = _declare_3d_truth(data)
    arguments = [
        mode,
        "--dataset",
        str(data.dataset),
        "--tracker",
        "eagermot",
        "--project",
        str(data.project),
        "--calibrate-kf",
    ]
    if mode == "tune":
        arguments += ["--n-trials", "1"]
    result = CliRunner().invoke(boxmot, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
    output = data.project / "val"
    report = json.loads((output / "kf-tuning/calibration.json").read_text())
    assert report["geometry"] == "box3d"
    assert report["coordinates"] == "world"
    assert report["statistics"]["matched"] == 4
    assert report["final_summary"]["HOTA"] == pytest.approx(100)
    assert str(truth_path) in json.dumps(report["ground_truth_sources"])
    assert set(report["classes"]) == {"car", "pedestrian"}
    profiles = load_kitti_profiles(output / "kf-tuning/calibrated.yaml")
    replay_output = output / "trials/0000" if mode == "tune" else output
    replay_manifest = json.loads((replay_output / "run.json").read_text())
    for class_id, name in KITTI_CLASSES.items():
        assert set(report["classes"][name]["parameters"]) == set(KALMAN_NOISE_OPTIONS)
        assert report["classes"][name]["parameters"]["kf_measurement_noise_scale"]["status"] == "fitted"
        assert profiles[class_id]["kf_measurement_noise_scale"] < 1.0
        for parameter in KALMAN_NOISE_OPTIONS:
            assert profiles[class_id][parameter] == report["classes"][name]["parameters"][parameter]["value"]
            assert replay_manifest["tracker_profiles"][str(class_id)][parameter] == profiles[class_id][parameter]


def test_existing_sensor_results_are_not_overwritten_by_replay(tmp_path: Path) -> None:
    """A calibration subfolder may predate replay; an existing run may not."""
    from types import SimpleNamespace

    data = _fixture(tmp_path)
    args = SimpleNamespace(dataset=data.dataset, split="val", sequence_names=("0002",))
    inputs = evaluation.prepare_eagermot_kitti(args)
    output = tmp_path / "results"
    evaluation.evaluate_eagermot_kitti(inputs, KITTI_PROFILES, output)
    original = (output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.evaluate_eagermot_kitti(inputs, KITTI_PROFILES, output)
    assert (output / "run.json").read_bytes() == original
