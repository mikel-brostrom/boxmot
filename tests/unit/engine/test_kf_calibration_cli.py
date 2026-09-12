"""Kalman calibration is explicit and saved configs reach live and cached trackers."""

from __future__ import annotations

import builtins
import importlib.util
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.engine.eval.evaluator import _tracker_options
from boxmot.engine.tracking.workflow import _tracker_spec


@pytest.mark.parametrize("mode", ["eval", "tune"])
@pytest.mark.parametrize(
    "flags, enabled",
    [([], False), (["--calibrate-kf"], True)],
)
def test_dispatch_preserves_kalman_calibration_selection(monkeypatch, mode, flags, enabled) -> None:
    captured = {}

    def run_workflow(module, args):
        captured.update(module=module, args=args)

    monkeypatch.setattr(_support, "_run_engine_workflow", run_workflow)
    result = CliRunner().invoke(
        boxmot,
        [mode, "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "fixture-build", *flags],
    )

    assert result.exit_code == 0, result.output
    assert captured["module"] == {"eval": "boxmot.engine.eval.evaluator", "tune": "boxmot.engine.tuning.tuner"}[mode]
    assert captured["args"].calibrate_kf is enabled
    assert not hasattr(captured["args"], "kf_trials")
    assert captured["args"].variable_dt is None


@pytest.mark.parametrize("mode", ["eval", "tune"])
@pytest.mark.parametrize("flags", [[], ["--calibrate-kf"]])
@pytest.mark.parametrize("removed_options", [["--kf-trials", "20"], ["--kf-tuning"]])
def test_removed_kalman_options_are_rejected(monkeypatch, mode, flags, removed_options) -> None:
    def unexpected_workflow(*args):
        pytest.fail("Removed search options must fail before materialization")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        [mode, "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "fixture-build", *flags, *removed_options],
    )

    assert result.exit_code == 2
    assert f"No such option '{removed_options[0]}'" in result.output


@pytest.mark.parametrize("mode", ["eval", "tune"])
@pytest.mark.parametrize(
    "tracker, backend",
    [("sfsort", "python"), ("maf_hda", "python"), ("botsort", "cpp")],
)
def test_unsupported_kalman_calibration_fails_before_workflow(monkeypatch, mode, tracker, backend) -> None:
    def unexpected_workflow(*args):
        pytest.fail("Unsupported trackers must fail before materialization")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        [
            mode,
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--calibrate-kf",
            "--tracker",
            tracker,
            "--tracker-backend",
            backend,
        ],
    )

    assert result.exit_code == 2
    assert "Kalman" in result.output or "Python" in result.output


def test_kalman_calibration_dispatch_needs_no_search_dependencies(monkeypatch) -> None:
    real_import = builtins.__import__
    real_find_spec = importlib.util.find_spec
    captured = {}

    def import_without_search(name, *args, **kwargs):
        if name.split(".")[0] in {"optuna", "ray"}:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_search)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: None if name.split(".")[0] in {"optuna", "ray"} else real_find_spec(name, *args),
    )
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "fixture-build", "--calibrate-kf"],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].calibrate_kf is True


@pytest.mark.parametrize("tracker,backend", [("sfsort", "python"), ("botsort", "cpp")])
@pytest.mark.parametrize("mode", ["eval", "tune"])
def test_unsupported_timestamp_mode_fails_before_materialization(monkeypatch, mode, tracker, backend) -> None:
    def unexpected_workflow(*args):
        pytest.fail("Unsupported timestamp modes must fail before materialization")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        [
            mode,
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--tracker",
            tracker,
            "--tracker-backend",
            backend,
            "--variable-dt",
        ],
    )
    assert result.exit_code == 2
    assert "does not support variable_dt" in result.output


def test_live_config_unit_conflict_fails_before_detector_loading(monkeypatch, tmp_path) -> None:
    from boxmot.engine.tracking import workflow

    profile = tmp_path / "seconds.yaml"
    profile.write_text(yaml.safe_dump({"variable_dt": True, "kf_time_unit": "seconds"}))

    def unexpected_detector(*args):
        pytest.fail("Conflicting calibration units must fail before loading perception")

    monkeypatch.setattr(workflow, "create_detector", unexpected_detector)
    with pytest.raises(ValueError, match="conflicts with variable_dt"):
        workflow.run_track(SimpleNamespace(tracker="bytetrack", tracker_config=profile, variable_dt=False))


@pytest.mark.parametrize("mode", ["eval", "tune"])
@pytest.mark.parametrize(
    "tracker",
    ["botsort", "boosttrack", "bytetrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "strongsort"],
)
def test_supported_kalman_trackers_reach_workflow(monkeypatch, mode, tracker) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    result = CliRunner().invoke(
        boxmot,
        [
            mode,
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture-build",
            "--tracker",
            tracker,
            "--calibrate-kf",
            "--fixed-dt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == tracker
    assert captured["args"].variable_dt is False


def test_tune_calibration_cannot_replace_a_resumed_search(monkeypatch) -> None:
    def unexpected_workflow(*args):
        pytest.fail("A resumed search cannot start a fresh calibration")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture-build",
            "--calibrate-kf",
            "--resume-tune",
            "previous-run",
        ],
    )
    assert result.exit_code == 2
    assert "--calibrate-kf cannot be combined with --resume-tune" in result.output


def test_tune_resume_dispatches_without_recalibrating(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture-build",
            "--resume-tune",
            "previous-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].calibrate_kf is False
    assert captured["args"].resume_tune == "previous-run"


def test_tune_calibration_rejects_conflicting_units_before_workflow(monkeypatch, tmp_path) -> None:
    path = tmp_path / "seconds.yaml"
    path.write_text("tracker: botsort\nvariable_dt: true\nkf_time_unit: seconds\n")

    def unexpected_workflow(*args):
        pytest.fail("Conflicting calibration units must fail before tuning")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture-build",
            "--tracker",
            "botsort",
            "--tracker-config",
            str(path),
            "--fixed-dt",
            "--calibrate-kf",
        ],
    )
    assert result.exit_code == 2
    assert "kf_time_unit" in result.output


@pytest.mark.parametrize("mode", ["track", "eval", "tune"])
def test_tracker_config_selector_reaches_runtime_namespace(monkeypatch, mode) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    argv = [mode, "--tracker", "botsort", "--tracker-config", "botsort-mot17-ablation"]
    if mode == "track":
        argv += ["--source", "video.mp4"]
    else:
        argv += ["--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "fixture-build"]
    result = CliRunner().invoke(boxmot, argv)

    assert result.exit_code == 0, result.output
    assert captured["args"].tracker_config == "botsort-mot17-ablation"


@pytest.mark.parametrize("mode", ["track", "eval"])
@pytest.mark.parametrize("saved_mode, override", [(False, None), (False, False), (True, None), (True, True)])
def test_saved_kalman_config_preserves_units_and_accepts_matching_overrides(
    tmp_path, mode, saved_mode, override
) -> None:
    saved = {
        "tracker": "bytetrack",
        "variable_dt": saved_mode,
        "kf_time_unit": "seconds" if saved_mode else "frames",
        "kf_reference_dt_s": 0.04,
        "asso_func": "giou",
        "kf_process_position_scale": 2.0,
        "kf_process_velocity_scale": 3.0,
        "kf_measurement_noise_scale": 0.5,
        "kf_initial_position_scale": 0.75,
        "kf_initial_velocity_scale": 4.0,
    }
    path = tmp_path / "calibrated.yaml"
    path.write_text(yaml.safe_dump(saved))
    args = SimpleNamespace(tracker="bytetrack", tracker_config=path, variable_dt=override, asso_func="iou")

    options = _tracker_spec(args, "aabb").option_dict if mode == "track" else dict(_tracker_options(args, None))

    assert options["variable_dt"] is saved_mode
    assert options["asso_func"] == "iou"
    assert "tracker" not in options
    for name in (
        "kf_time_unit",
        "kf_reference_dt_s",
        "kf_process_position_scale",
        "kf_process_velocity_scale",
        "kf_measurement_noise_scale",
        "kf_initial_position_scale",
        "kf_initial_velocity_scale",
    ):
        assert options[name] == saved[name]
    assert "track_thresh" in options


@pytest.mark.parametrize("mode", ["track", "eval"])
@pytest.mark.parametrize("saved_mode", [False, True])
def test_calibrated_config_rejects_conflicting_timing_override(tmp_path, mode, saved_mode) -> None:
    path = tmp_path / "calibrated.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tracker": "bytetrack",
                "variable_dt": saved_mode,
                "kf_time_unit": "seconds" if saved_mode else "frames",
                "kf_reference_dt_s": 1 / 30,
                "kf_process_velocity_scale": 2.0,
            }
        )
    )
    args = SimpleNamespace(tracker="bytetrack", tracker_config=path, variable_dt=not saved_mode)

    with pytest.raises(ValueError, match="kf_time_unit"):
        if mode == "track":
            _tracker_spec(args, "aabb")
        else:
            _tracker_options(args, None)


@pytest.mark.parametrize("saved_mode, flag", [(False, "--variable-dt"), (True, "--fixed-dt")])
@pytest.mark.parametrize("mode", ["eval", "tune"])
def test_rejects_calibrated_unit_flip_before_materialization(monkeypatch, tmp_path, mode, saved_mode, flag) -> None:
    path = tmp_path / "calibrated.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tracker": "bytetrack",
                "variable_dt": saved_mode,
                "kf_time_unit": "seconds" if saved_mode else "frames",
                "kf_reference_dt_s": 1 / 30,
            }
        )
    )

    def unexpected_workflow(*args):
        pytest.fail("Conflicting calibration units must fail before materialization")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        [mode, "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--tracker-config", str(path), flag],
    )

    assert result.exit_code == 2
    assert "kf_time_unit" in result.output


@pytest.mark.parametrize("mode", ["track", "eval"])
def test_tracker_config_accepts_builtin_preset(mode) -> None:
    args = SimpleNamespace(tracker="botsort", tracker_config="botsort-mot17-ablation")

    options = _tracker_spec(args, "aabb").option_dict if mode == "track" else dict(_tracker_options(args, None))

    assert options["track_buffer"] == 40
    assert "tracker" not in options


@pytest.mark.parametrize("mode", ["track", "eval"])
def test_tracker_config_rejects_metadata_for_a_different_tracker(tmp_path, mode) -> None:
    path = tmp_path / "calibrated.yaml"
    path.write_text("tracker: botsort\nkf_process_position_scale: 2.0\n")
    args = SimpleNamespace(tracker="bytetrack", tracker_config=path)

    with pytest.raises(ValueError, match="botsort.*bytetrack"):
        if mode == "track":
            _tracker_spec(args, "aabb")
        else:
            _tracker_options(args, None)
