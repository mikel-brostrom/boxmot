"""Kalman calibration is explicit and saved configs reach live and cached trackers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.engine.commands import eval as eval_command
from boxmot.engine.eval.evaluator import _tracker_options
from boxmot.engine.tracking.workflow import _tracker_spec
from boxmot.engine.tuning import kalman


@pytest.mark.parametrize(
    "flags, enabled, trials",
    [([], False, 20), (["--kf-tuning"], True, 20), (["--kf-tuning", "--kf-trials", "1"], True, 1)],
)
def test_eval_dispatch_preserves_kalman_tuning_selection(monkeypatch, flags, enabled, trials) -> None:
    captured = {}

    def run_workflow(module, args):
        captured.update(module=module, args=args)

    monkeypatch.setattr(_support, "_run_engine_workflow", run_workflow)
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "fixture", "--build", "fixture-build", *flags],
    )

    assert result.exit_code == 0, result.output
    assert captured["module"] == "boxmot.engine.eval.evaluator"
    assert captured["args"].kf_tuning is enabled
    assert captured["args"].kf_trials == trials
    assert captured["args"].variable_dt is None


@pytest.mark.parametrize("trials", ["1", "20"])
def test_explicit_kalman_trials_require_tuning_flag(monkeypatch, trials) -> None:
    def unexpected_workflow(*args):
        pytest.fail("Invalid tuning options must fail before materialization")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    monkeypatch.setattr(eval_command, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(boxmot, ["eval", "--experiment", "fixture", "--kf-trials", trials])

    assert result.exit_code == 2
    assert "--kf-trials requires --kf-tuning" in result.output


@pytest.mark.parametrize("trials", ["0", "-1"])
def test_kalman_trials_must_include_at_least_the_baseline(trials) -> None:
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "fixture", "--kf-tuning", "--kf-trials", trials],
    )

    assert result.exit_code == 2
    assert "--kf-trials" in result.output
    assert "range" in result.output


@pytest.mark.parametrize(
    "tracker, backend",
    [("sfsort", "python"), ("sam2mot", "python"), ("botsort", "cpp")],
)
def test_unsupported_kalman_tuning_fails_before_materialization(monkeypatch, tracker, backend) -> None:
    def unexpected_workflow(*args):
        pytest.fail("Unsupported trackers must fail before materialization")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    monkeypatch.setattr(eval_command, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "fixture",
            "--kf-tuning",
            "--tracker",
            tracker,
            "--tracker-backend",
            backend,
        ],
    )

    assert result.exit_code == 2
    assert "Kalman" in result.output or "Python" in result.output


def test_missing_optuna_fails_before_materialization(monkeypatch) -> None:
    find_spec = kalman.importlib.util.find_spec
    monkeypatch.setattr(kalman.importlib.util, "find_spec", lambda name: None if name == "optuna" else find_spec(name))

    def unexpected_workflow(*args):
        pytest.fail("Missing tuning dependencies must fail before materialization")

    monkeypatch.setattr(eval_command, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(boxmot, ["eval", "--experiment", "fixture", "--kf-tuning"])
    assert result.exit_code == 2
    assert "requires Optuna" in result.output


@pytest.mark.parametrize("tracker,backend", [("sfsort", "python"), ("botsort", "cpp")])
def test_unsupported_timestamp_mode_fails_before_materialization(monkeypatch, tracker, backend) -> None:
    def unexpected_workflow(*args):
        pytest.fail("Unsupported timestamp modes must fail before materialization")

    monkeypatch.setattr(eval_command, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot, ["eval", "--experiment", "fixture", "--tracker", tracker, "--tracker-backend", backend, "--variable-dt"]
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


@pytest.mark.parametrize(
    "tracker",
    ["botsort", "boosttrack", "bytetrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "strongsort"],
)
def test_supported_kalman_trackers_reach_evaluation(monkeypatch, tracker) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "fixture",
            "--build",
            "fixture-build",
            "--tracker",
            tracker,
            "--kf-tuning",
            "--fixed-dt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].tracker == tracker
    assert captured["args"].variable_dt is False


@pytest.mark.parametrize("mode", ["track", "eval", "tune"])
def test_tracker_config_selector_reaches_runtime_namespace(monkeypatch, mode) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    argv = [mode, "--tracker", "botsort", "--tracker-config", "botsort-mot17-ablation"]
    if mode == "track":
        argv += ["--source", "video.mp4"]
    else:
        argv += ["--experiment", "fixture", "--build", "fixture-build"]
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
    path = tmp_path / "best.yaml"
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
    path = tmp_path / "best.yaml"
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
def test_eval_rejects_calibrated_unit_flip_before_materialization(monkeypatch, tmp_path, saved_mode, flag) -> None:
    path = tmp_path / "best.yaml"
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
    monkeypatch.setattr(eval_command, "_run_engine_workflow", unexpected_workflow)
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--experiment", "fixture", "--tracker-config", str(path), flag],
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
    path = tmp_path / "best.yaml"
    path.write_text("tracker: botsort\nkf_process_position_scale: 2.0\n")
    args = SimpleNamespace(tracker="bytetrack", tracker_config=path)

    with pytest.raises(ValueError, match="botsort.*bytetrack"):
        if mode == "track":
            _tracker_spec(args, "aabb")
        else:
            _tracker_options(args, None)
