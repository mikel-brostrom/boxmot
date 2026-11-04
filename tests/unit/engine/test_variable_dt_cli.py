"""Timing mode is a tracker runtime option and a constant during tuning."""

import importlib
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.engine.eval.evaluator import _output_directory, _tracker_options
from boxmot.engine.tracking.workflow import _tracker_spec
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.postprocessing import write_trial_yaml
from boxmot.engine.tuning.search_space import (
    default_tune_config,
    flatten_yaml_config,
    load_yaml_config,
    validate_tuning_config,
    yaml_to_tune_space,
)
from boxmot.engine.tuning.tuner import Tuner
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.motion.kalman_filters.noise import DEFAULT_REFERENCE_DT_S, KALMAN_NOISE_OPTIONS


@pytest.mark.parametrize("mode", ["track", "eval", "tune"])
@pytest.mark.parametrize("flag, expected", [(None, None), ("--variable-dt", True), ("--fixed-dt", False)])
def test_cli_timing_flags_preserve_optional_override(monkeypatch, mode, flag, expected):
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    argv = [mode]
    if mode == "track":
        argv += ["--source", "video.mp4"]
    else:
        argv += ["--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "fixture-build"]
    if flag is not None:
        argv.append(flag)
    result = CliRunner().invoke(boxmot, argv)
    assert result.exit_code == 0, result.output
    assert captured["args"].variable_dt is expected


@pytest.mark.parametrize("mode", ["eval", "tune"])
@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_sfsort_variable_timing_fails_before_workflow(monkeypatch, mode, backend) -> None:
    def unexpected_workflow(*args):
        pytest.fail("SFSORT variable timing must fail before loading models or replay data")

    monkeypatch.setattr(_support, "_run_engine_workflow", unexpected_workflow)
    argv = [mode, "--tracker", "sfsort", "--tracker-backend", backend, "--variable-dt"]
    argv += ["--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "fixture-build"]
    result = CliRunner().invoke(boxmot, argv)
    assert result.exit_code == 2
    assert "sfsort" in result.output
    assert "does not support kalman" in result.output


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_live_sfsort_variable_timing_fails_before_detector_loading(monkeypatch, backend) -> None:
    from boxmot.engine.tracking import workflow

    def unexpected_detector(*args):
        pytest.fail("SFSORT variable timing must fail before loading perception")

    monkeypatch.setattr(workflow, "create_detector", unexpected_detector)
    args = SimpleNamespace(tracker="sfsort", tracker_backend=backend, variable_dt=True)
    with pytest.raises(ValueError, match="sfsort.*does not support kalman"):
        workflow.run_track(args)


def test_tune_cli_preserves_repeated_selected_kf_scales(monkeypatch):
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
            "--tracker",
            "botsort",
            "--tune-kf",
            "process_velocity_scale",
            "--tune-kf",
            "measurement_noise_scale",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["args"].tune_kf == ("process_velocity_scale", "measurement_noise_scale")


@pytest.mark.parametrize("field", ["time_unit", "reference_dt_s", "unknown"])
def test_tune_cli_rejects_non_scale_kf_fields(field):
    result = CliRunner().invoke(boxmot, ["tune", "--dataset", "fixture", "--tune-kf", field])
    assert result.exit_code == 2
    assert "Invalid value for '--tune-kf'" in result.output


def test_tune_cli_rejects_refinement_without_supported_filter(monkeypatch):
    command = importlib.import_module("boxmot.engine.commands.tune")
    monkeypatch.setattr(
        command, "_prepare_replay_build", lambda *args, **kwargs: pytest.fail("Must validate before building")
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--tracker",
            "sfsort",
            "--tune-kf",
            "process_velocity_scale",
        ],
    )
    assert result.exit_code == 2
    assert "sfsort" in result.output


def test_sensor_tune_accepts_selected_kf_scales(tmp_path, monkeypatch):
    from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture

    dataset = sensor_dataset_fixture(tmp_path / "sensor").dataset
    command = importlib.import_module("boxmot.engine.commands.tune")
    captured = {}
    monkeypatch.setattr(command, "_dispatch_cli_workflow", lambda ctx, mode, module, payload: captured.update(payload))
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--dataset",
            str(dataset),
            "--tracker",
            "eagermot",
            "--tune-kf",
            "measurement_noise_scale",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["tune_kf"] == ("measurement_noise_scale",)


@pytest.mark.parametrize("value", [None, False, True])
def test_runtime_specs_apply_only_explicit_timing_override(value):
    args = SimpleNamespace(tracker="bytetrack", variable_dt=value)
    track_options = _tracker_spec(args, "aabb").option_dict
    eval_options = dict(_tracker_options(args, {"kalman.variable_dt": True}))
    if value is None:
        assert "kalman.variable_dt" not in track_options
        assert eval_options["kalman.variable_dt"] is True
    else:
        assert track_options["kalman.variable_dt"] is value
        assert eval_options["kalman.variable_dt"] is value


@pytest.mark.parametrize(
    "tracker", ["bytetrack", "botsort", "boosttrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "strongsort"]
)
def test_builtin_timing_and_kf_noise_are_fixed_default_only(tracker):
    schema = flatten_yaml_config(load_yaml_config(tracker))
    assert schema["kalman.variable_dt"] == {"default": False}
    assert load_tracker_defaults(tracker)["kalman.variable_dt"] is False
    assert "kalman.variable_dt" not in default_tune_config(schema)
    for parameter in KALMAN_NOISE_OPTIONS:
        assert schema[parameter] == {"default": 1.0}
        assert parameter not in default_tune_config(schema)
    for parameter in ("kalman.noise.time_unit", "kalman.noise.reference_dt_s"):
        assert set(schema[parameter]) == {"default"}
        assert parameter not in default_tune_config(schema)


@pytest.mark.parametrize("tracker", ["maf_hda", "eagermot"])
def test_additional_trackers_keep_timing_out_of_search_space(tracker):
    schema = flatten_yaml_config(load_yaml_config(tracker))
    assert schema["kalman.variable_dt"] == {"default": False}
    assert load_tracker_defaults(tracker)["kalman.noise.time_unit"] is None
    assert "kalman.variable_dt" not in default_tune_config(schema)


def test_sfsort_defaults_and_saved_runtime_options_have_no_kalman_settings() -> None:
    from boxmot.engine.config.trackers import resolve_tracker_options

    schema = flatten_yaml_config(load_yaml_config("sfsort"))
    runtime = resolve_tracker_options(SimpleNamespace(tracker="sfsort"), include_defaults=True, stamp_timing=True)
    assert not any(key.startswith("kalman.") for key in schema)
    assert not any(key.startswith("kalman.") for key in runtime)


def test_search_backends_skip_fixed_timing_mode():
    schema = {
        "kalman.variable_dt": {"default": True},
        "threshold": {"type": "uniform", "default": 0.5, "range": [0.1, 0.9]},
    }
    ray = SimpleNamespace(uniform=lambda low, high: (low, high))
    assert yaml_to_tune_space(schema, ray) == {"threshold": (0.1, 0.9)}
    trial = SimpleNamespace(params={})
    trial.suggest_float = lambda parameter, *args: trial.params.setdefault(parameter, 0.1)
    yaml_to_optuna_define_space(schema)(trial)
    assert trial.params == {"threshold": 0.1}


@pytest.mark.parametrize("parameter", ["kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s"])
def test_timing_mode_cannot_be_changed_to_a_search_dimension(parameter):
    with pytest.raises(ValueError, match=f"{parameter}.*fixed"):
        validate_tuning_config("bytetrack", {parameter: {"type": "choice", "default": False, "options": [False, True]}})


def test_saved_runtime_config_and_trial_identity_include_timing(tmp_path):
    args = SimpleNamespace(project=tmp_path, dataset_id="dataset", name="eval", variable_dt=False)
    overrides = {"track_thresh": 0.4}
    fixed_path = _output_directory(
        args, TrackerSpec("bytetrack", options=_tracker_options(args, overrides)), trial=True
    )
    args.variable_dt = True
    timed_path = _output_directory(
        args, TrackerSpec("bytetrack", options=_tracker_options(args, overrides)), trial=True
    )
    assert timed_path != fixed_path
    output = tmp_path / "best.yaml"
    write_trial_yaml({}, overrides, output, base_config={"kalman.variable_dt": True, "track_thresh": 0.5})
    assert yaml.safe_load(output.read_text()) == {"kalman": {"variable_dt": True}, "track_thresh": 0.4}


@pytest.mark.parametrize("requested, saved", [(True, False), (False, True)])
@pytest.mark.parametrize("tracker", ["bytetrack", "maf_hda"])
def test_tune_resume_rejects_changed_timing_mode(requested, saved, tracker):
    tuner = Tuner(SimpleNamespace(tracker=tracker, variable_dt=requested))
    with pytest.raises(ValueError, match="same kalman.variable_dt"):
        tuner._validate_resumed_timing(
            [
                SimpleNamespace(
                    config={
                        "kalman.variable_dt": saved,
                        "kalman.noise.time_unit": "seconds" if saved else "frames",
                        "kalman.noise.reference_dt_s": DEFAULT_REFERENCE_DT_S,
                    }
                )
            ]
        )
    tuner._validate_resumed_timing(
        [
            SimpleNamespace(
                config={
                    "kalman.variable_dt": requested,
                    "kalman.noise.time_unit": "seconds" if requested else "frames",
                    "kalman.noise.reference_dt_s": DEFAULT_REFERENCE_DT_S,
                }
            )
        ]
    )


@pytest.mark.parametrize("replacement", [{"kalman.noise.time_unit": "frames"}, {"kalman.noise.reference_dt_s": 0.04}])
def test_resume_rejects_changed_units_or_reference(replacement):
    tuner = Tuner(SimpleNamespace(tracker="bytetrack", variable_dt=True))
    saved = {
        "kalman.variable_dt": True,
        "kalman.noise.time_unit": "seconds",
        "kalman.noise.reference_dt_s": DEFAULT_REFERENCE_DT_S,
        **replacement,
    }
    with pytest.raises(ValueError, match="same kalman.variable_dt.*kalman.noise.reference_dt_s"):
        tuner._validate_resumed_timing([SimpleNamespace(config=saved)])


def test_resume_requires_explicit_saved_units():
    tuner = Tuner(SimpleNamespace(tracker="bytetrack", variable_dt=True))
    with pytest.raises(ValueError, match="lack explicit timing"):
        tuner._validate_resumed_timing([SimpleNamespace(config={"kalman.variable_dt": True})])
