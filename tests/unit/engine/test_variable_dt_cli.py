"""Timing mode is a tracker runtime option and a constant during tuning."""

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
    load_yaml_config,
    validate_tuning_config,
    yaml_to_tune_space,
)
from boxmot.engine.tuning.tuner import Tuner
from boxmot.motion.kalman_filters.noise import DEFAULT_REFERENCE_DT_S, KALMAN_NOISE_OPTIONS
from boxmot.trackers.config import load_tracker_defaults


@pytest.mark.parametrize("mode", ["track", "eval", "tune"])
@pytest.mark.parametrize("flag, expected", [(None, None), ("--variable-dt", True), ("--fixed-dt", False)])
def test_cli_timing_flags_preserve_optional_override(monkeypatch, mode, flag, expected):
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.setdefault("args", args))
    argv = [mode]
    if mode == "track":
        argv += ["--source", "video.mp4"]
    else:
        argv += ["--experiment", "fixture", "--build", "fixture-build"]
    if flag is not None:
        argv.append(flag)
    result = CliRunner().invoke(boxmot, argv)
    assert result.exit_code == 0, result.output
    assert captured["args"].variable_dt is expected


@pytest.mark.parametrize("value", [None, False, True])
def test_runtime_specs_apply_only_explicit_timing_override(value):
    args = SimpleNamespace(tracker="bytetrack", variable_dt=value)
    track_options = _tracker_spec(args, "aabb").option_dict
    eval_options = dict(_tracker_options(args, {"variable_dt": True}))
    if value is None:
        assert "variable_dt" not in track_options
        assert eval_options["variable_dt"] is True
    else:
        assert track_options["variable_dt"] is value
        assert eval_options["variable_dt"] is value


@pytest.mark.parametrize(
    "tracker", ["bytetrack", "botsort", "boosttrack", "deepocsort", "hybridsort", "occluboost", "ocsort", "strongsort"]
)
def test_builtin_timing_mode_is_fixed_default_only(tracker):
    schema = load_yaml_config(tracker)
    assert schema["variable_dt"] == {"default": False}
    assert load_tracker_defaults(tracker)["variable_dt"] is False
    assert "variable_dt" not in default_tune_config(schema)
    for parameter in KALMAN_NOISE_OPTIONS:
        assert schema[parameter] == {"type": "loguniform", "default": 1.0, "range": [0.01, 100]}
        assert default_tune_config(schema)[parameter] == 1.0
    for parameter in ("kf_time_unit", "kf_reference_dt_s"):
        assert set(schema[parameter]) == {"default"}
        assert parameter not in default_tune_config(schema)


def test_search_backends_skip_fixed_timing_mode():
    schema = {"variable_dt": {"default": True}, "threshold": {"type": "uniform", "default": 0.5, "range": [0.1, 0.9]}}
    ray = SimpleNamespace(uniform=lambda low, high: (low, high))
    assert yaml_to_tune_space(schema, ray) == {"threshold": (0.1, 0.9)}
    trial = SimpleNamespace(params={})
    trial.suggest_float = lambda parameter, *args: trial.params.setdefault(parameter, 0.1)
    yaml_to_optuna_define_space(schema)(trial)
    assert trial.params == {"threshold": 0.1}


@pytest.mark.parametrize("parameter", ["variable_dt", "kf_time_unit", "kf_reference_dt_s"])
def test_timing_mode_cannot_be_changed_to_a_search_dimension(parameter):
    with pytest.raises(ValueError, match=f"{parameter}.*fixed"):
        validate_tuning_config("bytetrack", {parameter: {"type": "choice", "default": False, "options": [False, True]}})


def test_saved_runtime_config_and_trial_identity_include_timing(tmp_path):
    args = SimpleNamespace(project=tmp_path, dataset_id="dataset", name="eval", variable_dt=False)
    overrides = {"track_thresh": 0.4}
    fixed_path = _output_directory(args, overrides)
    args.variable_dt = True
    timed_path = _output_directory(args, overrides)
    assert timed_path != fixed_path
    output = tmp_path / "best.yaml"
    write_trial_yaml({}, overrides, output, base_config={"variable_dt": True, "track_thresh": 0.5})
    assert yaml.safe_load(output.read_text()) == {"variable_dt": True, "track_thresh": 0.4}


@pytest.mark.parametrize("requested, saved", [(True, False), (False, True)])
def test_tune_resume_rejects_changed_timing_mode(requested, saved):
    tuner = Tuner(SimpleNamespace(tracker="bytetrack", variable_dt=requested))
    with pytest.raises(ValueError, match="same variable_dt"):
        tuner._validate_resumed_timing(
            [
                SimpleNamespace(
                    config={
                        "variable_dt": saved,
                        "kf_time_unit": "seconds" if saved else "frames",
                        "kf_reference_dt_s": DEFAULT_REFERENCE_DT_S,
                    }
                )
            ]
        )
    tuner._validate_resumed_timing(
        [
            SimpleNamespace(
                config={
                    "variable_dt": requested,
                    "kf_time_unit": "seconds" if requested else "frames",
                    "kf_reference_dt_s": DEFAULT_REFERENCE_DT_S,
                }
            )
        ]
    )


@pytest.mark.parametrize("replacement", [{"kf_time_unit": "frames"}, {"kf_reference_dt_s": 0.04}])
def test_resume_rejects_changed_units_or_reference(replacement):
    tuner = Tuner(SimpleNamespace(tracker="bytetrack", variable_dt=True))
    saved = {"variable_dt": True, "kf_time_unit": "seconds", "kf_reference_dt_s": DEFAULT_REFERENCE_DT_S, **replacement}
    with pytest.raises(ValueError, match="same variable_dt.*kf_reference_dt_s"):
        tuner._validate_resumed_timing([SimpleNamespace(config=saved)])


def test_resume_requires_explicit_saved_units():
    tuner = Tuner(SimpleNamespace(tracker="bytetrack", variable_dt=True))
    with pytest.raises(ValueError, match="lack explicit timing"):
        tuner._validate_resumed_timing([SimpleNamespace(config={"variable_dt": True})])
