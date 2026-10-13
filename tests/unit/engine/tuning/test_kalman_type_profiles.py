"""Filter selection is searchable unless a runtime policy or profile fixes it."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.kalman_refinement import condition_kalman_type_schema
from boxmot.engine.tuning.search_space import (
    condition_tracker_schema,
    default_tune_config,
    flatten_yaml_config,
    load_yaml_config,
)
from boxmot.engine.tuning.tuner import Tuner
from boxmot.trackers import TrackerSpec, create_tracker
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.models import MotionModelKind, resolve_motion_model_kind
from tests.unit.engine.tuning.test_calibrated_tuner import _SuggestionTrial
from tests.unit.engine.tuning.test_calibrated_tuner import fake_tuning as fake_tuning
from tests.unit.engine.tuning.test_fixed_search_space import _sample


@pytest.mark.parametrize("name", sorted(KALMAN_TRACKER_NAMES))
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("backend", ["optuna", "hyperopt", "random"])
def test_default_filter_type_reaches_each_search_backend(name: str, geometry: str, backend: str) -> None:
    """Every Python box Kalman tracker searches all supported filters by default."""
    schema = load_yaml_config(name)
    options = [kind.value for kind in MotionModelKind]
    default_type = resolve_motion_model_kind(name).value
    assert flatten_yaml_config(schema)["kalman.type"] == {
        "type": "choice", "default": default_type, "options": options,
    }
    defaults = load_tracker_defaults(name)
    schema = condition_kalman_type_schema(schema, defaults, {})
    conditioned = condition_tracker_schema(schema, defaults, geometry=geometry)

    assert default_tune_config(conditioned)["kalman.type"] == default_type
    assert _sample(conditioned, backend)["kalman.type"] in options


@pytest.mark.parametrize("name", sorted(KALMAN_TRACKER_NAMES))
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_default_filter_type_constructs_tracker(name: str, geometry: str) -> None:
    """Factories use the concrete YAML filter for either box geometry."""
    default_type = resolve_motion_model_kind(name).value
    tracker = create_tracker(TrackerSpec(name, geometry=geometry))

    assert tracker.kalman_config.type == default_type


@pytest.mark.parametrize("name", sorted(KALMAN_TRACKER_NAMES))
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("filter_type", [None, "xywh"])
def test_filter_type_profile_constructs_tracker_and_stays_fixed_in_search(
    name: str, geometry: str, filter_type: str | None,
) -> None:
    """An explicit runtime selector, including null, fixes the supplied filter."""
    schema = load_yaml_config(name)
    defaults = load_tracker_defaults(name)
    assert defaults["kalman.type"] == resolve_motion_model_kind(name).value
    runtime = {**defaults, "kalman.type": filter_type}

    kalman = normalize_kalman_config(runtime, tracker_name=name)
    tracker = create_tracker(TrackerSpec(name, geometry=geometry), options={"kalman": {"type": filter_type}})
    assert tracker.kalman_config == kalman
    expected_kind = resolve_motion_model_kind(name, filter_type, is_obb=geometry == "obb")
    assert resolve_motion_model_kind(name, tracker.kalman_config.type, is_obb=tracker.is_obb) == expected_kind

    schema = condition_kalman_type_schema(schema, runtime, {"kalman.type": filter_type})
    conditioned = condition_tracker_schema(schema, runtime, geometry=geometry)
    assert flatten_yaml_config(conditioned)["kalman.type"] == {"default": filter_type}
    assert "kalman.type" not in default_tune_config(conditioned)
    trial = _SuggestionTrial()
    yaml_to_optuna_define_space(conditioned)(trial)
    assert "kalman.type" not in trial.params
    assert "asso_func" in trial.params


@pytest.mark.parametrize("backend", ["optuna", "hyperopt", "random"])
@pytest.mark.parametrize(
    "runtime_options,tracker_backend",
    [
        ({}, "cpp"),
        ({"kalman.adaptive_kf": True}, "python"),
        ({"calibration.filter": "xyhr"}, "python"),
        ({"kalman.noise.process_position_diagonal": [1.0] * 4}, "python"),
        ({"kalman.noise.by_class.0.process_position_diagonal": [1.0] * 4}, "python"),
    ],
)
def test_constrained_filter_type_is_excluded_from_search(
    backend: str, runtime_options: dict, tracker_backend: str,
) -> None:
    """Native filters, adaptive policies and coordinate priors keep their model."""
    runtime = {**load_tracker_defaults("occluboost"), **runtime_options}
    schema = condition_kalman_type_schema(
        load_yaml_config("occluboost"), runtime, {}, backend=tracker_backend,
    )
    conditioned = condition_tracker_schema(schema, runtime, geometry="aabb")

    assert flatten_yaml_config(conditioned)["kalman.type"] == {"default": "xyhr"}
    assert "kalman.type" not in _sample(conditioned, backend)


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
@pytest.mark.parametrize("filter_type", [None, "xywh"])
def test_driver_preserves_authored_filter_type(
    fake_tuning: SimpleNamespace, tmp_path: Path, search_alg: str, filter_type: str | None,
) -> None:
    """Loaded runtime selectors survive orchestration and every trial."""
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump({"kalman": {"type": filter_type}}), encoding="utf-8")
    driver = Tuner(fake_tuning.args(calibrate_kf=False, tracker_config=profile, search_alg=search_alg))
    driver.fit()

    assert flatten_yaml_config(driver._yaml_cfg)["kalman.type"] == {"default": filter_type}
    assert all(config["kalman.type"] == filter_type for config in fake_tuning.captured["trial_configs"])


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
def test_plain_driver_searches_filter_type_with_scalar_noise_overrides(
    fake_tuning: SimpleNamespace, search_alg: str,
) -> None:
    """Scalar noise refinement does not prevent ordinary filter selection."""
    driver = Tuner(
        fake_tuning.args(calibrate_kf=False, search_alg=search_alg),
        baseline_config={"kalman.noise.measurement_noise_scale": 2.0},
    )
    _, directory, _, _ = driver.fit()

    assert flatten_yaml_config(driver._yaml_cfg)["kalman.type"]["type"] == "choice"
    profile = json.loads((directory / "search-space.json").read_text())
    assert "kalman.type" not in profile["fixed_options"]
    assert _sample(driver._yaml_cfg, search_alg)["kalman.type"] == "xyah"


def test_direct_calibration_freezes_filter_type_on_resume(fake_tuning: SimpleNamespace) -> None:
    """Direct calibration binds its filter even when no type was authored."""
    first = Tuner(fake_tuning.args())
    _, directory, _, _ = first.fit()
    assert first._calibrated_fixed_options["kalman.type"] == "xywh"
    assert flatten_yaml_config(first._yaml_cfg)["kalman.type"] == {"default": "xywh"}
    fake_tuning.captured["restore_enabled"] = True

    resumed = Tuner(fake_tuning.args(calibrate_kf=False, resume_tune=directory, variable_dt=None))
    resumed.fit()

    assert fake_tuning.captured["events"].count("calibrate") == 1
    assert resumed._calibrated_fixed_options["kalman.type"] == "xywh"
    assert flatten_yaml_config(resumed._yaml_cfg)["kalman.type"] == {"default": "xywh"}
    assert all(config["kalman.type"] == "xywh" for config in fake_tuning.captured["trial_configs"])
