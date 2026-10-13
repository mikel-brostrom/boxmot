"""Authored filter types remain runtime settings throughout ordinary tuning."""

from __future__ import annotations

import pytest

from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.search_space import (
    condition_tracker_schema,
    default_tune_config,
    flatten_yaml_config,
    load_yaml_config,
)
from boxmot.trackers import TrackerSpec, create_tracker
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES
from boxmot.trackers.common.motion.models import resolve_motion_model_kind
from tests.unit.engine.tuning.test_calibrated_tuner import _SuggestionTrial


@pytest.mark.parametrize("name", sorted(KALMAN_TRACKER_NAMES))
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("filter_type", [None, "xywh"])
def test_filter_type_profile_constructs_tracker_and_stays_fixed_in_search(
    name: str, geometry: str, filter_type: str | None,
) -> None:
    """A component named type must not hide its group or become a search distribution."""
    schema = load_yaml_config(name)
    assert flatten_yaml_config(schema)["kalman.type"] == {"default": None}
    defaults = load_tracker_defaults(name)
    assert defaults["kalman.type"] is None
    runtime = {**defaults, "kalman.type": filter_type}

    kalman = normalize_kalman_config(runtime, tracker_name=name)
    tracker = create_tracker(TrackerSpec(name, geometry=geometry), options={"kalman": {"type": filter_type}})
    assert tracker.kalman_config == kalman
    expected_kind = resolve_motion_model_kind(name, filter_type, is_obb=geometry == "obb")
    assert resolve_motion_model_kind(name, tracker.kalman_config.type, is_obb=tracker.is_obb) == expected_kind

    conditioned = condition_tracker_schema(schema, runtime, geometry=geometry)
    assert flatten_yaml_config(conditioned)["kalman.type"] == {"default": filter_type}
    assert "kalman.type" not in default_tune_config(conditioned)
    trial = _SuggestionTrial()
    yaml_to_optuna_define_space(conditioned)(trial)
    assert "kalman.type" not in trial.params
    assert "asso_func" in trial.params
