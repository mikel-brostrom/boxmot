"""Native algorithm controls share the tracker search space with Python."""

from __future__ import annotations

import pytest

from boxmot.engine.tuning.conditions import map_entries
from boxmot.engine.tuning.search_space import (
    condition_tracker_schema,
    expand_yaml_groups,
    flatten_yaml_config,
    load_yaml_config,
    validate_tuning_config,
)
from boxmot.trackers.common.config import load_tracker_defaults
from tests.unit.engine.tuning.test_fixed_search_space import _sample

_ALGORITHM_CONTROLS = {
    "botsort": {"alpha_fixed_emb", "duplicate_distance_thresh", "fuse_unconfirmed_associate"},
    "bytetrack": {"duplicate_distance_thresh", "fuse_first_associate", "fuse_unconfirmed_associate"},
    "occluboost": {
        "alpha_fixed_emb", "appearance_thresh", "appearance_iou_fraction", "dlo_soft_alpha",
        "dlo_soft_power", "dlo_threshold_start", "dlo_threshold_end", "duo_mahalanobis_thresh",
        "duo_iou_thresh", "track_confidence_decay", "track_confidence_warmup",
    },
}


@pytest.mark.parametrize("name", _ALGORITHM_CONTROLS)
@pytest.mark.parametrize("search_backend", ["optuna", "hyperopt", "random"])
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("association", ["iou", "centroid"])
def test_new_native_algorithm_controls_remain_searchable(
    name: str, search_backend: str, geometry: str, association: str,
) -> None:
    """Every applicable new control reaches each optimizer without a backend restriction."""
    schema = load_yaml_config(name)
    authored = flatten_yaml_config(schema)
    defaults = load_tracker_defaults(name)
    assert all("backend" not in entry for entry in authored.values())
    assert not {"backend", "tracker_backend"}.intersection(authored)

    def enable_children(parameter: str, entry: dict) -> dict:
        if "activates" not in entry:
            return entry
        defaults[parameter] = True
        return {**entry, "options": [True]}

    schema = map_entries(expand_yaml_groups(schema), enable_children)
    defaults["asso_func"] = association
    schema["asso_func"] = {"default": association}
    conditioned = condition_tracker_schema(schema, defaults, geometry=geometry)
    controls = _ALGORITHM_CONTROLS[name]
    active = {
        key for key in controls
        if authored[key].get("geometry", geometry) == geometry
        and association in authored[key].get("association", [association])
    }

    assert controls.intersection(_sample(conditioned, search_backend)) == active


@pytest.mark.parametrize("parameter", ["backend", "tracker_backend"])
def test_tracker_backend_cannot_be_a_search_dimension(parameter: str) -> None:
    """Backend selection belongs to the run, not to sampled algorithm settings."""
    schema = {parameter: {"type": "choice", "default": "python", "options": ["python", "cpp"]}}
    with pytest.raises(ValueError, match="parameters absent from its runtime defaults"):
        validate_tuning_config("botsort", schema)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_algorithm_controls_cannot_be_restricted_to_one_backend(backend: str) -> None:
    """Reject metadata that could conceal missing native algorithm support."""
    schema = {
        "alpha_fixed_emb": {
            "backend": backend, "type": "uniform", "default": 0.9, "range": [0.7, 1.0],
        },
    }
    with pytest.raises(ValueError, match="unsupported metadata.*backend"):
        validate_tuning_config("botsort", schema)
