"""Exercise shipped centroid search profiles through real backend samplers."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from boxmot.engine.tuning.backends.hyperopt_backend import yaml_to_hyperopt_space
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.search_space import (
    condition_tracker_schema,
    flatten_yaml_config,
    load_yaml_config,
    normalize_trial_config,
    yaml_to_tune_space,
)
from boxmot.trackers.common.config import get_tracker_config_class, load_tracker_defaults

TRACKERS = (
    "botsort", "bytetrack", "boosttrack", "occluboost", "ocsort",
    "deepocsort", "hybridsort", "strongsort", "sfsort", "maf_hda",
)
OVERLAP_METRICS = ("iou", "giou", "diou", "ciou", "hmiou")
OVERLAP_CONTROLS = {
    "botsort": ("match_thresh", "second_match_thresh", "unconfirmed_match_thresh", "proximity_thresh"),
    "bytetrack": ("match_thresh", "second_match_thresh", "unconfirmed_match_thresh"),
    "boosttrack": ("iou_threshold",),
    "occluboost": (
        "iou_threshold", "recovery_iou_thresh", "second_iou_thresh",
    ),
    "ocsort": ("iou_threshold",),
    "deepocsort": ("iou_threshold",),
    "hybridsort": ("iou_threshold", "longterm_reid_correction_thresh"),
    "strongsort": ("max_iou_dist",),
    "sfsort": ("match_th_first", "match_th_second", "match_th_first_m"),
    "maf_hda": ("iou_threshold", "appearance_gate", "appearance_upper"),
}


def _authored_controls(name: str) -> dict:
    """Use distinguishable runtime settings with each control's validated type."""
    values = {key: False if key == "appearance_gate" else 0.37 for key in OVERLAP_CONTROLS[name]}
    if "appearance_upper" in values:
        # Active centroid lower-threshold samples may exceed this inactive value.
        values["appearance_upper"] = 0.2
    return values


def _samples(schema: dict, backend: str, count: int = 1) -> Iterator[dict]:
    """Sample distributions without starting a cluster or tracker evaluation."""
    if backend == "optuna":
        import optuna

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=19))
        suggest = yaml_to_optuna_define_space(schema)
        for _ in range(count):
            trial = study.ask()
            constants = suggest(trial) or {}
            yield normalize_trial_config({**trial.params, **constants})
            study.tell(trial, 0.0)
    elif backend == "hyperopt":
        from hyperopt.pyll import as_apply, clone, rec_eval
        from hyperopt.pyll.stochastic import recursive_set_rng_kwarg

        space = yaml_to_hyperopt_space(schema)
        rng = np.random.default_rng(19)
        # Hyperopt's sample() clones and binds this graph for every draw.
        # The graph and RNG are local to this invocation, so prepare them once;
        # Each rec_eval uses a fresh memo and advances the same RNG. Disabling
        # intermediate memo collection keeps node values only until that return.
        expression = recursive_set_rng_kwarg(clone(as_apply(space)), rng)
        for _ in range(count):
            yield normalize_trial_config(rec_eval(expression, memo_gc=False))
    else:
        from ray import tune
        from ray.tune.search.variant_generator import generate_variants

        space = yaml_to_tune_space(schema, tune)
        rng = np.random.RandomState(19)
        for _ in range(count):
            _, resolved = next(generate_variants({"config": space}, random_state=rng))
            yield normalize_trial_config(resolved["config"])


@pytest.mark.parametrize("name", TRACKERS)
def test_profiles_declare_metric_applicability_without_neutral_defaults(name: str) -> None:
    schema = flatten_yaml_config(load_yaml_config(name))
    for parameter in OVERLAP_CONTROLS[name]:
        assert schema[parameter]["association"] == list(OVERLAP_METRICS)
        assert "centroid" not in schema[parameter].get("cases", {})
    radius = schema["centroid_max_distance"]
    assert radius["association"] == ["centroid"]
    assert radius["type"] == "uniform"
    assert radius["range"] == [0.005, 0.1]
    assert "cases" not in radius


@pytest.mark.parametrize("name", TRACKERS)
@pytest.mark.parametrize("metric", (*OVERLAP_METRICS, "centroid"))
@pytest.mark.parametrize("backend", ("random", "optuna", "hyperopt"))
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_shipped_profiles_sample_only_effective_centroid_controls(
    name: str, metric: str, backend: str, geometry: str,
) -> None:
    """The radius is the only searched centroid gate, including with scalar overrides."""
    defaults = load_tracker_defaults(name)
    defaults["asso_func"] = metric
    defaults.update(_authored_controls(name))
    schema = load_yaml_config(name)
    original = flatten_yaml_config(schema)
    schema["asso_func"] = {"default": metric}
    schema = condition_tracker_schema(schema, defaults, geometry=geometry)
    flat = flatten_yaml_config(schema)
    fixed = {
        key: entry["default"]
        for key, entry in flat.items()
        if set(entry) == {"default"}
    }
    sampled = next(_samples(schema, backend))
    resolved = {**defaults, **fixed, **sampled}
    if metric == "centroid":
        assert 0.005 <= sampled["centroid_max_distance"] <= 0.1
        searched_gates = {key for key in (*OVERLAP_CONTROLS[name], "centroid_max_distance") if "type" in flat[key]}
        assert searched_gates == {"centroid_max_distance"}
        for key in OVERLAP_CONTROLS[name]:
            assert key not in sampled
            assert flat[key] == {"default": defaults[key]}
            assert resolved[key] == defaults[key]
    else:
        assert "centroid_max_distance" not in sampled
        assert resolved["centroid_max_distance"] == defaults["centroid_max_distance"]
        for key in OVERLAP_CONTROLS[name]:
            if original[key].get("geometry", geometry) != geometry:
                assert key not in sampled
                assert resolved[key] == defaults[key]
            elif key in sampled:
                if "range" in flat[key]:
                    low, high = flat[key]["range"]
                    assert low <= sampled[key] <= high
                else:
                    assert sampled[key] in flat[key]["options"]
    config_type = get_tracker_config_class(name)
    config = config_type.from_mapping({key: resolved[key] for key in config_type.fields()})
    assert config.asso_func == metric


@pytest.mark.parametrize("backend", ("random", "optuna", "hyperopt"))
@pytest.mark.parametrize("name", TRACKERS)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_sampled_metric_selects_only_one_centroid_gate(backend: str, name: str, geometry: str) -> None:
    """Inactive metric controls keep authored values without drawing distributions."""
    schema = load_yaml_config(name)
    original = flatten_yaml_config(schema)
    defaults = load_tracker_defaults(name)
    defaults.update(_authored_controls(name))
    schema = condition_tracker_schema(schema, defaults, geometry=geometry)
    flat = flatten_yaml_config(schema)
    fixed = {key: entry["default"] for key, entry in flat.items() if set(entry) == {"default"}}
    seen = set()
    config_type = get_tracker_config_class(name)
    for sampled in _samples(schema, backend, count=64):
        resolved = {**defaults, **fixed, **sampled}
        metric = sampled["asso_func"]
        seen.add(metric)
        if metric == "centroid":
            assert 0.005 <= sampled["centroid_max_distance"] <= 0.1
            for key in OVERLAP_CONTROLS[name]:
                assert resolved[key] == defaults[key]
        else:
            assert sampled["centroid_max_distance"] == defaults["centroid_max_distance"]
            if name == "bytetrack" and metric == "iou":
                assert 0.7 <= sampled["match_thresh"] <= 0.9
                assert 0.1 <= sampled["second_match_thresh"] <= 0.9
        for key in OVERLAP_CONTROLS[name]:
            if original[key].get("geometry", geometry) != geometry:
                assert resolved[key] == defaults[key]
        config = config_type.from_mapping({key: resolved[key] for key in config_type.fields()})
        assert config.asso_func == metric
    assert seen == {*OVERLAP_METRICS, "centroid"}


@pytest.mark.parametrize("name", TRACKERS)
def test_inactive_controls_keep_typed_defaults_without_authored_overrides(name: str) -> None:
    defaults = load_tracker_defaults(name)
    defaults["asso_func"] = "centroid"
    schema = load_yaml_config(name)
    schema["asso_func"] = {"default": "centroid"}
    flat = flatten_yaml_config(condition_tracker_schema(schema, defaults, geometry="aabb"))
    for parameter in OVERLAP_CONTROLS[name]:
        assert flat[parameter] == {"default": defaults[parameter]}
