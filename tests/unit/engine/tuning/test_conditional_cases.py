"""Categorical association cases retain correct values through sampling and resume."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from boxmot.engine.tuning.backends.hyperopt_backend import yaml_to_hyperopt_space
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.conditions import validate_dependencies
from boxmot.engine.tuning.postprocessing import write_trial_yaml
from boxmot.engine.tuning.search_space import (
    condition_tracker_schema,
    conditional_yaml_tree,
    default_tune_config,
    flatten_yaml_config,
    normalize_trial_config,
    validate_tuning_config,
    yaml_to_tune_space,
)
from boxmot.engine.tuning.tuner import Tuner
from boxmot.trackers.common.config import load_tracker_config, load_tracker_defaults
from tests.unit.engine.tuning.test_calibrated_tuner import fake_tuning as fake_tuning


def _schema() -> dict:
    return {
        "asso_func": {"type": "choice", "options": ["iou", "centroid"], "default": "iou"},
        "enabled": {
            "type": "choice", "options": [False, True], "default": False,
            "activates": {
                "threshold": {
                    "type": "uniform", "range": [0.1, 0.4], "default": 0.2,
                    "depends_on": "asso_func", "cases": {"centroid": {"range": [0.7, 1.0]}},
                },
            },
        },
        "centroid_max_distance": {
            "default": 0.05, "depends_on": "asso_func",
            "cases": {"centroid": {"type": "uniform", "range": [0.005, 0.1]}},
        },
    }


def _backend_samples(schema: dict, backend: str, *, count: int = 60) -> list[dict]:
    """Exercise native optimizer branches without starting tracker evaluations."""
    if backend == "random":
        from ray import tune
        from ray.tune.search.variant_generator import generate_variants

        space = yaml_to_tune_space(schema, tune)
        rng = np.random.RandomState(8)
        samples = [normalize_trial_config(next(generate_variants(space, random_state=rng))[1]) for _ in range(count)]
    elif backend == "hyperopt":
        from hyperopt.base import Domain
        from hyperopt.pyll.stochastic import sample

        space = yaml_to_hyperopt_space(schema)
        Domain(lambda config: config, space)  # Reject duplicated optimizer labels before sampling.
        rng = np.random.default_rng(8)
        samples = [normalize_trial_config(sample(space, rng=rng)) for _ in range(count)]
    else:
        import optuna

        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=8))
        space = yaml_to_optuna_define_space(schema)
        samples = []
        for _ in range(count):
            trial = study.ask()
            constants = space(trial) or {}
            samples.append(normalize_trial_config({**trial.params, **constants}))
            study.tell(trial, 1.0)
    return samples


@pytest.mark.parametrize("backend", ["random", "optuna", "hyperopt"])
@pytest.mark.parametrize("applicability", [False, True])
def test_cases_inside_activates_sample_only_enabled_ranges(backend: str, applicability: bool) -> None:
    """Native samplers must respect both dependencies, including inactive defaults."""
    schema = _schema()
    if applicability:
        schema["asso_func"]["options"] = ["iou", "giou", "centroid"]
        threshold = schema["enabled"]["activates"]["threshold"]
        threshold["association"] = ["iou", "giou"]
        threshold["cases"] = {"giou": {"range": [0.7, 1.0]}}
        schema["centroid_max_distance"] = {
            "default": 0.05, "type": "uniform", "range": [0.005, 0.1], "association": ["centroid"],
        }
    metrics = schema["asso_func"]["options"]
    schema = condition_tracker_schema(schema, {}, geometry="aabb")
    samples = _backend_samples(schema, backend)
    assert {(sample["asso_func"], sample["enabled"]) for sample in samples} == {
        (metric, enabled) for metric in metrics for enabled in (False, True)
    }
    for sample in samples:
        if not sample["enabled"] or (applicability and sample["asso_func"] == "centroid"):
            assert sample["threshold"] == 0.2
        else:
            case = "giou" if applicability else "centroid"
            low, high = (0.7, 1.0) if sample["asso_func"] == case else (0.1, 0.4)
            assert low <= sample["threshold"] <= high
        if sample["asso_func"] == "iou":
            assert sample["centroid_max_distance"] == 0.05


@pytest.mark.parametrize("backend", ["random", "optuna", "hyperopt"])
@pytest.mark.parametrize("fixed_metric", [False, True])
def test_inactive_association_parent_freezes_all_activation_descendants(backend: str, fixed_metric: bool) -> None:
    """A preserved true toggle must not activate children in an inapplicable metric."""
    schema = {
        "asso_func": {"default": "centroid", "type": "choice", "options": ["iou", "centroid"]},
        "max_age": {"default": 10, "type": "randint", "range": [5, 15]},
        "enabled": {
            "default": True, "type": "choice", "options": [False, True], "association": ["iou"],
            "activates": {
                "threshold": {
                    "default": 0.3, "type": "uniform", "range": [0.1, 0.9],
                    "depends_on": "asso_func", "cases": {"centroid": {"default": 0.8}},
                },
                "nested": {
                    "default": True, "type": "choice", "options": [False, True],
                    "activates": {
                        "weight": {
                            "default": 0.4, "type": "uniform", "range": [0.1, 0.9],
                            "depends_on": "asso_func", "cases": {"centroid": {"default": 0.9}},
                        },
                    },
                },
            },
        },
    }
    defaults = {"threshold": 0.37, "weight": 0.41}
    if fixed_metric:
        schema["asso_func"] = {"default": "centroid"}
    baseline = {"max_age": 10, **({} if fixed_metric else {"asso_func": "centroid"})}
    assert default_tune_config(schema, defaults=defaults) == baseline
    schema = condition_tracker_schema(schema, defaults, geometry="aabb")
    assert default_tune_config(schema) == baseline
    flat = flatten_yaml_config(schema)
    constants = {name: entry["default"] for name, entry in flat.items() if set(entry) == {"default"}}
    seen = set()
    for sampled in _backend_samples(schema, backend):
        config = {**constants, **sampled}
        seen.add(config["asso_func"])
        if config["asso_func"] == "centroid":
            assert {name: config[name] for name in ("enabled", "threshold", "nested", "weight")} == {
                "enabled": True, "threshold": 0.37, "nested": True, "weight": 0.41,
            }
    assert seen == ({"centroid"} if fixed_metric else {"iou", "centroid"})


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"depends_on": "missing"}, "unknown depends_on"),
        ({"cases": {"wrong": {"default": 1.0}}}, "unknown case"),
        ({"cases": []}, "cases mapping"),
        ({"cases": {"centroid": "wrong"}}, "nonempty mapping"),
        ({"cases": {"centroid": {"unknown": 1}}}, "unsupported case"),
    ],
)
def test_invalid_dependency_metadata_fails_before_sampling(update: dict, message: str) -> None:
    schema = _schema()
    schema["centroid_max_distance"].update(update)
    flat = flatten_yaml_config(schema)
    _, _, parents = conditional_yaml_tree(schema)
    with pytest.raises(ValueError, match=message):
        validate_dependencies(flat, parents)


def test_dependency_cycle_through_activation_is_rejected() -> None:
    schema = {
        "parent": {
            "default": True, "type": "choice", "options": [False, True],
            "depends_on": "child", "cases": {True: {"default": True}},
            "activates": {"child": {"default": True, "type": "choice", "options": [False, True]}},
        },
    }
    _, _, parents = conditional_yaml_tree(schema)
    with pytest.raises(ValueError, match="Cyclic"):
        validate_dependencies(flatten_yaml_config(schema), parents)


def test_invalid_case_distribution_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import boxmot.engine.tuning.search_space as search

    schema = _schema()
    schema["centroid_max_distance"]["cases"]["centroid"]["range"] = []
    monkeypatch.setattr(search, "load_tracker_defaults", lambda name: dict.fromkeys(flatten_yaml_config(schema)))
    with pytest.raises(ValueError, match="invalid search metadata"):
        validate_tuning_config("example", schema)


@pytest.mark.parametrize(
    ("association", "message"),
    [
        ([], "nonempty association list"),
        ("centroid", "nonempty association list"),
        (["unknown"], "unknown association metric"),
        ([None], "unknown association metric"),
        (["centroid", "centroid"], "duplicate association"),
    ],
)
def test_invalid_association_applicability_is_rejected(
    monkeypatch: pytest.MonkeyPatch, association: object, message: str,
) -> None:
    import boxmot.engine.tuning.search_space as search

    schema = _schema()
    schema["centroid_max_distance"]["association"] = association
    monkeypatch.setattr(search, "load_tracker_defaults", lambda name: dict.fromkeys(flatten_yaml_config(schema)))
    with pytest.raises(ValueError, match=message):
        validate_tuning_config("example", schema)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("missing_selector", "without an asso_func selector"),
        ("numeric_selector", "categorical or fixed"),
        ("unknown_selector", "recognized asso_func"),
        ("self", "for itself"),
        ("different_parent", "different depends_on parent"),
        ("missing_default", "runtime default"),
        ("cycle", "Cyclic"),
    ],
)
def test_association_dependency_contract_is_validated(problem: str, message: str) -> None:
    schema = _schema()
    distance = schema["centroid_max_distance"]
    distance["association"] = ["centroid"]
    if problem == "missing_selector":
        del schema["asso_func"]
    elif problem == "numeric_selector":
        schema["asso_func"] = {"type": "uniform", "range": [0.1, 0.9], "default": 0.2}
    elif problem == "unknown_selector":
        schema["asso_func"]["options"] = ["unknown"]
    elif problem == "self":
        schema["asso_func"]["association"] = ["centroid"]
    elif problem == "different_parent":
        distance["depends_on"] = "enabled"
    elif problem == "missing_default":
        del distance["default"]
    else:
        distance["activates"] = {"asso_func": schema.pop("asso_func")}
    # Isolate applicability validation from the generic test fixture's cases.
    del schema["enabled"]["activates"]["threshold"]["depends_on"]
    del schema["enabled"]["activates"]["threshold"]["cases"]
    _, _, parents = conditional_yaml_tree(schema)
    with pytest.raises(ValueError, match=message):
        validate_dependencies(flatten_yaml_config(schema), parents)


@pytest.mark.parametrize("metric", ["iou", "centroid"])
def test_applicability_baseline_contains_only_selected_distributions(metric: str) -> None:
    schema = {
        "asso_func": {"default": "iou", "type": "choice", "options": ["iou", "centroid"]},
        "threshold": {"default": 0.2, "type": "uniform", "range": [0.1, 0.9], "association": ["iou"]},
        "radius": {"default": 0.05, "type": "uniform", "range": [0.005, 0.1], "association": ["centroid"]},
    }
    baseline = default_tune_config(schema, defaults={"asso_func": metric, "threshold": 0.3, "radius": 0.04})
    assert baseline == {"asso_func": metric, **({"threshold": 0.3} if metric == "iou" else {"radius": 0.04})}


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
@pytest.mark.parametrize("explicit_threshold", [None, 0.73])
def test_driver_preserves_inactive_metric_values_and_scalar_overrides(
    fake_tuning, tmp_path: Path, search_alg: str, explicit_threshold: float | None,
) -> None:
    gates = ("match_thresh", "second_match_thresh", "unconfirmed_match_thresh")
    overrides = {} if explicit_threshold is None else dict.fromkeys(gates, explicit_threshold)
    driver = Tuner(
        fake_tuning.args(tracker="bytetrack", calibrate_kf=False, asso_func="centroid", search_alg=search_alg),
        baseline_config=overrides,
    )
    _, directory, _, _ = driver.fit()
    expected = {**load_tracker_defaults("bytetrack"), **overrides}
    profile = json.loads((directory / "search-space.json").read_text())
    for key in gates:
        assert driver._runtime_config[key] == expected[key]
        assert profile["fixed_options"][key] == expected[key]
    assert "centroid_max_distance" not in profile["fixed_options"]
    for index, trial in enumerate(fake_tuning.captured["trial_configs"]):
        assert trial["asso_func"] == "centroid"
        assert all(trial[key] == expected[key] for key in gates)
        assert 0.005 <= trial["centroid_max_distance"] <= 0.1
        path = tmp_path / f"trial-{index}.yaml"
        write_trial_yaml(driver._yaml_cfg, trial, path, base_config=driver._runtime_config, tracker_name="bytetrack")
        restored = load_tracker_config("bytetrack", path)
        assert all(restored[key] == expected[key] for key in gates)
        assert restored["centroid_max_distance"] == trial["centroid_max_distance"]


def test_explicit_profile_association_is_fixed_and_resumes_without_search_drift(
    fake_tuning, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = tmp_path / "tracker.yaml"
    profile.write_text(yaml.safe_dump({"asso_func": "centroid", "second_match_thresh": 0.73}))
    args = fake_tuning.args(tracker="bytetrack", calibrate_kf=False, tracker_config=profile)
    driver = Tuner(args)
    _, directory, _, _ = driver.fit()
    assert driver._yaml_cfg["asso_func"] == {"default": "centroid"}
    saved = deepcopy(fake_tuning.captured["trial_configs"])
    fake_tuning.captured["restore_enabled"] = True
    Tuner(fake_tuning.args(
        tracker="bytetrack", calibrate_kf=False, tracker_config=profile, resume_tune=directory,
    )).fit()
    assert fake_tuning.captured["trial_configs"][len(saved):] == saved
    import boxmot.engine.tuning.tuner as tuner_module

    load_schema = tuner_module.load_yaml_config

    def changed_applicability(name: str) -> dict:
        schema = load_schema(name)
        schema["second_match_thresh"]["association"].append("centroid")
        return schema

    monkeypatch.setattr(tuner_module, "load_yaml_config", changed_applicability)
    with pytest.raises(ValueError, match="search space does not match"):
        Tuner(fake_tuning.args(
            tracker="bytetrack", calibrate_kf=False, tracker_config=profile, resume_tune=directory,
        )).fit()


@pytest.mark.parametrize("metric", ["iou", "centroid"])
def test_fixed_cases_preserve_unmatched_baselines_and_inactive_values(metric: str) -> None:
    """Only the selected fixed case replaces its scalar starting value."""
    schema = _schema()
    schema["asso_func"] = {"default": metric}
    schema["enabled"]["activates"]["threshold"]["cases"]["centroid"] = {"default": 1.0}
    defaults = {"threshold": 0.3, "centroid_max_distance": 0.04}
    conditioned = condition_tracker_schema(schema, defaults, geometry="aabb")
    flat = flatten_yaml_config(conditioned)
    if metric == "centroid":
        assert flat["threshold"] == {"default": 1.0}
        assert flat["centroid_max_distance"]["default"] == 0.04
    else:
        assert flat["threshold"]["default"] == 0.3
        assert flat["threshold"]["type"] == "uniform"
        assert flat["centroid_max_distance"] == {"default": 0.04}


def test_baseline_uses_only_active_search_dimensions() -> None:
    schema = _schema()
    assert default_tune_config(schema) == {"asso_func": "iou", "enabled": False}
    baseline = default_tune_config(schema, defaults={"asso_func": "centroid", "enabled": True})
    assert baseline == {"asso_func": "centroid", "enabled": True, "threshold": 0.2, "centroid_max_distance": 0.05}


def test_baseline_follows_resolved_default_for_chained_cases() -> None:
    schema = {
        "mode": {"default": "centroid", "type": "choice", "options": ["iou", "centroid"]},
        "enabled": {
            "default": False, "type": "choice", "options": [False, True],
            "depends_on": "mode", "cases": {"centroid": {"default": True}},
        },
        "threshold": {
            "default": 0.5, "depends_on": "enabled",
            "cases": {True: {"type": "uniform", "range": [0.1, 0.9]}},
        },
    }
    assert default_tune_config(schema, defaults={"enabled": False}) == {"mode": "centroid", "threshold": 0.5}


def test_optuna_different_case_distributions_survive_searcher_restore(tmp_path: Path) -> None:
    """Real Ray suggestions keep categorical/type changes separate and export canonical keys."""
    from ray.tune.search.optuna import OptunaSearch

    schema = {
        "mode": {"default": "iou", "type": "choice", "options": ["iou", "centroid"]},
        "policy": {
            "default": "slow", "type": "choice", "options": ["slow", "fast"],
            "depends_on": "mode", "cases": {"centroid": {"options": ["near", "far"]}},
        },
        "threshold": {
            "default": 0.5, "type": "uniform", "range": [0.1, 0.9],
            "depends_on": "mode", "cases": {"centroid": {"type": "randint", "range": [2, 5]}},
        },
    }
    space = yaml_to_optuna_define_space(schema)
    search = OptunaSearch(space=space, metric="score", mode="max", seed=8)
    observed = set()
    for index in range(30):
        if index == 15:
            checkpoint = tmp_path / "search.pkl"
            search.save(str(checkpoint))
            search = OptunaSearch(space=space, metric="score", mode="max", seed=8)
            search.restore(str(checkpoint))
        config = normalize_trial_config(search.suggest(str(index)))
        assert set(config) == {"mode", "policy", "threshold"}
        observed.add(config["mode"])
        if config["mode"] == "centroid":
            assert config["policy"] in {"near", "far"}
            assert isinstance(config["threshold"], int) and 2 <= config["threshold"] < 5
        else:
            assert config["policy"] in {"slow", "fast"}
            assert 0.1 <= config["threshold"] <= 0.9
        search.on_trial_complete(str(index), result={"score": 1.0})
    assert observed == {"iou", "centroid"}


def test_optuna_first_point_retains_baseline_selector() -> None:
    from ray.tune.search.optuna import OptunaSearch

    schema = _schema()
    space = yaml_to_optuna_define_space(schema)
    baseline = space.baseline(default_tune_config(schema))
    assert baseline["asso_func"] == "iou"
    assert baseline["enabled"] is False
    search = OptunaSearch(space=space, metric="score", mode="max", seed=0, points_to_evaluate=[baseline])

    config = normalize_trial_config(search.suggest("baseline"))

    assert config["asso_func"] == "iou"
    assert config["enabled"] is False
    assert config["threshold"] == 0.2
    assert config["centroid_max_distance"] == 0.05
