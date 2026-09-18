"""Search backends receive only effective, explicitly searchable dimensions."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.tuning.backends.hyperopt_backend import yaml_to_hyperopt_space
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.kalman_refinement import refine_kalman_schema
from boxmot.engine.tuning.mask_guidance import condition_mask_guidance_schema, prepare_mask_guidance_tuning
from boxmot.engine.tuning.search_space import (
    condition_tracker_schema,
    flatten_yaml_config,
    load_yaml_config,
    normalize_trial_config,
    validate_tuning_config,
    yaml_to_tune_space,
)
from boxmot.engine.tuning.tuner import Tuner
from boxmot.trackers.common.mask_guidance import MASK_GUIDANCE_OPTIONS
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS, KALMAN_TIMING_OPTIONS
from tests.unit.engine.tuning.test_calibrated_tuner import _SuggestionTrial
from tests.unit.engine.tuning.test_calibrated_tuner import fake_tuning as fake_tuning


def test_driver_records_filtered_search_and_preserves_fixed_trial_values(fake_tuning, tmp_path: Path) -> None:
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"checkpoint identity only; no inference during tuning fixture")
    args = fake_tuning.args(
        tracker="occluboost", edgetam=True, mask_guidance_weights=checkpoint,
        mask_guidance_max_objects=64, device="cpu", calibrate_kf=True,
    )
    driver = Tuner(args)

    _, directory, _, _ = driver.fit()

    profile = json.loads((directory / "search-space.json").read_text())
    assert profile["schema"] == driver._yaml_cfg
    sampled = _sample(profile["schema"], "optuna")
    excluded = {*KALMAN_NOISE_OPTIONS, "asso_func", "edgetam.max_objects"}
    excluded.update(key for key in profile["fixed_options"] if key.startswith("obb_"))
    assert len([key for key in excluded if key.startswith("obb_")]) == 7
    assert not excluded.intersection(sampled)
    assert set(MASK_GUIDANCE_OPTIONS) - {"edgetam.max_objects"} <= sampled.keys()
    for trial in fake_tuning.captured["trial_configs"]:
        assert all(trial[key] == profile["fixed_options"][key] for key in excluded)


def _sample(schema: dict, backend: str) -> dict:
    """Exercise each converter without starting a tuning job or model."""
    if backend == "optuna":
        trial = _SuggestionTrial()
        yaml_to_optuna_define_space(schema)(trial)
        return trial.params
    if backend == "hyperopt":
        stochastic = pytest.importorskip("hyperopt.pyll.stochastic")
        return normalize_trial_config(
            stochastic.sample(yaml_to_hyperopt_space(schema), rng=np.random.default_rng(4))
        )
    fake_ray = SimpleNamespace(
        choice=lambda choices: choices[0],
        uniform=lambda low, high: (low + high) / 2,
        loguniform=lambda low, high: (low * high) ** 0.5,
        randint=lambda low, high: low,
        qrandint=lambda low, high, step: low,
    )
    return yaml_to_tune_space(schema, fake_ray)


@pytest.mark.parametrize("backend", ["optuna", "hyperopt", "random"])
@pytest.mark.parametrize("refine", [False, True])
@pytest.mark.parametrize("guided", [False, True])
def test_temporal_guidance_excludes_geometry_and_fixed_kf_parameters(
    backend: str, refine: bool, guided: bool,
) -> None:
    args = SimpleNamespace(
        tracker="occluboost", tracker_backend="python", geometry="aabb", asso_func="iou",
        edgetam=guided, device="cpu", sequence_workers=1,
    )
    calibrated = {name: float(index + 2) for index, name in enumerate(KALMAN_NOISE_OPTIONS)}
    runtime = resolve_tracker_options(args, calibrated, include_defaults=True)
    runtime = prepare_mask_guidance_tuning(args, runtime)
    original = load_yaml_config("occluboost")
    snapshot = deepcopy(original)
    selected = "kalman.noise.measurement_noise_scale"
    schema = refine_kalman_schema(original, runtime, ("measurement_noise_scale",) if refine else ())
    schema = condition_mask_guidance_schema(schema, args, runtime)
    conditioned = condition_tracker_schema(schema, runtime, geometry="aabb")
    sampled = _sample(conditioned, backend)
    flat = flatten_yaml_config(conditioned)

    fixed = {
        *KALMAN_NOISE_OPTIONS, *KALMAN_TIMING_OPTIONS, "kalman.variable_dt", "kalman.adaptive_kf",
        "asso_func", *(key for key in runtime if key.startswith("obb_")),
    }
    if refine:
        fixed.remove(selected)
        assert selected in sampled
        assert flat[selected]["range"] == [runtime[selected] / 4, runtime[selected] * 4]
    assert not fixed.intersection(sampled)
    assert all(flat[key] == {"default": runtime[key]} for key in fixed)
    assert "max_age" in sampled
    guidance_keys = set(MASK_GUIDANCE_OPTIONS) if guided else set()
    assert set(MASK_GUIDANCE_OPTIONS).intersection(sampled) == guidance_keys
    assert set(MASK_GUIDANCE_OPTIONS).intersection(flat) == guidance_keys
    assert all("geometry" not in entry for entry in flat.values())
    assert original == snapshot


def test_hyperopt_choice_values_share_conditional_child_distributions() -> None:
    """Several truthy choices reuse child labels while zero disables the branch."""
    domain_type = pytest.importorskip("hyperopt.base").Domain
    sample = pytest.importorskip("hyperopt.pyll.stochastic").sample
    schema = {
        "strength": {
            "type": "choice", "default": 1.0, "options": [0.0, 0.25, 0.5, 1.0, 2.0],
            "activates": {
                "lower_threshold": {"type": "uniform", "default": 0.1, "range": [0.0, 0.5]},
                "upper_threshold": {"type": "uniform", "default": 0.9, "range": [0.5, 1.0]},
            },
        },
    }
    space = yaml_to_hyperopt_space(schema)
    domain = domain_type(lambda config: config, space)
    children = {"lower_threshold", "upper_threshold"}
    assert set(domain.params) == {"strength", *children}
    rng = np.random.default_rng(2)
    observed = set()
    for _ in range(100):
        sampled = normalize_trial_config(sample(space, rng=rng))
        observed.add(sampled["strength"])
        assert children.intersection(sampled) == (children if sampled["strength"] else set())
    assert observed == {0.0, 0.25, 0.5, 1.0, 2.0}


@pytest.mark.parametrize("backend", ["optuna", "hyperopt", "random"])
@pytest.mark.parametrize("enabled", [False, True])
def test_fixed_conditional_parents_follow_runtime_values(backend: str, enabled: bool) -> None:
    schema = {
        "enabled": {
            "default": not enabled,
            "activates": {
                "weight": {"type": "uniform", "default": 0.3, "range": [0.1, 1.0]},
                "nested": {
                    "default": False,
                    "activates": {"nested_weight": {"type": "uniform", "default": 0.4, "range": [0.1, 1.0]}},
                },
            },
        },
        "independent": {"type": "uniform", "default": 0.2, "range": [0.1, 1.0]},
    }
    defaults = {"enabled": enabled, "weight": 0.7, "nested": True, "nested_weight": 0.8, "independent": 0.2}
    snapshot = deepcopy(schema)

    conditioned = condition_tracker_schema(schema, defaults, geometry="aabb")
    sampled = _sample(conditioned, backend)

    assert set(sampled) == ({"weight", "nested_weight", "independent"} if enabled else {"independent"})
    assert conditioned["enabled"] == {"default": enabled}
    assert conditioned["nested"] == {"default": True}
    if not enabled:
        assert conditioned["weight"] == {"default": 0.7}
        assert conditioned["nested_weight"] == {"default": 0.8}
    assert schema == snapshot


@pytest.mark.parametrize("backend", ["optuna", "hyperopt", "random"])
def test_geometry_mismatch_freezes_every_descendant(backend: str) -> None:
    schema = {
        "obb_parent": {
            "geometry": "obb", "type": "choice", "default": True, "options": [False, True],
            "activates": {
                "child": {"geometry": "aabb", "type": "uniform", "default": 0.2, "range": [0.0, 1.0]},
                "nested": {
                    "type": "choice", "default": True, "options": [False, True],
                    "activates": {"leaf": {"type": "uniform", "default": 0.3, "range": [0.0, 1.0]}},
                },
            },
        },
        "active": {"geometry": "aabb", "type": "uniform", "default": 0.5, "range": [0.0, 1.0]},
    }
    defaults = {"obb_parent": True, "child": 0.6, "nested": True, "leaf": 0.7, "active": 0.5}
    snapshot = deepcopy(schema)

    conditioned = condition_tracker_schema(schema, defaults, geometry="aabb")

    assert set(_sample(conditioned, backend)) == {"active"}
    assert all(conditioned[key] == {"default": defaults[key]} for key in ("obb_parent", "child", "nested", "leaf"))
    assert "geometry" not in conditioned["active"]
    assert schema == snapshot


def test_variable_conditional_parent_keeps_its_branch_structure() -> None:
    schema = {
        "enabled": {
            "type": "choice", "default": False, "options": [False, True],
            "activates": {"weight": {"geometry": "aabb", "type": "uniform", "default": 0.3, "range": [0, 1]}},
        }
    }
    snapshot = deepcopy(schema)

    conditioned = condition_tracker_schema(schema, {"enabled": False, "weight": 0.3}, geometry="aabb")

    assert conditioned["enabled"]["options"] == [False, True]
    assert "weight" in conditioned["enabled"]["activates"]
    assert "geometry" not in conditioned["enabled"]["activates"]["weight"]
    assert schema == snapshot


@pytest.mark.parametrize("geometry", ["polygon", "", True])
def test_tuning_metadata_rejects_invalid_geometry(geometry: object) -> None:
    schema = {"max_age": {"geometry": geometry, "type": "randint", "default": 30, "range": [1, 100]}}

    with pytest.raises(ValueError, match="geometry"):
        validate_tuning_config("occluboost", schema)


@pytest.mark.parametrize("backend", ["optuna", "hyperopt", "random"])
def test_obb_search_excludes_aabb_thresholds_and_ams_without_losing_shared_weights(backend: str) -> None:
    args = SimpleNamespace(tracker="occluboost", tracker_backend="python", geometry="obb")
    runtime = resolve_tracker_options(args, include_defaults=True)
    schema = condition_tracker_schema(load_yaml_config("occluboost"), runtime, geometry="obb")
    sampled = _sample(schema, backend)
    flat = flatten_yaml_config(schema)
    inactive = {
        "max_age", "det_thresh", "iou_threshold", "new_track_thresh", "instant_confirm_thresh",
        "recovery_max_age", "second_iou_thresh", "lambda_mhd", "lambda_shape", "s_sim_corr",
        "lambda_emb_multiplier", *(key for key in runtime if key.startswith("kalman.ams.")),
    }

    assert not inactive.intersection(sampled)
    assert all(flat[key] == {"default": runtime[key]} for key in inactive)
    assert {"obb_det_thresh", "obb_iou_threshold", "obb_max_age", "lambda_iou"}.issubset(sampled)
    assert flat["recovery_iou_thresh"]["type"] == "uniform"
