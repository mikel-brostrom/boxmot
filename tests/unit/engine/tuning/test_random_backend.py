"""Verify random-search reproducibility without starting a Ray cluster."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from boxmot.engine.tuning.backends.random_backend import RandomBackend
from boxmot.engine.tuning.search_space import normalize_trial_config, yaml_to_tune_space

tune = pytest.importorskip("ray.tune")

_SCHEMA = {
    "det_thresh": {"type": "uniform", "range": [0.1, 0.9]},
    "max_age": {"type": "randint", "range": [1, 100]},
    "asso_func": {"type": "choice", "options": ["iou", "giou", "diou"]},
}


@pytest.fixture
def trainable_name() -> Iterator[str]:
    """Register a trainable for trial validation without ever executing it."""
    from ray.tune.registry import TRAINABLE_CLASS, _global_registry, register_trainable

    name = "boxmot-test-random-backend"

    def trainable(_config: dict[str, Any]) -> None:
        pytest.fail("Sampling search configurations must not execute training.")

    register_trainable(name, trainable)
    try:
        yield name
    finally:
        _global_registry.unregister(TRAINABLE_CLASS, name)


def _backend(seed: int | None) -> RandomBackend:
    """Use continuous, integer, and categorical tracker tuning parameters."""
    return RandomBackend(yaml_cfg=_SCHEMA, opt_metrics=["HOTA"], opt_modes=["max"], seed=seed)


def _sample_configs(seed: int, trainable_name: str, storage_path: Path) -> list[dict[str, Any]]:
    """Request actual Ray trials so comparison covers the sampler's behavior."""
    searcher, param_space = _backend(seed).build(tune)
    assert searcher is not None
    searcher.add_configurations(
        {
            "seeded-random": {
                "run": trainable_name,
                "config": param_space,
                "num_samples": 6,
                "storage_path": str(storage_path),
            }
        }
    )
    configs = []
    for _ in range(6):
        trial = searcher.next_trial()
        assert trial is not None
        configs.append(trial.config)
        searcher.on_trial_complete(trial.trial_id)
    assert searcher.next_trial() is None
    return configs


def test_unseeded_random_search_preserves_default_ray_sampler() -> None:
    """Omitting a seed continues to let the caller configure Ray's default search."""
    searcher, param_space = _backend(None).build(tune)

    assert searcher is None
    assert set(param_space) == set(_SCHEMA)
    assert param_space["det_thresh"].lower == 0.1
    assert param_space["det_thresh"].upper == 0.9
    assert param_space["max_age"].lower == 1
    assert param_space["max_age"].upper == 100
    assert param_space["asso_func"].categories == ["iou", "giou", "diou"]


@pytest.mark.parametrize("seed", (0, 19))
def test_seed_controls_actual_random_trial_configurations(seed: int, trainable_name: str, tmp_path: Path) -> None:
    """Equal seeds replay the same samples, including the valid zero seed."""
    first = _sample_configs(seed, trainable_name, tmp_path / "first")
    repeated = _sample_configs(seed, trainable_name, tmp_path / "repeated")
    different = _sample_configs(seed + 1, trainable_name, tmp_path / "different")

    assert first == repeated
    assert first != different
    assert len({config["det_thresh"] for config in first}) > 1
    for config in first:
        assert 0.1 <= config["det_thresh"] <= 0.9
        assert 1 <= config["max_age"] < 100
        assert config["asso_func"] in ("iou", "giou", "diou")


@pytest.mark.parametrize("case_location", ("none", "independent", "child"))
def test_random_sampler_draws_only_active_children(case_location: str) -> None:
    """Check Ray's sampled dimensions, not just defaults restored in trial config."""
    from ray.tune.search.variant_generator import generate_variants

    schema = {
        "enabled": {
            "default": False, "type": "choice", "options": [False, True],
            "activates": {
                "weight": {"default": 0.4, "type": "uniform", "range": [0.1, 0.9]},
                "nested": {
                    "default": True, "type": "choice", "options": [False, True],
                    "activates": {
                        "nested_weight": {"default": 0.3, "type": "uniform", "range": [0.1, 0.9]},
                    },
                },
            },
        },
    }
    if case_location != "none":
        schema.update({
            "mode": {"default": "off", "type": "choice", "options": ["off", "on"]},
            "case_weight": {
                "default": 0.2, "depends_on": "mode",
                "cases": {"on": {"type": "uniform", "range": [0.1, 0.9]}},
            },
        })
    if case_location == "child":
        schema["enabled"]["activates"]["weight"].update({
            "depends_on": "mode", "cases": {"on": {"range": [0.2, 0.8]}},
        })
    space = yaml_to_tune_space(schema, tune)
    rng = np.random.RandomState(8)
    seen = set()
    for _ in range(48):
        sampled, raw_config = next(generate_variants(space, random_state=rng))
        config = normalize_trial_config(raw_config)
        dimensions = {path[-1] for path in sampled}
        enabled, nested = config["enabled"], config["nested"]
        seen.add((enabled, nested))
        assert ("weight" in dimensions) is enabled
        assert ("nested" in dimensions) is enabled
        assert ("nested_weight" in dimensions) is (enabled and nested)
        if not enabled:
            assert config["weight"] == 0.4
            assert config["nested"] is True
        if not (enabled and nested):
            assert config["nested_weight"] == 0.3
        if case_location != "none":
            assert ("case_weight" in dimensions) is (config["mode"] == "on")
            if config["mode"] == "off":
                assert config["case_weight"] == 0.2
    assert seen == {(False, True), (True, False), (True, True)}
