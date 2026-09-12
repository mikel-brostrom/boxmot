"""Verify random-search reproducibility without starting a Ray cluster."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from boxmot.engine.tuning.backends.random_backend import RandomBackend

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
