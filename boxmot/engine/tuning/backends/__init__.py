"""Tune search backends."""
from __future__ import annotations

from typing import Any

import click

from boxmot.engine.tuning.backends.base import BaseTuneBackend
from boxmot.engine.tuning.backends.hyperopt_backend import HyperOptBackend
from boxmot.engine.tuning.backends.optuna_backend import OptunaBackend
from boxmot.engine.tuning.backends.random_backend import RandomBackend

SEARCH_BACKENDS = ("optuna", "hyperopt", "random")

_BACKEND_REGISTRY: dict[str, type[BaseTuneBackend]] = {
    "optuna": OptunaBackend,
    "hyperopt": HyperOptBackend,
    "random": RandomBackend,
}


def resolve_search_backend(args: Any) -> str:
    """Resolve the requested search backend from args, with backwards-compatible aliases."""
    raw_backend = (
        getattr(args, "search_alg", None)
        or getattr(args, "search_backend", None)
        or getattr(args, "tune_search_alg", None)
        or "optuna"
    )
    backend = str(raw_backend).strip().lower().replace("_", "-")
    aliases = {
        "optuna": "optuna",
        "optunasearch": "optuna",
        "optuna-search": "optuna",
        "hyperopt": "hyperopt",
        "hyperoptsearch": "hyperopt",
        "hyperopt-search": "hyperopt",
        "random": "random",
        "basic": "random",
        "basicvariant": "random",
        "basic-variant": "random",
    }
    resolved = aliases.get(backend)
    if resolved is None:
        raise click.UsageError(
            f"Unknown tune search backend '{raw_backend}'. "
            f"Choose one of: {', '.join(SEARCH_BACKENDS)}."
        )
    return resolved


def build_search_backend(
    *,
    backend: str,
    yaml_cfg: dict,
    tune,
    opt_metrics: list[str],
    opt_modes: list[str],
    baseline_config: dict[str, Any] | None,
    seed: int | None,
    max_concurrent: int,
) -> tuple[Any | None, dict[str, Any]]:
    """Factory: create the appropriate backend and return (search_alg, param_space)."""
    cls = _BACKEND_REGISTRY.get(backend)
    if cls is None:
        raise click.UsageError(
            f"Unknown tune search backend '{backend}'. "
            f"Choose one of: {', '.join(SEARCH_BACKENDS)}."
        )
    instance = cls(
        yaml_cfg=yaml_cfg,
        opt_metrics=opt_metrics,
        opt_modes=opt_modes,
        baseline_config=baseline_config,
        seed=seed,
        max_concurrent=max_concurrent,
    )
    return instance.build(tune)


__all__ = [
    "BaseTuneBackend",
    "HyperOptBackend",
    "OptunaBackend",
    "RandomBackend",
    "SEARCH_BACKENDS",
    "build_search_backend",
    "resolve_search_backend",
]
