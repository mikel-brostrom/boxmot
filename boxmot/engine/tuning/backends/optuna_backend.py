"""Optuna search backend with conditional define-by-run search space."""
from __future__ import annotations

from typing import Any

from boxmot.engine.tuning.backends.base import BaseTuneBackend
from boxmot.engine.tuning.search_space import (
    conditional_yaml_tree,
    default_tune_config,
    flatten_yaml_config,
)


def _suggest_param(trial, param: str, details: dict):
    """Call the appropriate ``trial.suggest_*`` for a single parameter."""
    t = details.get("type")
    rng = details.get("range")
    opts = details.get("options") or details.get("values")

    if t == "uniform":
        return trial.suggest_float(param, rng[0], rng[1])
    elif t == "loguniform":
        return trial.suggest_float(param, rng[0], rng[1], log=True)
    elif t == "randint":
        return trial.suggest_int(param, rng[0], rng[1] - 1)
    elif t == "qrandint":
        step = rng[2] if len(rng) > 2 else 1
        return trial.suggest_int(param, rng[0], rng[1] - 1, step=step)
    elif t in ("choice", "grid_search"):
        return trial.suggest_categorical(param, list(opts))
    return None


class _OptunaDefineSpace:
    """Picklable callable for Optuna define-by-run search space.

    Ray Tune serializes the space function to send it to workers.  A nested
    closure can't be pickled, so we use a top-level class instead.
    """

    def __init__(self, flat: dict, parents_with_children: dict, child_params: set):
        self.flat = flat
        self.parents_with_children = parents_with_children
        self.child_params = child_params

    def __call__(self, trial):
        for param, details in self.flat.items():
            if not isinstance(details, dict):
                continue
            if param in self.child_params:
                continue
            t = details.get("type")
            rng = details.get("range")
            opts = details.get("options") or details.get("values")

            if t in ("uniform", "randint", "qrandint", "loguniform"):
                if not rng or len(rng) < 2:
                    continue
            elif t in ("choice", "grid_search"):
                if not opts:
                    continue
            else:
                continue

            _suggest_param(trial, param, details)

            if param in self.parents_with_children:
                parent_value = trial.params[param]
                if parent_value:
                    for child_name, child_details in self.parents_with_children[param].items():
                        if isinstance(child_details, dict):
                            _suggest_param(trial, child_name, child_details)


def yaml_to_optuna_define_space(config: dict) -> _OptunaDefineSpace:
    """Build an Optuna define-by-run callable from a tracker YAML config.

    Uses Optuna's native conditional parameter support: child params inside
    ``activates`` blocks are only suggested when their parent toggle is truthy.
    """
    flat = flatten_yaml_config(config)
    parents_with_children, child_params, _ = conditional_yaml_tree(config)
    return _OptunaDefineSpace(flat, parents_with_children, child_params)


class OptunaBackend(BaseTuneBackend):
    """Optuna TPE backend with conditional define-by-run search space."""

    def build(self, tune) -> tuple[Any, dict[str, Any]]:
        from ray.tune.search import ConcurrencyLimiter
        from ray.tune.search.optuna import OptunaSearch

        metric = self.opt_metrics[0] if len(self.opt_metrics) == 1 else self.opt_metrics
        mode = self.opt_modes[0] if len(self.opt_modes) == 1 else self.opt_modes

        kwargs: dict[str, Any] = {"metric": metric, "mode": mode}
        if self.seed is not None:
            kwargs["seed"] = self.seed

        baseline = self.baseline_config
        if baseline is None:
            baseline = default_tune_config(self.yaml_cfg) or None
        if baseline:
            kwargs["points_to_evaluate"] = [baseline]

        kwargs = self.filter_supported_kwargs(OptunaSearch, kwargs)
        searcher = OptunaSearch(space=yaml_to_optuna_define_space(self.yaml_cfg), **kwargs)
        return ConcurrencyLimiter(searcher, max_concurrent=self.max_concurrent), {}
