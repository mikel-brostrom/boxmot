"""HyperOpt TPE search backend with flat search space."""
from __future__ import annotations

import math
from typing import Any

import click

from boxmot.engine.tuning.backends.base import BaseTuneBackend
from boxmot.engine.tuning.search_space import (
    conditional_yaml_tree,
    default_tune_config,
    flatten_yaml_config,
    is_valid_search_param,
    yaml_to_tune_space,
)
from boxmot.utils import logger as LOGGER


def _hyperopt_param(hp, param: str, details: dict):
    """Create one HyperOpt distribution from one YAML search-space entry.

    Integer ranges follow the same upper-exclusive convention as the
    Ray/Optuna conversion: ``range: [lo, hi]``.
    """
    if not is_valid_search_param(param, details):
        return None

    t = details.get("type")
    rng = details.get("range")
    opts = details.get("options") or details.get("values")

    if t == "uniform":
        return hp.uniform(param, rng[0], rng[1])
    if t == "loguniform":
        return hp.loguniform(param, math.log(rng[0]), math.log(rng[1]))
    if t == "randint":
        values = list(range(int(rng[0]), int(rng[1])))
        if not values:
            raise ValueError(f"Invalid randint range for '{param}': {rng}")
        return hp.choice(param, values)
    if t == "qrandint":
        step = int(rng[2]) if len(rng) > 2 else 1
        values = list(range(int(rng[0]), int(rng[1]), step))
        if not values:
            raise ValueError(f"Invalid qrandint range for '{param}': {rng}")
        return hp.choice(param, values)
    if t in ("choice", "grid_search"):
        return hp.choice(param, list(opts))

    return None


def yaml_to_hyperopt_space(config: dict) -> dict:
    """Build a native HyperOpt search space from a tracker YAML config.

    Conditional ``activates`` blocks become branch dictionaries. The raw Ray
    trial config from HyperOpt is therefore nested, and must be flattened by
    ``normalize_trial_config`` before it reaches the tracker/evaluator.
    """
    from hyperopt import hp

    flat = flatten_yaml_config(config)
    parents_with_children, child_params, _ = conditional_yaml_tree(config)
    space: dict[str, Any] = {}

    for param, details in flat.items():
        if param in child_params:
            continue
        if not is_valid_search_param(param, details):
            continue

        if param not in parents_with_children:
            value = _hyperopt_param(hp, param, details)
            if value is not None:
                space[param] = value
            continue

        t = details.get("type")
        opts = details.get("options") or details.get("values")
        if t not in ("choice", "grid_search") or not opts:
            raise ValueError(
                f"HyperOpt conditional parent '{param}' must be a choice/grid_search "
                "parameter with boolean-like options."
            )

        branches = []
        for parent_value in list(opts):
            branch = {param: parent_value}
            if bool(parent_value):
                for child_name, child_details in parents_with_children[param].items():
                    if isinstance(child_details, dict):
                        child_value = _hyperopt_param(hp, child_name, child_details)
                        if child_value is not None:
                            branch[child_name] = child_value
            branches.append(branch)

        if not branches:
            raise ValueError(f"Conditional parent '{param}' has no valid branches")
        space[param] = hp.choice(param, branches)

    if not space:
        LOGGER.warning(
            "No valid HyperOpt search space parameters found in tracker config. "
            "Check that the YAML contains entries with valid 'type' and 'range'/'options' keys."
        )
    return space


class HyperOptBackend(BaseTuneBackend):
    """HyperOpt TPE backend with native conditional search space.

    Following Ray Tune best practices for conditional spaces:
    - Build a native HyperOpt space with ``hp.choice`` branches for conditionals
    - Pass ``space=native_space`` to ``HyperOptSearch``
    - Return empty ``param_space`` (the searcher owns the space)

    The nested trial config is flattened by ``normalize_trial_config`` before
    reaching the tracker.
    """

    def build(self, tune) -> tuple[Any, dict[str, Any]]:
        if len(self.opt_metrics) != 1:
            raise click.UsageError(
                "HyperOptSearch is only supported for single-objective tuning in this workflow. "
                "Use --search-alg optuna for Pareto/multi-objective tuning."
            )

        from ray.tune.search import ConcurrencyLimiter
        from ray.tune.search.hyperopt import HyperOptSearch

        # Build native HyperOpt space with conditional branches.
        # This lets HyperOpt TPE learn that child params only matter when
        # their parent toggle is active.
        native_space = yaml_to_hyperopt_space(self.yaml_cfg)

        # NOTE: points_to_evaluate is incompatible with native conditional
        # spaces (hp.choice with branch dicts). Nested baselines hang the
        # HyperOpt validator; flat baselines fail category lookup. We only
        # seed the baseline when the space has no conditional branches.
        baseline = self._build_flat_baseline()

        kwargs: dict[str, Any] = {
            "space": native_space,
            "metric": self.opt_metrics[0],
            "mode": self.opt_modes[0],
        }
        if self.seed is not None:
            kwargs["random_state_seed"] = self.seed
        if baseline:
            kwargs["points_to_evaluate"] = [baseline]

        kwargs = self.filter_supported_kwargs(HyperOptSearch, kwargs)
        searcher = HyperOptSearch(**kwargs)

        # Empty param_space — the searcher owns the space definition.
        return (
            ConcurrencyLimiter(searcher, max_concurrent=self.max_concurrent),
            {},
        )

    def _build_flat_baseline(self) -> dict[str, Any] | None:
        """Build a flat baseline for non-conditional params only.

        Returns ``None`` if any conditional branches exist in the space,
        since ``points_to_evaluate`` is incompatible with HyperOpt's native
        ``hp.choice`` conditional branching (causes hangs or lookup errors).
        """
        flat = flatten_yaml_config(self.yaml_cfg)
        _, child_params, _ = conditional_yaml_tree(self.yaml_cfg)

        # If conditionals exist, skip baseline entirely
        if child_params:
            return None

        baseline: dict[str, Any] = {}
        for param, details in flat.items():
            if not isinstance(details, dict) or "default" not in details:
                continue
            baseline[param] = details["default"]

        return baseline if baseline else None
