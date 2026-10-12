"""Optuna search backend with conditional define-by-run search space."""
from __future__ import annotations

from typing import Any

from boxmot.engine.tuning.backends.base import BaseTuneBackend
from boxmot.engine.tuning.conditions import (
    branch_parameters,
    categorical_branch,
    resolve_fixed_cases,
    scoped_parameter_name,
)
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


def _is_suggestable(details: dict) -> bool:
    t = details.get("type")
    rng = details.get("range")
    opts = details.get("options") or details.get("values")

    if t in ("uniform", "randint", "qrandint", "loguniform"):
        return bool(rng and len(rng) >= 2)
    if t in ("choice", "grid_search"):
        return bool(opts)
    return False


class _OptunaDefineSpace:
    """Picklable callable for Optuna define-by-run search space.

    Ray Tune serializes the space function to send it to workers.  A nested
    closure can't be pickled, so we use a top-level class instead.
    """

    def __init__(self, flat: dict, parents_with_children: dict, child_params: set):
        self.flat = flat
        self.parents_with_children = parents_with_children
        self.child_params = child_params

    def _suggest_children(self, trial, parent: str) -> None:
        for child_name, child_details in self.parents_with_children.get(parent, {}).items():
            if not isinstance(child_details, dict) or not _is_suggestable(child_details):
                continue
            _suggest_param(trial, child_name, child_details)
            if child_name in self.parents_with_children and trial.params.get(child_name):
                self._suggest_children(trial, child_name)

    def __call__(self, trial):
        for param, details in self.flat.items():
            if not isinstance(details, dict):
                continue
            if param in self.child_params:
                continue
            if not _is_suggestable(details):
                continue

            _suggest_param(trial, param, details)

            if param in self.parents_with_children and trial.params.get(param):
                self._suggest_children(trial, param)


class _CaseOptunaSpace:
    """Select cases before suggesting their active, independently named distributions."""

    def __init__(self, schema: dict) -> None:
        self.schema = schema
        self.controlled = branch_parameters(schema)
        self.dependent = {name for name, entry in flatten_yaml_config(schema).items() if "depends_on" in entry}

    def baseline(self, values: dict) -> dict:
        """Translate a baseline into the optimizer labels used by its selected cases."""
        schema = self.schema
        selectors = {}
        while (branch := categorical_branch(schema, self.controlled)) is not None:
            parent, cases = branch
            entry = flatten_yaml_config(schema)[parent]
            value = values.get(parent, entry["default"])
            label = scoped_parameter_name(parent, entry) if parent in self.dependent else parent
            selectors[label] = value
            schema = next(case for option, case in cases if type(option) is type(value) and option == value)
        flat = flatten_yaml_config(schema)
        distributions = {
            scoped_parameter_name(name, flat[name]) if name in self.dependent else name: value
            for name, value in values.items() if name in flat and _is_suggestable(flat[name])
        }
        return {**selectors, **distributions}

    def __call__(self, trial) -> dict:
        outer = self

        class CanonicalTrial:
            """Hide scoped optimizer labels from the recorded tracker configuration."""

            def __init__(self) -> None:
                self.params: dict[str, Any] = {}
                self.details: dict[str, dict] = {}

            def __getattr__(self, method: str):
                def suggest(name: str, *args, **kwargs):
                    label = scoped_parameter_name(name, self.details[name]) if name in outer.dependent else name
                    value = getattr(trial, method)(label, *args, **kwargs)
                    self.params[name] = value
                    return value

                return suggest

        canonical = CanonicalTrial()
        schema = self.schema
        controlled = set(self.controlled)
        while (branch := categorical_branch(schema, controlled)) is not None:
            parent, cases = branch
            controlled.add(parent)
            canonical.details = flatten_yaml_config(schema)
            value = _suggest_param(canonical, parent, canonical.details[parent])
            schema = next(case for option, case in cases if type(option) is type(value) and option == value)
        flat = flatten_yaml_config(schema)
        parents, children, _ = conditional_yaml_tree(schema)
        canonical.details = flat
        _OptunaDefineSpace(flat, parents, children)(canonical)
        constants = {
            name: entry["default"] for name, entry in flat.items()
            if name in controlled and set(entry) == {"default"}
        }
        return {**canonical.params, **constants}


def yaml_to_optuna_define_space(config: dict) -> _OptunaDefineSpace | _CaseOptunaSpace:
    """Build an Optuna define-by-run callable from a tracker YAML config.

    Uses Optuna's native conditional parameter support: child params inside
    ``activates`` blocks are only suggested when their parent toggle is truthy.
    """
    config = resolve_fixed_cases(config)
    flat = flatten_yaml_config(config)
    if any(isinstance(entry, dict) and "depends_on" in entry for entry in flat.values()):
        return _CaseOptunaSpace(config)
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

        baseline = default_tune_config(
            self.yaml_cfg,
            defaults=self.baseline_config,
        ) or None
        space = yaml_to_optuna_define_space(self.yaml_cfg)
        if baseline:
            kwargs["points_to_evaluate"] = [
                space.baseline(baseline) if isinstance(space, _CaseOptunaSpace) else baseline
            ]

        kwargs = self.filter_supported_kwargs(OptunaSearch, kwargs)
        searcher = OptunaSearch(space=space, **kwargs)
        return ConcurrencyLimiter(searcher, max_concurrent=self.max_concurrent), {}
