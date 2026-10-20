"""Optuna search backend with conditional define-by-run search space."""

from __future__ import annotations

from collections.abc import Callable
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


class _OptunaCase:
    """Prepare each visited conditional path once, without expanding the whole tree."""

    def __init__(self, schema: dict, controlled: set[str], dependent: set[str]) -> None:
        self.flat = flatten_yaml_config(schema)
        self.labels = {
            name: scoped_parameter_name(name, entry) if name in dependent else name for name, entry in self.flat.items()
        }
        self.branch = categorical_branch(schema, controlled)
        self.controlled = controlled
        self.dependent = dependent
        self.children: dict[tuple[type, Any], _OptunaCase] = {}
        self.space = None
        self.constants = {}
        if self.branch is None:
            parents, children, _ = conditional_yaml_tree(schema)
            self.space = _OptunaDefineSpace(self.flat, parents, children)
            self.constants = {
                name: entry["default"]
                for name, entry in self.flat.items()
                if name in controlled and set(entry) == {"default"}
            }

    def child(self, value: Any) -> _OptunaCase:
        """Keep boolean and numeric selectors distinct, as in authored cases."""
        key = (type(value), value)
        if key not in self.children:
            parent, cases = self.branch
            schema = next(case for option, case in cases if type(option) is type(value) and option == value)
            self.children[key] = _OptunaCase(schema, self.controlled | {parent}, self.dependent)
        return self.children[key]


class _CaseOptunaSpace:
    """Select cases before suggesting their active, independently named distributions."""

    def __init__(self, schema: dict) -> None:
        self.schema = schema
        dependent = {name for name, entry in flatten_yaml_config(schema).items() if "depends_on" in entry}
        self.root = _OptunaCase(schema, branch_parameters(schema), dependent)

    def __getstate__(self) -> dict:
        """Keep derived case trees out of optimizer checkpoints."""
        return {"schema": self.schema}

    def __setstate__(self, state: dict) -> None:
        """Prepare a fresh lazy cache from the checkpoint's authoritative schema."""
        self.__init__(state["schema"])

    def _select_leaf(self, select: Callable[[str, dict, str], Any]) -> _OptunaCase:
        """Share the prepared case traversal between baseline and trial sampling."""
        node = self.root
        while node.branch is not None:
            parent, _ = node.branch
            node = node.child(select(parent, node.flat[parent], node.labels[parent]))
        return node

    def baseline(self, values: dict) -> dict:
        """Translate a baseline into the optimizer labels used by its selected cases."""
        selectors = {}

        def select(name: str, entry: dict, label: str) -> Any:
            selectors[label] = values.get(name, entry["default"])
            return selectors[label]

        leaf = self._select_leaf(select)
        distributions = {
            leaf.labels[name]: value
            for name, value in values.items()
            if name in leaf.flat and _is_suggestable(leaf.flat[name])
        }
        return {**selectors, **distributions}

    def __call__(self, trial) -> dict:
        selectors = {}

        def select(name: str, entry: dict, label: str) -> Any:
            selectors[name] = _suggest_param(trial, label, entry)
            return selectors[name]

        leaf = self._select_leaf(select)

        class CanonicalTrial:
            """Hide scoped optimizer labels from the recorded tracker configuration."""

            def __init__(self) -> None:
                self.params: dict[str, Any] = selectors

            def __getattr__(self, method: str):
                def suggest(name: str, *args, **kwargs):
                    value = getattr(trial, method)(leaf.labels[name], *args, **kwargs)
                    self.params[name] = value
                    return value

                return suggest

        canonical = CanonicalTrial()
        leaf.space(canonical)
        return {**canonical.params, **leaf.constants}


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

        baseline = (
            default_tune_config(
                self.yaml_cfg,
                defaults=self.baseline_config,
            )
            or None
        )
        space = yaml_to_optuna_define_space(self.yaml_cfg)
        if baseline:
            kwargs["points_to_evaluate"] = [
                space.baseline(baseline) if isinstance(space, _CaseOptunaSpace) else baseline
            ]

        kwargs = self.filter_supported_kwargs(OptunaSearch, kwargs)
        searcher = OptunaSearch(space=space, **kwargs)
        return ConcurrencyLimiter(searcher, max_concurrent=self.max_concurrent), {}
