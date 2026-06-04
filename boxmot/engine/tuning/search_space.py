"""YAML config parsing and search-space helpers shared across all backends."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from boxmot.utils import TRACKER_CONFIGS
from boxmot.utils import logger as LOGGER

import yaml


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def load_yaml_config(tracker_name: str) -> dict:
    config_path = TRACKER_CONFIGS / f"{tracker_name}.yaml"
    if not config_path.exists():
        available = sorted(p.stem for p in TRACKER_CONFIGS.glob("*.yaml"))
        raise FileNotFoundError(
            f"Tracker config not found: {config_path}\n"
            f"Available trackers: {', '.join(available) or '(none)'}"
        )
    try:
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse tracker config {config_path}: {exc}"
        ) from exc
    if not isinstance(cfg, dict) or not cfg:
        raise ValueError(
            f"Tracker config {config_path} is empty or not a valid YAML mapping."
        )
    return cfg


# ---------------------------------------------------------------------------
# Flattening / conditional tree
# ---------------------------------------------------------------------------

def flatten_yaml_config(yaml_cfg: dict) -> dict:
    """Flatten a nested YAML config into a single-level dict.

    Entries with an ``activates`` block have their children promoted to the
    top level. The parent entry itself is preserved (without the ``activates``
    key in the flattened copy).
    """
    flat: dict = {}
    for param, details in yaml_cfg.items():
        if not isinstance(details, dict):
            flat[param] = details
            continue
        parent_copy = {k: v for k, v in details.items() if k != "activates"}
        flat[param] = parent_copy
        children = details.get("activates")
        if isinstance(children, dict):
            for child_param, child_details in children.items():
                if isinstance(child_details, dict):
                    flat[child_param] = child_details
    return flat


def conditional_yaml_tree(config: dict) -> tuple[dict[str, dict], set[str], dict[str, str]]:
    """Return conditional parent/child metadata from ``activates`` blocks."""
    parents_with_children: dict[str, dict] = {}
    child_params: set[str] = set()
    child_to_parent: dict[str, str] = {}

    for param, details in config.items():
        if not isinstance(details, dict):
            continue
        children = details.get("activates")
        if not isinstance(children, dict):
            continue

        parents_with_children[param] = children
        for child_name in children:
            if child_name in child_to_parent and child_to_parent[child_name] != param:
                LOGGER.warning(
                    f"Conditional parameter '{child_name}' is activated by both "
                    f"'{child_to_parent[child_name]}' and '{param}'. Using '{param}'."
                )
            child_params.add(child_name)
            child_to_parent[child_name] = param

    return parents_with_children, child_params, child_to_parent


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def is_valid_search_param(param: str, details: dict, *, warn: bool = True) -> bool:
    """Validate a YAML search-space entry."""
    if not isinstance(details, dict):
        if warn:
            LOGGER.warning(
                f"Skipping malformed config entry '{param}': expected a mapping, "
                f"got {type(details).__name__}"
            )
        return False

    t = details.get("type")
    rng = details.get("range")
    opts = details.get("options") or details.get("values")

    if t in ("uniform", "randint", "qrandint", "loguniform"):
        if not rng or len(rng) < 2:
            if warn:
                LOGGER.warning(f"Skipping param '{param}': '{t}' requires a range with 2 values")
            return False
        if t == "loguniform" and (rng[0] <= 0 or rng[1] <= 0):
            if warn:
                LOGGER.warning(f"Skipping param '{param}': 'loguniform' requires positive bounds")
            return False
        return True

    if t in ("choice", "grid_search"):
        if not opts:
            if warn:
                LOGGER.warning(f"Skipping param '{param}': '{t}' requires a non-empty 'options' list")
            return False
        return True

    if warn:
        LOGGER.warning(f"Skipping param '{param}': unknown type '{t}'")
    return False


# ---------------------------------------------------------------------------
# Flat Ray Tune search space (used by HyperOpt and random backends)
# ---------------------------------------------------------------------------

def yaml_to_tune_space(config: dict, tune) -> dict:
    """Convert a tracker YAML config into a flat Ray Tune search space dict.

    The ``activates`` children are treated as regular top-level params.
    """
    flat = flatten_yaml_config(config)
    space = {}
    for param, details in flat.items():
        if not isinstance(details, dict):
            LOGGER.warning(f"Skipping malformed config entry '{param}': expected a mapping, got {type(details).__name__}")
            continue
        t = details.get("type")
        rng = details.get("range")
        opts = details.get("options") or details.get("values")

        if t == "uniform":
            if not rng or len(rng) < 2:
                LOGGER.warning(f"Skipping param '{param}': 'uniform' requires a range with 2 values")
                continue
            space[param] = tune.uniform(*rng[:2])
        elif t == "randint":
            if not rng or len(rng) < 2:
                LOGGER.warning(f"Skipping param '{param}': 'randint' requires a range with 2 values")
                continue
            space[param] = tune.randint(*rng[:2])
        elif t == "qrandint":
            if not rng or len(rng) < 2:
                LOGGER.warning(f"Skipping param '{param}': 'qrandint' requires a range with 2 values")
                continue
            space[param] = tune.qrandint(*rng[:3])
        elif t in ("choice", "grid_search"):
            if not opts:
                LOGGER.warning(f"Skipping param '{param}': '{t}' requires a non-empty 'options' list")
                continue
            space[param] = tune.choice(list(opts))
        elif t == "loguniform":
            if not rng or len(rng) < 2:
                LOGGER.warning(f"Skipping param '{param}': 'loguniform' requires a range with 2 values")
                continue
            space[param] = tune.loguniform(*rng[:2])
        else:
            LOGGER.warning(f"Skipping param '{param}': unknown type '{t}'")

    if not space:
        LOGGER.warning(
            "No valid search space parameters found in tracker config. "
            "Check that the YAML contains entries with valid 'type' and 'range'/'options' keys."
        )
    return space


# ---------------------------------------------------------------------------
# Default baseline config
# ---------------------------------------------------------------------------

def default_tune_config(
    yaml_cfg: dict,
    search_space: dict | None = None,
    *,
    unconditional: bool = False,
) -> dict[str, Any]:
    """Return a flat default point.

    By default, children of inactive parents are excluded (Optuna mode).
    Set ``unconditional=True`` to include all defaults regardless of parent
    state (needed for flat search spaces like HyperOpt/random).
    """
    flat = flatten_yaml_config(yaml_cfg)
    parents_with_children, child_params, child_to_parent = conditional_yaml_tree(yaml_cfg)
    keys = list(search_space) if search_space is not None else list(flat)

    defaults: dict[str, Any] = {}
    for param in keys:
        details = flat.get(param)
        if not isinstance(details, dict) or "default" not in details:
            continue

        if not unconditional and param in child_params:
            parent = child_to_parent.get(param)
            parent_details = flat.get(parent, {})
            parent_default = parent_details.get("default") if isinstance(parent_details, dict) else None
            if not bool(parent_default):
                continue

        defaults[param] = details["default"]

    # Ensure parent defaults are present even if the search-space key set came
    # from a backend-specific representation.
    for parent in parents_with_children:
        details = flat.get(parent)
        if isinstance(details, dict) and "default" in details and parent not in defaults:
            defaults[parent] = details["default"]

    return defaults


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def to_builtin_value(value: Any) -> Any:
    """Convert NumPy scalar values returned by search backends to Python scalars."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def unpack_nested_dict(dct: dict[str, Any]) -> dict[str, Any]:
    """Recursively flatten nested dicts produced by conditional HyperOpt branches."""
    out: dict[str, Any] = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            out.update(unpack_nested_dict(value))
        else:
            out[key] = to_builtin_value(value)
    return out


def normalize_trial_config(config: dict | None) -> dict[str, Any]:
    """Return the flat config shape expected by tracker evaluation and reporting."""
    if config is None:
        return {}
    return unpack_nested_dict(dict(config))
