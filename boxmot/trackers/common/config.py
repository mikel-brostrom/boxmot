"""Tracker configuration loading.

Built-in tracker YAMLs colocate runtime defaults and tuning metadata. This
module resolves scalar runtime values;
interpretation of search metadata remains owned by :mod:`boxmot.engine.tuning`.
Reusable presets and custom runtime configs group Kalman settings under
``kalman`` and mask-guidance settings under ``edgetam``. Engine/search code
addresses those leaves using dotted paths.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from boxmot.configs import CONFIG_ROOT
from boxmot.trackers.common.mask_guidance import MASK_GUIDANCE_OPTIONS

TRACKER_CONFIGS_DIR = CONFIG_ROOT / "trackers"
TRACKER_PRESETS_DIR = TRACKER_CONFIGS_DIR / "presets"
TRACKER_METADATA_KEY = "tracker"


def flatten_tracker_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize authored Kalman and EdgeTAM settings to scalar parameter paths.

    Nested YAML mappings and the public immutable configuration object resolve
    to the same fields. Other tracker options retain their original values.
    Supplying a field twice through nested and dotted notation is an error.
    """
    flattened: dict[str, Any] = {}
    parents: set[str] = set()

    def insert(name: str, value: Any) -> None:
        if name in flattened:
            raise ValueError(f"Tracker configuration specifies {name!r} more than once.")
        if name.startswith("kalman.") and not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError(
                "Dotted Kalman option paths must address scalar leaves; use the 'kalman' group for mappings."
            )
        if name.startswith("edgetam."):
            if name not in MASK_GUIDANCE_OPTIONS:
                raise TypeError(f"Unknown edgetam parameter {name!r}.")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise TypeError("Dotted EdgeTAM option paths must address scalar leaves.")
        parts = name.split(".")
        prefixes = {".".join(parts[:length]) for length in range(1, len(parts))}
        if name in parents or prefixes.intersection(flattened):
            raise ValueError(f"Tracker option {name!r} conflicts with another parameter path.")
        parents.update(prefixes)
        flattened[name] = value

    def visit_group(value: Any, prefix: str) -> None:
        if isinstance(value, tuple):
            value = dict(value)
        if not isinstance(value, Mapping):
            from boxmot.trackers.common.motion.kalman_filters.config import (
                AbnormalMotionSuppressionConfig,
                KalmanConfig,
            )
            from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig

            expected = (
                KalmanConfig
                if prefix == "kalman."
                else (AbnormalMotionSuppressionConfig if prefix == "kalman.ams." else KalmanNoiseConfig)
            )
            if not isinstance(value, expected):
                raise TypeError(f"{prefix.rstrip('.')} must be a {expected.__name__} or a configuration mapping.")
            value = value.to_dict()
        if not value:
            field, default = (
                ("variable_dt", False)
                if prefix == "kalman."
                else (("enabled", True) if prefix == "kalman.ams." else ("time_unit", None))
            )
            insert(prefix + field, default)
            return
        for field, setting in value.items():
            if not isinstance(field, str) or "." in field:
                raise ValueError("Kalman field names must be unqualified strings.")
            if prefix == "kalman." and field in {"noise", "ams"}:
                if field == "ams" and setting is None:
                    continue
                visit_group(setting, prefix + field + ".")
            elif field == "by_class":
                if prefix != "kalman.noise.":
                    raise ValueError("Kalman class overrides belong under kalman.noise.by_class and cannot nest.")
                if isinstance(setting, tuple):
                    setting = dict(setting)
                if not isinstance(setting, Mapping):
                    raise TypeError("kalman.noise.by_class must map class IDs to noise configurations.")
                for class_id, child in setting.items():
                    text_id = str(class_id)
                    if not text_id.isascii() or not text_id.isdecimal() or str(int(text_id)) != text_id:
                        raise ValueError("kalman.noise.by_class keys must be non-negative integer class IDs.")
                    visit_group(child, f"{prefix}by_class.{text_id}.")
            else:
                insert(prefix + field, setting)

    for name, value in config.items():
        if not isinstance(name, str):
            raise TypeError("Tracker option names must be strings.")
        if name.startswith("mask_guidance_"):
            raise TypeError(f"Unknown tracker option {name!r}; configure guidance parameters under 'edgetam'.")
        if (
            name.startswith("kf_")
            or name in {"kalman_noise", "variable_dt", "adaptive_kf", "is_angular"}
            or name.startswith(("kalman_noise.", "ams_"))
        ):
            raise TypeError(f"Unknown tracker option {name!r}; configure Kalman settings under 'kalman'.")
        if name == "calibration":
            if not isinstance(value, Mapping) or not value:
                raise ValueError("calibration must contain saved profile metadata.")
            for field, setting in value.items():
                if not isinstance(field, str) or "." in field:
                    raise ValueError("calibration field names must be unqualified strings.")
                insert(f"calibration.{field}", setting)
            continue
        if name == "edgetam":
            if isinstance(value, tuple):
                value = dict(value)
            if not isinstance(value, Mapping):
                raise TypeError("edgetam must be a mapping of guidance parameters.")
            for field, setting in value.items():
                if not isinstance(field, str) or "." in field:
                    raise ValueError("EdgeTAM field names must be unqualified strings.")
                insert(f"edgetam.{field}", setting)
            continue
        if name != "kalman":
            insert(name, value)
            continue
        if value is None:
            insert("kalman.variable_dt", False)
            continue
        count = len(flattened)
        visit_group(value, "kalman.")
        if len(flattened) == count:
            insert("kalman.variable_dt", False)
    return flattened


def nest_tracker_options(config: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize resolved tracker settings using the authored YAML structure."""
    nested: dict[str, Any] = {}
    for name, value in flatten_tracker_options(config).items():
        if name.startswith(("kalman.", "calibration.", "edgetam.")):
            target = nested
            parts = name.split(".")
            for part in parts[:-1]:
                existing = target.setdefault(part, {})
                if not isinstance(existing, dict):
                    raise ValueError(f"Tracker option {name!r} conflicts with another parameter path.")
                target = existing
            target[parts[-1]] = value
        else:
            nested[name] = value
    return nested


def get_tracker_config_path(tracker_name: str) -> Path:
    """Return the built-in combined config path for ``tracker_name``."""

    return TRACKER_CONFIGS_DIR / f"{tracker_name}.yaml"


def get_tracker_preset_path(preset_name: str) -> Path:
    """Return the built-in scalar preset path for ``preset_name``."""

    return TRACKER_PRESETS_DIR / f"{preset_name}.yaml"


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse {label} config {path}: {exc}") from exc

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"{label.capitalize()} config {path} must contain a non-empty YAML mapping.")

    return dict(payload)


def _load_scalar_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = flatten_tracker_options(_load_mapping(path, label=label))

    non_scalar = [
        str(key) for key, value in payload.items() if not isinstance(value, (str, int, float, bool, type(None)))
    ]
    if non_scalar:
        names = ", ".join(non_scalar)
        raise ValueError(
            f"{label.capitalize()} config {path} must contain runtime parameter values, "
            f"with fields grouped under kalman or edgetam; invalid entries: {names}"
        )
    return dict(payload)


def _flatten_tracker_entries(config: Mapping[str, Any], *, path: Path) -> dict[str, Mapping[str, Any]]:
    """Flatten tracker parameters, including conditional ``activates`` entries."""

    flattened: dict[str, Mapping[str, Any]] = {}

    def _visit(entries: Mapping[str, Any], prefix: str = "") -> None:
        for parameter, details in entries.items():
            if (parameter in {"kalman", "edgetam"} and not prefix) or (
                prefix == "kalman." and parameter in {"noise", "ams"}
            ):
                if not isinstance(details, Mapping) or not details:
                    raise ValueError(f"Tracker config {path} {parameter} must contain parameter definitions.")
                _visit(details, prefix + str(parameter) + ".")
                continue
            parameter = prefix + str(parameter)
            if prefix == "edgetam." and parameter not in MASK_GUIDANCE_OPTIONS:
                raise TypeError(f"Unknown edgetam parameter {parameter!r} in tracker config {path}.")
            if not isinstance(details, Mapping):
                raise ValueError(f'Tracker config {path} entry "{parameter}" must be a mapping containing a default.')
            if parameter in flattened:
                raise ValueError(f'Tracker config {path} defines parameter "{parameter}" more than once.')
            flattened[str(parameter)] = details

            children = details.get("activates")
            if children is None:
                continue
            if not isinstance(children, Mapping):
                raise ValueError(f'Tracker config {path} entry "{parameter}" has a non-mapping activates block.')
            _visit(children, prefix)

    _visit(config)
    return flattened


def _strip_tracker_metadata(
    config: dict[str, Any],
    *,
    expected_tracker: str,
    path: Path,
    required: bool = False,
) -> dict[str, Any]:
    resolved = dict(config)
    declared_tracker = resolved.pop(TRACKER_METADATA_KEY, None)
    if declared_tracker in (None, ""):
        if required:
            raise ValueError(
                f'Built-in tracker preset {path} must declare "{TRACKER_METADATA_KEY}: {expected_tracker}".'
            )
        return resolved

    if declared_tracker != expected_tracker:
        raise ValueError(f'Tracker config {path} is for "{declared_tracker}", not "{expected_tracker}".')
    return resolved


def load_tracker_schema(tracker_name: str) -> dict[str, Any]:
    """Load one built-in combined runtime/tuning tracker schema."""

    path = get_tracker_config_path(tracker_name)
    if not path.is_file():
        available = sorted(candidate.stem for candidate in TRACKER_CONFIGS_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Tracker config not found: {path}\nAvailable trackers: {', '.join(available) or '(none)'}"
        )
    return _load_mapping(path, label="tracker")


def load_tracker_defaults(tracker_name: str) -> dict[str, Any]:
    """Extract runtime defaults from the tracker schema."""

    path = get_tracker_config_path(tracker_name)
    entries = _flatten_tracker_entries(load_tracker_schema(tracker_name), path=path)
    missing = sorted(parameter for parameter, details in entries.items() if "default" not in details)
    if missing:
        raise ValueError(f"Tracker config {path} must define a runtime default for: {', '.join(missing)}")

    defaults = {parameter: details["default"] for parameter, details in entries.items()}
    non_scalar = sorted(
        parameter for parameter, value in defaults.items() if not isinstance(value, (str, int, float, bool, type(None)))
    )
    if non_scalar:
        raise ValueError(
            f"Tracker config {path} runtime defaults must be scalar values; invalid entries: {', '.join(non_scalar)}"
        )
    return defaults


def resolve_tracker_config_path(reference: str | Path) -> Path:
    """Resolve a custom tracker config path or a built-in default/preset name."""

    path = Path(reference)
    if path.is_file():
        return path.resolve()

    filename = path.name if path.suffix else f"{path.name}.yaml"
    candidates = [
        TRACKER_CONFIGS_DIR / filename,
        TRACKER_PRESETS_DIR / filename,
    ]
    matches = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = "\n  - ".join(str(candidate) for candidate in matches)
        raise ValueError(f'Ambiguous tracker config "{reference}":\n  - {choices}')
    raise FileNotFoundError(f'Tracker config not found for "{reference}".')


def load_tracker_config(
    tracker_name: str,
    tracker_config: str | Path | None = None,
    *overrides: Mapping[str, Any] | None,
    include_defaults: bool = True,
) -> dict[str, Any]:
    """Resolve one tracker config to scalar paths using deterministic overlays.

    Built-in defaults are loaded first. ``tracker_config`` may be a partial
    runtime YAML or a built-in preset and overlays those defaults. Additional
    mappings are then applied from left to right. ``include_defaults=False``
    returns only authored values and overrides, allowing backend factories to
    apply their own defaults without treating them as explicit user choices.
    """

    resolved = load_tracker_defaults(tracker_name) if include_defaults else {}
    if tracker_config is not None:
        config_path = resolve_tracker_config_path(tracker_config)
        default_path = get_tracker_config_path(tracker_name).resolve()
        if config_path != default_path:
            is_builtin_config = config_path.parent == TRACKER_CONFIGS_DIR.resolve()
            if is_builtin_config and config_path.stem != tracker_name:
                raise ValueError(f'Tracker config {config_path} is for "{config_path.stem}", not "{tracker_name}".')
            is_builtin_preset = config_path.parent == TRACKER_PRESETS_DIR.resolve()
            overlay = _strip_tracker_metadata(
                _load_scalar_mapping(config_path, label="tracker"),
                expected_tracker=tracker_name,
                path=config_path,
                required=is_builtin_preset,
            )
            resolved.update(overlay)

    for override in overrides:
        if override:
            resolved.update(flatten_tracker_options(override))
    return resolved


__all__ = (
    "TRACKER_CONFIGS_DIR",
    "TRACKER_METADATA_KEY",
    "TRACKER_PRESETS_DIR",
    "get_tracker_config_path",
    "get_tracker_preset_path",
    "flatten_tracker_options",
    "load_tracker_config",
    "load_tracker_defaults",
    "load_tracker_schema",
    "nest_tracker_options",
    "resolve_tracker_config_path",
)
