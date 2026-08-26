"""Tracker configuration loading.

Built-in tracker YAMLs colocate runtime defaults and tuning metadata. This
module extracts only the runtime values; interpretation of search metadata
remains owned by :mod:`boxmot.engine.tuning`. Reusable presets and custom
runtime configs are plain scalar mappings.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from boxmot.configs import CONFIG_ROOT

TRACKER_CONFIGS_DIR = CONFIG_ROOT / "trackers"
TRACKER_PRESETS_DIR = TRACKER_CONFIGS_DIR / "presets"
TRACKER_METADATA_KEY = "tracker"


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
    payload = _load_mapping(path, label=label)

    non_scalar = [
        str(key)
        for key, value in payload.items()
        if not isinstance(value, (str, int, float, bool, type(None)))
    ]
    if non_scalar:
        names = ", ".join(non_scalar)
        raise ValueError(
            f"{label.capitalize()} config {path} must contain runtime parameter values, "
            f"not nested or collection values; invalid entries: {names}"
        )
    return dict(payload)


def _flatten_tracker_entries(config: Mapping[str, Any], *, path: Path) -> dict[str, Mapping[str, Any]]:
    """Flatten tracker parameters, including conditional ``activates`` entries."""

    flattened: dict[str, Mapping[str, Any]] = {}

    def _visit(entries: Mapping[str, Any]) -> None:
        for parameter, details in entries.items():
            if not isinstance(details, Mapping):
                raise ValueError(
                    f'Tracker config {path} entry "{parameter}" must be a mapping containing a default.'
                )
            if parameter in flattened:
                raise ValueError(f'Tracker config {path} defines parameter "{parameter}" more than once.')
            flattened[str(parameter)] = details

            children = details.get("activates")
            if children is None:
                continue
            if not isinstance(children, Mapping):
                raise ValueError(
                    f'Tracker config {path} entry "{parameter}" has a non-mapping activates block.'
                )
            _visit(children)

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

    if str(declared_tracker).strip().lower() != str(expected_tracker).strip().lower():
        raise ValueError(
            f'Tracker config {path} is for "{declared_tracker}", not "{expected_tracker}".'
        )
    return resolved


def load_tracker_schema(tracker_name: str) -> dict[str, Any]:
    """Load one built-in combined runtime/tuning tracker schema."""

    path = get_tracker_config_path(tracker_name)
    if not path.is_file():
        available = sorted(candidate.stem for candidate in TRACKER_CONFIGS_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Tracker config not found: {path}\n"
            f"Available trackers: {', '.join(available) or '(none)'}"
        )
    return _load_mapping(path, label="tracker")


def load_tracker_defaults(tracker_name: str) -> dict[str, Any]:
    """Extract scalar runtime defaults from one built-in tracker schema."""

    path = get_tracker_config_path(tracker_name)
    entries = _flatten_tracker_entries(load_tracker_schema(tracker_name), path=path)
    missing = sorted(parameter for parameter, details in entries.items() if "default" not in details)
    if missing:
        raise ValueError(
            f"Tracker config {path} must define a runtime default for: {', '.join(missing)}"
        )

    defaults = {parameter: details["default"] for parameter, details in entries.items()}
    non_scalar = sorted(
        parameter
        for parameter, value in defaults.items()
        if not isinstance(value, (str, int, float, bool, type(None)))
    )
    if non_scalar:
        raise ValueError(
            f"Tracker config {path} runtime defaults must be scalar values; invalid entries: "
            f"{', '.join(non_scalar)}"
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
) -> dict[str, Any]:
    """Resolve one tracker config using deterministic overlay precedence.

    Built-in defaults are loaded first. ``tracker_config`` may be a partial
    scalar YAML or a built-in preset and overlays those defaults. Additional
    mappings are then applied from left to right.
    """

    resolved = load_tracker_defaults(tracker_name)
    if tracker_config is not None:
        config_path = resolve_tracker_config_path(tracker_config)
        default_path = get_tracker_config_path(tracker_name).resolve()
        if config_path != default_path:
            is_builtin_config = config_path.parent == TRACKER_CONFIGS_DIR.resolve()
            if is_builtin_config and config_path.stem != tracker_name:
                raise ValueError(
                    f'Tracker config {config_path} is for "{config_path.stem}", not "{tracker_name}".'
                )
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
            resolved.update(dict(override))
    return resolved


__all__ = (
    "TRACKER_CONFIGS_DIR",
    "TRACKER_METADATA_KEY",
    "TRACKER_PRESETS_DIR",
    "get_tracker_config_path",
    "get_tracker_preset_path",
    "load_tracker_config",
    "load_tracker_defaults",
    "load_tracker_schema",
    "resolve_tracker_config_path",
)
