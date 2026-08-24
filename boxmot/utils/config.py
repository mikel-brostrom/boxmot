"""Shared helpers for loading and resolving BoxMOT YAML configuration files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration file is malformed or internally inconsistent."""


CONFIG_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def validate_config_id(value: Any, *, path: str | Path, label: str) -> str:
    """Return a validated, portable kebab-case catalog identifier."""
    config_id = str(value or "")
    if not CONFIG_ID_PATTERN.fullmatch(config_id):
        raise ConfigurationError(
            f'{label.capitalize()} config "{path}" has malformed id "{config_id}"; '
            "ids must use lowercase kebab-case."
        )
    return config_id


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Load *path* and require its top-level YAML value to be a mapping."""
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f'Failed to parse configuration "{config_path}": {exc}') from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(f'Configuration "{config_path}" must contain a YAML mapping.')
    return payload


def iter_config_paths(config_dir: str | Path) -> list[Path]:
    """Return all YAML files below *config_dir* in deterministic order."""
    directory = Path(config_dir)
    paths = {*directory.glob("**/*.yaml"), *directory.glob("**/*.yml")}
    return sorted(path for path in paths if path.is_file())


def index_config_ids(config_dir: str | Path, label: str) -> dict[str, Path]:
    """Index a catalog by declared id, rejecting missing, malformed, or duplicate ids."""
    indexed: dict[str, Path] = {}
    for path in iter_config_paths(config_dir):
        payload = load_yaml_mapping(path)
        config_id = validate_config_id(payload.get("id"), path=path, label=label)
        previous = indexed.get(config_id)
        if previous is not None:
            raise ConfigurationError(
                f'Duplicate {label} id "{config_id}" in "{previous}" and "{path}".'
            )
        indexed[config_id] = path.resolve()
    return indexed


def resolve_config_path(config_dir: str | Path, reference: str | Path, label: str) -> Path:
    """Resolve a config by explicit path, relative path, filename, stem, or declared id."""
    directory = Path(config_dir)
    if not str(reference).strip():
        raise FileNotFoundError(f"{label.capitalize()} config reference must not be empty.")
    path = Path(reference)

    if path.suffix.lower() in {".yaml", ".yml"} and path.is_file():
        return path.resolve()

    relative = path if path.suffix.lower() in {".yaml", ".yml"} else path.with_suffix(".yaml")
    exact = directory / relative
    if exact.is_file():
        return exact.resolve()

    if path.is_absolute() or path.parent != Path("."):
        raise FileNotFoundError(f'{label.capitalize()} config path does not exist: "{path}"')

    reference_text = str(reference)
    reference_stem = relative.stem.casefold()
    id_index = index_config_ids(directory, label)
    candidates: list[Path] = [id_index[reference_text]] if reference_text in id_index else []
    for candidate in id_index.values():
        if candidate.stem.casefold() == reference_stem:
            candidates.append(candidate)

    unique = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        choices = "\n  - ".join(str(candidate) for candidate in unique)
        raise ConfigurationError(f'Ambiguous {label} reference "{reference}". Use an id or path:\n  - {choices}')
    raise FileNotFoundError(f'{label.capitalize()} config not found for "{reference}" in {directory}')


__all__ = (
    "CONFIG_ID_PATTERN",
    "ConfigurationError",
    "index_config_ids",
    "iter_config_paths",
    "load_yaml_mapping",
    "resolve_config_path",
    "validate_config_id",
)
