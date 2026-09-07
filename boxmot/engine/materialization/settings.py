"""Strict YAML/CLI settings for the local materialization executor."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_EXECUTOR_SETTINGS: dict[str, dict[str, Any]] = {
    "detect": {
        "batch_size": 16,
        "workers": 1,
        "executor": "inline",
        "retries": 2,
        "retry_backoff_s": 0.0,
    },
    "segment": {
        "batch_size": 8,
        "workers": 1,
        "executor": "inline",
        "retries": 2,
        "retry_backoff_s": 0.0,
    },
    "embed": {
        "batch_size": 64,
        "workers": 1,
        "executor": "inline",
        "retries": 2,
        "retry_backoff_s": 0.0,
    },
    "finalize": {
        "batch_size": 1,
        "workers": 1,
        "executor": "inline",
        "retries": 0,
        "retry_backoff_s": 0.0,
    },
    "decode": {"workers": min(4, os.cpu_count() or 1)},
    "writer": {"instance_rows_per_shard": 50_000, "workers": 1},
}

_ALLOWED_FIELDS = {name: frozenset(values) for name, values in DEFAULT_EXECUTOR_SETTINGS.items()}


def _validate(settings: Mapping[str, Mapping[str, Any]]) -> None:
    unknown_sections = set(settings) - set(_ALLOWED_FIELDS)
    if unknown_sections:
        raise ValueError(f"Unknown executor plan sections: {', '.join(sorted(unknown_sections))}.")
    for section, values in settings.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"Executor plan section {section!r} must be a mapping.")
        unknown = set(values) - set(_ALLOWED_FIELDS[section])
        if unknown:
            raise ValueError(f"Unknown executor plan keys for {section}: {', '.join(sorted(unknown))}.")

    for stage in ("detect", "segment", "embed", "finalize"):
        if stage not in settings:
            continue
        values = settings[stage]
        for key in ("batch_size", "workers"):
            if key not in values:
                continue
            value = values[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{stage}.{key} must be a positive integer.")
        if "retries" in values:
            retries = values["retries"]
            if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
                raise ValueError(f"{stage}.retries must be a non-negative integer.")
        if "retry_backoff_s" in values:
            backoff = values["retry_backoff_s"]
            if isinstance(backoff, bool) or not isinstance(backoff, (int, float)) or backoff < 0:
                raise ValueError(f"{stage}.retry_backoff_s must be non-negative.")
        executor = values.get("executor", "inline")
        if executor not in {"inline", "process"}:
            raise ValueError(f"{stage}.executor must be 'inline' or 'process'.")
        workers = values.get("workers", 1)
        if executor == "inline" and workers != 1:
            raise ValueError(
                f"{stage}.workers > 1 requires executor=process and a serializable component spec."
            )

    if "decode" in settings:
        if "workers" in settings["decode"]:
            decode_workers = settings["decode"]["workers"]
            if isinstance(decode_workers, bool) or not isinstance(decode_workers, int) or decode_workers <= 0:
                raise ValueError("decode.workers must be a positive integer.")
    if "writer" in settings:
        for key in ("workers", "instance_rows_per_shard"):
            if key not in settings["writer"]:
                continue
            value = settings["writer"][key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"writer.{key} must be a positive integer.")
        if settings["writer"].get("workers", 1) != 1:
            raise ValueError("Materialization v1 requires exactly one writer worker.")


def load_executor_settings(
    plan_path: str | Path | None = None,
    overrides: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    """Merge defaults, a plan YAML, and strict ``section.field=value`` overrides."""

    settings = deepcopy(DEFAULT_EXECUTOR_SETTINGS)
    if plan_path is not None:
        with Path(plan_path).open("r", encoding="utf-8") as stream:
            authored = yaml.safe_load(stream) or {}
        if not isinstance(authored, Mapping):
            raise TypeError("Executor plan YAML must contain a mapping.")
        _validate({str(key): value for key, value in authored.items()})
        for section, values in authored.items():
            settings[str(section)].update(dict(values))

    for override in overrides:
        key, separator, raw_value = override.partition("=")
        section, dot, field = key.partition(".")
        if not separator or not dot or not section or not field:
            raise ValueError(f"Invalid --set {override!r}; expected stage.field=value.")
        if section not in _ALLOWED_FIELDS or field not in _ALLOWED_FIELDS[section]:
            raise ValueError(f"Unknown executor plan key {section}.{field}.")
        value = yaml.safe_load(raw_value)
        settings[section][field] = value

    _validate(settings)
    return settings


__all__ = ("DEFAULT_EXECUTOR_SETTINGS", "load_executor_settings")
