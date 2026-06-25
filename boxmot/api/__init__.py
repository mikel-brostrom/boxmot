# Mikel Brostrom - BoxMOT - AGPL-3.0 license

"""Public BoxMOT Python API."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Boxmot": ("boxmot.api.client", "Boxmot"),
    "evaluate": ("boxmot.api.functional", "evaluate"),
    "track": ("boxmot.api.functional", "track"),
    "ExportResult": ("boxmot.api.results", "ExportResult"),
    "GenerateResult": ("boxmot.api.results", "GenerateResult"),
    "ResearchResult": ("boxmot.api.results", "ResearchResult"),
    "TrackRunResult": ("boxmot.api.results", "TrackRunResult"),
    "TrainResult": ("boxmot.api.results", "TrainResult"),
    "TuneResult": ("boxmot.api.results", "TuneResult"),
    "TuneTrialResult": ("boxmot.api.results", "TuneTrialResult"),
    "ValidationResult": ("boxmot.api.results", "ValidationResult"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
