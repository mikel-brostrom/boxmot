from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ExportResult": ("boxmot.engine.workflows.results", "ExportResult"),
    "GenerateResult": ("boxmot.engine.workflows.results", "GenerateResult"),
    "ResearchResult": ("boxmot.engine.research", "ResearchResult"),
    "TrackRunResult": ("boxmot.engine.workflows.results", "TrackRunResult"),
    "TrainResult": ("boxmot.reid.training.trainer", "TrainResult"),
    "TuneResult": ("boxmot.engine.workflows.results", "TuneResult"),
    "TuneTrialResult": ("boxmot.engine.workflows.results", "TuneTrialResult"),
    "ValidationResult": ("boxmot.engine.workflows.results", "ValidationResult"),
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
