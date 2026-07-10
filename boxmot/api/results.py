from __future__ import annotations

from importlib import import_module

from boxmot.engine.research import models as _research_models
from boxmot.engine.workflows import results as _workflow_results

ExportResult = _workflow_results.ExportResult
GenerateResult = _workflow_results.GenerateResult
ResearchResult = _research_models.ResearchResult
TrackRunResult = _workflow_results.TrackRunResult
TuneResult = _workflow_results.TuneResult
TuneTrialResult = _workflow_results.TuneTrialResult
ValidationResult = _workflow_results.ValidationResult

_LAZY_EXPORTS = {
    "TrainResult": ("boxmot.reid.training.trainer", "TrainResult"),
}

_STATIC_EXPORTS = (
    "ExportResult",
    "GenerateResult",
    "ResearchResult",
    "TrackRunResult",
    "TuneResult",
    "TuneTrialResult",
    "ValidationResult",
)
__all__ = _STATIC_EXPORTS + tuple(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
