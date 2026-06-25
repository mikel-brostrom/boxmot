"""Command-specific Rich workflow reporters."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EvalWorkflowReporter": ("boxmot.utils.rich.reporters.eval", "EvalWorkflowReporter"),
    "ExportWorkflowReporter": ("boxmot.utils.rich.reporters.export", "ExportWorkflowReporter"),
    "GenerateWorkflowReporter": ("boxmot.utils.rich.reporters.generate", "GenerateWorkflowReporter"),
    "ResearchWorkflowReporter": ("boxmot.utils.rich.reporters.research", "ResearchWorkflowReporter"),
    "TrackWorkflowReporter": ("boxmot.utils.rich.reporters.track", "TrackWorkflowReporter"),
    "TuneWorkflowReporter": ("boxmot.utils.rich.reporters.tune", "TuneWorkflowReporter"),
}

__all__ = [
    "EvalWorkflowReporter",
    "ExportWorkflowReporter",
    "GenerateWorkflowReporter",
    "ResearchWorkflowReporter",
    "TrackWorkflowReporter",
    "TuneWorkflowReporter",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
