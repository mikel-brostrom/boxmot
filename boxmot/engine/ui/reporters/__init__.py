"""Command-specific Rich workflow reporters."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "EvalWorkflowReporter": ("boxmot.engine.ui.reporters.eval", "EvalWorkflowReporter"),
    "ExportWorkflowReporter": ("boxmot.engine.ui.reporters.export", "ExportWorkflowReporter"),
    "MaterializeWorkflowReporter": (
        "boxmot.engine.ui.reporters.materialize",
        "MaterializeWorkflowReporter",
    ),
    "ResearchWorkflowReporter": ("boxmot.engine.ui.reporters.research", "ResearchWorkflowReporter"),
    "TrackWorkflowReporter": ("boxmot.engine.ui.reporters.track", "TrackWorkflowReporter"),
    "TuneWorkflowReporter": ("boxmot.engine.ui.reporters.tune", "TuneWorkflowReporter"),
}

__all__ = [
    "EvalWorkflowReporter",
    "ExportWorkflowReporter",
    "MaterializeWorkflowReporter",
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
