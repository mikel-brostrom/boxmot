"""Workflow plumbing for Rich progress panels."""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "PipelineTracker": ("boxmot.engine.ui.workflow.pipeline", "PipelineTracker"),
    "StepRecord": ("boxmot.engine.ui.workflow.pipeline", "StepRecord"),
    "RichTqdm": ("boxmot.engine.ui.workflow.progress", "RichTqdm"),
    "RichWorkflowCallback": ("boxmot.engine.ui.workflow.reporting", "RichWorkflowCallback"),
    "RichWorkflowReporter": ("boxmot.engine.ui.workflow.reporting", "RichWorkflowReporter"),
    "SilentProgressReporter": ("boxmot.engine.ui.workflow.reporting", "SilentProgressReporter"),
    "WorkflowDetailCallback": ("boxmot.engine.ui.workflow.reporting", "WorkflowDetailCallback"),
}

__all__ = [
    "PipelineTracker",
    "RichTqdm",
    "RichWorkflowCallback",
    "RichWorkflowReporter",
    "SilentProgressReporter",
    "StepRecord",
    "WorkflowDetailCallback",
]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
