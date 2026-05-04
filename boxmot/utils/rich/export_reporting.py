"""Rich workflow reporter for the ``export`` command."""

from __future__ import annotations

from typing import Any

import boxmot.utils.rich.ui as ui
from boxmot.utils.rich.reporting import RichWorkflowReporter
from boxmot.utils.rich.steps import (
    EXPORT as EXPORT_RUN_STEP,
    EXPORT_STEPS,
    SETUP as EXPORT_SETUP_STEP,
)


def _build_export_workflow_fields(args: Any) -> list[tuple[str, object]]:
    fields: list[tuple[str, object]] = [
        ("Weights", getattr(args, "weights", None)),
    ]
    include = getattr(args, "include", None)
    if include:
        fields.append(("Formats", ", ".join(include)))
    device = getattr(args, "device", None)
    if device not in {None, ""}:
        fields.append(("Device", device))
    fields.append(("Half", bool(getattr(args, "half", False))))
    fields.append(("Dynamic", bool(getattr(args, "dynamic", False))))
    fields.append(("Simplify", bool(getattr(args, "simplify", False))))
    return fields


class ExportWorkflowReporter(RichWorkflowReporter):
    title = "ReID Export"
    SETUP = 0
    RUN = 1
    steps = EXPORT_STEPS

    def fields(self) -> list[tuple[str, object]]:
        return _build_export_workflow_fields(self.args)


def log_export_pipeline_intro(args: Any) -> ui.WorkflowProgress:
    return ExportWorkflowReporter(args).create()


__all__ = [
    "EXPORT_SETUP_STEP",
    "EXPORT_RUN_STEP",
    "ExportWorkflowReporter",
    "log_export_pipeline_intro",
]
