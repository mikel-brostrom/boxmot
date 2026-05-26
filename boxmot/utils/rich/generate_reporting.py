"""Rich workflow reporter for the ``generate`` command."""

from __future__ import annotations

import argparse

import boxmot.utils.rich.ui as ui
from boxmot.utils.rich.reporting import RichWorkflowReporter
from boxmot.utils.rich.steps import (
    GENERATE as GENERATE_RUN_STEP,
)
from boxmot.utils.rich.steps import (
    GENERATE_STEPS,
)
from boxmot.utils.rich.steps import (
    SETUP as GENERATE_SETUP_STEP,
)


def _build_generate_workflow_fields(args: argparse.Namespace) -> list[tuple[str, object]]:
    fields: list[tuple[str, object]] = []

    detector = getattr(args, "detector", None)
    if detector:
        primary = detector[0] if isinstance(detector, (list, tuple)) else detector
        fields.append(("Detector", primary))

    reid = getattr(args, "reid", None)
    if reid:
        primary = reid[0] if isinstance(reid, (list, tuple)) else reid
        fields.append(("ReID", primary))

    dataset = (
        getattr(args, "data", None)
        or getattr(args, "benchmark", None)
        or getattr(args, "source", None)
    )
    if dataset:
        fields.append(("Dataset", dataset))

    imgsz = getattr(args, "imgsz", None)
    if imgsz is not None:
        fields.append(("Image size", imgsz))

    return fields


class GenerateWorkflowReporter(RichWorkflowReporter):
    title = "Generate"
    prefer_compact_layout = True
    SETUP = 0
    RUN = 1
    steps = GENERATE_STEPS

    def fields(self) -> list[tuple[str, object]]:
        return _build_generate_workflow_fields(self.args)


def log_generate_pipeline_intro(args: argparse.Namespace) -> ui.WorkflowProgress:
    return GenerateWorkflowReporter(args).create()


__all__ = [
    "GENERATE_SETUP_STEP",
    "GENERATE_RUN_STEP",
    "GenerateWorkflowReporter",
    "log_generate_pipeline_intro",
]
