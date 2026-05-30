"""Rich workflow reporter for the ``research`` command."""

from __future__ import annotations

import argparse

import boxmot.utils.rich.ui as ui
from boxmot.utils.rich.reporting import RichWorkflowReporter
from boxmot.utils.rich.steps import (
    BASELINE as RESEARCH_BASELINE_STEP,
)
from boxmot.utils.rich.steps import (
    BEST_CANDIDATE as RESEARCH_BEST_STEP,
)
from boxmot.utils.rich.steps import (
    PREPARE as RESEARCH_PREPARE_STEP,
)
from boxmot.utils.rich.steps import (
    RESEARCH_OPTIMIZE as RESEARCH_OPTIMIZE_STEP,
)
from boxmot.utils.rich.steps import (
    RESEARCH_STEPS,
)


def _build_research_workflow_fields(args: argparse.Namespace) -> list[tuple[str, object]]:
    fields: list[tuple[str, object]] = []

    tracker = getattr(args, "tracker", None)
    if tracker:
        fields.append(("Tracker", tracker))

    benchmark = (
        getattr(args, "benchmark", None)
        or getattr(args, "data", None)
    )
    if benchmark:
        fields.append(("Benchmark", benchmark))

    proposal_model = getattr(args, "proposal_model", None)
    if proposal_model:
        fields.append(("Proposal model", proposal_model))

    max_metric_calls = getattr(args, "max_metric_calls", None)
    if max_metric_calls is not None:
        fields.append(("Max metric calls", max_metric_calls))

    return fields


class ResearchWorkflowReporter(RichWorkflowReporter):
    title = "Tracker Research"
    prefer_compact_layout = True
    PREPARE = 0
    BASELINE = 1
    OPTIMIZE = 2
    BEST = 3
    steps = RESEARCH_STEPS

    def fields(self) -> list[tuple[str, object]]:
        return _build_research_workflow_fields(self.args)


def log_research_pipeline_intro(args: argparse.Namespace) -> ui.WorkflowProgress:
    return ResearchWorkflowReporter(args).create()


__all__ = [
    "RESEARCH_PREPARE_STEP",
    "RESEARCH_BASELINE_STEP",
    "RESEARCH_OPTIMIZE_STEP",
    "RESEARCH_BEST_STEP",
    "ResearchWorkflowReporter",
    "log_research_pipeline_intro",
]
