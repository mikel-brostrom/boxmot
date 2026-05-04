"""Rich workflow reporter for the ``research`` command."""

from __future__ import annotations

import argparse

import boxmot.utils.rich.ui as ui
from boxmot.utils.rich.reporting import RichWorkflowReporter


RESEARCH_PREPARE_STEP = "Prepare workspace"
RESEARCH_BASELINE_STEP = "Baseline evaluation"
RESEARCH_OPTIMIZE_STEP = "GEPA optimization"
RESEARCH_BEST_STEP = "Best candidate evaluation"


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
    steps = (
        (RESEARCH_PREPARE_STEP, "active"),
        (RESEARCH_BASELINE_STEP, "todo"),
        (RESEARCH_OPTIMIZE_STEP, "todo"),
        (RESEARCH_BEST_STEP, "todo"),
    )

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
