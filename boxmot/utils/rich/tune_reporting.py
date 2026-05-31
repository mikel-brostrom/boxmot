from __future__ import annotations

import math
from typing import Any, Sequence

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

import boxmot.utils.rich.ui as ui
from boxmot.engine.workflows.reporting import (
    SUMMARY_COLUMNS,
    format_core_summary,
)
from boxmot.utils.rich.reporting import RichWorkflowCallback, RichWorkflowReporter, SilentProgressReporter
from boxmot.utils.rich.steps import (
    GENERATE as TUNE_GENERATE_STEP,
)
from boxmot.utils.rich.steps import (
    OPTIMIZE as TUNE_OPTIMIZE_STEP,
)
from boxmot.utils.rich.steps import (
    SETUP as TUNE_SETUP_STEP,
)
from boxmot.utils.rich.steps import (
    TUNE_STEPS,
)


def _score_summary(
    summary: dict[str, Any],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> tuple[float, ...]:
    score: list[float] = []
    for metric in maximize:
        score.append(float(summary.get(metric, float("-inf"))))
    for metric in minimize:
        score.append(-float(summary.get(metric, float("inf"))))
    return tuple(score)


def build_tune_artifacts_renderable(saved_artifacts: dict[str, Any]) -> RenderableType:
    artifact_table = Table.grid(expand=True, padding=(0, 1))
    artifact_table.add_column(style=ui.STYLE_ACCENT, no_wrap=True)
    artifact_table.add_column(style=ui.STYLE_TEXT, ratio=1)
    artifact_table.add_row("Results CSV", str(saved_artifacts["csv_path"]))
    artifact_table.add_row(
        f"Best config ({saved_artifacts['best_trial_id']})",
        str(saved_artifacts["best_yaml_path"]),
    )
    artifact_table.add_row("Summary", str(saved_artifacts["summary_path"]))
    return Group(
        Text("Saved Artifacts", style=ui.STYLE_TITLE),
        artifact_table,
    )


def combine_tune_result_renderables(
    best_renderable: RenderableType,
    artifacts_renderable: RenderableType | None,
) -> RenderableType:
    if artifacts_renderable is None:
        return best_renderable
    return Group(best_renderable, artifacts_renderable)


def build_tune_workflow_fields(args: Any, *, maximize: list[str], minimize: list[str]) -> list[tuple[str, object]]:
    fields: list[tuple[str, object]] = [
        ("Tracker", getattr(args, "tracker", None)),
        ("Detector", getattr(args, "detector", [None])[0]),
        ("ReID", getattr(args, "reid", [None])[0]),
        ("Dataset", getattr(args, "data", getattr(args, "benchmark", None))),
        ("Trials", getattr(args, "n_trials", None)),
        ("Objective", _format_tune_objective(maximize=maximize, minimize=minimize)),
    ]
    return fields


def _format_tune_objective(*, maximize: Sequence[str], minimize: Sequence[str]) -> str:
    objective_count = len(maximize) + len(minimize)
    mode = "Pareto" if objective_count > 1 else "Single-objective"
    parts = [f"max {', '.join(str(metric) for metric in maximize)}"] if maximize else []
    if minimize:
        parts.append(f"min {', '.join(str(metric) for metric in minimize)}")
    objective = " / ".join(parts)
    return f"{mode}: {objective}" if objective else mode


class TuneWorkflowProgress(ui.WorkflowProgress):
    def renderable(self, *, compact: bool = False, include_setup: bool = True):
        return ui.build_workflow_intro(
            self.title,
            self.fields,
            steps=self.steps,
            detail_title=self.detail_title,
            detail_text=self.detail_text,
            detail_renderable=self.detail_renderable,
            include_setup=include_setup,
            compact=compact,
        )


class TuneWorkflowReporter(RichWorkflowReporter):
    title = "Tuning"
    prefer_compact_layout = True
    SETUP = 0
    GENERATE = 1
    OPTIMIZE = 2
    steps = TUNE_STEPS
    start_on_create = False

    def __init__(self, args: Any, *, maximize: list[str], minimize: list[str]) -> None:
        super().__init__(args)
        self.maximize = maximize
        self.minimize = minimize

    def fields(self) -> list[tuple[str, object]]:
        return build_tune_workflow_fields(self.args, maximize=self.maximize, minimize=self.minimize)

    def create(self) -> ui.WorkflowProgress:
        workflow = TuneWorkflowProgress(
            self.title,
            self.fields(),
            steps=list(self.steps),
            stderr=self.stderr,
            transient=self.transient,
            prefer_alt_screen=self.prefer_alt_screen,
            prefer_compact_layout=self.prefer_compact_layout,
        )
        if self.start_on_create:
            workflow.start()
        return workflow


def log_tune_pipeline_intro(args: Any, *, maximize: list[str], minimize: list[str]) -> ui.WorkflowProgress:
    return TuneWorkflowReporter(args, maximize=maximize, minimize=minimize).create()


def format_initial_tune_progress(total: int) -> str:
    return format_tune_progress(0, int(total), current_trial=1)


def set_tune_progress_workflow(workflow: ui.WorkflowProgress | None) -> None:
    """Register the driver-local workflow used by Ray Tune callbacks."""
    TuneWorkflowCallback.set_workflow(workflow)


class TuneSilentReporter(SilentProgressReporter):
    """Suppress Ray Tune's terminal reporter while Rich owns the workflow UI."""


class TuneWorkflowCallback(RichWorkflowCallback):
    """Serializable Ray callback that keeps Rich workflow state driver-local."""

    detail_step = TUNE_OPTIMIZE_STEP

    def __init__(self, *, total: int, maximize: list[str], minimize: list[str]) -> None:
        self.total = int(total)
        self.maximize = list(maximize)
        self.minimize = list(minimize)
        self.completed = 0
        self._trial_index_offset = 0
        self.trial_durations: list[float] = []
        self.trial_indices: dict[str, int] = {}
        self.active_trials: set[str] = set()
        self.best_score: tuple[float, ...] | None = None

    def _trial_id(self, trial: Any) -> str:
        return str(getattr(trial, "trial_id", getattr(trial, "trial_name", trial)))

    def _trial_index(self, trial: Any) -> int:
        trial_id = self._trial_id(trial)
        if trial_id not in self.trial_indices:
            self.trial_indices[trial_id] = len(self.trial_indices) + 1 + self._trial_index_offset
        return self.trial_indices[trial_id]

    def _running_index(self) -> int | None:
        if self.active_trials:
            return min(self.trial_indices[trial_id] for trial_id in self.active_trials)
        if self.completed < self.total:
            return min(self.completed + 1, self.total)
        return None

    def _remaining_seconds(self) -> float | None:
        remaining_trials = max(self.total - self.completed, 0)
        return estimate_tune_remaining(self.trial_durations, remaining_trials)

    def _set_progress(self, summary: dict[str, Any] | None = None, *, is_new_best: bool = False) -> None:
        self.set_workflow_detail(
            format_tune_progress(
                self.completed,
                self.total,
                summary,
                current_trial=self._running_index(),
                is_new_best=is_new_best,
                remaining_seconds=self._remaining_seconds(),
            )
        )

    def on_trial_start(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        trial_id = self._trial_id(trial)
        self._trial_index(trial)
        self.active_trials.add(trial_id)
        self._set_progress()

    def on_trial_complete(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        trial_id = self._trial_id(trial)
        self.active_trials.discard(trial_id)
        self.completed += 1
        result = getattr(trial, "last_result", {}) or {}
        duration = result.get("time_total_s")
        if duration is not None:
            self.trial_durations.append(float(duration))
        summary = {
            key: float(result.get(key, 0.0))
            for key in SUMMARY_COLUMNS
            if key in result
        }
        score = _score_summary(summary, maximize=self.maximize, minimize=self.minimize) if summary else None
        is_new_best = score is not None and (self.best_score is None or score > self.best_score)
        if is_new_best and score is not None:
            self.best_score = score
        self._set_progress(summary if summary else None, is_new_best=is_new_best)

    def on_trial_error(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        trial_id = self._trial_id(trial)
        self.active_trials.discard(trial_id)
        self.completed += 1
        self._set_progress()


# ── Tune progress formatting (moved from workflow_reporting) ─────────────


def format_remaining_time(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"

    total_seconds = 0 if seconds <= 0 else int(math.ceil(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_tune_remaining(trial_durations: Sequence[float], remaining_trials: int) -> float | None:
    if remaining_trials <= 0:
        return 0.0
    if not trial_durations:
        return None
    avg_trial_seconds = sum(trial_durations) / len(trial_durations)
    return avg_trial_seconds * remaining_trials


def format_progress_bar(current: int, total: int, *, bar_width: int = 20) -> tuple[str, float]:
    if total <= 0:
        pct = 1.0 if current >= total else 0.0
    else:
        pct = min(max(current / total, 0.0), 1.0)

    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    return bar, pct


def format_named_progress(label: str, current: int, total: int, *, detail: str = "") -> str:
    bar, pct = format_progress_bar(current, total)
    message = f"  {label:<8s} {bar} {pct:>5.0%}  ({current}/{total})"
    if detail:
        message = f"{message}  {detail}"
    return message


def format_tune_progress(
    completed: int,
    total: int,
    summary: dict[str, Any] | None = None,
    *,
    current_trial: int | None = None,
    is_new_best: bool = False,
    remaining_seconds: float | None = None,
) -> str:
    remaining = format_remaining_time(remaining_seconds)
    if summary is None:
        running = current_trial if current_trial is not None else (completed + 1)
        return format_named_progress(
            "Tune",
            completed,
            total,
            detail=f"running trial {running}/{total}  remaining {remaining}",
        )

    core = format_core_summary(summary)
    suffix = "  best" if is_new_best else ""
    if current_trial is not None and current_trial > completed:
        detail = f"running trial {current_trial}/{total}  last {core}{suffix}  remaining {remaining}"
        return format_named_progress("Tune", completed, total, detail=detail)

    detail = f"{core}{suffix}  remaining {remaining}"
    return format_named_progress("Tune", completed, total, detail=detail)
