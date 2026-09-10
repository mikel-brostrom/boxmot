from __future__ import annotations

import math
from typing import Any, Sequence

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.eval.results import CORE_SUMMARY_COLUMNS, SUMMARY_COLUMNS
from boxmot.engine.ui.workflow import steps as step_labels
from boxmot.engine.ui.workflow.reporting import RichWorkflowCallback, RichWorkflowReporter, SilentProgressReporter
from boxmot.engine.ui.workflow.task_progress import create_task_progress

TUNE_SETUP_STEP = step_labels.SETUP
TUNE_OPTIMIZE_STEP = step_labels.OPTIMIZE


def _format_core_summary(summary: dict[str, Any] | None) -> str:
    """Format the three main metrics without inventing missing measurements."""
    metrics = []
    for metric in CORE_SUMMARY_COLUMNS:
        value = (summary or {}).get(metric)
        formatted = f"{float(value):.3f}" if value is not None and math.isfinite(float(value)) else "--"
        metrics.append(f"{metric}={formatted}")
    return " ".join(metrics)


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
    """Present the artifacts produced by the selected tuning backend."""
    artifact_table = Table.grid(expand=True, padding=(0, 1))
    artifact_table.add_column(style=ui.STYLE_ACCENT, no_wrap=True)
    artifact_table.add_column(style=ui.STYLE_TEXT, ratio=1, overflow="fold")
    if saved_artifacts.get("csv_path") is not None:
        artifact_table.add_row("Results CSV", str(saved_artifacts["csv_path"]))
    artifact_table.add_row(
        f"Best config ({saved_artifacts['best_trial_id']})",
        str(saved_artifacts["best_yaml_path"]),
    )
    for key, label in (("summary_path", "Summary"), ("study_path", "Study"), ("manifest_path", "Run manifest")):
        if saved_artifacts.get(key) is not None:
            artifact_table.add_row(label, str(saved_artifacts[key]))
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
        ("Build", getattr(args, "build", None)),
        ("Experiment", getattr(args, "experiment", None) or getattr(args, "benchmark", None)),
        ("Dataset", getattr(args, "dataset_id", None) or getattr(args, "dataset", None)),
        ("Trials", getattr(args, "n_trials", None)),
        ("Sequence workers", getattr(args, "sequence_workers", None)),
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


class TuneWorkflowReporter(RichWorkflowReporter):
    title = "Tuning"
    prefer_compact_layout = True
    SETUP = 0
    OPTIMIZE = 1
    steps = step_labels.TUNE_STEPS
    start_on_create = False

    def __init__(self, args: Any, *, maximize: list[str], minimize: list[str]) -> None:
        super().__init__(args)
        self.maximize = maximize
        self.minimize = minimize
        self.steps = step_labels.tune_steps()

    def fields(self) -> list[tuple[str, object]]:
        return build_tune_workflow_fields(self.args, maximize=self.maximize, minimize=self.minimize)


def log_tune_pipeline_intro(args: Any, *, maximize: list[str], minimize: list[str]) -> ui.WorkflowProgress:
    return TuneWorkflowReporter(args, maximize=maximize, minimize=minimize).create()


def format_initial_tune_progress(total: int) -> Group:
    """Show the shared progress row while the first trial waits to start."""
    return format_tune_progress(0, int(total))


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
        self.failed = 0
        self._trial_index_offset = 0
        self.trial_durations: list[float] = []
        self.trial_indices: dict[str, int] = {}
        self.active_trials: set[str] = set()
        self.best_score: tuple[float, ...] | None = None
        self.best_summary: dict[str, float] | None = None
        self.last_summary: dict[str, float] | None = None

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
        return None

    def _remaining_seconds(self) -> float | None:
        remaining_trials = max(self.total - self.completed, 0)
        return estimate_tune_remaining(self.trial_durations, remaining_trials)

    def _set_progress(self) -> None:
        """Keep completed-trial metrics visible between callback events."""
        self.set_workflow_detail_renderable(
            format_tune_progress(
                self.completed,
                self.total,
                self.last_summary,
                best_summary=self.best_summary,
                current_trial=self._running_index(),
                remaining_seconds=self._remaining_seconds(),
                failed=self.failed,
                active_trials=len(self.active_trials),
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
        metrics = dict.fromkeys((*SUMMARY_COLUMNS, *self.maximize, *self.minimize))
        summary = {key: float(result[key]) for key in metrics if result.get(key) is not None}
        if summary:
            self.last_summary = summary
            score = _score_summary(summary, maximize=self.maximize, minimize=self.minimize)
            if all(math.isfinite(value) for value in score) and (self.best_score is None or score > self.best_score):
                self.best_score = score
                self.best_summary = summary.copy()
        self._set_progress()

    def on_trial_error(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        trial_id = self._trial_id(trial)
        self.active_trials.discard(trial_id)
        self.completed += 1
        self.failed += 1
        self._set_progress()


# ── Tune progress presentation ────────────────────────────────────────


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


def format_tune_progress(
    completed: int,
    total: int,
    last_summary: dict[str, Any] | None = None,
    *,
    best_summary: dict[str, Any] | None = None,
    current_trial: int | None = None,
    remaining_seconds: float | None = None,
    failed: int = 0,
    active_trials: int | None = None,
) -> Group:
    """Use evaluation's Rich bar, count, and status columns for trial progress.

    This renderable is created in the driver and only stored in its workflow.
    Ray callbacks retain ordinary counts and metrics, never Rich progress locks.
    """
    total = max(0, int(total))
    completed = min(max(0, int(completed)), total)
    failed = min(max(0, int(failed)), completed)
    active = max(0, int(active_trials)) if active_trials is not None else int(current_trial is not None)
    finished = completed >= total
    if finished:
        active = 0
    status = ("failed" if failed else "completed") if finished else ("running" if active else "queued")

    aggregate = Text("Tuning: ", style=ui.STYLE_TEXT_STRONG)
    aggregate.append(f"{completed - failed}/{total} trials done", style=ui.STYLE_STATUS_DONE)
    if active:
        aggregate.append(f" · {active} running", style=ui.STYLE_STATUS_ACTIVE)
    if failed:
        aggregate.append(f" · {failed} failed", style=ui.STYLE_STATUS_FAILED)

    detail = []
    if not finished and current_trial is not None:
        detail.append(f"trial {current_trial}/{total}")
    remaining = format_remaining_time(0.0 if finished else remaining_seconds)
    detail.append(f"remaining {remaining}")
    progress = create_task_progress(unit="trials")
    task_id = progress.add_task(
        "Tune",
        total=total,
        completed=completed,
        start=bool(active or completed or finished),
        status=status,
        detail=" · ".join(detail),
    )
    if finished:
        progress.update(task_id, completed=completed)
        progress.stop_task(task_id)
    parts: list[RenderableType] = [aggregate, progress]
    if best_summary is not None or last_summary is not None:
        metrics = Text(f"Best trial: {_format_core_summary(best_summary)}", style=ui.STYLE_STATUS_DONE)
        metrics.append(f" · Last trial: {_format_core_summary(last_summary)}", style=ui.STYLE_MUTED)
        parts.append(metrics)
    return Group(*parts)
