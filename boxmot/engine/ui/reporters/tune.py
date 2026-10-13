from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.eval.results import CORE_SUMMARY_COLUMNS, SUMMARY_COLUMNS
from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter
from boxmot.engine.ui.workflow import steps as step_labels
from boxmot.engine.ui.workflow.reporting import (
    RichWorkflowCallback,
    RichWorkflowReporter,
    SilentProgressReporter,
    WorkflowDetailCallback,
)
from boxmot.engine.ui.workflow.task_progress import create_task_progress

if TYPE_CHECKING:
    from boxmot.engine.eval.replay import ReplayProgressEvent

TUNE_SETUP_STEP = step_labels.SETUP
TUNE_OPTIMIZE_STEP = step_labels.OPTIMIZE


def _metric_value(summary: Mapping[str, Any] | None, metric: str) -> float | None:
    """Distinguish unavailable metrics from measured zero."""
    value = (summary or {}).get(metric)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_comparison(
    value: float | None, baseline: float | None, *, compare: bool, minimize: bool
) -> Text:
    """Show raw metric differences, coloring them by the objective direction."""
    cell = Text("--" if value is None else f"{value:.3f}", style=ui.STYLE_TEXT)
    if not compare:
        return cell
    if value is None or baseline is None:
        cell.append(" (--)", style=ui.STYLE_MUTED)
        return cell
    delta = round(value - baseline, 3)
    if delta == 0:
        delta = 0.0
        style = ui.STYLE_MUTED
    else:
        improved = delta < 0 if minimize else delta > 0
        style = ui.STYLE_STATUS_DONE if improved else ui.STYLE_STATUS_FAILED
    cell.append(f" ({delta:+.3f})", style=style)
    return cell


def _trial_comparison_table(
    first: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None,
    last: Mapping[str, Any] | None,
    *,
    best_trial: int | None,
    last_trial: int | None,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> Table:
    """Keep the baseline and both moving results aligned in a compact table."""
    metrics = tuple(dict.fromkeys((*CORE_SUMMARY_COLUMNS, *maximize, *minimize)))
    table = Table(box=None, padding=(0, 1), pad_edge=False, header_style=ui.STYLE_TABLE_HEADER)
    table.add_column("Run", style=ui.STYLE_TEXT_STRONG)
    for metric in metrics:
        table.add_column(metric, justify="right", overflow="fold")
    for label, index, summary, compare in (
        ("First", None, first, False),
        ("Best", best_trial, best, True),
        ("Last", last_trial, last, True),
    ):
        name = f"{label} #{index}" if index is not None else label
        table.add_row(
            name,
            *(
                _metric_comparison(
                    _metric_value(summary, metric),
                    _metric_value(first, metric),
                    compare=compare,
                    minimize=metric in minimize,
                )
                for metric in metrics
            ),
        )
    return table


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
    *,
    comparison_label: str | None = None,
) -> RenderableType:
    """Combine the best result with its comparison context and saved artifacts."""
    if comparison_label is not None:
        best_renderable = Group(Text(comparison_label, style=ui.STYLE_MUTED), best_renderable)
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
        self.first_summary: dict[str, float] | None = None
        self.best_summary: dict[str, float] | None = None
        self.last_summary: dict[str, float] | None = None
        self.best_trial_index: int | None = None
        self.last_trial_index: int | None = None
        self.sequence_totals: dict[str, int | None] = {}
        self.sequence_progress_dir: Path | None = None
        self.sequence_events: dict[str, tuple[ReplayProgressEvent, ...]] = {}
        self.last_sequence_trial: str | None = None
        self.last_sequence_failed = False

    def configure_sequence_progress(self, directory: Path, totals: Mapping[str, int | None]) -> None:
        """Bind this driver's progress channel without retaining Rich state."""
        self.sequence_progress_dir = Path(directory)
        self.sequence_totals = dict(totals)
        self.sequence_events.clear()
        self.last_sequence_trial = None
        self.last_sequence_failed = False

    def _read_sequence_progress(self, trial_id: str) -> bool:
        """Read the latest bounded worker snapshot only when it has changed."""
        if self.sequence_progress_dir is None:
            return False
        from boxmot.engine.tuning.progress import read_trial_sequence_progress

        events = read_trial_sequence_progress(self.sequence_progress_dir, trial_id)
        if not events or events == self.sequence_events.get(trial_id):
            return False
        self.sequence_events[trial_id] = events
        return True

    def _clear_sequence_progress(self, trial_id: str) -> None:
        """Remove a finished or retried trial's transport snapshot."""
        if self.sequence_progress_dir is not None:
            from boxmot.engine.tuning.progress import clear_trial_sequence_progress

            clear_trial_sequence_progress(self.sequence_progress_dir, trial_id)

    def _retain_sequence_progress(self) -> None:
        """Retain only running trials and the latest finished trial's rows."""
        self.sequence_events = {
            trial_id: events
            for trial_id, events in self.sequence_events.items()
            if trial_id in self.active_trials or trial_id == self.last_sequence_trial
        }

    def on_step_end(self, iteration: int, trials: list, **info: Any) -> None:
        """Refresh frames while Ray is waiting for complete trial metrics."""
        changed = [self._read_sequence_progress(trial_id) for trial_id in self.active_trials]
        if any(changed):
            self._set_progress()

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
        workflow = type(self)._workflow
        if workflow is None:
            return
        parts: list[RenderableType] = [
            format_tune_progress(
                self.completed,
                self.total,
                self.last_summary,
                first_summary=self.first_summary,
                best_summary=self.best_summary,
                best_trial=self.best_trial_index,
                last_trial=self.last_trial_index,
                maximize=self.maximize,
                minimize=self.minimize,
                current_trial=self._running_index(),
                remaining_seconds=self._remaining_seconds(),
                failed=self.failed,
                active_trials=len(self.active_trials),
            )
        ]
        shown = sorted(self.active_trials, key=self.trial_indices.__getitem__)
        if not shown and self.last_sequence_trial is not None:
            shown = [self.last_sequence_trial]
        for trial_id in shown:
            events = self.sequence_events.get(trial_id, ())
            totals = dict(self.sequence_totals)
            for event in events:
                totals.setdefault(event.sequence_id, event.total)
            if not totals:
                continue
            presenter = EvalSequenceProgressPresenter(
                WorkflowDetailCallback(workflow, TUNE_OPTIMIZE_STEP, render=False), totals
            )
            for event in events:
                # Sequence names and frame counts are enough during replay;
                # retain diagnostic details only for failed sequences.
                presenter(replace(event, detail=event.detail if event.status == "failed" else None))
            if trial_id == self.last_sequence_trial and self.last_sequence_failed:
                terminal = {event.sequence_id for event in events if event.status in {"completed", "failed"}}
                for sequence_id in totals.keys() - terminal:
                    presenter.fail(sequence_id, "trial failed")
            parts.extend(
                (Text(f"Trial {self.trial_indices[trial_id]}/{self.total}", style=ui.STYLE_TITLE), presenter.renderable)
            )
        self.set_workflow_detail_renderable(Group(*parts))

    def on_trial_start(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        """Retain submission order even when workers start trials out of order."""
        for scheduled_trial in trials:
            self._trial_index(scheduled_trial)
        trial_id = self._trial_id(trial)
        self._trial_index(trial)
        self.active_trials.add(trial_id)
        self._clear_sequence_progress(trial_id)
        self.sequence_events.pop(trial_id, None)
        self.last_sequence_trial = None
        self.last_sequence_failed = False
        self._retain_sequence_progress()
        self._set_progress()

    def on_trial_complete(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        """Refresh the latest result and replace the best only on improvement."""
        trial_id = self._trial_id(trial)
        trial_index = self._trial_index(trial)
        self._read_sequence_progress(trial_id)
        self._clear_sequence_progress(trial_id)
        self.last_sequence_trial = trial_id
        self.last_sequence_failed = False
        self.active_trials.discard(trial_id)
        self._retain_sequence_progress()
        self.completed += 1
        result = getattr(trial, "last_result", {}) or {}
        self.record_result(result, trial_index=trial_index)
        self._set_progress()

    def record_result(self, result: Mapping[str, Any], *, trial_index: int) -> None:
        """Retain completed metrics, including restored results in completion order."""
        duration = result.get("time_total_s")
        if duration is not None and math.isfinite(float(duration)) and float(duration) >= 0:
            self.trial_durations.append(float(duration))
        metrics = dict.fromkeys((*SUMMARY_COLUMNS, *self.maximize, *self.minimize))
        summary = {key: float(result[key]) for key in metrics if result.get(key) is not None}
        if summary:
            # The first scheduled trial is the baseline even if other trials
            # finish sooner. A failed baseline must not become another trial.
            if trial_index == 1:
                self.first_summary = summary.copy()
            self.last_summary = summary
            self.last_trial_index = trial_index
            score = _score_summary(summary, maximize=self.maximize, minimize=self.minimize)
            if all(math.isfinite(value) for value in score) and (self.best_score is None or score > self.best_score):
                self.best_score = score
                self.best_summary = summary.copy()
                self.best_trial_index = trial_index

    def on_trial_error(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        """Retain completed metrics when a trial fails, including before startup."""
        for scheduled_trial in trials:
            self._trial_index(scheduled_trial)
        trial_id = self._trial_id(trial)
        self._trial_index(trial)
        self._read_sequence_progress(trial_id)
        self._clear_sequence_progress(trial_id)
        self.last_sequence_trial = trial_id
        self.last_sequence_failed = True
        events = self.sequence_events.get(trial_id, ())
        self.sequence_events[trial_id] = tuple(
            replace(event, status="failed", detail=event.detail or "trial failed")
            if event.status in {"queued", "running"} else event
            for event in events
        )
        self.active_trials.discard(trial_id)
        self._retain_sequence_progress()
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
    first_summary: dict[str, Any] | None = None,
    best_summary: dict[str, Any] | None = None,
    best_trial: int | None = None,
    last_trial: int | None = None,
    maximize: Sequence[str] = (),
    minimize: Sequence[str] = (),
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
    if any(summary is not None for summary in (first_summary, best_summary, last_summary)):
        parts.extend((
            Text("Δ vs first trial (metric units)", style=ui.STYLE_MUTED),
            _trial_comparison_table(
                first_summary,
                best_summary,
                last_summary,
                best_trial=best_trial,
                last_trial=last_trial,
                maximize=maximize,
                minimize=minimize,
            ),
        ))
    return Group(*parts)
