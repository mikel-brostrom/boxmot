"""Rich workflow reporter for the ``eval`` command."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rich.console import Group
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TextColumn
from rich.table import Table
from rich.text import Text

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.ui.workflow.fields import panel_field
from boxmot.engine.ui.workflow.reporting import RichWorkflowReporter, WorkflowDetailCallback
from boxmot.engine.ui.workflow.steps import (
    EVALUATE as EVAL_EVALUATE_STEP,
)
from boxmot.engine.ui.workflow.steps import (
    SETUP as EVAL_SETUP_STEP,
)
from boxmot.engine.ui.workflow.steps import (
    TRACK as EVAL_TRACK_STEP,
)
from boxmot.engine.ui.workflow.steps import (
    eval_steps,
)

SequenceProgressStatus = Literal["queued", "running", "completed", "failed"]
_TERMINAL_SEQUENCE_STATES = frozenset({"completed", "failed"})
_SEQUENCE_STATE_RANK = {"queued": 0, "running": 1, "completed": 2, "failed": 2}


class _SequenceProgressEvent(Protocol):
    """Structural event contract consumed without importing the replay engine."""

    sequence_id: str
    status: SequenceProgressStatus
    completed: int
    total: int | None
    detail: str | None


@dataclass(slots=True)
class _SequenceProgressState:
    task_id: int
    total: int | None
    completed: int = 0
    status: SequenceProgressStatus = "queued"
    detail: str | None = None


class _SequenceCountColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        completed = int(task.completed)
        if task.total is None:
            return Text(f"{completed:,} frames", style=ui.STYLE_MUTED)
        return Text(f"{completed:,}/{int(task.total):,} frames", style=ui.STYLE_MUTED)


class _SequenceStatusColumn(ProgressColumn):
    _STYLES = {
        "queued": ("○", "pending", ui.STYLE_STATUS_TODO),
        "running": ("▶", "running", ui.STYLE_STATUS_ACTIVE),
        "completed": ("✓", "done", ui.STYLE_STATUS_DONE),
        "failed": ("✕", "failed", ui.STYLE_STATUS_FAILED),
    }

    def render(self, task: Task) -> Text:
        status = str(task.fields["status"])
        marker, label, style = self._STYLES[status]
        rendered = Text()
        rendered.append(marker, style=style)
        rendered.append(f" {label}", style=style)
        detail = task.fields.get("detail")
        if detail:
            rendered.append(f" · {detail}", style=ui.STYLE_MUTED)
        return rendered


class _SequenceProgressView:
    """Dynamic Rich renderable backed by a sequence presenter."""

    def __init__(self, presenter: "EvalSequenceProgressPresenter") -> None:
        self._presenter = presenter

    def __rich__(self) -> Group:
        progress = self._presenter.progress
        tasks = progress.tasks
        if len(tasks) <= 10:
            return Group(self._presenter.summary, progress)

        midpoint = (len(tasks) + 1) // 2
        columns = Table.grid(expand=True, padding=(0, 2))
        columns.add_column(ratio=1)
        columns.add_column(ratio=1)
        columns.add_row(
            progress.make_tasks_table(tasks[:midpoint]),
            progress.make_tasks_table(tasks[midpoint:]),
        )
        return Group(self._presenter.summary, columns)


class EvalSequenceProgressPresenter:
    """Render ordered per-sequence replay progress in the active eval step.

    The presenter is intentionally UI-only and driver-local. Replay workers
    may send any picklable event object exposing ``sequence_id``, ``status``,
    ``completed``, ``total``, and ``detail``; Rich objects never cross the
    worker boundary.
    """

    def __init__(
        self,
        callback: WorkflowDetailCallback,
        sequence_totals: Mapping[str, int | None],
        *,
        clock: Callable[[], float] = time.monotonic,
        refresh_interval_s: float | None = 0.08,
    ) -> None:
        if refresh_interval_s is not None and refresh_interval_s <= 0:
            raise ValueError("refresh_interval_s must be positive or None.")
        if not sequence_totals:
            raise ValueError("sequence_totals must not be empty.")

        self._callback = callback
        self._clock = clock
        self._refresh_interval_s = refresh_interval_s
        self._last_refresh_s: float | None = None
        self._detail_scope: Any | None = None
        self._active = False
        self._progress = Progress(
            TextColumn("{task.description}", style=ui.STYLE_TEXT_STRONG, markup=False),
            BarColumn(),
            _SequenceCountColumn(),
            _SequenceStatusColumn(),
            expand=True,
            auto_refresh=False,
        )
        self._states: dict[str, _SequenceProgressState] = {}
        for sequence_id, total in sequence_totals.items():
            sequence_id = self._validate_sequence_id(sequence_id)
            total = self._validate_count(total, name="total", optional=True)
            task_id = self._progress.add_task(
                sequence_id,
                total=total,
                completed=0,
                start=False,
                status="queued",
                detail=None,
            )
            self._states[sequence_id] = _SequenceProgressState(task_id=task_id, total=total)
        self._view = _SequenceProgressView(self)

    @staticmethod
    def _validate_sequence_id(sequence_id: object) -> str:
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("sequence_id must be a non-empty string.")
        return sequence_id

    @staticmethod
    def _validate_count(value: object, *, name: str, optional: bool = False) -> int | None:
        if value is None and optional:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            suffix = " or None" if optional else ""
            raise ValueError(f"{name} must be a non-negative integer{suffix}.")
        return value

    @staticmethod
    def _normalize_status(status: object) -> SequenceProgressStatus:
        raw = getattr(status, "value", status)
        if raw not in _SEQUENCE_STATE_RANK:
            allowed = ", ".join(_SEQUENCE_STATE_RANK)
            raise ValueError(f"Unknown sequence progress status {raw!r}; expected one of {allowed}.")
        return raw

    @staticmethod
    def _normalize_detail(detail: object) -> str | None:
        if detail is None:
            return None
        compact = " ".join(str(detail).split())
        if not compact:
            return None
        return compact if len(compact) <= 120 else f"{compact[:117]}..."

    @property
    def progress(self) -> Progress:
        """The embedded Rich progress object, exposed for composition and tests."""

        return self._progress

    @property
    def renderable(self) -> _SequenceProgressView:
        """The dynamic Rich renderable installed in the workflow detail panel."""

        return self._view

    @property
    def summary(self) -> Text:
        """Build the current aggregate sequence status line."""

        completed = sum(state.status == "completed" for state in self._states.values())
        failed = sum(state.status == "failed" for state in self._states.values())
        total = len(self._states)
        summary = Text("Tracking: ", style=ui.STYLE_TEXT_STRONG)
        summary.append(f"{completed}/{total} sequences done", style=ui.STYLE_STATUS_DONE)
        if failed:
            summary.append(f" · {failed} failed", style=ui.STYLE_STATUS_FAILED)
        return summary

    def __enter__(self) -> "EvalSequenceProgressPresenter":
        if self._active:
            raise RuntimeError("EvalSequenceProgressPresenter is already active.")
        self._detail_scope = self._callback._scoped_detail(self._view)
        self._detail_scope.__enter__()
        self._active = True
        self._last_refresh_s = self._clock()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        scope = self._detail_scope
        if scope is None:
            return
        self.flush()
        self._active = False
        self._detail_scope = None
        scope.__exit__(exc_type, exc, traceback)

    def __call__(self, event: _SequenceProgressEvent) -> None:
        """Consume a structural replay progress event."""

        self.update(
            event.sequence_id,
            event.completed,
            event.total,
            status=event.status,
            detail=event.detail,
        )

    def update(
        self,
        sequence_id: str,
        completed: int,
        total: int | None = None,
        *,
        status: SequenceProgressStatus | None = None,
        detail: str | None = None,
    ) -> None:
        """Update one row while preventing stale events from regressing it."""

        sequence_id = self._validate_sequence_id(sequence_id)
        try:
            state = self._states[sequence_id]
        except KeyError as exc:
            raise KeyError(f"Unknown evaluation sequence {sequence_id!r}.") from exc

        completed = self._validate_count(completed, name="completed")
        total = self._validate_count(total, name="total", optional=True)
        normalized_status = self._normalize_status(
            status
            if status is not None
            else ("completed" if state.total is not None and completed == state.total else "running")
        )
        # A worker that has not opened its sequence yet may report the
        # transport-level placeholder ``0/0`` when queued or when startup
        # fails. The catalog total supplied to this presenter remains
        # authoritative.
        if (
            normalized_status in {"queued", "failed"}
            and completed == 0
            and total == 0
            and state.total is not None
        ):
            total = None
        if state.total is not None and total is not None and total != state.total:
            raise ValueError(
                f"Sequence {sequence_id!r} total changed from {state.total} to {total}."
            )
        if state.total is None and total is not None:
            state.total = total
        if state.total is not None and completed > state.total:
            raise ValueError(
                f"Sequence {sequence_id!r} completed count {completed} exceeds total {state.total}."
            )

        if state.status in _TERMINAL_SEQUENCE_STATES and normalized_status != state.status:
            return
        if _SEQUENCE_STATE_RANK[normalized_status] < _SEQUENCE_STATE_RANK[state.status]:
            return

        state.status = normalized_status
        state.completed = max(state.completed, completed)
        if normalized_status == "completed" and state.total is not None:
            state.completed = state.total
        state.detail = self._normalize_detail(detail)

        if normalized_status == "running":
            self._progress.start_task(state.task_id)
        update_fields: dict[str, object] = {
            "completed": state.completed,
            "status": state.status,
            "detail": state.detail,
        }
        if state.total is not None:
            update_fields["total"] = state.total
        self._progress.update(state.task_id, **update_fields)
        if normalized_status in _TERMINAL_SEQUENCE_STATES:
            self._progress.stop_task(state.task_id)
        self._refresh()

    def complete(self, sequence_id: str, total: int | None = None) -> None:
        """Mark a sequence complete, filling its known progress bar."""

        sequence_id = self._validate_sequence_id(sequence_id)
        try:
            state = self._states[sequence_id]
        except KeyError as exc:
            raise KeyError(f"Unknown evaluation sequence {sequence_id!r}.") from exc
        resolved_total = state.total if total is None else total
        completed = state.completed if resolved_total is None else resolved_total
        self.update(sequence_id, completed, resolved_total, status="completed")

    def fail(self, sequence_id: str, detail: str | BaseException) -> None:
        """Mark a sequence failed without discarding its completed count."""

        sequence_id = self._validate_sequence_id(sequence_id)
        try:
            state = self._states[sequence_id]
        except KeyError as exc:
            raise KeyError(f"Unknown evaluation sequence {sequence_id!r}.") from exc
        self.update(
            sequence_id,
            state.completed,
            state.total,
            status="failed",
            detail=str(detail),
        )

    def flush(self) -> None:
        """Force the latest progress state into the parent workflow."""

        self._refresh(force=True)

    def _refresh(self, *, force: bool = False) -> None:
        if not self._active or not self._callback.render:
            return
        now = self._clock()
        if (
            force
            or self._refresh_interval_s is None
            or self._last_refresh_s is None
            or now - self._last_refresh_s >= self._refresh_interval_s
        ):
            self._callback.workflow._update_live(render=True, force=True)
            self._last_refresh_s = now


def _effective_eval_tracker_backend(args: argparse.Namespace) -> str | None:
    raw_tracker_backend = getattr(args, "tracker_backend", None)
    if raw_tracker_backend in {None, ""}:
        return None

    return str(raw_tracker_backend)


def _build_eval_workflow_fields(args: argparse.Namespace) -> list[tuple[str, object]]:
    """Build workflow fields as subsystem summary cards.

    Instead of dumping every tracker parameter, each subsystem gets a
    compact one-line summary with only the most relevant settings.
    """
    dataset_ref = getattr(args, "dataset", None)
    experiment_ref = getattr(args, "experiment", None)
    dataset = (
        dataset_ref
        or experiment_ref
        or getattr(args, "benchmark", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "experiment_id", None)
        or getattr(args, "source", None)
    )

    fields: list[tuple[str, object]] = []

    # ── Tracker card ──────────────────────────────────────────────
    tracker = getattr(args, "tracker", None)
    tracker_backend = _effective_eval_tracker_backend(args)
    cmc_method = getattr(args, "cmc_method", None)

    tracker_items: list[tuple[str, object]] = []
    if tracker:
        tracker_items.append(("Name", tracker))
    if tracker_backend:
        tracker_items.append(("Backend", tracker_backend))
    n_threads = getattr(args, "n_threads", None)
    if n_threads is not None:
        sequence_info = getattr(args, "seq_info", None)
        sequence_count = len(sequence_info) if isinstance(sequence_info, Mapping) else 0
        active_processes = min(int(n_threads), sequence_count) if sequence_count else int(n_threads)
        tracker_items.append(("Processes", active_processes))
    if cmc_method not in {None, "", "none"}:
        tracker_items.append(("CMC", cmc_method))
    # Key thresholds only
    det_thresh = getattr(args, "det_thresh", None)
    if det_thresh is not None:
        tracker_items.append(("Det thresh", f"{det_thresh:.2f}"))
    new_track = getattr(args, "new_track_thresh", None)
    if new_track is not None:
        tracker_items.append(("New track", f"{new_track:.2f}"))
    if tracker_items:
        fields.append(panel_field("Tracker", tracker_items))

    # ── Detector card ─────────────────────────────────────────────
    # Perception is represented by the immutable build selected below.

    # ── ReID card ─────────────────────────────────────────────────
    # Eval never constructs detector or ReID runtimes.

    # ── Dataset card ──────────────────────────────────────────────
    dataset_items: list[tuple[str, object]] = []
    if dataset:
        label = "Dataset" if dataset_ref else ("Experiment" if experiment_ref else "Benchmark")
        dataset_items.append((label, dataset))
    split = getattr(args, "split", None)
    if split:
        dataset_items.append(("Split", split))
    if dataset_items:
        fields.append(panel_field("Dataset", dataset_items))

    build = getattr(args, "build_path", None) or getattr(args, "build", None)
    if build:
        fields.append(panel_field("Build", [("Path/ID", build)]))

    return fields


def _refresh_eval_pipeline_intro(
    workflow: ui.WorkflowProgress | None,
    args: argparse.Namespace,
) -> None:
    if workflow is None:
        return

    updated_fields = _build_eval_workflow_fields(args)
    if hasattr(workflow, "set_fields"):
        workflow.set_fields(updated_fields)
        return

    if hasattr(workflow, "fields"):
        workflow.fields = updated_fields


class EvalWorkflowReporter(RichWorkflowReporter):
    title = "Evaluation"
    prefer_compact_layout = True
    SETUP = 0
    TRACK = 1
    EVALUATE = 2

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self.steps = eval_steps(postprocess=False)

    def fields(self) -> list[tuple[str, object]]:
        return _build_eval_workflow_fields(self.args)


def log_eval_pipeline_intro(args: argparse.Namespace) -> ui.WorkflowProgress:
    # Engine module is responsible for normalizing args before constructing
    # the reporter; keep this function side-effect free here so it can be
    # imported without pulling engine internals.
    return EvalWorkflowReporter(args).create()


__all__ = [
    "EVAL_SETUP_STEP",
    "EVAL_TRACK_STEP",
    "EVAL_EVALUATE_STEP",
    "EvalSequenceProgressPresenter",
    "EvalWorkflowReporter",
    "SequenceProgressStatus",
    "log_eval_pipeline_intro",
    "_refresh_eval_pipeline_intro",
]
