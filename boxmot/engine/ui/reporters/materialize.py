"""Rich workflow reporter for immutable dataset materialization."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from threading import Event, Thread
from typing import TYPE_CHECKING, Any

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.ui.workflow.fields import panel_field
from boxmot.engine.ui.workflow.reporting import RichWorkflowReporter
from boxmot.engine.ui.workflow.steps import MATERIALIZE, MATERIALIZE_STEPS, SETUP

if TYPE_CHECKING:
    from boxmot.engine.ui.workflow.pipeline import PipelineTracker


def _source_count(plan: Any) -> int | None:
    metadata = getattr(plan, "metadata", {})
    value = metadata.get("source_count") if isinstance(metadata, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _total_shards(plan: Any, stage: Any) -> int | None:
    name = str(getattr(stage, "name", ""))
    if name == "finalize":
        return 1
    if name not in {"detect", "segment", "embed"}:
        return None
    count = _source_count(plan)
    batch_size = getattr(stage, "batch_size", None)
    if count is None or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        return None
    return (count + batch_size - 1) // batch_size


def _error_summary(error: BaseException) -> str:
    detail = str(error).strip()
    summary = type(error).__name__ if not detail else f"{type(error).__name__}: {detail}"
    return summary if len(summary) <= 120 else f"{summary[:117]}..."


def _stage_label(name: str) -> str:
    return name.replace("_", " ").title()


def _duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, whole_seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {whole_seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _description(name: str, state: str, suffix: str = "") -> str:
    markers = {
        "todo": ("○", ui.STYLE_STATUS_TODO),
        "active": ("▶", ui.STYLE_STATUS_ACTIVE),
        "done": ("✓", ui.STYLE_STATUS_DONE),
        "failed": ("✕", ui.STYLE_STATUS_FAILED),
    }
    marker, style = markers[state]
    trailing = "" if not suffix else f"  [boxmot.text.muted]{suffix}[/]"
    return f"[{style}]{marker}[/] {_stage_label(name)}{trailing}"


def _initial_fields(args: Any) -> list[tuple[str, object]]:
    input_items: list[tuple[str, object]] = []
    experiment = getattr(args, "experiment", None)
    if experiment:
        input_items.append(("Experiment", experiment))

    fields: list[tuple[str, object]] = []
    if input_items:
        fields.append(panel_field("Input", input_items))
    build_root = getattr(args, "build_root", None)
    if build_root:
        fields.append(panel_field("Storage", [("Build root", build_root)]))
    explicit_keys = set(getattr(args, "materialize_explicit_keys", ()) or ())
    if "device" in explicit_keys:
        fields.append(panel_field("Execution", [("Device override", getattr(args, "device", "unknown"))]))
    fields.append(
        panel_field(
            "Publish",
            [
                ("Image refs", bool(getattr(args, "publish_image_refs", True))),
                ("Masks", bool(getattr(args, "publish_masks", False))),
                ("Embeddings", bool(getattr(args, "publish_embeddings", True))),
            ],
        )
    )
    return fields


def _component_summary(component: object) -> str | None:
    if not isinstance(component, Mapping):
        return None
    spec = component.get("spec")
    if not isinstance(spec, Mapping):
        return None
    backend = spec.get("backend")
    if not isinstance(backend, str) or not backend:
        return None
    device = spec.get("device")
    if isinstance(device, str) and device:
        return f"{backend} · {device}"
    return backend


def _build_fields(plan: Any) -> list[tuple[str, object]]:
    metadata = getattr(plan, "metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    count = _source_count(plan)
    input_items: list[tuple[str, object]] = [
        ("Dataset", getattr(plan, "dataset_name", "unknown")),
        ("Samples", "unknown" if count is None else f"{count:,}"),
        ("Geometry", getattr(plan, "box_type", "unknown")),
    ]
    split = metadata.get("split")
    if split:
        input_items.insert(1, ("Split", split))

    fields: list[tuple[str, object]] = [panel_field("Input", input_items)]
    components = metadata.get("components")
    if isinstance(components, Mapping):
        component_items = [
            (name.title(), backend)
            for name in ("detector", "segmentor", "reid")
            if (backend := _component_summary(components.get(name))) is not None
        ]
        if component_items:
            fields.append(panel_field("Components", component_items))

    publish = getattr(plan, "publish", None)
    if publish is not None:
        fields.append(
            panel_field(
                "Publish",
                [
                    ("Image refs", bool(getattr(publish, "image_references", False))),
                    ("Masks", bool(getattr(publish, "masks", False))),
                    ("Embeddings", bool(getattr(publish, "embeddings", False))),
                ],
            )
        )
    fields.append(
        panel_field(
            "Build",
            [
                ("ID", getattr(plan, "build_id", "unknown")),
                ("Output", getattr(plan, "output_root", "unknown")),
                ("Staging", getattr(plan, "staging_root", "unknown")),
            ],
        )
    )
    return fields


class MaterializeWorkflowReporter(RichWorkflowReporter):
    """Host materialization stages in one non-scrolling Rich Live panel."""

    title = "Dataset Materialization"
    steps = MATERIALIZE_STEPS
    start_on_create = False
    prefer_compact_layout = True

    def __init__(
        self,
        args: Any,
        *,
        workflow: ui.WorkflowProgress | None = None,
        clock: Callable[[], float] = time.monotonic,
        refresh_interval_s: float | None = 1.0,
    ) -> None:
        super().__init__(args)
        if refresh_interval_s is not None and refresh_interval_s <= 0:
            raise ValueError("refresh_interval_s must be positive or None.")
        self._workflow = workflow
        self._clock = clock
        self._refresh_interval_s = refresh_interval_s
        self._refresh_stop = Event()
        self._refresh_thread: Thread | None = None
        self._display_started = False
        self._build_started = False
        self._plan: Any | None = None
        self._pipeline: PipelineTracker | None = None
        self._progress: Progress | None = None
        self._task_ids: dict[str, int] = {}
        self._totals: dict[str, int | None] = {}
        self._started_s: dict[str, float] = {}
        self._initial_completed: dict[str, int] = {}

    @property
    def workflow(self) -> ui.WorkflowProgress | None:
        """Expose the retained workflow for engine integration tests."""

        return self._workflow

    @property
    def stage_progress(self) -> Progress | None:
        """Expose the embedded progress renderable for deterministic tests."""

        return self._progress

    def fields(self) -> Sequence[tuple[str, object]]:
        return _initial_fields(self.args)

    def start(self) -> None:
        if self._display_started:
            return
        if self._workflow is None:
            self._workflow = self.create()
        from boxmot.engine.ui.workflow.pipeline import PipelineTracker

        self._pipeline = PipelineTracker(self._workflow)
        self._pipeline.__enter__()
        self._workflow.set_detail(SETUP, "Cataloging sources and resolving component artifacts…")
        self._display_started = True

    def stop(self) -> None:
        self._stop_refresher()
        if self._pipeline is not None and self._display_started:
            self._pipeline.__exit__(None, None, None)
        elif self._workflow is not None and self._display_started:
            self._workflow.stop()
        self._pipeline = None
        self._display_started = False

    def setup_status(self, message: str) -> None:
        if not self._display_started:
            self.start()
        assert self._workflow is not None
        self._workflow.set_detail(SETUP, message)

    def unhandled_failure(self, error: BaseException) -> None:
        if not self._display_started:
            self.start()
        assert self._workflow is not None
        target = MATERIALIZE if self._build_started else SETUP
        self._workflow.fail(target, error)
        try:
            error._workflow_rendered_error = True  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

    def build_started(self, plan: Any) -> None:
        if not self._display_started:
            self.start()
        assert self._workflow is not None
        self._build_started = True
        self._plan = plan
        self._workflow.set_fields(_build_fields(plan))
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[summary]}", style=ui.STYLE_MUTED),
            expand=True,
            auto_refresh=False,
        )
        self._task_ids.clear()
        self._totals.clear()
        for stage in getattr(plan, "ordered_stages")():
            name = str(stage.name)
            total = _total_shards(plan, stage)
            task_id = self._progress.add_task(
                _description(name, "todo"),
                total=total,
                completed=0,
                start=False,
                summary="queued",
            )
            self._task_ids[name] = task_id
            self._totals[name] = total
        if self._pipeline is not None:
            self._pipeline.advance()
            self._pipeline.set_detail_renderable("Build progress", self._progress)
        else:
            self._workflow.transition(SETUP, MATERIALIZE)
            self._workflow.set_detail_renderable("Build progress", self._progress)
        self._start_refresher()

    def build_reused(self, output_root: Path) -> None:
        for name in self._task_ids:
            self._mark_done(name, "validated")
        self._finish(output_root, reused=True)

    def build_lock_waiting(self, lock_path: Path) -> None:
        if not self._task_ids:
            return
        first_stage = next(iter(self._task_ids))
        self._update_task(first_stage, summary=f"waiting for build lock • {lock_path.name}")

    def build_lock_acquired(self) -> None:
        if not self._task_ids:
            return
        first_stage = next(iter(self._task_ids))
        self._update_task(first_stage, summary="queued • build lock acquired")

    def build_completed(self, output_root: Path) -> None:
        self._finish(output_root, reused=False)

    def stage_started(self, plan: Any, stage: Any, *, completed_shards: int) -> None:
        del plan
        name = str(stage.name)
        self._started_s[name] = self._clock()
        self._initial_completed[name] = completed_shards
        total = self._totals.get(name)
        pending = "unknown" if total is None else str(max(total - completed_shards, 0))
        self._update_task(
            name,
            description=_description(name, "active"),
            completed=completed_shards,
            summary=(
                f"{completed_shards}/{total if total is not None else '?'} shards • {pending} pending "
                f"• batch {stage.batch_size} • {stage.workers} worker(s)"
            ),
            start=True,
        )

    def shard_completed(
        self,
        stage_name: str,
        shard_id: str,
        *,
        completed_shards: int,
        items: int | None = None,
        rows: int | None = None,
    ) -> None:
        del shard_id
        now = self._clock()
        started = self._started_s.get(stage_name, now)
        elapsed = max(now - started, 0.0)
        processed = max(completed_shards - self._initial_completed.get(stage_name, 0), 1)
        rate = processed / elapsed if elapsed > 0 else None
        total = self._totals.get(stage_name)
        counts = []
        if items is not None:
            counts.append(f"{items} items")
        if rows is not None:
            counts.append(f"{rows} rows")
        timing = _duration(elapsed)
        if rate is not None:
            timing += f" • {rate:.2f} shards/s"
            if total is not None:
                timing += f" • ETA {_duration(max(total - completed_shards, 0) / rate)}"
        summary = f"{completed_shards}/{total if total is not None else '?'} shards"
        if counts:
            summary += f" • {' • '.join(counts)}"
        summary += f" • {timing}"
        self._update_task(stage_name, completed=completed_shards, summary=summary)

    def stage_completed(self, stage: Any, outcome: Any, *, completed_shards: int) -> None:
        name = str(stage.name)
        now = self._clock()
        elapsed = max(now - self._started_s.get(name, now), 0.0)
        metrics = getattr(outcome, "metrics", {})
        counters = []
        if isinstance(metrics, Mapping):
            for key in sorted(metrics):
                value = metrics[key]
                if value is not None and isinstance(value, str | int | float | bool):
                    counters.append(f"{key}={value}")
        summary = f"complete in {_duration(elapsed)}"
        if counters:
            summary += f" • {' • '.join(counters)}"
        self._mark_done(name, summary, completed_shards=completed_shards)

    def stage_skipped(self, stage_name: str) -> None:
        self._mark_done(stage_name, "resumed")

    def stage_retry(
        self,
        stage_name: str,
        error: Exception,
        *,
        attempt: int,
        max_attempts: int,
        delay_s: float,
    ) -> None:
        self._update_task(
            stage_name,
            description=_description(stage_name, "active", f"retry {attempt + 1}/{max_attempts}"),
            summary=f"{_error_summary(error)} • retry in {delay_s:.1f}s",
        )

    def stage_failed(
        self,
        stage_name: str,
        error: BaseException,
        *,
        attempt: int,
        max_attempts: int,
        staging_root: Path,
    ) -> None:
        try:
            error.add_note(
                f"Materialization stage {stage_name!r} failed on attempt {attempt}/{max_attempts}. "
                f"Resumable state remains at {staging_root}."
            )
        except (AttributeError, TypeError):
            pass
        self._update_task(
            stage_name,
            description=_description(stage_name, "failed"),
            summary=_error_summary(error),
            stop=True,
        )
        assert self._workflow is not None
        self._workflow.fail(
            MATERIALIZE,
            f"{_error_summary(error)}\nResume data: {staging_root}",
        )

    def _mark_done(self, name: str, summary: str, *, completed_shards: int | None = None) -> None:
        total = self._totals.get(name)
        if total is None:
            completed = completed_shards
        else:
            completed = total
        self._update_task(
            name,
            description=_description(name, "done"),
            completed=completed,
            summary=summary,
            stop=True,
        )

    def _update_task(self, name: str, *, start: bool = False, stop: bool = False, **fields: Any) -> None:
        if self._progress is None or name not in self._task_ids:
            return
        task_id = self._task_ids[name]
        if start:
            self._progress.start_task(task_id)
        self._progress.update(task_id, **fields)
        if stop:
            self._progress.stop_task(task_id)
        if self._workflow is not None:
            self._workflow._update_live(render=True, force=True)

    def _start_refresher(self) -> None:
        if self._refresh_interval_s is None or self._refresh_thread is not None:
            return
        self._refresh_stop.clear()

        def refresh() -> None:
            assert self._refresh_interval_s is not None
            while not self._refresh_stop.wait(self._refresh_interval_s):
                workflow = self._workflow
                if workflow is not None:
                    try:
                        workflow._update_live(render=True, force=True)
                    except Exception:
                        self._refresh_stop.set()
                        return

        self._refresh_thread = Thread(
            target=refresh,
            name="boxmot-materialize-ui",
            daemon=True,
        )
        self._refresh_thread.start()

    def _stop_refresher(self) -> None:
        thread = self._refresh_thread
        if thread is None:
            return
        self._refresh_stop.set()
        thread.join(timeout=max(float(self._refresh_interval_s or 0.0) * 2.0, 0.1))
        self._refresh_thread = None

    def _finish(self, output_root: Path, *, reused: bool) -> None:
        if self._workflow is None:
            return
        self._stop_refresher()
        build_id = getattr(self._plan, "build_id", output_root.name)
        action = "Validated and reused" if reused else "Published"
        message = Text()
        message.append(f"{action}: ", style=ui.STYLE_STATUS_DONE)
        message.append(str(output_root), style=ui.STYLE_TEXT_STRONG)
        message.append(f"\nBuild ID: {build_id}", style=ui.STYLE_MUTED)
        if self._pipeline is not None:
            self._pipeline.finish(message, title="Build complete", include_steps=False)
        else:
            self._workflow.complete(MATERIALIZE, render=False)
            self._workflow.set_detail_renderable("Build complete", message)


__all__ = ("MATERIALIZE", "MaterializeWorkflowReporter")
