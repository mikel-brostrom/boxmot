from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, ClassVar, Iterator, Sequence

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

import boxmot.utils.rich.ui as ui

# ---- shared helpers reused by per-command reporters ----------------------

_PARAM_LABEL_REPLACEMENTS = {
    "Id": "ID",
    "Ids": "IDs",
    "Idsw": "IDSW",
    "Reid": "ReID",
    "Cmc": "CMC",
    "Fps": "FPS",
    "Imgsz": "Image Size",
}


def format_param_label(name: str) -> str:
    """Human-readable label for a tracker/pipeline parameter name."""
    label = str(name).replace("_", " ").title()
    for source, target in _PARAM_LABEL_REPLACEMENTS.items():
        label = label.replace(source, target)
    return label


def primary_model_ref(value: Any) -> Any:
    """Extract the first element from a list/tuple model spec, or pass through."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _make_progress(*, unit: str | None = None) -> Progress:
    """Build a Rich Progress bar with standard columns for workflow panels."""
    columns: list[Any] = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ]
    if unit:
        columns.append(TextColumn(f"• {{task.completed}}/{{task.total}} {unit}"))
    else:
        columns.append(TextColumn("• {task.completed}/{task.total}"))
    columns.append(TimeRemainingColumn())
    return Progress(*columns, transient=True, expand=True)


class RichWorkflowReporter:
    """Base helper for commands that render a Rich workflow panel."""

    title: ClassVar[str]
    steps: ClassVar[Sequence[tuple[str, ui.StepState]]] = ()
    stderr: ClassVar[bool] = True
    transient: ClassVar[bool] = False
    start_on_create: ClassVar[bool] = True
    prefer_alt_screen: ClassVar[bool] = False
    prefer_compact_layout: ClassVar[bool] = False

    def __init__(self, args: Any) -> None:
        self.args = args

    def fields(self) -> Sequence[tuple[str, object]]:
        return ()

    def create(self) -> ui.WorkflowProgress:
        workflow = ui.create_workflow_progress(
            self.title,
            self.fields(),
            steps=self.steps,
            stderr=self.stderr,
            transient=self.transient,
        )
        workflow.prefer_alt_screen = self.prefer_alt_screen
        workflow.prefer_compact_layout = self.prefer_compact_layout
        if self.start_on_create:
            workflow.start()
        return workflow

    def pipeline(self, **kwargs):
        """Create a :class:`PipelineTracker` for this reporter."""
        from boxmot.utils.rich.pipeline import PipelineTracker

        return PipelineTracker(self.create(), **kwargs)


class WorkflowDetailCallback:
    """Callable adapter that routes progress text into one workflow step."""

    def __init__(self, workflow: ui.WorkflowProgress, step: str, *, render: bool = True) -> None:
        self.workflow = workflow
        self.step = step
        self.render = render

    def __call__(self, message: str) -> None:
        self.workflow.set_detail(self.step, message, render=self.render)

    def _save_detail(self) -> tuple[str | None, str | None, Any]:
        return (
            self.workflow.detail_title,
            self.workflow.detail_text,
            self.workflow.detail_renderable,
        )

    def _restore_detail(
        self,
        saved: tuple[str | None, str | None, Any],
    ) -> None:
        title, text, renderable = saved
        if renderable is not None:
            self.workflow.set_detail_renderable(title, renderable, render=self.render)
        elif text is not None:
            self.workflow.set_detail(title, text, render=self.render)
        else:
            self.workflow.clear_detail(render=self.render)

    @contextmanager
    def _scoped_detail(self, renderable: Any) -> Iterator[None]:
        """Save current detail, show *renderable*, restore on exit."""
        saved = self._save_detail()
        self.workflow.set_detail_renderable(self.step, renderable, render=self.render)
        try:
            yield
        finally:
            self._restore_detail(saved)

    @contextmanager
    def bar(
        self,
        description: str,
        total: int | float | None,
        *,
        unit: str | None = None,
    ) -> Iterator[Callable[[int], None]]:
        """Render a Rich progress bar inside the workflow's detail panel.

        Yields an ``advance(n)`` callable. The bar is automatically removed
        from the panel on exit, regardless of success or exception.

        Use this in place of ``tqdm`` whenever a workflow is active so the
        bar is hosted by the same Rich Live region — preventing duplicated
        panel renders that result from tqdm's carriage-return updates
        racing with Rich's repaints.
        """
        progress = _make_progress(unit=unit)
        task_id = progress.add_task(description, total=total)

        def _advance(n: int = 1) -> None:
            progress.update(task_id, advance=n)
            if self.render:
                self.workflow._update_live(render=True, force=True)

        with self._scoped_detail(progress):
            yield _advance

    @contextmanager
    def tqdm_proxy(self, description: str, *, unit: str | None = None) -> Iterator[type]:
        """Yield a ``tqdm.tqdm``-compatible class that renders into this step.

        Useful for monkey-patching third-party libraries (e.g. ``gdown``) so
        their progress output is hosted by the workflow's Rich Live region
        instead of leaking raw carriage-return text to the terminal.
        """
        progress = _make_progress(unit=unit)
        callback = self

        class _RichTqdm:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                total = kwargs.get("total")
                initial = kwargs.get("initial", 0) or 0
                desc = kwargs.get("desc") or description
                self._task_id = progress.add_task(
                    str(desc), total=int(total) if total else None, completed=int(initial)
                )

            def update(self, n: int = 1) -> None:
                progress.update(self._task_id, advance=int(n))
                if callback.render:
                    callback.workflow._update_live(render=True, force=True)

            def set_description(self, desc: str, refresh: bool = True) -> None:
                progress.update(self._task_id, description=str(desc))

            def set_postfix(self, *args: Any, **kwargs: Any) -> None:
                pass

            def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
                pass

            def refresh(self) -> None:
                if callback.render:
                    callback.workflow._update_live(render=True, force=True)

            def close(self) -> None:
                pass

            def __enter__(self) -> "_RichTqdm":
                return self

            def __exit__(self, *exc: Any) -> None:
                self.close()

        with self._scoped_detail(progress):
            yield _RichTqdm

    @contextmanager
    def parallel_bars(
        self,
        descriptions: Sequence[str],
        *,
        unit: str | None = "B",
    ) -> Iterator[list["_ParallelTaskCallback"]]:
        """Render N progress bars stacked in this step's detail panel."""
        progress = _make_progress(unit=unit)

        def _render() -> None:
            if self.render:
                self.workflow._update_live(render=True, force=True)

        callbacks: list[_ParallelTaskCallback] = []
        for desc in descriptions:
            task_id = progress.add_task(str(desc), total=None)
            callbacks.append(_ParallelTaskCallback(progress, task_id, _render))

        with self._scoped_detail(progress):
            yield callbacks


class _ParallelTaskCallback:
    """Per-task callback used by :meth:`WorkflowDetailCallback.parallel_bars`.

    Mimics the small subset of :class:`WorkflowDetailCallback` used by
    download helpers: ``__call__(message)`` (currently a no-op so the panel
    layout stays stable) and ``bar()`` (yields an ``advance`` function that
    updates this callback's task in the shared ``Progress``).
    """

    def __init__(self, progress: Progress, task_id: int, render: Callable[[], None]) -> None:
        self._progress = progress
        self._task_id = task_id
        self._render = render

    def __call__(self, message: str) -> None:  # noqa: D401 - matches WorkflowDetailCallback
        return None

    @contextmanager
    def bar(
        self,
        description: str,
        total: int | float | None,
        *,
        unit: str | None = None,
    ) -> Iterator[Callable[[int], None]]:
        if total is not None:
            self._progress.update(self._task_id, total=int(total))
        if description:
            self._progress.update(self._task_id, description=str(description))

        def _advance(n: int = 1) -> None:
            self._progress.update(self._task_id, advance=int(n))
            self._render()

        try:
            yield _advance
        finally:
            self._render()

    @contextmanager
    def tqdm_proxy(self, description: str, *, unit: str | None = None) -> Iterator[type]:
        """Yield a ``tqdm.tqdm``-compatible class advancing this callback's task.

        Mirrors :meth:`WorkflowDetailCallback.tqdm_proxy` but updates the
        task already allocated for this callback within the shared
        ``parallel_bars`` panel, instead of replacing the panel renderable.
        Required so monkey-patched libraries (e.g. ``gdown``) participate in
        a parallel-download panel rather than printing to stderr.
        """
        if description:
            self._progress.update(self._task_id, description=str(description))

        progress = self._progress
        task_id = self._task_id
        render = self._render

        class _RichTqdm:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                total = kwargs.get("total")
                initial = kwargs.get("initial", 0) or 0
                desc = kwargs.get("desc")
                if total is not None:
                    progress.update(task_id, total=int(total))
                if initial:
                    progress.update(task_id, completed=int(initial))
                if desc:
                    progress.update(task_id, description=str(desc))

            def update(self, n: int = 1) -> None:
                progress.update(task_id, advance=int(n))
                render()

            def set_description(self, desc: str, refresh: bool = True) -> None:
                progress.update(task_id, description=str(desc))

            def set_postfix(self, *args: Any, **kwargs: Any) -> None:
                pass

            def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
                pass

            def refresh(self) -> None:
                render()

            def close(self) -> None:
                pass

            def __enter__(self) -> "_RichTqdm":
                return self

            def __exit__(self, *exc: Any) -> None:
                self.close()

        try:
            yield _RichTqdm
        finally:
            render()


class SilentProgressReporter:
    """No-op progress reporter used when another UI owns the terminal."""

    def setup(self, *args: Any, **kwargs: Any) -> None:
        return None

    def should_report(self, trials: Any, done: bool = False) -> bool:
        return False

    def report(self, trials: Any, done: bool, *sys_info: Any) -> None:
        return None


class RichWorkflowCallback:
    """Serializable callback base that keeps Rich workflow state driver-local."""

    detail_step: ClassVar[str | None] = None
    _workflow: ClassVar[ui.WorkflowProgress | None] = None

    @classmethod
    def set_workflow(cls, workflow: ui.WorkflowProgress | None) -> None:
        cls._workflow = workflow

    def set_workflow_detail(self, detail: str | None) -> None:
        workflow = type(self)._workflow
        if workflow is not None:
            workflow.set_detail(type(self).detail_step, detail)

    def setup(self, **info: Any) -> None:
        return None

    def on_step_begin(self, iteration: int, trials: list, **info: Any) -> None:
        return None

    def on_step_end(self, iteration: int, trials: list, **info: Any) -> None:
        return None

    def on_trial_start(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_restore(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_save(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_result(self, iteration: int, trials: list, trial: Any, result: dict, **info: Any) -> None:
        return None

    def on_trial_complete(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_error(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_trial_recover(self, iteration: int, trials: list, trial: Any, **info: Any) -> None:
        return None

    def on_checkpoint(self, iteration: int, trials: list, trial: Any, checkpoint: Any, **info: Any) -> None:
        return None

    def on_experiment_end(self, trials: list, **info: Any) -> None:
        return None

    def get_state(self) -> None:
        return None

    def set_state(self, state: dict[str, Any] | None) -> None:
        return None
