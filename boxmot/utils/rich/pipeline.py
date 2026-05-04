"""Unified pipeline state tracker for Rich workflow UIs.

Provides :class:`PipelineTracker` — a linear state machine that wraps
:class:`~boxmot.utils.rich.ui.WorkflowProgress` with a clean, index-based
API.  Each engine ``main()`` replaces ~15 lines of boilerplate with::

    with TrackPipeline(args) as pipeline:
        # ... setup happens under step 0 ...
        pipeline.advance("Running tracker...")
        result = run_track(args, pipeline=pipeline)
        pipeline.finish(result.renderable)

The tracker automatically wires download / build status callbacks to the
current step and tears them down on exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

from boxmot.utils.rich.reporting import (
    RichWorkflowReporter,
    WorkflowDetailCallback,
)
import boxmot.utils.rich.ui as ui


class PipelineTracker:
    """Linear state machine over a :class:`WorkflowProgress` step list.

    Parameters
    ----------
    workflow:
        An already-created :class:`WorkflowProgress` (from a reporter's
        ``.create()``).  The tracker reads the step labels from it.
    auto_start:
        When ``True`` (default), entering the context manager calls
        ``workflow.start()``.  Set to ``False`` for pipelines like *tune*
        that need to start the Live region manually at a later point.
    wire_status_fns:
        When ``True`` (default), ``__enter__`` registers the first step's
        :class:`WorkflowDetailCallback` as the global download / build
        status sink.  ``__exit__`` resets both to ``None``.
    """

    def __init__(
        self,
        workflow: ui.WorkflowProgress,
        *,
        auto_start: bool = True,
        wire_status_fns: bool = True,
    ) -> None:
        self._workflow = workflow
        self._step_labels: list[str] = [label for label, _ in workflow.steps]
        self._current_idx: int = 0
        self._auto_start = auto_start
        self._wire_status_fns = wire_status_fns

    # -- properties --------------------------------------------------------

    @property
    def workflow(self) -> ui.WorkflowProgress:
        """The underlying :class:`WorkflowProgress`."""
        return self._workflow

    @property
    def current_step(self) -> str:
        """Label of the currently-active step."""
        return self._step_labels[self._current_idx]

    @property
    def step_index(self) -> int:
        """Zero-based index of the current step."""
        return self._current_idx

    @property
    def total_steps(self) -> int:
        return len(self._step_labels)

    # -- step label access -------------------------------------------------

    def step(self, index: int) -> str:
        """Return the label of the step at *index*."""
        return self._step_labels[index]

    # -- callbacks ---------------------------------------------------------

    def callback(self, step: str | int | None = None, *, render: bool = True) -> WorkflowDetailCallback:
        """Return a :class:`WorkflowDetailCallback` for *step*.

        *step* can be a label string, a zero-based integer index, or
        ``None`` (defaults to the current step).
        """
        if step is None:
            label = self.current_step
        elif isinstance(step, int):
            label = self._step_labels[step]
        else:
            label = step
        return WorkflowDetailCallback(self._workflow, label, render=render)

    # -- step transitions --------------------------------------------------

    def _resolve_step(self, step: str | int | None) -> str:
        """Resolve a step reference to its label string."""
        if step is None:
            return self.current_step
        if isinstance(step, int):
            return self._step_labels[step]
        return step

    def update(self, detail: str, step: int | None = None) -> None:
        """Update the detail text for *step* (default: current)."""
        self._workflow.set_detail(self._resolve_step(step), detail)

    def complete_step(self, step: int | None = None, *, render: bool = False) -> None:
        """Mark *step* (default: current) as done."""
        self._workflow.complete(self._resolve_step(step), render=render)

    def set_detail_renderable(
        self,
        title: str,
        renderable: Any,
        *,
        step: int | None = None,
        render: bool = False,
    ) -> None:
        """Show *renderable* in the detail panel."""
        self._workflow.set_detail_renderable(title, renderable, render=render)

    def refresh_fields(self, fields: list[tuple[str, object]]) -> None:
        """Update the header fields displayed in the workflow panel."""
        wf = self._workflow
        if hasattr(wf, "set_fields"):
            wf.set_fields(fields)
        elif hasattr(wf, "fields"):
            wf.fields = fields

    def advance(self, detail: str | None = None) -> None:
        """Complete the current step and activate the next one.

        Does nothing if already on the last step.
        """
        if self._current_idx >= len(self._step_labels) - 1:
            return
        done_label = self._step_labels[self._current_idx]
        self._current_idx += 1
        next_label = self._step_labels[self._current_idx]
        self._workflow.transition(done_label, next_label, detail)

        # Keep the global status callbacks pointing at the new step so that
        # downloads / builds triggered after an ``advance()`` land in the
        # correct detail panel.
        if self._wire_status_fns:
            _set_status_fns(self.callback())

    def finish(
        self,
        renderable: Any | None = None,
        *,
        title: str = "Results",
    ) -> None:
        """Mark the final step done and (optionally) show *renderable*."""
        self._workflow.complete(self.current_step, render=False)
        if renderable is not None:
            self._workflow.set_detail_renderable(title, renderable, render=False)

    def start(self) -> None:
        """Explicitly start the Live region (for deferred-start pipelines)."""
        self._workflow.start()

    def stop(self) -> None:
        """Explicitly stop the Live region."""
        self._workflow.stop()

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> PipelineTracker:
        if self._wire_status_fns:
            _set_status_fns(self.callback())
        if self._auto_start:
            self._workflow.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_val is not None:
            self._workflow.fail(error=exc_val)
            # Signal to the CLI wrapper that the panel already rendered the
            # error so it can suppress the duplicate traceback.
            exc_val._workflow_rendered_error = True  # type: ignore[union-attr]
        self._workflow.stop()
        if self._wire_status_fns:
            _set_status_fns(None)


def create_pipeline(
    reporter: RichWorkflowReporter,
    *,
    auto_start: bool = True,
    wire_status_fns: bool = True,
) -> PipelineTracker:
    """Convenience: build a :class:`PipelineTracker` from a reporter."""
    workflow = reporter.create()
    return PipelineTracker(
        workflow,
        auto_start=auto_start,
        wire_status_fns=wire_status_fns,
    )


# -- internal helpers ------------------------------------------------------

def _set_status_fns(callback: WorkflowDetailCallback | None) -> None:
    """Wire (or clear) the global download / build status sinks."""
    from boxmot.utils.download import set_download_status_fn
    from boxmot.native._common import set_build_status_fn

    set_download_status_fn(callback)
    set_build_status_fn(callback)
