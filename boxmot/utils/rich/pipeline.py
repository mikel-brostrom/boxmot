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

import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text
from rich.tree import Tree

import boxmot.utils.rich.ui as ui
from boxmot.utils.rich.reporting import WorkflowDetailCallback


def _slugify(label: str) -> str:
    """Turn a step label into a filesystem-safe slug."""
    return label.lower().replace(" ", "_").replace("/", "_")


@dataclass
class StepRecord:
    """Per-step snapshot captured when a pipeline step completes."""

    label: str
    started: float = 0.0
    ended: float = 0.0
    detail_text: str | None = None
    detail_renderable: RenderableType | None = None
    detail_file: Path | None = None

    @property
    def elapsed(self) -> float:
        if self.ended and self.started:
            return self.ended - self.started
        return 0.0

    @property
    def elapsed_str(self) -> str:
        secs = self.elapsed
        if secs < 0.01:
            return ""
        if secs < 60:
            return f"{secs:.1f}s"
        mins = int(secs // 60)
        return f"{mins}m {secs - mins * 60:.0f}s"


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
        self._step_records: dict[str, StepRecord] = {}
        self._step_start: float = _time.monotonic()
        self._interactive_on_exit: bool = False

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

    # -- per-step recording ------------------------------------------------

    def _capture_step(self, label: str) -> StepRecord:
        """Snapshot the current detail into a :class:`StepRecord`."""
        now = _time.monotonic()
        record = self._step_records.get(label)
        if record is None:
            record = StepRecord(label=label, started=self._step_start)
        record.ended = now
        # Grab whatever the workflow currently shows as detail.
        wf = self._workflow
        if record.detail_renderable is None:
            record.detail_renderable = wf.detail_renderable
        if record.detail_text is None:
            record.detail_text = wf.detail_text
        self._step_records[label] = record
        return record

    def store_step_info(
        self,
        renderable: RenderableType | None = None,
        *,
        text: str | None = None,
        step: int | None = None,
    ) -> None:
        """Explicitly store a result renderable for a step.

        This is the primary API for engine code to attach per-step output
        (tables, summaries, timings) that will appear in the final pipeline
        recap panel.
        """
        label = self._resolve_step(step)
        record = self._step_records.get(label)
        if record is None:
            record = StepRecord(label=label)
            self._step_records[label] = record
        if renderable is not None:
            record.detail_renderable = renderable
        if text is not None:
            record.detail_text = text

    @property
    def step_records(self) -> dict[str, StepRecord]:
        """Read-only access to all captured step records."""
        return dict(self._step_records)

    def build_step_summary(self, *, exp_dir: Path | None = None) -> RenderableType:
        """Build a compact Rich Tree of pipeline steps.

        Each completed step with stored detail becomes a clickable
        ``file://`` hyperlink in terminals that support OSC 8 (VS Code,
        iTerm2, etc.).  The detail is written to a ``.txt`` file under
        *exp_dir* so clicking the step name opens the full output.
        """
        # Write per-step detail files when an output directory is available.
        steps_dir: Path | None = None
        if exp_dir is not None:
            steps_dir = Path(exp_dir) / "pipeline_steps"
            steps_dir.mkdir(parents=True, exist_ok=True)
            for label, record in self._step_records.items():
                content = record.detail_text or ""
                if record.detail_renderable is not None:
                    content = ui.capture_renderable(record.detail_renderable)
                if not content:
                    continue
                slug = _slugify(label)
                detail_path = steps_dir / f"{slug}.txt"
                detail_path.write_text(content, encoding="utf-8")
                record.detail_file = detail_path

        tree = Tree(
            Text("Pipeline Steps", style=ui.STYLE_TITLE),
            guide_style=ui.STYLE_MUTED,
        )
        for label in self._step_labels:
            record = self._step_records.get(label)
            if record is None:
                node_text = Text()
                node_text.append("[ ] ", style=ui.STYLE_STATUS_TODO)
                node_text.append(label, style=ui.STYLE_MUTED)
                tree.add(node_text)
                continue

            # Build step header: [✓] Label  3.2s
            node_text = Text()
            node_text.append(f"{ui.STEP_DONE_MARKER} ", style=ui.STYLE_STATUS_DONE)

            # Make label a clickable link if a detail file was written.
            if record.detail_file is not None:
                file_uri = record.detail_file.resolve().as_uri()
                node_text.append(
                    label,
                    style=f"bold link {file_uri}",
                )
                node_text.append("  ↗", style=ui.STYLE_MUTED)
            else:
                node_text.append(label, style=ui.STYLE_TEXT_STRONG)

            elapsed = record.elapsed_str
            if elapsed:
                node_text.append(f"  {elapsed}", style=ui.STYLE_MUTED)

            tree.add(node_text)

        return tree

    # -- step transitions --------------------------------------------------

    def complete_step(self, step: int | None = None, *, render: bool = False) -> None:
        """Mark *step* (default: current) as done and capture its record."""
        label = self._resolve_step(step)
        self._capture_step(label)
        self._workflow.complete(label, render=render)

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

        Captures a :class:`StepRecord` for the finishing step before
        transitioning.  Does nothing if already on the last step.
        """
        if self._current_idx >= len(self._step_labels) - 1:
            return
        done_label = self._step_labels[self._current_idx]
        self._capture_step(done_label)
        self._current_idx += 1
        self._step_start = _time.monotonic()
        next_label = self._step_labels[self._current_idx]
        self._workflow.transition(done_label, next_label, detail)

        # Keep the global status callbacks pointing at the new step so that
        # downloads / builds triggered after an ``advance()`` land in the
        # correct detail panel.
        if self._wire_status_fns:
            _set_status_fns(self.callback())

    def show_interactive(self) -> None:
        """Launch a live interactive TUI to browse step details.

        Arrow keys navigate, Enter/Space expands or collapses a step,
        ``q`` exits.  Only activates in a real terminal (``isatty``).
        """
        from boxmot.utils.rich.interactive import interactive_step_viewer

        interactive_step_viewer(
            labels=self._step_labels,
            records=self._step_records,
            workflow=self._workflow,
        )

    def finish(
        self,
        renderable: Any | None = None,
        *,
        title: str = "Results",
        include_steps: bool = True,
        exp_dir: Path | str | None = None,
        interactive: bool = False,
    ) -> None:
        """Mark the final step done and (optionally) show *renderable*.

        When *include_steps* is ``True`` (default), the final detail panel
        includes a compact tree of all pipeline steps with per-step timing.
        If *exp_dir* is given, each step's detail is written to a file
        under ``exp_dir/pipeline_steps/`` and the step label becomes a
        clickable ``file://`` link in the terminal.

        When *interactive* is ``True``, an interactive TUI is launched
        after the workflow panel stops, letting the user browse step
        details with arrow keys and Enter.
        """
        self._capture_step(self.current_step)
        self._workflow.complete(self.current_step, render=False)

        resolved_dir = Path(exp_dir) if exp_dir is not None else None
        parts: list[RenderableType] = []
        # Write per-step detail files for clickable links, but don't add
        # the tree to the panel — the pipeline bar already shows all steps.
        if include_steps and self._step_records:
            self.build_step_summary(exp_dir=resolved_dir)
        if renderable is not None:
            parts.append(renderable)

        if parts:
            combined = Group(*parts) if len(parts) > 1 else parts[0]
            self._workflow.set_detail_renderable(title, combined, render=False)

        self._interactive_on_exit = interactive

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

        want_interactive = (
            exc_val is None
            and self._interactive_on_exit
            and self._step_records
        )

        # Always use transient live + clean reprint for the final panel
        if self._workflow._live is not None:
            self._workflow._live.transient = True

        self._workflow.stop()
        if self._wire_status_fns:
            _set_status_fns(None)

        if exc_val is not None:
            # On crash, print the full untruncated panel so no traceback
            # lines are hidden by the live-view terminal height limit.
            ui.print_renderable(
                self._workflow.renderable(compact=False, include_setup=False),
                stderr=self._workflow.stderr,
            )
        else:
            if want_interactive:
                self.show_interactive()
            # Print the clean final panel (no Setup, consistent with interactive)
            ui.print_renderable(
                self._workflow.renderable(compact=True, include_setup=False),
                stderr=self._workflow.stderr,
            )


# -- internal helpers ------------------------------------------------------

def _set_status_fns(callback: WorkflowDetailCallback | None) -> None:
    """Wire (or clear) the global download / build status sinks."""
    from boxmot.native._common import set_build_status_fn
    from boxmot.utils.download import set_download_status_fn

    set_download_status_fn(callback)
    set_build_status_fn(callback)
