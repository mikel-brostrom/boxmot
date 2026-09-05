"""Regression tests for the engine-owned Rich workflow state machine."""

from __future__ import annotations

from io import StringIO

import pytest
from rich.console import Console, Group
from rich.text import Text

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.ui.workflow.pipeline import PipelineTracker


class _FakeLive:
    """Minimal active Live object needed by ``WorkflowProgress.stop``."""

    def __init__(self) -> None:
        self.transient = False
        self.updates: list[object] = []

    def update(self, renderable, *, refresh: bool) -> None:
        assert refresh is False
        self.updates.append(renderable)

    def stop(self) -> None:
        return None


def test_workflow_stop_refreshes_pending_state_by_default() -> None:
    """Direct WorkflowProgress users retain the existing final refresh."""

    workflow = ui.create_workflow_progress(
        "Tracking",
        (),
        steps=(("Track", "active"),),
    )
    live = _FakeLive()
    workflow._started = True
    workflow._live = live  # type: ignore[assignment]
    workflow._last_rendered_state = workflow._state_snapshot()
    workflow.detail_title = "Results"
    workflow.detail_text = "pending final state"

    workflow.stop()

    assert len(live.updates) == 1
    assert "pending final state" in ui.capture_renderable(live.updates[0])


def test_pipeline_failure_prints_one_panel_after_alt_screen(monkeypatch) -> None:
    """An oversized failure must not leave both live and final panels behind."""

    workflow = ui.create_workflow_progress(
        "Tracking",
        (("Source", "0"),),
        steps=(("Set up", "active"), ("Run tracker", "todo")),
        stderr=True,
    )
    live = _FakeLive()
    workflow._started = True
    workflow._uses_alt_screen = True
    workflow._live = live  # type: ignore[assignment]
    monkeypatch.setattr(workflow, "_update_live", lambda **_kwargs: None)
    rendered: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        ui,
        "print_renderable",
        lambda renderable, *, stderr=False: rendered.append((renderable, stderr)),
    )

    error = RuntimeError("invalid detector selector")
    pipeline = PipelineTracker(workflow, auto_start=False, wire_status_fns=False)
    with pytest.raises(RuntimeError, match="invalid detector selector"):
        with pipeline:
            raise error

    assert workflow.transient is True
    assert live.transient is True
    assert live.updates == []
    assert getattr(error, "_workflow_rendered_error") is True
    assert len(rendered) == 1
    final_panel, stderr = rendered[0]
    assert stderr is True
    final_text = ui.capture_renderable(final_panel, width=120)
    assert "Set up failed" in final_text
    assert "invalid detector selector" in final_text
    assert "SOURCE" not in final_text


def _run_finished_pipeline(monkeypatch, *, force_terminal: bool) -> str:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=force_terminal,
        width=80,
        height=8,
        highlight=False,
        theme=ui.BOXMOT_THEME,
        _environ={"TERM": "xterm-256color"},
    )
    monkeypatch.setattr(ui, "_stderr_console", console)
    workflow = ui.create_workflow_progress(
        "Tracking",
        (),
        steps=(("Track", "active"),),
        stderr=True,
    )
    result = Group(
        Text("FINAL SUMMARY"),
        *(Text(f"result row {index}") for index in range(20)),
    )
    pipeline = PipelineTracker(workflow, wire_status_fns=False)

    with pipeline:
        pipeline.finish(result)

    return output.getvalue()


def test_pipeline_does_not_paint_tall_final_panel_into_short_terminal_live(monkeypatch) -> None:
    """The final report must bypass the transient normal-screen Live."""

    output = _run_finished_pipeline(monkeypatch, force_terminal=True)

    # ANSI cursor operations cause the bounded progress frame to occur more
    # than once in the raw stream.  The pending final report, however, must
    # occur exactly once: in the clean static print after Live has stopped.
    assert output.count("FINAL SUMMARY") == 1
    assert output.count("result row 0") == 1
    assert output.count("result row 19") == 1


def test_pipeline_prints_one_final_panel_to_non_tty_capture(monkeypatch) -> None:
    """Captured output contains no hidden Live frame or duplicate report."""

    output = _run_finished_pipeline(monkeypatch, force_terminal=False)

    assert output.count("Tracking") == 1
    assert output.count("FINAL SUMMARY") == 1
    assert output.count("result row 0") == 1
    assert output.count("result row 19") == 1
