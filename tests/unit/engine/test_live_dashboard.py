"""Interaction and lifecycle contracts for the opt-in Rich dashboard."""

from __future__ import annotations

import threading
import time
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console, Group
from rich.text import Text

import boxmot.engine.ui.core.ui as ui
import boxmot.engine.ui.dashboard as dashboard
from boxmot.engine.ui.dashboard import LiveDashboard, TrialRow, TuningProgressView, dashboard_mode
from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter
from boxmot.engine.ui.reporters.materialize import MaterializeWorkflowReporter
from boxmot.engine.ui.reporters.tune import format_tune_progress
from boxmot.engine.ui.workflow.pipeline import PipelineTracker
from boxmot.engine.ui.workflow.reporting import WorkflowDetailCallback


def _dashboard(width: int = 100, height: int = 24) -> LiveDashboard:
    """Build a deterministic renderer without opening a terminal or thread."""
    console = Console(file=StringIO(), width=width, height=height, theme=ui.BOXMOT_THEME)
    workflow = ui.WorkflowProgress(
        "Evaluation", [("Experiment", "mot17/ablation-yolox-lmbn.yaml")],
        steps=[("Set up", "done"), ("Run tracker", "active"), ("Evaluate results", "todo")],
    )
    return LiveDashboard(workflow, console)


def _render(view: LiveDashboard) -> str:
    """Capture exactly the frame a terminal would receive."""
    with view.console.capture() as capture:
        view.console.print(view.render())
    return capture.get()


@pytest.mark.parametrize("width,height", [(48, 16), (80, 24), (140, 40)])
def test_dashboard_navigation_fits_terminal_and_keeps_literal_configuration(width: int, height: int) -> None:
    view = _dashboard(width, height)
    view.workflow.set_fields([("Model", "weights/[untrusted].pt")], render=False)
    view.handle_key("4")

    rendered = _render(view)

    assert "[untrusted].pt" in rendered
    assert len(rendered.splitlines()) == height
    assert max(map(len, rendered.splitlines())) <= width
    view.handle_key("tab")
    assert view.tab == 4
    view.handle_key("shift+tab")
    assert view.tab == 3


def test_sequence_inspection_retains_real_failed_progress_after_step_finishes() -> None:
    view = _dashboard()
    presenter = EvalSequenceProgressPresenter(
        WorkflowDetailCallback(view.workflow, "Run tracker", render=False), {"MOT17-02": 100, "MOT17-04": 200}
    )
    with presenter:
        presenter.update("MOT17-02", 42, status="running")
        presenter.fail("MOT17-02", "invalid [frame]")
        view.inspect()
    assert view.workflow.detail_renderable is None

    view.handle_key("2")
    view.handle_key("enter")
    rendered = _render(view)

    assert "42 / 100" in rendered
    assert "failed" in rendered
    assert "invalid [frame]" in rendered
    assert "MOT17-02" in rendered
    assert view.sequences[0].completed == 42


def test_tuning_leaderboard_preserves_trials_and_respects_minimized_objective() -> None:
    view = _dashboard()
    view.workflow.detail_renderable = format_tune_progress(
        1, 3, {"IDs": 12}, first_summary={"IDs": 12}, best_summary={"IDs": 12},
        best_trial=1, last_trial=1, minimize=["IDs"],
    )
    view.inspect()
    view.workflow.detail_renderable = format_tune_progress(
        2, 3, {"IDs": 5}, first_summary={"IDs": 12}, best_summary={"IDs": 5},
        best_trial=2, last_trial=2, minimize=["IDs"],
    )
    view.inspect()
    view.workflow.detail_renderable = Text("Saving best configuration")
    view.inspect()
    view.handle_key("3")
    rendered = _render(view)

    assert list(view.trials) == [1, 2]
    assert view._best_history == [12, 5]
    trial_lines = [line for line in rendered.splitlines() if "#" in line and "Δ" not in line]
    assert "#2" in trial_lines[0]
    assert "#1" in trial_lines[1]
    assert "-7.000" in rendered
    view.handle_key("enter")
    assert "Trial #2" in _render(view)


def test_unknown_baseline_is_not_displayed_as_a_measured_zero() -> None:
    view = _dashboard()
    view.workflow.detail_renderable = TuningProgressView(
        Text("Trial complete"), trials=[TrialRow(3, {"HOTA": 50})], maximize=["HOTA"]
    )
    view.inspect()
    view.handle_key("3")
    view.handle_key("enter")
    rendered = _render(view)

    assert "50.000" in rendered
    assert "+50.000" not in rendered
    assert "—" in rendered


def test_logs_search_filter_freeze_and_scroll_do_not_pause_workflow() -> None:
    view = _dashboard()
    for index in range(50):
        view.add_log(f"frame {index}", "INFO")
    view.add_log("Decoder [warning]", "WARNING")
    view.handle_key("5")
    assert "frame 49" in _render(view)
    view.handle_key("up")
    assert view.frozen_logs is not None
    assert view.offset > 0
    view.add_log("latest frame", "INFO")
    assert "latest frame" not in _render(view)
    view.handle_key("f")
    assert "latest frame" in _render(view)
    view.handle_key("w")
    view.handle_key("/")
    for char in "DECODER":
        view.handle_key(char)
    view.handle_key("enter")
    rendered = _render(view)

    assert "Decoder [warning]" in rendered
    assert "latest frame" not in rendered
    assert "frame 49" not in rendered
    view.handle_key("escape")
    assert view.query == ""


def test_pipeline_completed_step_remains_inspectable() -> None:
    view = _dashboard()
    pipeline = PipelineTracker(view.workflow, auto_start=False, wire_status_fns=False)
    pipeline.store_step_info(Text("Detector resolved to yolox-x-mot17"), step=0)
    pipeline.advance("Tracking now")
    view.handle_key("3")
    view.handle_key("enter")

    assert "Detector resolved to yolox-x-mot17" in _render(view)


def test_opt_in_redirected_output_never_acquires_keyboard(monkeypatch) -> None:
    view = _dashboard()
    monkeypatch.setattr(ui, "get_console", lambda **_kwargs: view.console)
    monkeypatch.setattr("boxmot.engine.ui.keyboard.is_interactive", lambda _console: False)
    monkeypatch.setattr(LiveDashboard, "start", lambda _self: pytest.fail("Opened dashboard for redirected output"))
    with dashboard_mode():
        with view.workflow:
            view.workflow.set_detail("Tracking", "done")
    assert view.workflow._dashboard is None
    assert dashboard._enabled.get() is False


def test_dashboard_restores_input_and_screen_after_workflow_failure(monkeypatch) -> None:
    view = _dashboard()
    events: list[str] = []

    class Input:
        def __enter__(self):
            events.append("input start")
            return self

        def read_key(self, timeout=0.05):
            time.sleep(min(timeout, 0.01))
            return None

        def __exit__(self, *_args):
            events.append("input stop")

    class Live:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self, **_kwargs):
            events.append("live start")

        def update(self, *_args, **_kwargs):
            pass

        def stop(self):
            events.append("live stop")

    monkeypatch.setattr(ui, "get_console", lambda **_kwargs: view.console)
    monkeypatch.setattr("boxmot.engine.ui.keyboard.is_interactive", lambda _console: True)
    monkeypatch.setattr("boxmot.engine.ui.keyboard.TerminalInput", Input)
    monkeypatch.setattr(dashboard, "Live", Live)
    with pytest.raises(RuntimeError, match="tracker failed"), dashboard_mode():
        with view.workflow:
            assert dashboard.capture_dashboard_log("visible error", "ERROR")
            raise RuntimeError("tracker failed")

    assert events == ["input start", "live start", "live stop", "input stop"]
    assert dashboard._active is None
    assert view.workflow._dashboard is None
    assert not dashboard.capture_dashboard_log("after stop")
    assert not any(thread.name == "boxmot-dashboard" for thread in threading.enumerate())


def test_input_eof_stops_polling_without_spinning() -> None:
    view = _dashboard()
    reads: list[float] = []

    class Input:
        def read_key(self, timeout=0.05):
            reads.append(timeout)
            return "eof"

    view._input = Input()
    view._poll()

    assert reads == [0.1]
    assert view._stop.is_set()


def test_successful_dashboard_keeps_final_results_until_return() -> None:
    view = _dashboard()
    view.workflow.set_detail("Results", "Final HOTA: 61.25", render=False)
    waiter = threading.Thread(target=view.finish)
    waiter.start()
    try:
        deadline = time.monotonic() + 1.0
        while not view.completed and time.monotonic() < deadline:
            time.sleep(0.001)
        assert view.completed
        rendered = _render(view)
        assert "Final HOTA: 61.25" in rendered
        assert "q return" in rendered
        assert waiter.is_alive()
        view.handle_key("4")
        assert "mot17/ablation-yolox-lmbn.yaml" in _render(view)
        view.handle_key("q")
        waiter.join(timeout=1.0)
        assert not waiter.is_alive()
    finally:
        view._returned.set()
        waiter.join(timeout=1.0)


@pytest.mark.parametrize("state,allowed", [("failed", True), ("todo", True), ("done", False)])
def test_failed_or_intermediate_pipeline_does_not_wait_for_browsing(monkeypatch, state, allowed) -> None:
    view = _dashboard()
    view.workflow.steps = [("Materialize dataset", state)]
    view.workflow.allow_dashboard_browse = allowed
    view.workflow._started = True
    stopped = []
    view.workflow._dashboard = SimpleNamespace(
        finish=lambda: pytest.fail("Waited for results on an unsuccessful or intermediate workflow"),
        stop=lambda: stopped.append(True),
    )
    monkeypatch.setattr(ui, "print_renderable", lambda *_args, **_kwargs: None)

    PipelineTracker(view.workflow, auto_start=False, wire_status_fns=False).__exit__(None, None, None)

    assert stopped == [True]
    assert view.workflow._dashboard is None


@pytest.mark.parametrize("mode,expected", [("eval", False), ("tune", False), ("materialize", True)])
def test_materialize_reporter_only_browses_when_it_is_the_requested_command(monkeypatch, mode, expected) -> None:
    view = _dashboard()
    monkeypatch.setattr(PipelineTracker, "__enter__", lambda self: self)
    reporter = MaterializeWorkflowReporter(SimpleNamespace(materialize_mode=mode), workflow=view.workflow)

    reporter.start()

    assert view.workflow.allow_dashboard_browse is expected


def test_grouped_trial_sequences_are_all_inspectable() -> None:
    view = _dashboard()
    presenters = [
        EvalSequenceProgressPresenter(
            WorkflowDetailCallback(view.workflow, "Run tracker", render=False), {"MOT17-02": 100},
            dashboard_label=f"Trial {trial}",
        ) for trial in (1, 2)
    ]
    view.workflow.detail_renderable = Group(*(presenter.renderable for presenter in presenters))
    view.inspect()
    view.handle_key("2")
    rendered = _render(view)

    assert "Trial 1 / MOT17-02" in rendered
    assert "Trial 2 / MOT17-02" in rendered
