"""Driver-local Rich progress for sequence-parallel evaluation replay."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter, EvalWorkflowReporter
from boxmot.engine.ui.workflow.reporting import WorkflowDetailCallback
from boxmot.engine.ui.workflow.steps import TRACK, eval_steps


class FixedClock:
    def __init__(self, *values: float) -> None:
        self._values: Iterator[float] = iter(values)

    def __call__(self) -> float:
        return next(self._values)


def _presenter(
    *,
    render: bool = False,
    clock=FixedClock(0.0),
) -> tuple[ui.WorkflowProgress, EvalSequenceProgressPresenter]:
    workflow = ui.create_workflow_progress("Evaluation", (), steps=eval_steps(), stderr=True)
    callback = WorkflowDetailCallback(workflow, TRACK, render=render)
    presenter = EvalSequenceProgressPresenter(
        callback,
        {"MOT17-05": 3, "MOT17-02": 2},
        clock=clock,
    )
    return workflow, presenter


def test_eval_sequence_progress_consumes_structural_events_in_input_order() -> None:
    workflow, presenter = _presenter()

    with presenter:
        presenter(
            SimpleNamespace(
                sequence_id="MOT17-05",
                status="running",
                completed=1,
                total=3,
                detail=None,
            )
        )
        presenter.complete("MOT17-05")
        presenter.fail("MOT17-02", RuntimeError("tracker failed"))

        rendered = ui.capture_renderable(presenter.renderable, width=140)
        assert "Tracking: 1/2 sequences done · 1 failed" in rendered
        assert rendered.index("MOT17-05") < rendered.index("MOT17-02")
        assert "3/3 frames" in rendered
        assert "done" in rendered
        assert "failed · tracker failed" in rendered
        assert workflow.detail_renderable is presenter.renderable

    assert workflow.detail_renderable is None


def test_eval_sequence_progress_ignores_delayed_updates_after_completion() -> None:
    _workflow, presenter = _presenter()

    presenter.update("MOT17-05", 2, 3, status="running")
    presenter.complete("MOT17-05")
    presenter.update("MOT17-05", 1, 3, status="running", detail="stale")

    task = presenter.progress.tasks[0]
    assert task.completed == 3
    assert task.fields["status"] == "completed"
    assert task.fields["detail"] is None


def test_eval_sequence_progress_keeps_catalog_total_for_zero_placeholders() -> None:
    _workflow, presenter = _presenter()

    presenter(
        SimpleNamespace(
            sequence_id="MOT17-05",
            status="queued",
            completed=0,
            total=0,
            detail=None,
        )
    )

    task = presenter.progress.tasks[0]
    assert task.total == 3
    assert task.fields["status"] == "queued"

    presenter(
        SimpleNamespace(
            sequence_id="MOT17-02",
            status="failed",
            completed=0,
            total=0,
            detail="worker startup failed",
        )
    )

    failed_task = presenter.progress.tasks[1]
    assert failed_task.total == 2
    assert failed_task.fields["status"] == "failed"


def test_eval_sequence_progress_throttles_parent_workflow_refresh(monkeypatch) -> None:
    workflow, presenter = _presenter(render=True, clock=FixedClock(0.0, 0.01, 0.2, 0.21))
    refreshes: list[dict[str, object]] = []
    monkeypatch.setattr(workflow, "_update_live", lambda **kwargs: refreshes.append(kwargs))

    with presenter:
        refreshes.clear()
        presenter.update("MOT17-05", 1, 3, status="running")
        assert refreshes == []

        presenter.update("MOT17-05", 2, 3, status="running")
        assert refreshes == [{"render": True, "force": True}]


def test_eval_sequence_progress_uses_two_columns_above_ten_sequences() -> None:
    workflow = ui.create_workflow_progress("Evaluation", (), steps=eval_steps(), stderr=True)
    presenter = EvalSequenceProgressPresenter(
        WorkflowDetailCallback(workflow, TRACK, render=False),
        {f"sequence-{index:02d}": 2 for index in range(12)},
    )

    rendered = ui.capture_renderable(presenter.renderable, width=240)
    sequence_lines = [line for line in rendered.splitlines() if "sequence-" in line]
    assert len(sequence_lines) == 6
    assert "sequence-00" in sequence_lines[0]
    assert "sequence-06" in sequence_lines[0]


def test_eval_workflow_fields_show_sequence_worker_processes() -> None:
    fields = dict(EvalWorkflowReporter(SimpleNamespace(tracker="ocsort", sequence_workers=4)).fields())
    capped_fields = dict(
        EvalWorkflowReporter(
            SimpleNamespace(tracker="ocsort", sequence_workers=8, seq_info={"seq-a": 1, "seq-b": 2})
        ).fields()
    )

    assert ("Processes", 4) in fields["__panel__:Tracker"]
    assert ("Processes", 2) in capped_fields["__panel__:Tracker"]
