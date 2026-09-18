"""Offline per-sequence progress stays inside the evaluation workflow."""

from __future__ import annotations

from types import SimpleNamespace

import boxmot.engine.ui.core.ui as ui
from boxmot.engine.ui.reporters.postprocessing import EvalPostprocessingProgressPresenter
from boxmot.engine.ui.workflow.reporting import WorkflowDetailCallback
from boxmot.engine.ui.workflow.steps import POSTPROCESS, eval_steps


def _presenter(
    *, sequence_ids: tuple[str, ...] = ("MOT20-03", "MOT20-01", "MOT20-02", "MOT20-05"), render: bool = False
) -> tuple[ui.WorkflowProgress, EvalPostprocessingProgressPresenter]:
    """Create an embedded presenter without starting a second live display."""
    workflow = ui.create_workflow_progress("Evaluation", (), steps=eval_steps(postprocess=True), stderr=True)
    presenter = EvalPostprocessingProgressPresenter(
        WorkflowDetailCallback(workflow, POSTPROCESS, render=render),
        sequence_ids,
    )
    return workflow, presenter


def test_postprocessing_embeds_sequence_states_counts_and_method_details() -> None:
    workflow, presenter = _presenter()

    with presenter:
        presenter(
            SimpleNamespace(
                sequence_id="MOT20-03",
                status="running",
                completed=11,
                total=40,
                detail="GTA · split tracklets",
                phase_index=1,
            )
        )
        presenter.update("MOT20-01", 30, 50, status="running", detail="GSI · smooth tracks")
        presenter.complete("MOT20-02")
        presenter.fail("MOT20-05", ValueError("invalid embeddings"))

        rendered = ui.capture_renderable(presenter.renderable, width=180)
        assert "Postprocessing: 1/4 sequences done · 1 failed" in rendered
        assert "Tracking:" not in rendered
        assert "11/40 items" in rendered
        assert "30/50 items" in rendered
        assert "GTA · split tracklets" in rendered
        assert "GSI · smooth tracks" in rendered
        assert "failed · invalid embeddings" in rendered
        assert rendered.index("MOT20-03") < rendered.index("MOT20-01") < rendered.index("MOT20-02")
        assert workflow.detail_renderable is presenter.renderable

    assert workflow.detail_renderable is None


def test_postprocessing_starts_queued_and_discards_stale_worker_events() -> None:
    _workflow, presenter = _presenter()
    assert all(task.fields["status"] == "queued" for task in presenter.progress.tasks)
    assert all(task.start_time is None for task in presenter.progress.tasks)

    presenter.update("MOT20-03", 400, 1000, status="running", detail="GTA · associate")
    presenter.update("MOT20-03", 100, 1000, status="queued", detail="stale")
    task = presenter.progress.tasks[0]
    assert task.completed == 400
    assert task.fields["detail"] == "GTA · associate"

    presenter.complete("MOT20-03")
    presenter.update("MOT20-03", 600, 1000, status="running", detail="GSI · stale")
    assert task.completed == 1000
    assert task.fields["status"] == "completed"
    assert task.fields["detail"] is None


def test_postprocessing_only_animates_running_sequences_with_unknown_totals() -> None:
    _workflow, presenter = _presenter()
    task = presenter.progress.tasks[0]
    column = presenter.progress.columns[1]
    queued = column.render(task)
    assert not queued.pulse
    assert queued.total == 1
    assert queued.completed == 0

    presenter.update("MOT20-03", 4, status="running", detail="GTA · merge candidates")
    assert column.render(task).total is None

    presenter.complete("MOT20-03")
    completed = column.render(task)
    assert not completed.pulse
    assert completed.completed == completed.total == 1
    assert task.completed == 4
    assert task.total is None

    presenter.fail("MOT20-01", "worker failed")
    failed = column.render(presenter.progress.tasks[1])
    assert not failed.pulse
    assert failed.total == 1
    assert failed.completed == 0


def test_postprocessing_failed_sequence_retains_completed_work() -> None:
    _workflow, presenter = _presenter()
    presenter.update("MOT20-03", 450, 1000, status="running", detail="GTA · associate")
    presenter.fail("MOT20-03", "worker failed")
    presenter.update("MOT20-03", 300, 1000, status="running", detail="late")

    task = presenter.progress.tasks[0]
    assert task.completed == 450
    assert task.fields["status"] == "failed"
    assert task.fields["detail"] == "worker failed"


def test_postprocessing_uses_two_columns_for_many_sequences() -> None:
    _workflow, presenter = _presenter(sequence_ids=tuple(f"sequence-{index:02d}" for index in range(12)))

    rendered = ui.capture_renderable(presenter.renderable, width=240)
    rows = [line for line in rendered.splitlines() if "sequence-" in line]
    assert len(rows) == 6
    assert "sequence-00" in rows[0]
    assert "sequence-06" in rows[0]


def test_postprocessing_restores_outer_detail_after_failure() -> None:
    workflow, presenter = _presenter()
    existing = ui.Text("existing detail")
    workflow.detail_renderable = existing

    try:
        with presenter:
            assert workflow.detail_renderable is presenter.renderable
            raise ValueError("worker failed")
    except ValueError:
        pass

    assert workflow.detail_renderable is existing


def test_postprocessing_resets_phase_counters_and_rejects_late_phase_events() -> None:
    _workflow, presenter = _presenter()

    def publish(phase_index: int, completed: int, total: int | None, detail: str) -> None:
        presenter(
            SimpleNamespace(
                sequence_id="MOT20-03",
                status="running",
                completed=completed,
                total=total,
                detail=detail,
                phase_index=phase_index,
            )
        )

    publish(1, 39, 40, "GTA · split tracklets")
    publish(2, 2, None, "GTA · merge candidates")
    publish(1, 40, 40, "late split")

    task = presenter.progress.tasks[0]
    assert task.total is None
    assert task.completed == 2
    assert task.fields["detail"] == "GTA · merge candidates"
    rendered = ui.capture_renderable(presenter.renderable, width=180)
    assert "2 items" in rendered
    assert "2/40 items" not in rendered

    publish(3, 1, 8, "GSI · smooth tracks")
    assert task.total == 8
    assert task.completed == 1
    assert task.fields["detail"] == "GSI · smooth tracks"
    presenter.complete("MOT20-03")
    publish(4, 0, 15, "late phase")
    assert task.total == 8
    assert task.completed == 8
    assert task.fields["status"] == "completed"
