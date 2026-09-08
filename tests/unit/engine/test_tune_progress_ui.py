"""Tune progress presents the same useful task rows as evaluation replay."""

from __future__ import annotations

import pickle
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from rich.console import Group
from rich.progress import BarColumn, Progress

from boxmot.engine.tuning.tuner import Tuner
from boxmot.engine.ui.core.ui import capture_renderable
from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter
from boxmot.engine.ui.reporters.tune import (
    TUNE_OPTIMIZE_STEP,
    TuneWorkflowCallback,
    format_initial_tune_progress,
    format_tune_progress,
    set_tune_progress_workflow,
)


def _progress(renderable: Group) -> Progress:
    """Inspect the live Rich bar so a determinate count cannot hide a pulse."""
    return next(part for part in renderable.renderables if isinstance(part, Progress))


@pytest.mark.parametrize("width", [80, 160])
def test_initial_progress_is_queued_without_claiming_a_trial_is_running(width: int) -> None:
    renderable = format_initial_tune_progress(200)

    rendered = capture_renderable(renderable, width=width)

    assert "Tuning: 0/200 trials done" in rendered
    assert "Tune" in rendered
    assert "0/200 trials" in rendered
    assert "○ pending" in rendered
    assert "remaining --:--" in rendered
    assert "running" not in rendered
    assert "trial 1/200" not in rendered
    assert "Best trial:" not in rendered
    assert "Last trial:" not in rendered
    assert max(map(len, rendered.splitlines())) <= width
    task = _progress(renderable).tasks[0]
    assert not task.started
    assert task.completed == 0


@pytest.mark.parametrize("width", [80, 160])
def test_active_progress_shows_counts_running_jobs_and_a_quantitative_bar(width: int) -> None:
    renderable = format_tune_progress(
        52,
        200,
        current_trial=54,
        active_trials=2,
        remaining_seconds=65,
    )

    rendered = capture_renderable(renderable, width=width)

    assert "Tuning: 52/200 trials done · 2 running" in rendered
    assert "52/200 trials" in rendered
    assert "▶ running" in rendered
    assert "trial 54/200" in rendered
    assert "remaining 01:05" in " ".join(rendered.split())
    assert max(map(len, rendered.splitlines())) <= width
    task = _progress(renderable).tasks[0]
    assert task.started
    assert not task.finished
    assert task.percentage == pytest.approx(26.0)
    assert BarColumn().render(task).pulse is False


@pytest.mark.parametrize("width", [80, 160])
@pytest.mark.parametrize("failed", [0, 2])
def test_final_progress_distinguishes_successful_trials_from_failures(width: int, failed: int) -> None:
    renderable = format_tune_progress(10, 10, current_trial=10, active_trials=1, failed=failed, remaining_seconds=125)

    rendered = capture_renderable(renderable, width=width)

    assert f"Tuning: {10 - failed}/10 trials done" in rendered
    assert "10/10 trials" in rendered
    assert "remaining 00:00" in rendered
    assert "running" not in rendered
    assert "trial 10/10" not in rendered
    assert ("✕ failed" if failed else "✓ done") in rendered
    if failed:
        assert "2 failed" in rendered
    task = _progress(renderable).tasks[0]
    assert task.started
    assert task.finished
    assert task.percentage == 100.0
    assert BarColumn().render(task).pulse is False


@pytest.mark.parametrize("width", [80, 160])
@pytest.mark.parametrize("completed,status", [(52, "running"), (200, "completed")])
def test_tune_and_eval_render_identical_task_columns_apart_from_units(width: int, completed: int, status: str) -> None:
    tune = format_tune_progress(completed, 200, current_trial=53 if status == "running" else None)
    tune_progress = _progress(tune)
    task = tune_progress.tasks[0]
    evaluation = EvalSequenceProgressPresenter(MagicMock(), {"Tune": 200})
    evaluation.update("Tune", 0, status="running")
    evaluation.update("Tune", completed, status=status, detail=task.fields["detail"])

    tune_row = capture_renderable(tune_progress, width=width)
    eval_row = capture_renderable(evaluation.progress, width=width)

    # "frames" and "trials" occupy the same width. Equal row output verifies
    # label, filled bar, count, and status positioning at both terminal sizes.
    assert tune_row == eval_row.replace("frames", "trials")


@pytest.mark.parametrize("width", [80, 160])
def test_best_and_last_metrics_remain_visible_without_corrupting_bar(width: int) -> None:
    renderable = format_tune_progress(
        3,
        10,
        {"HOTA": 67.4, "MOTA": 75.2, "IDF1": 82.1},
        best_summary={"HOTA": 69.0, "MOTA": 74.0, "IDF1": 84.0},
    )

    rendered = capture_renderable(renderable, width=width)

    compact = " ".join(rendered.split())
    assert "Best trial: HOTA=69.000 MOTA=74.000 IDF1=84.000" in compact
    assert "Last trial: HOTA=67.400 MOTA=75.200 IDF1=82.100" in compact
    assert "3/10 trials" in rendered
    assert "pending" in rendered
    assert "running" not in rendered
    assert max(map(len, rendered.splitlines())) <= width


def test_missing_metrics_are_distinguished_from_measured_zero() -> None:
    renderable = format_tune_progress(
        1,
        3,
        {"HOTA": 0.0, "MOTA": None, "IDF1": float("nan")},
        best_summary={"HOTA": 5.0},
    )

    rendered = capture_renderable(renderable, width=160)

    assert "Best trial: HOTA=5.000 MOTA=-- IDF1=--" in rendered
    assert "Last trial: HOTA=0.000 MOTA=-- IDF1=--" in rendered


def test_callbacks_publish_rich_progress_for_real_trial_states_and_remain_pickle_safe() -> None:
    publications = []
    workflow = SimpleNamespace(
        _lock=threading.RLock(),
        set_detail=MagicMock(),
        set_detail_renderable=lambda title, renderable: publications.append((title, renderable)),
    )
    callback = TuneWorkflowCallback(total=4, maximize=["HOTA"], minimize=[])
    first = SimpleNamespace(
        trial_id="first", last_result={"HOTA": 60.0, "MOTA": 70.0, "IDF1": 80.0, "time_total_s": 5.0}
    )
    second = SimpleNamespace(trial_id="second", last_result={})
    third = SimpleNamespace(
        trial_id="third", last_result={"HOTA": 55.0, "MOTA": 90.0, "IDF1": 90.0, "time_total_s": 6.0}
    )
    fourth = SimpleNamespace(trial_id="fourth", last_result={"HOTA": 70.0, "MOTA": 65.0, "IDF1": 75.0})
    first_best = "Best trial: HOTA=60.000 MOTA=70.000 IDF1=80.000"
    first_last = "Last trial: HOTA=60.000 MOTA=70.000 IDF1=80.000"
    set_tune_progress_workflow(workflow)
    try:
        callback.on_trial_start(0, [], first)
        callback.on_trial_start(0, [], second)
        running = capture_renderable(publications[-1][1], width=160)
        assert "0/4 trials done · 2 running" in running
        assert "Best trial:" not in running
        assert "Last trial:" not in running

        callback.on_trial_complete(0, [], first)
        completed = capture_renderable(publications[-1][1], width=160)
        assert "1/4 trials done · 1 running" in completed
        assert first_best in completed
        assert first_last in completed

        callback.on_trial_error(0, [], second)
        waiting = capture_renderable(publications[-1][1], width=160)
        assert "1/4 trials done · 1 failed" in waiting
        assert "pending" in waiting
        assert "running" not in waiting
        assert "trial 3/4" not in waiting
        assert first_best in waiting
        assert first_last in waiting

        # Rich Progress contains locks. Publishing a renderable must not retain
        # it on the callback that Ray serializes to manage trial execution.
        restored = pickle.loads(pickle.dumps(callback))
        assert restored.completed == 2
        assert restored.failed == 1
        assert restored.best_summary == callback.best_summary
        assert restored.last_summary == callback.last_summary
        assert not any(isinstance(value, (Group, Progress)) for value in vars(callback).values())

        callback.on_trial_start(0, [], third)
        restarted = capture_renderable(publications[-1][1], width=160)
        assert first_best in restarted
        assert first_last in restarted
        callback.on_trial_complete(0, [], third)
        worse = capture_renderable(publications[-1][1], width=160)
        assert first_best in worse
        assert "Last trial: HOTA=55.000 MOTA=90.000 IDF1=90.000" in worse

        callback.on_trial_start(0, [], fourth)
        callback.on_trial_complete(0, [], fourth)
        final = capture_renderable(publications[-1][1], width=160)
        assert "3/4 trials done · 1 failed" in final
        assert "4/4 trials" in final
        assert "✕ failed" in final
        assert "remaining 00:00" in final
        assert "running" not in final
        assert "Best trial: HOTA=70.000 MOTA=65.000 IDF1=75.000" in final
        assert "Last trial: HOTA=70.000 MOTA=65.000 IDF1=75.000" in final
        assert all(title == TUNE_OPTIMIZE_STEP for title, _ in publications)
        assert all(isinstance(renderable, Group) for _, renderable in publications)
        workflow.set_detail.assert_not_called()
    finally:
        set_tune_progress_workflow(None)


def test_best_summary_uses_the_configured_objective_including_identity_switch_rate() -> None:
    callback = TuneWorkflowCallback(total=3, maximize=[], minimize=["IDSW_rate"])
    first = SimpleNamespace(trial_id="first", last_result={"HOTA": 70.0, "IDSW_rate": 0.8})
    second = SimpleNamespace(trial_id="second", last_result={"HOTA": 60.0, "IDSW_rate": 0.2})
    third = SimpleNamespace(trial_id="third", last_result={"HOTA": 80.0, "IDSW_rate": 0.7})

    for trial in (first, second, third):
        callback.on_trial_start(0, [], trial)
        callback.on_trial_complete(0, [], trial)

    assert callback.best_summary["HOTA"] == 60.0
    assert callback.last_summary["HOTA"] == 80.0
    assert callback.best_score == (-0.2,)


def test_restored_tuner_publishes_saved_trial_count_before_any_new_trial_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    publications = []
    workflow = SimpleNamespace(set_detail_renderable=lambda title, renderable: publications.append((title, renderable)))
    monkeypatch.setitem(
        sys.modules,
        "ray.tune",
        SimpleNamespace(ExperimentAnalysis=lambda path: SimpleNamespace(dataframe=lambda: [None] * 3)),
    )
    callback = TuneWorkflowCallback(total=10, maximize=["HOTA"], minimize=[])
    restored = SimpleNamespace(_local_tuner=SimpleNamespace(_run_config=SimpleNamespace()))
    set_tune_progress_workflow(workflow)
    try:
        Tuner(SimpleNamespace(n_trials=10))._inject_callback_into_restored(restored, callback, MagicMock(), tmp_path)

        assert len(publications) == 1
        title, renderable = publications[0]
        assert title == TUNE_OPTIMIZE_STEP
        assert isinstance(renderable, Group)
        rendered = capture_renderable(renderable, width=120)
        assert "3/10 trials done" in rendered
        assert "3/10 trials" in rendered
        assert "pending" in rendered
        assert "running" not in rendered
        assert "trial 4/10" not in rendered
        assert callback.completed == 3
        assert _progress(renderable).tasks[0].started
        assert BarColumn().render(_progress(renderable).tasks[0]).pulse is False
    finally:
        set_tune_progress_workflow(None)
