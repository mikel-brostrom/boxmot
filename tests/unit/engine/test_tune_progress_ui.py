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

from boxmot.engine.eval.replay import ReplayProgressEvent
from boxmot.engine.tuning.progress import TrialSequenceProgressWriter, read_trial_sequence_progress
from boxmot.engine.tuning.tuner import Tuner
from boxmot.engine.ui.core.ui import STYLE_STATUS_DONE, STYLE_STATUS_FAILED, capture_renderable, get_console
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


def _metric_row(rendered: str, label: str) -> str:
    """Read one visible comparison row without depending on table borders."""
    row = next(line for line in rendered.splitlines() if label in line)
    return " ".join(row.translate(str.maketrans({"│": " ", "┃": " ", "|": " "})).split())


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
    assert "First #1" not in rendered
    assert "Best #" not in rendered
    assert "Last #" not in rendered
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
def test_first_best_and_last_metrics_show_absolute_deltas_without_corrupting_bar(width: int) -> None:
    renderable = format_tune_progress(
        3,
        10,
        {"HOTA": 67.4, "MOTA": 75.2, "IDF1": 82.1},
        first_summary={"HOTA": 66.0, "MOTA": 74.0, "IDF1": 83.0},
        best_summary={"HOTA": 69.0, "MOTA": 74.0, "IDF1": 84.0},
        best_trial=2,
        last_trial=3,
        maximize=["HOTA"],
    )

    rendered = capture_renderable(renderable, width=width)

    assert "Run HOTA MOTA IDF1" in _metric_row(rendered, "HOTA")
    assert "First #1 (baseline) 66.000 74.000 83.000" == _metric_row(rendered, "First #1")
    assert "Best #2 69.000 (+3.000) 74.000 (+0.000) 84.000 (+1.000)" == _metric_row(rendered, "Best #2")
    assert "Last #3 67.400 (+1.400) 75.200 (+1.200) 82.100 (-0.900)" == _metric_row(rendered, "Last #3")
    assert "Δ vs first trial (metric units)" in rendered
    assert "3/10 trials" in rendered
    assert "pending" in rendered
    assert "running" not in rendered
    assert max(map(len, rendered.splitlines())) <= width


def test_missing_metrics_are_distinguished_from_measured_zero() -> None:
    renderable = format_tune_progress(
        1,
        3,
        {"HOTA": 0.0, "MOTA": 7.0, "IDF1": float("nan")},
        first_summary={"HOTA": 0.0, "MOTA": None, "IDF1": float("inf")},
        best_summary={"HOTA": 5.0, "IDF1": float("-inf")},
    )

    rendered = capture_renderable(renderable, width=160)

    assert _metric_row(rendered, "First #1") == "First #1 (baseline) 0.000 -- --"
    assert _metric_row(rendered, "Best") == "Best 5.000 (+5.000) -- (--) -- (--)"
    assert _metric_row(rendered, "Last") == "Last 0.000 (+0.000) 7.000 (--) -- (--)"


def test_custom_objectives_are_visible_once_and_delta_colors_follow_the_objective_direction() -> None:
    renderable = format_tune_progress(
        3,
        10,
        {"HOTA": 69.0, "MOTA": 69.0, "IDF1": 75.0, "IDSW_rate": 0.9},
        first_summary={"HOTA": 70.0, "MOTA": 70.0, "IDF1": 75.0, "IDSW_rate": 0.8},
        best_summary={"HOTA": 72.0, "MOTA": 68.0, "IDF1": 75.0, "IDSW_rate": 0.2},
        best_trial=2,
        last_trial=3,
        maximize=["HOTA", "IDF1"],
        minimize=["IDSW_rate"],
    )

    rendered = capture_renderable(renderable, width=160)

    assert _metric_row(rendered, "HOTA") == "Run HOTA MOTA IDF1 IDSW_rate"
    assert "0.200 (-0.600)" in _metric_row(rendered, "Best #2")
    assert "0.900 (+0.100)" in _metric_row(rendered, "Last #3")
    console = get_console()
    segments = list(console.render(renderable, console.options.update(width=160)))
    for delta, expected_style in (
        ("(+2.000)", STYLE_STATUS_DONE),
        ("(-2.000)", STYLE_STATUS_FAILED),
        ("(-0.600)", STYLE_STATUS_DONE),
        ("(+0.100)", STYLE_STATUS_FAILED),
    ):
        segment = next(segment for segment in segments if delta in segment.text)
        assert segment.style.color == console.get_style(expected_style).color


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
        trial_id="third", last_result={"HOTA": 70.0, "MOTA": 90.0, "IDF1": 90.0, "time_total_s": 6.0}
    )
    fourth = SimpleNamespace(trial_id="fourth", last_result={"HOTA": 65.0, "MOTA": 65.0, "IDF1": 75.0})
    first_baseline = "First #1 (baseline) 60.000 70.000 80.000"
    first_best = "Best #1 60.000 (+0.000) 70.000 (+0.000) 80.000 (+0.000)"
    first_last = "Last #1 60.000 (+0.000) 70.000 (+0.000) 80.000 (+0.000)"
    set_tune_progress_workflow(workflow)
    try:
        callback.on_trial_start(0, [], first)
        callback.on_trial_start(0, [], second)
        running = capture_renderable(publications[-1][1], width=160)
        assert "0/4 trials done · 2 running" in running
        assert "First #1" not in running
        assert "Best #" not in running
        assert "Last #" not in running

        callback.on_trial_complete(0, [], first)
        completed = capture_renderable(publications[-1][1], width=160)
        assert "1/4 trials done · 1 running" in completed
        assert _metric_row(completed, "First #1") == first_baseline
        assert _metric_row(completed, "Best #1") == first_best
        assert _metric_row(completed, "Last #1") == first_last

        callback.on_trial_error(0, [], second)
        waiting = capture_renderable(publications[-1][1], width=160)
        assert "1/4 trials done · 1 failed" in waiting
        assert "pending" in waiting
        assert "running" not in waiting
        assert "trial 3/4" not in waiting
        assert _metric_row(waiting, "First #1") == first_baseline
        assert _metric_row(waiting, "Best #1") == first_best
        assert _metric_row(waiting, "Last #1") == first_last

        # Rich Progress contains locks. Publishing a renderable must not retain
        # it on the callback that Ray serializes to manage trial execution.
        restored = pickle.loads(pickle.dumps(callback))
        assert restored.completed == 2
        assert restored.failed == 1
        assert restored.first_summary == callback.first_summary
        assert restored.best_summary == callback.best_summary
        assert restored.last_summary == callback.last_summary
        assert restored.best_trial_index == 1
        assert restored.last_trial_index == 1
        assert not any(isinstance(value, (Group, Progress)) for value in vars(callback).values())

        callback.on_trial_start(0, [], third)
        restarted = capture_renderable(publications[-1][1], width=160)
        assert _metric_row(restarted, "Best #1") == first_best
        assert _metric_row(restarted, "Last #1") == first_last
        callback.on_trial_complete(0, [], third)
        improved = capture_renderable(publications[-1][1], width=160)
        assert _metric_row(improved, "First #1") == first_baseline
        assert _metric_row(improved, "Best #3") == "Best #3 70.000 (+10.000) 90.000 (+20.000) 90.000 (+10.000)"
        assert _metric_row(improved, "Last #3") == "Last #3 70.000 (+10.000) 90.000 (+20.000) 90.000 (+10.000)"

        callback.on_trial_start(0, [], fourth)
        callback.on_trial_complete(0, [], fourth)
        final = capture_renderable(publications[-1][1], width=160)
        assert "3/4 trials done · 1 failed" in final
        assert "4/4 trials" in final
        assert "✕ failed" in final
        assert "remaining 00:00" in final
        assert "running" not in final
        assert _metric_row(final, "First #1") == first_baseline
        assert _metric_row(final, "Best #3") == "Best #3 70.000 (+10.000) 90.000 (+20.000) 90.000 (+10.000)"
        assert _metric_row(final, "Last #4") == "Last #4 65.000 (+5.000) 65.000 (-5.000) 75.000 (-5.000)"
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
    assert callback.first_summary["HOTA"] == 70.0
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 3


def test_baseline_is_the_first_started_trial_even_when_results_finish_out_of_order() -> None:
    callback = TuneWorkflowCallback(total=3, maximize=["HOTA"], minimize=[])
    first = SimpleNamespace(trial_id="first", last_result={"HOTA": 60.0})
    second = SimpleNamespace(trial_id="second", last_result={"HOTA": 70.0})
    third = SimpleNamespace(trial_id="third", last_result={"HOTA": 65.0})
    for trial in (first, second, third):
        callback.on_trial_start(0, [], trial)

    callback.on_trial_complete(0, [], second)
    assert callback.first_summary is None
    assert callback.best_trial_index == callback.last_trial_index == 2

    callback.on_trial_complete(0, [], first)
    assert callback.first_summary == {"HOTA": 60.0}
    assert callback.last_summary == {"HOTA": 60.0}
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 1

    callback.on_trial_complete(0, [], third)
    assert callback.first_summary == {"HOTA": 60.0}
    assert callback.best_summary == {"HOTA": 70.0}
    assert callback.last_summary == {"HOTA": 65.0}
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 3


def test_ray_creation_order_identifies_baseline_when_workers_start_out_of_order() -> None:
    callback = TuneWorkflowCallback(total=2, maximize=["HOTA"], minimize=[])
    first = SimpleNamespace(trial_id="first", last_result={"HOTA": 60.0})
    second = SimpleNamespace(trial_id="second", last_result={"HOTA": 70.0})
    trials = [first, second]
    callback.on_trial_start(0, trials, second)
    callback.on_trial_complete(0, trials, second)
    assert callback.first_summary is None
    assert callback.best_trial_index == callback.last_trial_index == 2

    callback.on_trial_start(0, trials, first)
    callback.on_trial_complete(0, trials, first)
    assert callback.first_summary == {"HOTA": 60.0}
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 1


@pytest.mark.parametrize("started", [False, True])
def test_failed_baseline_is_not_replaced_by_the_next_successful_trial(started: bool) -> None:
    callback = TuneWorkflowCallback(total=2, maximize=["HOTA"], minimize=[])
    first = SimpleNamespace(trial_id="first", last_result={})
    second = SimpleNamespace(trial_id="second", last_result={"HOTA": 70.0})
    if started:
        callback.on_trial_start(0, [first, second], first)
    callback.on_trial_error(0, [first, second], first)
    callback.on_trial_start(0, [], second)
    callback.on_trial_complete(0, [], second)

    assert callback.first_summary is None
    assert callback.best_summary == callback.last_summary == {"HOTA": 70.0}
    assert callback.best_trial_index == callback.last_trial_index == 2
    rendered = capture_renderable(
        format_tune_progress(
            callback.completed,
            callback.total,
            callback.last_summary,
            first_summary=callback.first_summary,
            best_summary=callback.best_summary,
            best_trial=callback.best_trial_index,
            last_trial=callback.last_trial_index,
            failed=callback.failed,
        ),
        width=160,
    )
    assert _metric_row(rendered, "First #1") == "First #1 (baseline) -- -- --"
    assert _metric_row(rendered, "Best #2") == "Best #2 70.000 (--) -- (--) -- (--)"
    assert _metric_row(rendered, "Last #2") == "Last #2 70.000 (--) -- (--) -- (--)"


def test_best_trial_uses_strict_lexicographic_improvement_and_last_keeps_ties() -> None:
    callback = TuneWorkflowCallback(total=4, maximize=["HOTA"], minimize=["IDSW_rate"])
    results = (
        {"HOTA": 70.0, "IDSW_rate": 0.8},
        {"HOTA": 69.0, "IDSW_rate": 0.1},
        {"HOTA": 70.0, "IDSW_rate": 0.2},
        {"HOTA": 70.0, "IDSW_rate": 0.2},
    )
    for index, result in enumerate(results, start=1):
        callback.record_result(result, trial_index=index)
        assert callback.best_trial_index == (1 if index < 3 else 3)
        assert callback.last_trial_index == index
    assert callback.first_summary == results[0]
    assert callback.best_score == (70.0, -0.2)


@pytest.mark.parametrize("result", [{"MOTA": 10.0}, {"HOTA": float("nan")}, {"HOTA": float("inf")}])
def test_unavailable_objective_does_not_replace_a_valid_zero_best(result: dict[str, float]) -> None:
    callback = TuneWorkflowCallback(total=2, maximize=["HOTA"], minimize=[])
    callback.record_result({"HOTA": 0.0}, trial_index=1)
    callback.record_result(result, trial_index=2)

    assert callback.first_summary == {"HOTA": 0.0}
    assert callback.best_summary == {"HOTA": 0.0}
    assert callback.best_trial_index == 1
    assert callback.last_trial_index == 2


@pytest.fixture
def sequence_progress(tmp_path: Path):
    """Connect actual worker snapshots to a driver-local Rich workflow."""
    publications = []
    workflow = SimpleNamespace(
        _lock=threading.RLock(),
        set_detail_renderable=lambda title, renderable: publications.append(renderable),
    )
    callback = TuneWorkflowCallback(total=2, maximize=["HOTA"], minimize=[])
    callback.configure_sequence_progress(tmp_path, {"MOT17-02-FRCNN": 100, "MOT17-04-FRCNN": 50})
    set_tune_progress_workflow(workflow)
    try:
        yield callback, publications, tmp_path
    finally:
        set_tune_progress_workflow(None)


@pytest.mark.parametrize("width", [80, 160])
def test_sequence_progress_updates_before_trial_results_and_resets_for_next_trial(sequence_progress, width):
    callback, publications, directory = sequence_progress
    first = SimpleNamespace(trial_id="first", last_result={"HOTA": 60.0})
    callback.on_trial_start(0, [], first)
    queued = capture_renderable(publications[-1], width=width)
    assert "Trial 1/2" in queued
    assert "MOT17-02-FRCNN" in queued
    assert "MOT17-04-FRCNN" in queued
    assert "0/100 frames" in queued
    assert "0/50 frames" in queued

    writer = TrialSequenceProgressWriter(directory, "first")
    writer(ReplayProgressEvent("MOT17-02-FRCNN", "running", 12, 100, 4, "long sample id", 0))
    callback.on_step_end(0, [first])
    running = capture_renderable(publications[-1], width=width)
    assert "12/100 frames" in running
    assert "0/50 frames" in running
    assert "long sample id" not in running
    assert callback.completed == 0
    assert callback.last_summary is None
    assert max(map(len, running.splitlines())) <= width
    restored = pickle.loads(pickle.dumps(callback))
    assert restored.sequence_events == callback.sequence_events

    writer(ReplayProgressEvent("MOT17-02-FRCNN", "completed", 100, 100, 40, None, 0))
    writer(ReplayProgressEvent("MOT17-04-FRCNN", "completed", 50, 50, 20, None, 1))
    callback.on_trial_complete(0, [], first)
    finished = capture_renderable(publications[-1], width=width)
    assert "2/2 sequences done" in finished
    assert "100/100 frames" in finished
    assert "50/50 frames" in finished
    assert read_trial_sequence_progress(directory, "first") == ()

    second = SimpleNamespace(trial_id="second", last_result={})
    callback.on_trial_start(0, [], second)
    restarted = capture_renderable(publications[-1], width=width)
    assert "Trial 2/2" in restarted
    assert "0/100 frames" in restarted
    assert "0/50 frames" in restarted
    assert _metric_row(restarted, "First #1") == "First #1 (baseline) 60.000 -- --"
    assert _metric_row(restarted, "Best #1").startswith("Best #1 60.000 (+0.000)")
    assert "Trial 1/2" not in restarted
    assert "first" not in callback.sequence_events


def test_concurrent_trials_keep_separate_sequences_and_prune_completed_snapshots(sequence_progress):
    callback, publications, directory = sequence_progress
    trials = [SimpleNamespace(trial_id=name, last_result={}) for name in ("first", "second")]
    for index, trial in enumerate(trials):
        callback.on_trial_start(0, trials, trial)
        writer = TrialSequenceProgressWriter(directory, trial.trial_id)
        writer(ReplayProgressEvent("MOT17-02-FRCNN", "running", 12 + index, 100, 0, None, 0))
    callback.on_step_end(0, trials)
    rendered = capture_renderable(publications[-1], width=160)
    first_rows, second_rows = rendered.split("Trial 2/2")
    assert "12/100 frames" in first_rows
    assert "13/100 frames" not in first_rows
    assert "13/100 frames" in second_rows
    assert "12/100 frames" not in second_rows

    for trial in trials:
        callback.on_trial_complete(0, trials, trial)
    assert set(callback.sequence_events) == {"second"}


@pytest.mark.parametrize("started", [False, True])
def test_failed_and_retried_trials_do_not_leave_stale_sequence_progress(sequence_progress, started):
    callback, publications, directory = sequence_progress
    trial = SimpleNamespace(trial_id="retry", last_result={})
    callback.on_trial_start(0, [], trial)
    if started:
        writer = TrialSequenceProgressWriter(directory, "retry")
        writer(ReplayProgressEvent("MOT17-02-FRCNN", "running", 12, 100, 0, None, 0))
    callback.on_trial_error(0, [], trial)
    failed = capture_renderable(publications[-1], width=160)
    assert "0/2 sequences done · 2 failed" in failed
    assert "pending" not in failed.split("Trial 1/2")[1]

    # A retry clears a stale file even if a previous worker left it behind.
    writer = TrialSequenceProgressWriter(directory, "retry")
    writer(ReplayProgressEvent("MOT17-02-FRCNN", "completed", 100, 100, 0, None, 0))
    callback.on_trial_start(0, [], trial)
    callback.on_step_end(0, [trial])
    restarted = capture_renderable(publications[-1], width=160)
    assert "0/100 frames" in restarted
    assert "0/50 frames" in restarted
    assert "100/100 frames" not in restarted
    assert read_trial_sequence_progress(directory, "retry") == ()


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
