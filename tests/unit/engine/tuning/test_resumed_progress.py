"""Resumed tuning preserves its original baseline and trial identities."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Group

from boxmot.engine.tuning.tuner import Tuner
from boxmot.engine.ui.core.ui import capture_renderable
from boxmot.engine.ui.reporters.tune import TuneWorkflowCallback, set_tune_progress_workflow


@pytest.fixture
def progress() -> Iterator[tuple[TuneWorkflowCallback, list[Group]]]:
    """Capture publications without constructing a Ray runtime."""
    publications: list[Group] = []
    workflow = SimpleNamespace(set_detail_renderable=lambda _step, renderable: publications.append(renderable))
    callback = TuneWorkflowCallback(total=8, maximize=["HOTA"], minimize=[])
    set_tune_progress_workflow(workflow)
    try:
        yield callback, publications
    finally:
        set_tune_progress_workflow(None)


def _result(tmp_path: Path, directory_id: str, *, error: Exception | None = None, **metrics: Any) -> SimpleNamespace:
    """Represent the public fields used from Ray's stored ResultGrid."""
    metrics.setdefault("trial_id", directory_id)
    return SimpleNamespace(path=str(tmp_path / f"trial_{directory_id}"), metrics=metrics, error=error)


def _restore(
    tmp_path: Path,
    callback: TuneWorkflowCallback,
    results: list[SimpleNamespace],
    *,
    trial_order: list[str] | None = None,
    search_alg: str = "optuna",
    include_metadata: bool = True,
    completion_times: dict[str, float] | None = None,
) -> SimpleNamespace:
    """Install a new driver callback using stored experiment results."""
    class StoredResults(list):
        """Retain trial metadata alongside the public ResultGrid iteration."""

    grid = StoredResults(results)
    if include_metadata:
        order = trial_order or [result.metrics["trial_id"] for result in results]
        grid._experiment_analysis = SimpleNamespace(
            trials=[
                SimpleNamespace(
                    trial_id=trial_id,
                    experiment_tag=f"{index}_sampled=value",
                    run_metadata=SimpleNamespace(last_result_time=completion_times[trial_id])
                    if completion_times is not None else None,
                )
                for index, trial_id in enumerate(order, start=0 if search_alg == "random" else 1)
            ]
        )
    tuner = SimpleNamespace(
        get_results=lambda: grid,
        _local_tuner=SimpleNamespace(_run_config=SimpleNamespace()),
    )
    Tuner(SimpleNamespace(n_trials=8, search_alg=search_alg))._inject_callback_into_restored(
        tuner, callback, None, tmp_path
    )
    return tuner


def test_resume_restores_baseline_best_and_latest_completion_with_original_indices(
    tmp_path: Path, progress: tuple[TuneWorkflowCallback, list[Group]]
) -> None:
    callback, publications = progress
    results = [
        _result(tmp_path, "first", trial_id="first", HOTA=60.0, timestamp=20, done=True, time_total_s=8.0),
        _result(tmp_path, "best", trial_id="best", HOTA=70.0, timestamp=10, done=True, time_total_s=4.0),
        _result(tmp_path, "latest", trial_id="latest", HOTA=65.0, timestamp=30, done=True, time_total_s=6.0),
        _result(tmp_path, "pending"),
    ]

    tuner = _restore(tmp_path, callback, results)

    assert callback.first_summary == {"HOTA": 60.0}
    assert callback.best_summary == {"HOTA": 70.0}
    assert callback.last_summary == {"HOTA": 65.0}
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 3
    assert callback.completed == 3
    assert callback.trial_durations == [4.0, 8.0, 6.0]
    assert callback.trial_indices == {"first": 1, "best": 2, "latest": 3, "pending": 4}
    assert tuner._local_tuner._run_config.callbacks == [callback]
    rendered = capture_renderable(publications[-1], width=160)
    assert "First #1 (baseline)" in rendered
    assert "Best #2" in rendered
    assert "Last #3" in rendered
    assert "+10.000" in rendered
    assert "+5.000" in rendered

    pending = SimpleNamespace(trial_id="pending", last_result={"HOTA": 80.0})
    callback.on_trial_start(0, [], pending)
    callback.on_trial_complete(0, [], pending)
    new = SimpleNamespace(trial_id="new", last_result={})
    callback.on_trial_start(0, [], new)

    assert callback.completed == 4
    assert callback.best_trial_index == 4
    assert callback.last_trial_index == 4
    assert callback.trial_indices["new"] == 5
    assert callback.first_summary == {"HOTA": 60.0}


def test_resume_does_not_replace_failed_baseline_or_count_trials_that_will_restart(
    tmp_path: Path, progress: tuple[TuneWorkflowCallback, list[Group]]
) -> None:
    callback, _publications = progress
    results = [
        _result(tmp_path, "baseline", error=RuntimeError("failed"), HOTA=90.0, done=True),
        _result(tmp_path, "complete", HOTA=60.0, timestamp=20, done=True),
        _result(tmp_path, "interrupted", HOTA=99.0, done=False),
    ]

    _restore(tmp_path, callback, results)

    assert callback.completed == 1
    assert callback.failed == 0
    assert callback.first_summary is None
    assert callback.best_summary == {"HOTA": 60.0}
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 2
    baseline = SimpleNamespace(trial_id="baseline", last_result={"HOTA": 50.0})
    callback.on_trial_start(0, [], baseline)
    callback.on_trial_complete(0, [], baseline)

    assert callback.first_summary == {"HOTA": 50.0}
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 1
    assert callback.completed == 2


def test_resume_keeps_first_trial_as_last_when_it_finished_after_later_trials(
    tmp_path: Path, progress: tuple[TuneWorkflowCallback, list[Group]]
) -> None:
    callback, _publications = progress
    results = [
        _result(tmp_path, "first", HOTA=60.0, timestamp=30, done=True),
        _result(tmp_path, "second", HOTA=65.0, timestamp=10, done=True),
    ]

    _restore(tmp_path, callback, results)

    assert callback.first_summary == callback.last_summary == {"HOTA": 60.0}
    assert callback.last_trial_index == 1
    assert callback.best_trial_index == 2


@pytest.mark.parametrize("search_alg", ["optuna", "random"])
def test_resume_uses_submission_tags_when_checkpoint_order_is_shuffled(
    tmp_path: Path, progress: tuple[TuneWorkflowCallback, list[Group]], search_alg: str
) -> None:
    callback, _publications = progress
    results = [
        _result(tmp_path, "best", HOTA=75.0, timestamp=10, done=True),
        _result(tmp_path, "pending"),
        _result(tmp_path, "baseline", HOTA=60.0, timestamp=10, done=True),
    ]

    _restore(
        tmp_path,
        callback,
        results,
        trial_order=["baseline", "best", "pending"],
        search_alg=search_alg,
        completion_times={"baseline": 10.8, "best": 10.2, "pending": 0.0},
    )

    assert callback.first_summary == callback.last_summary == {"HOTA": 60.0}
    assert callback.best_trial_index == 2
    assert callback.last_trial_index == 1
    assert callback.completed == 2
    assert callback.trial_indices == {"best": 2, "pending": 3, "baseline": 1}
    assert callback._trial_index_offset == 0


def test_resume_without_submission_metadata_does_not_invent_a_baseline(
    tmp_path: Path, progress: tuple[TuneWorkflowCallback, list[Group]]
) -> None:
    callback, _publications = progress
    results = [_result(tmp_path, "unknown", HOTA=65.0, timestamp=10, done=True)]

    _restore(tmp_path, callback, results, include_metadata=False)

    assert callback.completed == 1
    assert callback.first_summary is None
    assert callback.best_summary is None
    assert callback.last_summary is None
    assert callback.trial_indices == {}
    assert callback._trial_index_offset == 1
