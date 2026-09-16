"""Trial sequence progress stays visible without producing partial trial results."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from boxmot.engine.eval.replay import ReplayProgressEvent
from boxmot.engine.tuning import progress as progress_module
from boxmot.engine.tuning import tuner
from boxmot.engine.tuning.progress import (
    TrialSequenceProgressWriter,
    clear_trial_sequence_progress,
    read_trial_sequence_progress,
)
from boxmot.engine.tuning.trainable import build_tracker_trainable


def _event(sequence: str = "MOT17-02", ordinal: int = 0, **kwargs) -> ReplayProgressEvent:
    """Build a replay event with enough frames to exercise repeated updates."""
    return ReplayProgressEvent(
        **{
            "sequence_id": sequence,
            "status": "running",
            "completed": 1,
            "total": 1000,
            "track_rows": 1,
            "detail": None,
            "ordinal": ordinal,
            **kwargs,
        }
    )


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        raw={"HOTA": 50.0}, benchmark="fixture", summary_label="all", summary={"HOTA": 50.0}, timings={}, exp_dir=None
    )


def test_progress_is_bounded_throttled_and_atomically_replaced(tmp_path, monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(progress_module.time, "monotonic", lambda: clock[0])
    replacements = []
    atomic_replace = progress_module.os.replace

    def capture_replace(source, destination) -> None:
        assert source.parent == destination.parent
        replacements.append(read_trial_sequence_progress(tmp_path, "trial-a"))
        atomic_replace(source, destination)

    monkeypatch.setattr(progress_module.os, "replace", capture_replace)
    writer = TrialSequenceProgressWriter(tmp_path, "trial-a", interval=1.0)
    writer(_event())
    for completed in range(2, 501):
        writer(_event(completed=completed))
    assert len(replacements) == 2
    assert read_trial_sequence_progress(tmp_path, "trial-a")[0].completed == 1

    clock[0] = 1.0
    writer(_event(completed=501))
    assert read_trial_sequence_progress(tmp_path, "trial-a")[0].completed == 501
    assert replacements[-1][0].completed == 1
    writer(_event(completed=1000, status="completed"))
    assert read_trial_sequence_progress(tmp_path, "trial-a")[0].status == "completed"
    assert len(list(tmp_path.iterdir())) == 1
    assert len(read_trial_sequence_progress(tmp_path, "trial-a")) == 1


def test_trial_snapshots_are_isolated_sorted_and_reset_for_retry(tmp_path) -> None:
    first = TrialSequenceProgressWriter(tmp_path, "trial-a")
    second = TrialSequenceProgressWriter(tmp_path, "trial-b")
    first(_event("MOT17-04", ordinal=1))
    first(_event("MOT17-02", ordinal=0))
    second(_event("MOT17-05", ordinal=0))
    assert [event.sequence_id for event in read_trial_sequence_progress(tmp_path, "trial-a")] == [
        "MOT17-02",
        "MOT17-04",
    ]
    assert read_trial_sequence_progress(tmp_path, "trial-b")[0].sequence_id == "MOT17-05"
    TrialSequenceProgressWriter(tmp_path, "trial-a")
    assert read_trial_sequence_progress(tmp_path, "trial-a") == ()
    clear_trial_sequence_progress(tmp_path, "trial-a")
    clear_trial_sequence_progress(tmp_path, "trial-a")
    assert len(list(tmp_path.iterdir())) == 1
    assert read_trial_sequence_progress(tmp_path, "trial-b")[0].sequence_id == "MOT17-05"


@pytest.mark.parametrize("content", ["{", "[]", '{"trial_id":"other"}', '{"trial_id":"trial-a","sequences":[{}]}'])
def test_missing_or_invalid_snapshot_is_observational(tmp_path, content) -> None:
    assert read_trial_sequence_progress(tmp_path, "trial-a") == ()
    TrialSequenceProgressWriter(tmp_path, "trial-a")
    snapshot = next(tmp_path.iterdir())
    snapshot.write_text(content)
    assert read_trial_sequence_progress(tmp_path, "trial-a") == ()


def test_unwritable_snapshot_storage_does_not_interrupt_replay(tmp_path, monkeypatch) -> None:
    writer = TrialSequenceProgressWriter(tmp_path, "trial-a")

    def unavailable(*_args) -> None:
        raise OSError("storage unavailable")

    monkeypatch.setattr(progress_module.os, "replace", unavailable)
    writer(_event())
    writer.flush()
    assert read_trial_sequence_progress(tmp_path, "trial-a") == ()
    assert len(list(tmp_path.iterdir())) == 1


def test_retired_progress_directory_is_not_recreated_by_a_late_actor(tmp_path) -> None:
    directory = tmp_path / "progress"
    directory.mkdir()
    writer = TrialSequenceProgressWriter(directory, "trial-a")
    clear_trial_sequence_progress(directory, "trial-a")
    directory.rmdir()
    writer(_event())
    writer.flush()
    TrialSequenceProgressWriter(directory, "trial-b")
    assert not directory.exists()


def test_reused_actor_streams_each_trial_before_returning_only_final_score(tmp_path, monkeypatch) -> None:
    sessions = []
    options = SimpleNamespace(sequence_workers=1, cache_inputs=True, _tune_sequence_progress_dir=tmp_path)
    actor = build_tracker_trainable(SimpleNamespace(Trainable=object), options)()

    def evaluate(_args, *, evolve_config, replay_session, progress_callback, **kwargs):
        assert kwargs["show_progress"] is False
        assert read_trial_sequence_progress(tmp_path, actor.trial_id) == ()
        sessions.append(replay_session)
        event = _event(evolve_config["sequence"], completed=20)
        progress_callback(event)
        assert read_trial_sequence_progress(tmp_path, actor.trial_id) == (event,)
        progress_callback(replace(event, completed=1000, status="completed"))
        return _result()

    monkeypatch.setattr(tuner, "run_eval", evaluate)
    monkeypatch.setattr(tuner, "aggregate_results", dict)
    actor.setup({})
    try:
        for trial_id, sequence in [("trial-a", "MOT17-02"), ("trial-b", "MOT17-04")]:
            actor.trial_id = trial_id
            actor.reset_config({"sequence": sequence})
            result = actor.step()
            assert result["HOTA"] == 50.0 and result["done"] is True
            assert read_trial_sequence_progress(tmp_path, trial_id)[0].sequence_id == sequence
            assert read_trial_sequence_progress(tmp_path, trial_id)[0].status == "completed"
        assert sessions[0] is sessions[1]
        assert read_trial_sequence_progress(tmp_path, "trial-a")[0].sequence_id == "MOT17-02"
    finally:
        actor.cleanup()


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt])
def test_failed_trial_publishes_failure_and_closes_retained_workers(tmp_path, monkeypatch, error) -> None:
    sessions = []

    def evaluate(_args, *, replay_session, progress_callback, **kwargs):
        sessions.append(replay_session)
        progress_callback(_event("MOT17-02", status="completed", completed=1000))
        progress_callback(_event("MOT17-04", ordinal=1))
        progress_callback(_event("MOT17-05", ordinal=2, status="queued", completed=0))
        raise error("evaluation failed")

    monkeypatch.setattr(tuner, "run_eval", evaluate)
    actor = build_tracker_trainable(
        SimpleNamespace(Trainable=object), SimpleNamespace(sequence_workers=1, _tune_sequence_progress_dir=tmp_path)
    )()
    actor.trial_id = "trial-a"
    actor.setup({})
    actor.reset_config({})
    with pytest.raises(error, match="evaluation failed"):
        actor.step()
    events = read_trial_sequence_progress(tmp_path, "trial-a")
    assert [event.status for event in events] == ["completed", "failed", "failed"]
    assert all(event.detail for event in events[1:])
    assert sessions[0]._closed
    actor.cleanup()


@pytest.mark.parametrize("emit_first", [False, True])
@pytest.mark.parametrize("totals_field", ["sequence_frame_counts", "seq_info"])
def test_failure_reports_configured_sequences_without_replay_events(
    tmp_path, monkeypatch, emit_first, totals_field
) -> None:
    def evaluate(_args, *, progress_callback, **kwargs):
        if emit_first:
            progress_callback(_event("MOT17-04", ordinal=0, status="completed", completed=1000))
        raise ValueError("setup failed")

    monkeypatch.setattr(tuner, "run_eval", evaluate)
    options = SimpleNamespace(sequence_workers=1, **{totals_field: {"MOT17-02": 500, "MOT17-04": 1000}})
    objective = tuner.TrackerObjective(options)
    writer = TrialSequenceProgressWriter(tmp_path, "trial-a")
    try:
        with pytest.raises(ValueError, match="setup failed"):
            objective({}, progress_callback=writer)
    finally:
        objective.close()
    events = {event.sequence_id: event for event in read_trial_sequence_progress(tmp_path, "trial-a")}
    assert set(events) == {"MOT17-02", "MOT17-04"}
    assert events["MOT17-02"].status == "failed"
    assert events["MOT17-02"].total == 500
    assert events["MOT17-02"].detail == "ValueError: setup failed"
    assert events["MOT17-04"].status == ("completed" if emit_first else "failed")
    assert events["MOT17-04"].total == 1000


def test_progress_callback_failure_does_not_change_objective_metrics(monkeypatch) -> None:
    def report(_event: ReplayProgressEvent) -> None:
        raise OSError("progress unavailable")

    def evaluate(_args, *, progress_callback, **kwargs):
        progress_callback(_event())
        return _result()

    monkeypatch.setattr(tuner, "run_eval", evaluate)
    monkeypatch.setattr(tuner, "aggregate_results", dict)
    objective = tuner.TrackerObjective(SimpleNamespace(sequence_workers=1))
    try:
        assert objective({}, progress_callback=report)["HOTA"] == 50.0
    finally:
        objective.close()
