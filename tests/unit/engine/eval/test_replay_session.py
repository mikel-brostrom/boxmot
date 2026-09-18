"""Retained replay processes must never retain per-trial tracker state."""

from __future__ import annotations

import os
import queue
from concurrent.futures.process import BrokenProcessPool

import pytest

from boxmot.engine.eval import motmetrics, replay
from boxmot.engine.eval.session import ReplaySession
from boxmot.trackers import TrackerSpec
from tests.unit.datasets import conftest as dataset_fixtures


@pytest.fixture
def materialized_build(tmp_path):
    return dataset_fixtures.materialized_build.__wrapped__(tmp_path)


def _worker_state(_: object) -> tuple[int, tuple[tuple[str, int], ...]]:
    return os.getpid(), tuple((str(key), id(value)) for key, value in replay._WORKER_INPUTS.items())


def _fail_worker(kind: str) -> None:
    if kind == "broken":
        os._exit(17)
    if kind == "interrupt":
        raise KeyboardInterrupt("cancelled trial")
    raise ValueError("failed trial")


@pytest.mark.parametrize("cache_inputs", [False, True])
def test_repeated_replay_reuses_inputs_with_fresh_tracker_state(materialized_build, tmp_path, cache_inputs) -> None:
    build = materialized_build["root"]
    tracker = TrackerSpec(name="bytetrack")
    baseline = replay.replay_build(build, tracker, output_dir=tmp_path / "baseline", workers=1)
    expected = [path.read_bytes() for path in baseline.sequence_files]
    events = []
    with ReplaySession(1, cache_inputs=cache_inputs) as session:
        first = replay.replay_build(
            build, tracker, output_dir=tmp_path / "first", workers=1, session=session, progress_callback=events.append
        )
        first_run = {event.run_id for event in events}
        first_worker = session.map(_worker_state, [None])[0]
        assert len(first_worker[1]) == 2
        assert len(first_run) == 1 and None not in first_run
        events.clear()
        second = replay.replay_build(
            build, tracker, output_dir=tmp_path / "second", workers=1, session=session, progress_callback=events.append
        )
        second_worker = session.map(_worker_state, [None])[0]
        assert first_worker == second_worker
        assert len({event.run_id for event in events}) == 1
        assert first_run.isdisjoint({event.run_id for event in events})
        assert [path.read_bytes() for path in first.sequence_files] == expected
        assert [path.read_bytes() for path in second.sequence_files] == expected
    assert session._executor is None
    assert session._progress_queue is None
    with pytest.raises(RuntimeError, match="closed"):
        session.map(_worker_state, [None])


@pytest.mark.parametrize(
    ("kind", "error"), [("exception", ValueError), ("interrupt", KeyboardInterrupt), ("broken", BrokenProcessPool)]
)
def test_session_discards_failed_or_cancelled_pool_and_restarts(kind, error) -> None:
    with ReplaySession(1) as session:
        original_pid = session.map(_worker_state, [None])[0][0]
        with pytest.raises(error):
            session.map(_fail_worker, [kind])
        assert session._executor is None
        assert session._progress_queue is None
        replacement_pid = session.map(_worker_state, [None])[0][0]
        assert replacement_pid != original_pid


def test_session_resolves_relative_outputs_after_parent_changes_directory(materialized_build, tmp_path, monkeypatch):
    with ReplaySession(1) as session:
        outputs = []
        for directory in (tmp_path / "one", tmp_path / "two"):
            directory.mkdir()
            monkeypatch.chdir(directory)
            result = replay.replay_build(
                materialized_build["root"],
                TrackerSpec(name="bytetrack"),
                output_dir="relative",
                workers=1,
                session=session,
            )
            assert result.output_dir == directory / "relative"
            outputs.append([path.read_bytes() for path in result.sequence_files])
        assert outputs[0] == outputs[1]


def test_metrics_use_the_session_executor_and_restore_outer_context(monkeypatch) -> None:
    calls = []
    tasks = [object(), object()]
    session = ReplaySession(1)

    def ordered_map(function, values):
        calls.append((function, values))
        return ["first", "second"]

    monkeypatch.setattr(session, "map", ordered_map)
    sentinel = lambda _function, _tasks: ["outer"]
    with motmetrics.use_metric_executor(sentinel):
        with session.metric_execution():
            assert motmetrics._evaluate_sequence_tasks(tasks) == ["first", "second"]
        assert motmetrics._evaluate_sequence_tasks(tasks) == ["outer"]
    assert calls == [(motmetrics._evaluate_sequence_task, tasks)]
    assert motmetrics._METRIC_EXECUTOR.get() is None
    session.close()


def test_progress_discards_previous_trial_messages() -> None:
    events = []
    messages = queue.Queue()
    for run_id in ("previous", "current"):
        messages.put(replay.ReplayProgressEvent("seq", "running", 1, 2, 1, None, 0, run_id))
    replay._drain_progress_queue(messages, events.append, {}, "current")
    assert [event.run_id for event in events] == ["current"]


def test_worker_metadata_is_bounded_and_refreshed_after_source_changes(monkeypatch, tmp_path) -> None:
    created = []

    class Inputs:
        current = True

        def source_is_current(self):
            return self.current

    def open_inputs(*_args, **_kwargs):
        dataset = Inputs()
        created.append(dataset)
        return dataset

    monkeypatch.setattr(replay, "_WORKER_INPUT_LIMIT", 2)
    monkeypatch.setattr(replay.CachedVisionDataset, "_stream_sequence", open_inputs)
    replay._clear_worker_inputs()
    task = replay._SequenceReplayTask(
        str(tmp_path), TrackerSpec(name="bytetrack"), None, "a", 1, "output", 0, run_id="r"
    )
    options = dict(load_images=False, load_masks=False, load_embeddings=False)
    try:
        first = replay._worker_inputs(task, **options)
        assert replay._worker_inputs(task, **options) is first
        first.current = False
        assert replay._worker_inputs(task, **options) is not first
        from dataclasses import replace

        replay._worker_inputs(replace(task, sequence_id="b"), **options)
        replay._worker_inputs(replace(task, sequence_id="c"), **options)
        assert len(replay._WORKER_INPUTS) == 2
        assert len(created) == 4
        assert all(key[1] != "a" for key in replay._WORKER_INPUTS)
    finally:
        replay._clear_worker_inputs()


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_session_rejects_invalid_worker_budget(workers) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ReplaySession(workers)
