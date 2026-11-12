"""Coverage, identity, and lifecycle contracts for background track attributes."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
from dataclasses import replace

import pytest
import torch

from boxmot.attributes import AttributeConfig, AttributeEnricher
from boxmot.attributes.errors import AttributeAnalysisError, TransientAttributeError
from boxmot.attributes.spool import AttributeSpool
from boxmot.structures import Boxes, Frame, Tracks


def _frame(index: int, sequence: str = "video") -> Frame:
    return Frame(torch.zeros((3, 16, 16), dtype=torch.uint8), f"{sequence}/{index}", sequence, index)


def _tracks(frame: Frame, ids: tuple[int, ...], *, detection_index: int = 0) -> Tracks:
    count = len(ids)
    return Tracks(
        Boxes(torch.tensor([[1, 1, 14, 14]] * count, dtype=torch.float32).reshape(-1, 4)),
        torch.tensor(ids, dtype=torch.int64),
        torch.ones(count),
        torch.zeros(count, dtype=torch.int64),
        torch.full((count,), detection_index, dtype=torch.int64),
        frame.sample_id,
    )


def _config(**kwargs) -> AttributeConfig:
    return AttributeConfig(model="fake", prompt="Describe clothing.", schema={"type": "object"}, **kwargs)


class _Backend:
    def __init__(self, *, blocked: bool = False) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()
        if not blocked:
            self.release.set()
        self.calls = []
        self.closed = False

    async def extract(self, track_id, samples, images):
        self.calls.append((track_id, tuple(samples), tuple(images)))
        self.entered.set()
        while not self.release.is_set():
            await asyncio.sleep(0.005)
        return {"source": samples[-1].sample_id, "nested": {"color": "red"}}

    async def aclose(self):
        self.closed = True


def _poll_until(session, count=1):
    deadline = time.monotonic() + 3
    outcomes = []
    while len(outcomes) < count and time.monotonic() < deadline:
        outcomes.extend(session.poll())
        time.sleep(0.005)
    assert len(outcomes) == count
    return outcomes


def test_partial_departed_track_and_eof_are_analyzed(tmp_path):
    backend = _Backend()
    session = AttributeEnricher(_config(), backend=backend, spool_dir=tmp_path)
    try:
        first = _frame(0)
        assert session.update(first, _tracks(first, (7,))) == {}
        assert not backend.calls  # A full batch is not required after disappearance.
        next_frame = _frame(1)
        session.update(next_frame, _tracks(next_frame, ()))
        outcomes = list(session.finish())
        assert len(outcomes) == 1
        assert outcomes[0].track_id == 7
        assert outcomes[0].status == "completed"
        assert outcomes[0].samples[0].frame_index == 0
        assert session.stats["analyzed_tracks"] == 1
    finally:
        session.close()
    assert backend.closed
    assert not session.spool_path.exists()


def test_fifo_refresh_replacement_retains_latest_samples(tmp_path):
    backend = _Backend(blocked=True)
    session = AttributeEnricher(_config(batch_size=1, sample_every=1), backend=backend, spool_dir=tmp_path)
    try:
        first = _frame(0)
        session.update(first, _tracks(first, (0,)))
        assert backend.entered.wait(2)
        for index in range(1, 8):
            frame = _frame(index)
            session.update(frame, _tracks(frame, (1, 2)))
        assert session.pending_count == 3
        with sqlite3.connect(session.spool_path) as db:
            assert db.execute("SELECT max(n) FROM (SELECT count(*) n FROM crops GROUP BY job_id)").fetchone()[0] == 1
        backend.release.set()
        outcomes = list(session.finish())
        assert [call[0] for call in backend.calls] == [0, 1, 2]
        assert [(event.track_id, event.samples[-1].frame_index) for event in outcomes] == [(0, 0), (1, 7), (2, 7)]
    finally:
        session.close()


def test_overload_preserves_thousands_of_short_tracks_on_disk(tmp_path, monkeypatch):
    # Avoid image encoding cost: this test exercises spool capacity, not crops.
    monkeypatch.setattr("boxmot.attributes.session.encode_track_crop", lambda *_args: b"small JPEG")
    backend = _Backend(blocked=True)
    session = AttributeEnricher(_config(batch_size=1), backend=backend, spool_dir=tmp_path)
    try:
        first = _frame(0)
        session.update(first, _tracks(first, (0,)))
        assert backend.entered.wait(2)
        frame = _frame(1)
        ids = tuple(range(1, 2001))
        session.update(frame, _tracks(frame, ids))
        assert session.pending_count == 2001
        with sqlite3.connect(session.spool_path) as db:
            assert db.execute("SELECT count(*) FROM crops").fetchone()[0] == 2001
            assert db.execute("SELECT count(*) FROM jobs WHERE state = 'running'").fetchone()[0] == 1
        backend.release.set()
        completed_ids = {event.track_id for event in session.finish() if event.status == "completed"}
        assert completed_ids == set(range(2001))
        assert session.stats["tracks"] == session.stats["analyzed_tracks"] == 2001
    finally:
        session.close()


def test_resets_isolate_late_results_even_with_same_sequence_and_id(tmp_path):
    backend = _Backend(blocked=True)
    session = AttributeEnricher(_config(batch_size=1, sample_every=1000), backend=backend, spool_dir=tmp_path)
    try:
        frame = _frame(10)
        session.update(frame, _tracks(frame, (1,)))
        assert backend.entered.wait(2)
        session.reset()
        second = _frame(0)
        assert session.update(second, _tracks(second, (1,))) == {}
        backend.release.set()
        events = _poll_until(session, 2)
        assert {event.generation for event in events} == {0, 1}
        later = _frame(1)
        snapshot = session.update(later, _tracks(later, (1,)))
        assert snapshot[1].generation == 1
        assert snapshot[1].values["source"] == "video/0"
        snapshot[1].values["nested"]["color"] = "changed by caller"
        last = _frame(2)
        next_snapshot = session.update(last, _tracks(last, (1,)))
        assert next_snapshot[1].values["nested"]["color"] == "red"
        assert snapshot[1].values["nested"]["color"] == "changed by caller"
    finally:
        session.close()


def test_late_completion_cannot_populate_new_generation_before_new_result(tmp_path):
    second_entered = threading.Event()
    second_release = threading.Event()

    class Backend(_Backend):
        async def extract(self, track_id, samples, images):
            if self.calls:
                second_entered.set()
                while not second_release.is_set():
                    await asyncio.sleep(0.005)
            return await super().extract(track_id, samples, images)

    backend = Backend(blocked=True)
    session = AttributeEnricher(_config(batch_size=1, sample_every=1000), backend=backend, spool_dir=tmp_path)
    try:
        frame = _frame(0, "first")
        session.update(frame, _tracks(frame, (1,)))
        assert backend.entered.wait(2)
        session.reset()
        frame = _frame(0, "second")
        session.update(frame, _tracks(frame, (1,)))
        backend.release.set()
        assert second_entered.wait(2)
        event = _poll_until(session)[0]
        assert event.sequence_id == "first"
        current = _frame(1, "second")
        assert session.update(current, _tracks(current, (1,))) == {}
        second_release.set()
        assert len(list(session.finish())) == 1
    finally:
        session.close()


@pytest.mark.parametrize(
    ("retry_after_s", "expected_delay"), [(None, 1), (0, 1), (0.25, 1), (10, 10), (60, 60)]
)
def test_transient_retry_yields_to_other_tracks_and_preserves_previous_values(
    tmp_path, monkeypatch, retry_after_s: float | None, expected_delay: float
) -> None:
    """Respect provider cooldowns while retaining the queue's per-track fairness."""
    delays = []
    original_retry = AttributeSpool.retry

    def record_retry(spool: AttributeSpool, job, delay: float) -> None:
        """Capture the deadline passed to the real spool without a wall-clock wait."""
        delays.append(delay)
        # Record the worker's requested delay without making the test sleep.
        original_retry(spool, job, delay=0)

    monkeypatch.setattr(AttributeSpool, "retry", record_retry)

    class Backend(_Backend):
        async def extract(self, track_id, samples, images):
            result = await super().extract(track_id, samples, images)
            if track_id == 1 and len([call for call in self.calls if call[0] == 1]) == 1:
                raise TransientAttributeError("temporarily unavailable", retry_after_s=retry_after_s)
            return result

    backend = Backend()
    with AttributeEnricher(_config(batch_size=1), backend=backend, spool_dir=tmp_path) as session:
        frame = _frame(0)
        session.update(frame, _tracks(frame, (1, 2)))
        events = list(session.finish())
        assert delays == [expected_delay]
        assert [call[0] for call in backend.calls] == [1, 2, 1]
        assert [event.track_id for event in events] == [2, 1]
        assert all(event.status == "completed" for event in events)


def test_failed_refresh_keeps_previous_snapshot(tmp_path):
    class Backend(_Backend):
        async def extract(self, track_id, samples, images):
            result = await super().extract(track_id, samples, images)
            if len(self.calls) > 1:
                raise AttributeAnalysisError("invalid JSON")
            return result

    backend = Backend()
    with AttributeEnricher(_config(batch_size=1, sample_every=1), backend=backend, spool_dir=tmp_path) as session:
        first = _frame(0)
        session.update(first, _tracks(first, (1,)))
        _poll_until(session)
        refresh = _frame(1)
        old = session.update(refresh, _tracks(refresh, (1,)))
        assert _poll_until(session)[0].status == "failed"
        prediction = _frame(2)
        latest = session.update(prediction, _tracks(prediction, (1,), detection_index=-1))
        assert latest[1].values == old[1].values
        assert latest[1].samples[-1].frame_index == 0


def test_cancellation_is_prompt_and_records_running_and_partial_jobs(tmp_path):
    backend = _Backend(blocked=True)
    session = AttributeEnricher(_config(batch_size=1, request_timeout_s=120), backend=backend, spool_dir=tmp_path)
    frame = _frame(0)
    session.update(frame, _tracks(frame, (1, 2)))
    assert backend.entered.wait(2)
    started = time.monotonic()
    session.cancel()
    events = session.poll()
    session.close()
    assert time.monotonic() - started < 2
    assert backend.closed
    assert {event.track_id for event in events} == {1, 2}
    assert {event.status for event in events} == {"cancelled"}
    assert session.pending_count == 0


def test_no_visual_observation_is_reported_and_class_filter_is_respected(tmp_path):
    with AttributeEnricher(_config(class_ids=(0,)), backend=_Backend(), spool_dir=tmp_path) as session:
        frame = _frame(0)
        tracks = replace(_tracks(frame, (1, 2), detection_index=-1), class_ids=torch.tensor([0, 3]))
        session.update(frame, tracks)
        events = list(session.finish())
        assert [(event.track_id, event.status) for event in events] == [(1, "ineligible")]


def test_session_input_order_and_completion_contracts(tmp_path):
    with AttributeEnricher(_config(), backend=_Backend(), spool_dir=tmp_path) as session:
        frame = _frame(0)
        session.update(frame, _tracks(frame, ()))
        with pytest.raises(ValueError, match="increase"):
            session.update(frame, _tracks(frame, ()))
        other = _frame(1, "other")
        with pytest.raises(ValueError, match="Reset"):
            session.update(other, _tracks(other, ()))
        list(session.finish())
        with pytest.raises(RuntimeError, match="after finishing"):
            session.update(other, _tracks(other, ()))


def test_spool_write_failure_propagates_instead_of_dropping_work(tmp_path, monkeypatch):
    session = AttributeEnricher(_config(), backend=_Backend(), spool_dir=tmp_path)

    def fail(*_args):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(session._spool, "append", fail)
    try:
        frame = _frame(0)
        with pytest.raises(sqlite3.OperationalError, match="disk is full"):
            session.update(frame, _tracks(frame, (1,)))
    finally:
        session.close()


def test_background_spool_failure_is_reported(tmp_path, monkeypatch):
    backend = _Backend(blocked=True)
    session = AttributeEnricher(_config(batch_size=1), backend=backend, spool_dir=tmp_path)
    original = session._spool.complete

    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    try:
        frame = _frame(0)
        session.update(frame, _tracks(frame, (1,)))
        assert backend.entered.wait(2)
        monkeypatch.setattr(session._spool, "complete", fail)
        backend.release.set()
        with pytest.raises(RuntimeError, match="worker failed") as exc:
            list(session.finish())
        assert isinstance(exc.value.__cause__, sqlite3.OperationalError)
    finally:
        monkeypatch.setattr(session._spool, "complete", original)
        with pytest.raises(RuntimeError, match="worker failed"):
            session.close()
    assert not session.spool_path.exists()


def test_disk_failure_during_cancellation_still_stops_request(tmp_path, monkeypatch):
    backend = _Backend(blocked=True)
    session = AttributeEnricher(_config(batch_size=1), backend=backend, spool_dir=tmp_path)
    frame = _frame(0)
    session.update(frame, _tracks(frame, (1,)))
    assert backend.entered.wait(2)

    def fail(*_args, **_kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(session._spool, "cancel", fail)
    started = time.monotonic()
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        session.cancel()
    with pytest.raises(sqlite3.OperationalError, match="disk is full"):
        session.close()
    assert backend.closed
    assert not session._thread.is_alive()
    assert not session.spool_path.exists()
    assert time.monotonic() - started < 2


def test_cancel_interrupts_a_backend_waiting_on_a_long_io_operation(tmp_path):
    class Backend(_Backend):
        async def extract(self, track_id, samples, images):
            self.entered.set()
            await asyncio.sleep(120)
            return {}

    backend = Backend()
    session = AttributeEnricher(_config(batch_size=1), backend=backend, spool_dir=tmp_path)
    frame = _frame(0)
    session.update(frame, _tracks(frame, (1,)))
    assert backend.entered.wait(2)
    started = time.monotonic()
    session.cancel()
    assert session.poll()[0].status == "cancelled"
    session.close()
    assert backend.closed
    assert time.monotonic() - started < 2


def test_direct_api_revalidates_mutated_canonical_tracks(tmp_path):
    with AttributeEnricher(_config(), backend=_Backend(), spool_dir=tmp_path) as session:
        frame = _frame(0)
        tracks = _tracks(frame, (1, 2))
        tracks.track_ids[1] = 1
        with pytest.raises(ValueError, match="unique"):
            session.update(frame, tracks)
        assert session.stats["tracks"] == 0


def test_request_timeout_has_finite_retries_without_blocking_other_tracks(tmp_path):
    class Backend(_Backend):
        async def extract(self, track_id, samples, images):
            self.calls.append(track_id)
            if track_id == 1:
                await asyncio.sleep(10)
            return {"track_color": "blue"}

    backend = Backend()
    config = _config(batch_size=1, request_timeout_s=0.02, retries=1)
    with AttributeEnricher(config, backend=backend, spool_dir=tmp_path) as session:
        frame = _frame(0)
        session.update(frame, _tracks(frame, (1, 2)))
        events = list(session.finish())
        assert backend.calls == [1, 2, 1]
        assert [(event.track_id, event.status) for event in events] == [(2, "completed"), (1, "failed")]
        assert "timed out" in events[-1].error
        assert session.pending_count == 0
