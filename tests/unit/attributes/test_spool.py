"""Exercise PostgreSQL isolation, transactions, and retained attribute outcomes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.models import AttributeSample
from boxmot.attributes.spool import AttributeSpool
from boxmot.utils.config import ConfigurationError


def _spool() -> AttributeSpool:
    """Create an independent run in the test database."""
    return AttributeSpool(
        AttributeConfig(model="fixture", prompt="Describe", schema={"type": "object"}, batch_size=1),
        uuid4().hex,
    )


def _enqueue(spool: AttributeSpool, track_id: int = 7, image: bytes = b"jpeg") -> None:
    """Submit one ready crop using the same generation and IDs across runs."""
    sample = AttributeSample("video/0", 0, 0.0)
    with spool.transaction():
        spool.observe(0, "video", track_id, 0, 1, sample)
        spool.append(0, track_id, 1, sample, image)


def test_database_url_is_required_without_creating_a_local_fallback(monkeypatch) -> None:
    """Missing database configuration fails before any storage is initialized."""
    monkeypatch.delenv("BOXMOT_ATTRIBUTES_DATABASE_URL", raising=False)
    with pytest.raises(ConfigurationError, match="BOXMOT_ATTRIBUTES_DATABASE_URL"):
        _spool()


def test_invalid_database_url_does_not_expose_credentials(monkeypatch) -> None:
    """Driver diagnostics must not leak an authored connection string."""
    pytest.importorskip("psycopg")
    monkeypatch.setenv("BOXMOT_ATTRIBUTES_DATABASE_URL", "password=private-test-secret invalid-connection")
    with pytest.raises(ConfigurationError, match="PostgreSQL initialization failed") as error:
        _spool()
    assert "private-test-secret" not in str(error.value)
    assert "invalid-connection" not in str(error.value)
    assert error.value.__suppress_context__


def test_runs_isolate_identical_tracks_and_retain_history_after_close(attribute_database) -> None:
    """Polling or closing one run must not erase or expose another run's values."""
    first, second = _spool(), _spool()
    try:
        _enqueue(first, image=b"first")
        _enqueue(second, image=b"second")
        first_job, second_job = first.claim(), second.claim()
        assert first_job.images == (b"first",)
        assert second_job.images == (b"second",)
        assert first.claim() is None
        first.complete(first_job, values={"color": "red"})
        assert second.latest(0, [7]) == {}
        assert second.poll(10) == []
        second.complete(second_job, values={"color": "blue"})
        assert first.poll(10)[0].values == {"color": "red"}
        assert first.poll(10) == []
        assert second.poll(10)[0].values == {"color": "blue"}
        assert first.latest(0, [7])[7].values == {"color": "red"}
        assert second.latest(0, [7])[7].values == {"color": "blue"}
    finally:
        first.close()
        second.close()
    for spool, color in [(first, "red"), (second, "blue")]:
        row = attribute_database.execute(
            "SELECT latest FROM boxmot_attributes.tracks WHERE run_id = %s", (spool.run_id,),
        ).fetchone()
        assert row["latest"]["values"] == {"color": color}
        events = attribute_database.execute(
            "SELECT payload, delivered_at FROM boxmot_attributes.events WHERE run_id = %s", (spool.run_id,),
        ).fetchall()
        assert len(events) == 1 and events[0]["delivered_at"] is not None
        assert events[0]["payload"]["values"] == {"color": color}
        assert attribute_database.execute(
            "SELECT count(*) AS n FROM boxmot_attributes.crops WHERE run_id = %s", (spool.run_id,),
        ).fetchone()["n"] == 0


def test_frame_transaction_rolls_back_all_crops_and_allows_retry(attribute_database) -> None:
    """A failed frame cannot leave a partly observed track or orphaned crop."""
    spool = _spool()
    try:
        with pytest.raises(ValueError, match="frame failed"):
            with spool.transaction():
                _enqueue(spool)
                raise ValueError("frame failed")
        assert spool.pending_count == 0
        assert spool.stats["tracks"] == 0
        _enqueue(spool)
        job = spool.claim()
        assert job.track_id == 7
        spool.complete(job, values={"color": "red"})
        assert spool.poll(10)[0].status == "completed"
    finally:
        spool.close()


def test_retry_uses_database_deadline_and_yields_to_other_tracks(attribute_database) -> None:
    """A cooldown must survive scheduling without a process-local monotonic epoch."""
    spool = _spool()
    try:
        _enqueue(spool, 1)
        _enqueue(spool, 2)
        first = spool.claim()
        assert first.track_id == 1
        spool.retry(first, delay=60)
        remaining = attribute_database.execute(
            "SELECT extract(epoch FROM available - clock_timestamp()) AS seconds "
            "FROM boxmot_attributes.jobs WHERE run_id = %s AND id = %s", (spool.run_id, first.id),
        ).fetchone()["seconds"]
        assert 40 < remaining <= 60
        second = spool.claim()
        assert second.track_id == 2
        spool.complete(second, values={})
        assert spool.claim() is None
        attribute_database.execute(
            "UPDATE boxmot_attributes.jobs SET available = clock_timestamp() - interval '1 second' "
            "WHERE run_id = %s AND id = %s", (spool.run_id, first.id),
        )
        retried = spool.claim()
        assert retried.track_id == 1 and retried.attempts == 2
        spool.complete(retried, values={})
        assert [event.track_id for event in spool.poll(10)] == [2, 1]
    finally:
        spool.close()


def test_concurrent_sessions_initialize_and_claim_independent_runs(attribute_database) -> None:
    """Concurrent startup and identical track IDs remain safe in one schema."""
    def run(index: int) -> str:
        """Exercise a complete independent database session from another thread."""
        spool = _spool()
        try:
            _enqueue(spool)
            spool.complete(spool.claim(), values={"index": index})
            assert spool.poll(10)[0].values == {"index": index}
            return spool.run_id
        finally:
            spool.close()

    with ThreadPoolExecutor(max_workers=4) as executor:
        runs = list(executor.map(run, range(4)))
    assert len(set(runs)) == 4
