"""Real PostgreSQL contracts for continuous camera ownership and retention."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from uuid import uuid4

import pytest

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.database import ensure_schema
from boxmot.attributes.models import AttributeSample
from boxmot.attributes.spool import AttributeSpool
from boxmot.attributes.streams import CameraJournal, CameraLease, OwnershipLost, maintain_database

ENV = "BOXMOT_ATTRIBUTES_DATABASE_URL"


@pytest.fixture
def camera(attribute_database):
    """Allocate isolated owners and close every connection before fixture cleanup."""
    camera_id = "test-" + uuid4().hex
    with ExitStack() as resources:

        def owner(**kwargs):
            lease = CameraLease(camera_id, ENV, renew_s=kwargs.pop("renew_s", 60), **kwargs)
            resources.callback(lease.close)
            return lease

        def spool(lease, **kwargs):
            config = AttributeConfig(model="fake", prompt="Describe", schema={"type": "object"}, batch_size=1)
            result = AttributeSpool(config, uuid4().hex, ownership=lease, **kwargs)
            resources.callback(result.close)
            return result

        def journal(lease):
            result = CameraJournal(lease)
            resources.callback(result.close)
            return result

        yield owner, spool, journal
    attribute_database.execute("DELETE FROM boxmot_attributes.camera_leases WHERE camera_id=%s", (camera_id,))


def enqueue(spool, track_id=7, image=b"jpeg"):
    sample = AttributeSample("sample", 0, 0.0)
    with spool.transaction():
        spool.observe(1, "camera", track_id, 0, 1, sample)
        spool.append(1, track_id, 1, sample, image)


def expire(db, owner):
    db.execute(
        "UPDATE boxmot_attributes.camera_leases SET expires_at=clock_timestamp()-interval '1 second' "
        "WHERE camera_id=%s",
        (owner.camera_id,),
    )


def test_competing_owner_takeover_preserves_completed_and_cancels_pending(camera, attribute_database):
    owner, spool, journal = camera
    first = owner()
    with pytest.raises(OwnershipLost, match="active owner"):
        owner()
    work = spool(first)
    enqueue(work, 1)
    work.complete(work.claim(), values={"color": "red"})
    enqueue(work, 2)
    late_job = work.claim()
    enqueue(work, 3)
    expire(attribute_database, first)
    replacement = owner()
    assert replacement.epoch == first.epoch + 1
    events = journal(replacement).read()
    assert [(e["track_id"], e["status"]) for e in events] == [(1, "completed"), (2, "cancelled"), (3, "cancelled")]
    with pytest.raises(OwnershipLost):
        work.complete(late_job, values={"late": True})
    assert (
        attribute_database.execute(
            "SELECT count(*) AS n FROM boxmot_attributes.crops WHERE run_id=%s", (work.run_id,)
        ).fetchone()["n"]
        == 0
    )
    assert (
        attribute_database.execute(
            "SELECT state FROM boxmot_attributes.runs WHERE run_id=%s", (work.run_id,)
        ).fetchone()["state"]
        == "abandoned"
    )


def test_end_session_rejects_late_completion_and_cancellation_is_idempotent(camera):
    owner, spool, journal = camera
    lease = owner()
    work, events = spool(lease), journal(lease)
    enqueue(work)
    job = work.claim()
    events.end_session(work.run_id)
    events.end_session(work.run_id)
    with pytest.raises(OwnershipLost, match="no longer active"):
        work.complete(job, values={"late": True})
    assert [e["status"] for e in events.read()] == ["cancelled"]


def test_read_ack_replays_same_event_ids_after_owner_restart(camera):
    owner, spool, journal = camera
    first = owner()
    work, events = spool(first), journal(first)
    enqueue(work)
    work.complete(work.claim(), values={"color": "blue"})
    page = events.read()
    assert events.read() == page
    events.end_session(work.run_id)
    first.close()
    # close is idempotent, including ExitStack teardown.
    replacement = owner()
    replay = journal(replacement)
    assert replay.read() == page
    replay.acknowledge(page)
    assert replay.read() == []


@pytest.mark.parametrize(
    "limits,images", [({"max_pending_jobs": 2}, [b"x"] * 3), ({"max_crop_bytes": 5}, [b"1234", b"1234"])]
)
def test_capacity_cancels_oldest_pending_with_explicit_reason(camera, limits, images):
    owner, spool, journal = camera
    lease = owner()
    work = spool(lease, **limits)
    for index, image in enumerate(images):
        enqueue(work, index, image)
    event = journal(lease).read()[0]
    assert event["track_id"] == 0 and event["status"] == "cancelled"
    assert "capacity" in event["error"]
    assert work.pending_count == len(images) - 1


def test_expired_lease_cannot_renew_and_connection_loss_stops_admission(camera, attribute_database):
    owner, spool, _ = camera
    lease = owner(renew_s=0.05)
    work = spool(lease)
    expire(attribute_database, lease)
    assert lease.lost.wait(2)
    with pytest.raises(OwnershipLost):
        enqueue(work)


def test_renewal_connection_failure_marks_owner_lost(camera, attribute_database):
    owner, _, _ = camera
    lease = owner(renew_s=0.05)
    attribute_database.execute("SELECT pg_terminate_backend(%s)", (lease._db.info.backend_pid,))
    assert lease.lost.wait(2)
    with pytest.raises(OwnershipLost):
        lease.check()


def test_guard_rechecks_database_time_after_waiting_for_lock(camera, attribute_database):
    owner, spool, _ = camera
    lease = owner()
    work = spool(lease)
    attribute_database.execute(
        "UPDATE boxmot_attributes.camera_leases SET expires_at=clock_timestamp()+interval '0.5 second' "
        "WHERE camera_id=%s",
        (lease.camera_id,),
    )
    with ThreadPoolExecutor(1) as executor:
        with attribute_database.transaction():
            attribute_database.execute(
                "SELECT camera_id FROM boxmot_attributes.camera_leases WHERE camera_id=%s FOR UPDATE",
                (lease.camera_id,),
            )
            pending = executor.submit(enqueue, work)
            time.sleep(0.7)
        with pytest.raises(OwnershipLost):
            pending.result(timeout=2)


def test_guard_rechecks_renewal_failure_after_waiting_for_lock(camera, attribute_database, monkeypatch):
    import threading

    owner, spool, _ = camera
    lease = owner()
    work = spool(lease)
    admitted = threading.Event()
    original = lease.check

    def check():
        original()
        admitted.set()

    monkeypatch.setattr(lease, "check", check)
    with ThreadPoolExecutor(1) as executor:
        with attribute_database.transaction():
            attribute_database.execute(
                "SELECT camera_id FROM boxmot_attributes.camera_leases WHERE camera_id=%s FOR UPDATE",
                (lease.camera_id,),
            )
            pending = executor.submit(enqueue, work)
            assert admitted.wait(2)
            lease.lost.set()
        with pytest.raises(OwnershipLost):
            pending.result(timeout=2)


def test_renewal_keeps_export_owner_alive_past_initial_expiry(camera):
    owner, _, journal = camera
    lease = owner(ttl_s=1, renew_s=0.05)
    events = journal(lease)
    deadline = time.monotonic() + 1.2
    while time.monotonic() < deadline:
        assert events.read() == []
        time.sleep(0.05)
    with pytest.raises(OwnershipLost, match="active owner"):
        owner()


def test_retention_preserves_current_pending_and_unacknowledged(camera, attribute_database):
    owner, spool, journal = camera
    lease = owner()
    work, events = spool(lease), journal(lease)
    for track in (1, 2, 3):
        enqueue(work, track)
        work.complete(work.claim(), values={"track": track})
    events.acknowledge(events.read())
    enqueue(work, 4)
    # Track 1 remains current; track 2 is inactive; track 3 is old but has
    # unacknowledged history until the second maintenance pass.
    attribute_database.execute(
        "UPDATE boxmot_attributes.tracks SET observed_at=clock_timestamp()-interval '8 days', "
        "active=(track_id=1) WHERE run_id=%s",
        (work.run_id,),
    )
    attribute_database.execute(
        "UPDATE boxmot_attributes.events SET created_at=clock_timestamp()-interval '8 days', "
        "delivered_at=CASE WHEN (payload->>'track_id')::int=3 THEN NULL ELSE delivered_at END "
        "WHERE run_id=%s",
        (work.run_id,),
    )
    maintain_database(ENV)
    assert len(events.read()) == 1
    assert work.latest(1, [1, 2, 3]).keys() == {1, 2, 3}
    events.acknowledge(events.read())
    maintain_database(ENV)
    assert work.latest(1, [1, 2, 3]).keys() == {1}
    assert work.pending_count == 1
    expire(attribute_database, lease)
    maintain_database(ENV)
    replacement = owner()
    assert journal(replacement).read()[0]["status"] == "cancelled"


def test_additive_v1_upgrade_preserves_history_and_requires_explicit_action(attribute_database):
    from psycopg import Rollback, sql

    from boxmot.attributes.database import _SCHEMA_SQL
    from boxmot.utils.config import ConfigurationError

    ensure_schema(attribute_database)
    # DDL is transactionally rolled back; never reset the configured test DB.
    with attribute_database.transaction():
        attribute_database.execute(
            sql.SQL("ALTER SCHEMA boxmot_attributes RENAME TO {}").format(sql.Identifier("saved_" + uuid4().hex))
        )
        attribute_database.execute(_SCHEMA_SQL, prepare=False)
        attribute_database.execute(
            "INSERT INTO boxmot_attributes.runs(run_id,model,backend) VALUES ('existing','model','vllm')"
        )
        with pytest.raises(ConfigurationError, match="upgrade"):
            ensure_schema(attribute_database)
        ensure_schema(attribute_database, upgrade=True)
        assert (
            attribute_database.execute("SELECT model FROM boxmot_attributes.runs WHERE run_id='existing'").fetchone()[
                "model"
            ]
            == "model"
        )
        ensure_schema(attribute_database, upgrade=True)
        raise Rollback()


def test_retention_skips_camera_with_inflight_admission(camera, attribute_database):
    owner, spool, journal = camera
    lease = owner()
    work, events = spool(lease), journal(lease)
    enqueue(work)
    work.complete(work.claim(), values={"color": "red"})
    events.acknowledge(events.read())
    attribute_database.execute(
        "UPDATE boxmot_attributes.tracks SET active=FALSE,observed_at=clock_timestamp()-interval '8 days' "
        "WHERE run_id=%s",
        (work.run_id,),
    )
    attribute_database.execute(
        "UPDATE boxmot_attributes.events SET created_at=clock_timestamp()-interval '8 days' WHERE run_id=%s",
        (work.run_id,),
    )
    with ThreadPoolExecutor(1) as executor:
        with work.transaction():
            # Maintenance must skip, rather than select stale deletion
            # candidates while this transaction can still admit another job.
            executor.submit(maintain_database, ENV).result(timeout=2)
            assert work.latest(1, [7])[7].values == {"color": "red"}
            assert attribute_database.execute(
                "SELECT 1 FROM boxmot_attributes.events WHERE run_id=%s", (work.run_id,)
            ).fetchone()
            enqueue(work)
    maintain_database(ENV)
    assert work.pending_count == 1


def test_retention_processes_multiple_pages_of_inactive_history(camera, attribute_database):
    owner, spool, _ = camera
    work = spool(owner())
    attribute_database.execute(
        """INSERT INTO boxmot_attributes.tracks
        (run_id,generation,track_id,class_id,last_seen,observation,observed_at,active)
        SELECT %s,1,n,0,1,'{}',clock_timestamp()-interval '8 days',FALSE FROM generate_series(1,1100) n""",
        (work.run_id,),
    )
    attribute_database.execute(
        """INSERT INTO boxmot_attributes.events(run_id,payload,created_at,delivered_at)
        SELECT %s,'{}',clock_timestamp()-interval '8 days',clock_timestamp() FROM generate_series(1,1100)""",
        (work.run_id,),
    )
    maintain_database(ENV)
    for table in ("tracks", "events"):
        from psycopg import sql

        assert (
            attribute_database.execute(
                sql.SQL("SELECT count(*) AS n FROM boxmot_attributes.{} WHERE run_id=%s").format(sql.Identifier(table)),
                (work.run_id,),
            ).fetchone()["n"]
            == 0
        )


def test_upgrade_timeout_is_scoped_to_explicit_upgrade(attribute_database, monkeypatch):
    from boxmot.attributes import database

    original = database.ensure_schema

    def ensure_with_ddl_time(connection, *, upgrade):
        assert upgrade
        assert connection.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "10min"
        assert connection.execute("SHOW lock_timeout").fetchone()["lock_timeout"] == "30s"
        original(connection, upgrade=upgrade)

    monkeypatch.setattr(database, "ensure_schema", ensure_with_ddl_time)
    database.upgrade_database(ENV)
    with database.connect(ENV) as connection:
        assert connection.execute("SHOW statement_timeout").fetchone()["statement_timeout"] == "3s"


def test_output_maintenance_exports_and_cleans_camera_that_never_restarts(camera, attribute_database, tmp_path):
    import json
    import os

    from boxmot.engine.streaming.maintenance import maintain_outputs

    owner, spool, journal = camera
    lease = owner()
    work = spool(lease)
    enqueue(work)
    work.complete(work.claim(), values={"color": "red"})
    event_id = journal(lease).read()[0]["event_id"]
    old = tmp_path / lease.camera_id / work.run_id / "tracks-old.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text("{}\n")
    age = time.time() - 8 * 86400
    os.utime(old, (age, age))
    maintain_outputs(ENV, lease.camera_id, tmp_path, 7)
    assert old.exists()  # A live owner keeps responsibility for pending export.
    expire(attribute_database, lease)
    maintain_database(ENV)
    maintain_outputs(ENV, lease.camera_id, tmp_path, 7)
    assert not old.exists()
    exported = [
        json.loads(line) for path in old.parent.glob("attributes-*.jsonl") for line in path.read_text().splitlines()
    ]
    assert [e["event_id"] for e in exported] == [event_id]
    assert (
        attribute_database.execute(
            "SELECT 1 FROM boxmot_attributes.events WHERE run_id=%s AND delivered_at IS NULL", (work.run_id,)
        ).fetchone()
        is None
    )
