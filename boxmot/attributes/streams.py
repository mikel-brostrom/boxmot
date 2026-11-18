"""Camera ownership, fenced stream sessions, and durable event maintenance."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator
from uuid import uuid4

from boxmot.attributes.database import connect, ensure_schema


class OwnershipLost(RuntimeError):
    """The process can no longer publish for its camera."""


def cancel_jobs(db: Any, run_ids: list[str], reason: str, job_ids: list[int] | None = None) -> None:
    """Publish cancellation and delete crops in bulk inside the caller's transaction."""
    rows = db.execute(
        """SELECT j.*, t.sequence_id, t.class_id, r.model, r.display_fields,
        COALESCE((SELECT jsonb_agg(c.sample ORDER BY c.id) FROM boxmot_attributes.crops c
          WHERE c.run_id=j.run_id AND c.job_id=j.id), '[]') AS samples
        FROM boxmot_attributes.jobs j JOIN boxmot_attributes.tracks t USING(run_id,generation,track_id)
        JOIN boxmot_attributes.runs r USING(run_id)
        WHERE j.run_id=ANY(%s) AND (%s::bigint[] IS NULL OR j.id=ANY(%s)) ORDER BY j.ticket FOR UPDATE OF j""",
        (run_ids, job_ids, job_ids),
    ).fetchall()
    if not rows:
        return
    import json

    payloads = []
    for row in rows:
        payload = {
            name: row[name]
            for name in (
                "run_id",
                "sequence_id",
                "generation",
                "track_id",
                "class_id",
                "model",
                "samples",
                "display_fields",
            )
        }
        payload.update(status="cancelled", values=None, error=reason)
        payloads.append({"run_id": row["run_id"], "payload": payload})
    db.execute(
        """INSERT INTO boxmot_attributes.events(run_id,payload)
        SELECT run_id,payload FROM jsonb_to_recordset(%s::jsonb) AS rows(run_id text,payload jsonb)""",
        (json.dumps(payloads, allow_nan=False),),
    )
    # One statement per aggregate/table keeps cancellation bounded by the queue cap.
    ids = [row["id"] for row in rows]
    db.execute(
        """INSERT INTO boxmot_attributes.counters AS c(run_id,status,count)
        SELECT run_id,'cancelled',count(*) FROM boxmot_attributes.jobs WHERE run_id=ANY(%s) AND id=ANY(%s)
        GROUP BY run_id ON CONFLICT(run_id,status) DO UPDATE SET count=c.count+excluded.count""",
        (run_ids, ids),
    )
    db.execute(
        """UPDATE boxmot_attributes.tracks t SET last_status='cancelled'
        FROM boxmot_attributes.jobs j WHERE j.run_id=ANY(%s) AND j.id=ANY(%s)
        AND (t.run_id,t.generation,t.track_id)=(j.run_id,j.generation,j.track_id)""",
        (run_ids, ids),
    )
    db.execute("DELETE FROM boxmot_attributes.jobs WHERE run_id=ANY(%s) AND id=ANY(%s)", (run_ids, ids))


class CameraLease:
    """Renew a database-time lease on its own connection and fence every writer."""

    def __init__(self, camera_id: str, database_url_env: str, *, ttl_s: int = 30, renew_s: float = 5) -> None:
        self.camera_id = camera_id
        self.database_url_env = database_url_env
        self.owner_id = uuid4().hex
        self.ttl_s = ttl_s
        self.renew_s = renew_s
        self.epoch = 0
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._db = connect(database_url_env)
        try:
            ensure_schema(self._db)
            with self._db.transaction():
                self._db.execute(
                    "INSERT INTO boxmot_attributes.camera_leases(camera_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (camera_id,),
                )
                row = self._db.execute(
                    """UPDATE boxmot_attributes.camera_leases SET owner_id=%s,epoch=epoch+1,
                    expires_at=clock_timestamp()+%s*interval '1 second'
                    WHERE camera_id=%s AND (owner_id IS NULL OR expires_at<=clock_timestamp()) RETURNING epoch""",
                    (self.owner_id, ttl_s, camera_id),
                ).fetchone()
                if row is None:
                    raise OwnershipLost("Camera already has an active owner.")
                self.epoch = row["epoch"]
                runs = self._db.execute(
                    """UPDATE boxmot_attributes.runs SET state='abandoned',ended_at=clock_timestamp()
                    WHERE camera_id=%s AND state='active' RETURNING run_id""",
                    (camera_id,),
                ).fetchall()
                cancel_jobs(self._db, [row["run_id"] for row in runs], "Camera owner was replaced.")
            self._thread = threading.Thread(target=self._renew, name="boxmot-camera-lease", daemon=True)
            self._thread.start()
        except BaseException:
            self._db.close()
            raise

    def check(self) -> None:
        """Reject local work as soon as renewal becomes uncertain."""
        if self.lost.is_set() or self._stop.is_set():
            raise OwnershipLost("Camera ownership was lost.")

    def guard(self, db: Any, run_id: str | None = None) -> None:
        """Lock ownership until commit, serializing takeover with persisted writes."""
        self.check()
        # Validate time *after* obtaining the lock: a waiter may cross expiry
        # without the locked tuple itself changing.
        db.execute(
            "SELECT camera_id FROM boxmot_attributes.camera_leases WHERE camera_id=%s FOR SHARE", (self.camera_id,)
        )
        self.check()
        row = db.execute(
            """SELECT epoch FROM boxmot_attributes.camera_leases WHERE camera_id=%s
            AND owner_id=%s AND epoch=%s AND expires_at>clock_timestamp()""",
            (self.camera_id, self.owner_id, self.epoch),
        ).fetchone()
        if row is None:
            self.lost.set()
            raise OwnershipLost("Camera ownership was lost.")
        if run_id is not None:
            row = db.execute(
                """SELECT run_id FROM boxmot_attributes.runs WHERE run_id=%s AND camera_id=%s
                AND owner_id=%s AND epoch=%s AND state='active' FOR SHARE""",
                (run_id, self.camera_id, self.owner_id, self.epoch),
            ).fetchone()
            if row is None:
                raise OwnershipLost("Stream session is no longer active.")
            self.check()

    def _renew(self) -> None:
        try:
            while not self._stop.wait(self.renew_s):
                with self._db.transaction():
                    self._db.execute(
                        "SELECT camera_id FROM boxmot_attributes.camera_leases WHERE camera_id=%s FOR UPDATE",
                        (self.camera_id,),
                    )
                    row = self._db.execute(
                        """UPDATE boxmot_attributes.camera_leases
                        SET expires_at=clock_timestamp()+%s*interval '1 second'
                        WHERE camera_id=%s AND owner_id=%s AND epoch=%s
                        AND expires_at>clock_timestamp() RETURNING epoch""",
                        (self.ttl_s, self.camera_id, self.owner_id, self.epoch),
                    ).fetchone()
                if row is None:
                    raise OwnershipLost("Camera lease expired.")
        except Exception:
            self.lost.set()

    def close(self) -> None:
        """Stop renewal before releasing this owner's lease."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)
            if self._thread.is_alive():
                self._db.cancel()
                self._thread.join(timeout=1)
        try:
            if not self._db.closed:
                self._db.execute(
                    """UPDATE boxmot_attributes.camera_leases SET owner_id=NULL,expires_at=clock_timestamp()
                    WHERE camera_id=%s AND owner_id=%s AND epoch=%s""",
                    (self.camera_id, self.owner_id, self.epoch),
                )
        except Exception:
            # Releasing a dead connection is best effort; expiry enables takeover.
            self.lost.set()
        finally:
            self._db.close()


class CameraJournal:
    """Replay a camera's unacknowledged events, including earlier sessions."""

    def __init__(self, lease: CameraLease) -> None:
        self.lease = lease
        self._db = connect(lease.database_url_env)

    @contextmanager
    def transaction(self) -> Iterator[Any]:
        """Guard reads/acknowledgments against obsolete camera owners."""
        try:
            with self._db.transaction():
                self.lease.guard(self._db)
                yield self._db
        except OwnershipLost:
            raise
        except Exception as exc:
            raise RuntimeError(f"Camera journal operation failed ({type(exc).__name__}).") from None

    def read(self, limit: int = 100) -> list[dict[str, Any]]:
        """Read without acknowledgment; the exporter must sync before ack."""
        with self.transaction() as db:
            rows = db.execute(
                """SELECT e.id,e.run_id,e.payload,e.created_at FROM boxmot_attributes.events e
                JOIN boxmot_attributes.runs r USING(run_id)
                WHERE r.camera_id=%s AND e.delivered_at IS NULL ORDER BY e.id LIMIT %s""",
                (self.lease.camera_id, limit),
            ).fetchall()
            return [
                {
                    **row["payload"],
                    "event_id": f"{row['run_id']}:{row['id']}",
                    "camera_id": self.lease.camera_id,
                    "session_id": row["run_id"],
                    "created_at": row["created_at"].isoformat(),
                }
                for row in rows
            ]

    def acknowledge(self, events: list[dict[str, Any]]) -> None:
        """Acknowledge only this camera's successfully synchronized records."""
        ids = [int(event["event_id"].rsplit(":", 1)[1]) for event in events]
        with self.transaction() as db:
            db.execute(
                """UPDATE boxmot_attributes.events e SET delivered_at=clock_timestamp()
                FROM boxmot_attributes.runs r WHERE e.run_id=r.run_id
                AND r.camera_id=%s AND e.id=ANY(%s)""",
                (self.lease.camera_id, ids),
            )

    def end_session(self, run_id: str) -> None:
        """Fence further worker activity and retain cancellation outcomes."""
        with self.transaction() as db:
            rows = db.execute(
                """UPDATE boxmot_attributes.runs SET state='stopped',ended_at=clock_timestamp()
                WHERE run_id=%s AND camera_id=%s AND owner_id=%s AND epoch=%s RETURNING run_id""",
                (run_id, self.lease.camera_id, self.lease.owner_id, self.lease.epoch),
            ).fetchall()
            cancel_jobs(db, [row["run_id"] for row in rows], "Stream session ended.")

    def close(self) -> None:
        self._db.close()


def _prune_camera_history(db: Any, camera_id: str, retention_days: int) -> int:
    """Prune one bounded page while excluding concurrent admission and export."""
    with db.transaction():
        # Writers hold this row FOR SHARE. Take the exclusive lock before
        # selecting deletion candidates, so a new job/event cannot commit
        # between a retention snapshot and a cascading track deletion.
        camera = db.execute(
            "SELECT camera_id FROM boxmot_attributes.camera_leases WHERE camera_id=%s FOR UPDATE SKIP LOCKED",
            (camera_id,),
        ).fetchone()
        if camera is None:
            return 0
        deleted = db.execute(
            """DELETE FROM boxmot_attributes.events WHERE (run_id,id) IN (
            SELECT e.run_id,e.id FROM boxmot_attributes.events e JOIN boxmot_attributes.runs r USING(run_id)
            WHERE r.camera_id=%s AND e.delivered_at IS NOT NULL
            AND e.created_at<clock_timestamp()-%s*interval '1 day' LIMIT 1000)""",
            (camera_id, retention_days),
        ).rowcount
        deleted += db.execute(
            """DELETE FROM boxmot_attributes.tracks WHERE (run_id,generation,track_id) IN (
            SELECT t.run_id,t.generation,t.track_id FROM boxmot_attributes.tracks t
            JOIN boxmot_attributes.runs r USING(run_id)
            WHERE r.camera_id=%s AND (r.state<>'active' OR NOT t.active)
            AND t.observed_at<clock_timestamp()-%s*interval '1 day'
            AND NOT EXISTS(SELECT 1 FROM boxmot_attributes.jobs j
                WHERE (j.run_id,j.generation,j.track_id)=(t.run_id,t.generation,t.track_id))
            AND NOT EXISTS(SELECT 1 FROM boxmot_attributes.events e WHERE e.run_id=t.run_id
                AND e.delivered_at IS NULL) LIMIT 1000)""",
            (camera_id, retention_days),
        ).rowcount
        deleted += db.execute(
            """DELETE FROM boxmot_attributes.runs WHERE run_id IN (
            SELECT r.run_id FROM boxmot_attributes.runs r WHERE camera_id=%s AND state<>'active'
            AND ended_at<clock_timestamp()-%s*interval '1 day'
            AND NOT EXISTS(SELECT 1 FROM boxmot_attributes.events e WHERE e.run_id=r.run_id)
            AND NOT EXISTS(SELECT 1 FROM boxmot_attributes.jobs j WHERE j.run_id=r.run_id)
            AND NOT EXISTS(SELECT 1 FROM boxmot_attributes.tracks t WHERE t.run_id=r.run_id) LIMIT 1000)""",
            (camera_id, retention_days),
        ).rowcount
        return deleted


def maintain_database(database_url_env: str, retention_days: int = 7) -> None:
    """Cancel orphan jobs and prune exported history without touching finite runs."""
    if retention_days < 1:
        raise ValueError("retention_days must be positive")
    with connect(database_url_env) as db:
        ensure_schema(db)
        # Bound memory and transaction duration to one camera's queue. Subsequent
        # scheduled sweeps continue any backlog after this bounded camera page.
        for _ in range(100):
            with db.transaction():
                camera = db.execute(
                    """SELECT camera_id FROM boxmot_attributes.camera_leases c
                    WHERE expires_at<=clock_timestamp() AND EXISTS(
                      SELECT 1 FROM boxmot_attributes.runs r WHERE r.camera_id=c.camera_id AND r.state='active')
                    ORDER BY expires_at LIMIT 1 FOR UPDATE SKIP LOCKED"""
                ).fetchone()
                if camera is None:
                    break
                runs = db.execute(
                    """UPDATE boxmot_attributes.runs SET state='abandoned',ended_at=clock_timestamp()
                    WHERE camera_id=%s AND state='active' RETURNING run_id""",
                    (camera["camera_id"],),
                ).fetchall()
                cancel_jobs(db, [row["run_id"] for row in runs], "Camera ownership expired.")
        cursor = ""
        while True:
            cameras = db.execute(
                "SELECT camera_id FROM boxmot_attributes.camera_leases WHERE camera_id>%s ORDER BY camera_id LIMIT 100",
                (cursor,),
            ).fetchall()
            if not cameras:
                break
            for camera in cameras:
                # Release ownership locks between pages so renewal and live
                # writes can proceed. Busy cameras are retried next sweep.
                for _ in range(100):
                    if not _prune_camera_history(db, camera["camera_id"], retention_days):
                        break
            cursor = cameras[-1]["camera_id"]
