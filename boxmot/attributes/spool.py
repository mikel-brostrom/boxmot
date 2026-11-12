"""PostgreSQL crop scheduling and persistent results for attribute extraction."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.models import AttributeResult, AttributeSample
from boxmot.utils.config import ConfigurationError
from boxmot.utils.dependencies import require_extra

_SCHEMA_VERSION = "boxmot_attributes:1"

_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS boxmot_attributes;
COMMENT ON SCHEMA boxmot_attributes IS 'boxmot_attributes:1';
CREATE TABLE IF NOT EXISTS boxmot_attributes.runs (
    run_id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    backend TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ticket BIGINT NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS boxmot_attributes.tracks (
    run_id TEXT NOT NULL REFERENCES boxmot_attributes.runs(run_id) ON DELETE CASCADE,
    generation BIGINT NOT NULL,
    track_id BIGINT NOT NULL,
    sequence_id TEXT,
    class_id BIGINT NOT NULL,
    last_seen BIGINT NOT NULL,
    last_sample BIGINT,
    observation JSONB NOT NULL,
    ever_sampled BOOLEAN NOT NULL DEFAULT FALSE,
    unavailable_reported BOOLEAN NOT NULL DEFAULT FALSE,
    latest JSONB,
    last_status TEXT,
    PRIMARY KEY (run_id, generation, track_id)
);
CREATE INDEX IF NOT EXISTS tracks_seen ON boxmot_attributes.tracks(run_id, generation, last_seen);
CREATE TABLE IF NOT EXISTS boxmot_attributes.jobs (
    run_id TEXT NOT NULL,
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    ticket BIGINT NOT NULL,
    generation BIGINT NOT NULL,
    track_id BIGINT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    ready BOOLEAN NOT NULL DEFAULT FALSE,
    attempts INTEGER NOT NULL DEFAULT 0,
    available TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (run_id, id),
    FOREIGN KEY (run_id, generation, track_id)
        REFERENCES boxmot_attributes.tracks(run_id, generation, track_id) ON DELETE CASCADE
);
CREATE UNIQUE INDEX IF NOT EXISTS one_pending
    ON boxmot_attributes.jobs(run_id, generation, track_id) WHERE state = 'pending';
CREATE UNIQUE INDEX IF NOT EXISTS one_frozen
    ON boxmot_attributes.jobs(run_id, generation, track_id) WHERE state IN ('running', 'retry');
CREATE INDEX IF NOT EXISTS ready_jobs ON boxmot_attributes.jobs(run_id, state, ready, ticket);
CREATE TABLE IF NOT EXISTS boxmot_attributes.crops (
    run_id TEXT NOT NULL,
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    job_id BIGINT NOT NULL,
    sample JSONB NOT NULL,
    image BYTEA NOT NULL,
    PRIMARY KEY (run_id, id),
    FOREIGN KEY (run_id, job_id) REFERENCES boxmot_attributes.jobs(run_id, id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS job_crops ON boxmot_attributes.crops(run_id, job_id, id);
CREATE TABLE IF NOT EXISTS boxmot_attributes.events (
    run_id TEXT NOT NULL REFERENCES boxmot_attributes.runs(run_id) ON DELETE CASCADE,
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at TIMESTAMPTZ,
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS undelivered_events
    ON boxmot_attributes.events(run_id, id) WHERE delivered_at IS NULL;
CREATE TABLE IF NOT EXISTS boxmot_attributes.counters (
    run_id TEXT NOT NULL REFERENCES boxmot_attributes.runs(run_id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    count BIGINT NOT NULL,
    PRIMARY KEY (run_id, status)
);
"""


@dataclass(frozen=True, slots=True)
class _Job:
    """One immutable request copied out of the spool for the worker."""

    id: int
    generation: int
    track_id: int
    sequence_id: str | None
    class_id: int
    attempts: int
    samples: tuple[AttributeSample, ...]
    images: tuple[bytes, ...]


class AttributeSpool:
    """Serialize PostgreSQL transactions across one run's caller and worker.

    Runs share a schema, but every operation is scoped to this run. Only the
    active request and requested result pages enter memory. Completed outcomes
    and latest values remain in PostgreSQL after closing the connection; this
    does not resume abandoned jobs or attach additional workers to a run.
    """

    def __init__(self, config: AttributeConfig, run_id: str) -> None:
        """Connect using an environment-owned URL and create this run atomically."""
        database_url = os.environ.get(config.database_url_env, "").strip()
        if not database_url:
            raise ConfigurationError(
                f"Attribute database environment variable {config.database_url_env!r} must be set and nonempty."
            )
        require_extra("attributes", purpose="PostgreSQL track attribute storage")
        self._driver = import_module("psycopg")
        dict_row = import_module("psycopg.rows").dict_row
        self.config = config
        self.run_id = run_id
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._closed = False
        self._db = None
        try:
            self._db = self._driver.connect(
                database_url,
                autocommit=True,
                row_factory=dict_row,
                connect_timeout=5,
                options="-c statement_timeout=5000 -c lock_timeout=5000",
                application_name="boxmot-attributes",
                # Bound dead TCP peers as well as server-side query/lock waits.
                # libpq ignores unsupported TCP options and Unix-socket peers.
                keepalives=1,
                keepalives_idle=2,
                keepalives_interval=1,
                keepalives_count=3,
                tcp_user_timeout=5000,
            )
            with self._db.transaction():
                # DDL IF NOT EXISTS alone can race when two runs first start.
                # The transaction-scoped key is stable across processes.
                self._db.execute("SELECT pg_advisory_xact_lock(1112496205, 1096045650)")
                schema = self._db.execute(
                    """SELECT pg_catalog.obj_description(oid, 'pg_namespace') AS version
                    FROM pg_catalog.pg_namespace WHERE nspname = 'boxmot_attributes'"""
                ).fetchone()
                if schema is None:
                    self._db.execute(_SCHEMA_SQL, prepare=False)
                elif schema["version"] != _SCHEMA_VERSION:
                    raise ConfigurationError(
                        "Attribute PostgreSQL schema version is unsupported. "
                        "Use a database with the matching BoxMOT attributes schema or an empty schema destination."
                    )
                self._db.execute(
                    "INSERT INTO boxmot_attributes.runs(run_id, model, backend) VALUES (%s, %s, %s)",
                    (self.run_id, config.model, config.backend),
                )
        except self._driver.Error as exc:
            if self._db is not None:
                self._db.close()
            # Driver messages may contain the connection string or credentials.
            raise ConfigurationError(
                f"Attribute PostgreSQL initialization failed ({type(exc).__name__}). "
                f"Check {config.database_url_env}, database availability, and schema permissions."
            ) from None
        except BaseException:
            if self._db is not None:
                self._db.close()
            raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Batch a frame's writes into one transaction, sanitizing database errors."""
        with self._lock:
            outermost = self._transaction_depth == 0
            self._transaction_depth += 1
            try:
                if outermost:
                    with self._db.transaction():
                        yield
                else:
                    yield
            except self._driver.Error as exc:
                raise RuntimeError(f"Attribute PostgreSQL operation failed ({type(exc).__name__}).") from None
            finally:
                self._transaction_depth -= 1

    def _ticket(self) -> int:
        """Allocate the next FIFO position within this run's transaction."""
        row = self._db.execute(
            "UPDATE boxmot_attributes.runs SET ticket = ticket + 1 WHERE run_id = %s RETURNING ticket",
            (self.run_id,),
        ).fetchone()
        return int(row["ticket"])

    def observe(
        self,
        generation: int,
        sequence_id: str | None,
        track_id: int,
        class_id: int,
        step: int,
        sample: AttributeSample,
    ) -> int | None:
        """Record a visible identity and return its most recent sampling step."""
        with self.transaction():
            row = self._db.execute(
                """INSERT INTO boxmot_attributes.tracks
                (run_id, generation, track_id, sequence_id, class_id, last_seen, observation)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT(run_id, generation, track_id) DO UPDATE SET
                last_seen = excluded.last_seen, class_id = excluded.class_id, observation = excluded.observation
                RETURNING last_sample""",
                (self.run_id, generation, track_id, sequence_id, class_id, step, json.dumps(asdict(sample))),
            ).fetchone()
            return row["last_sample"]

    def append(self, generation: int, track_id: int, step: int, sample: AttributeSample, image: bytes) -> None:
        """Persist a sample, coalescing the pending batch without changing its rank."""
        with self.transaction():
            row = self._db.execute(
                """SELECT id FROM boxmot_attributes.jobs
                WHERE run_id = %s AND generation = %s AND track_id = %s AND state = 'pending'""",
                (self.run_id, generation, track_id),
            ).fetchone()
            if row is None:
                row = self._db.execute(
                    """INSERT INTO boxmot_attributes.jobs(run_id, ticket, generation, track_id)
                    VALUES (%s, %s, %s, %s) RETURNING id""",
                    (self.run_id, self._ticket(), generation, track_id),
                ).fetchone()
            job_id = row["id"]
            self._db.execute(
                "INSERT INTO boxmot_attributes.crops(run_id, job_id, sample, image) VALUES (%s, %s, %s::jsonb, %s)",
                (self.run_id, job_id, json.dumps(asdict(sample)), image),
            )
            self._db.execute(
                """DELETE FROM boxmot_attributes.crops WHERE run_id = %s AND job_id = %s AND id NOT IN
                (SELECT id FROM boxmot_attributes.crops WHERE run_id = %s AND job_id = %s
                ORDER BY id DESC LIMIT %s)""",
                (self.run_id, job_id, self.run_id, job_id, self.config.batch_size),
            )
            count = self._db.execute(
                "SELECT count(*) AS count FROM boxmot_attributes.crops WHERE run_id = %s AND job_id = %s",
                (self.run_id, job_id),
            ).fetchone()["count"]
            if count >= self.config.batch_size:
                self._db.execute(
                    "UPDATE boxmot_attributes.jobs SET ready = TRUE WHERE run_id = %s AND id = %s",
                    (self.run_id, job_id),
                )
            self._db.execute(
                """UPDATE boxmot_attributes.tracks SET last_sample = %s, ever_sampled = TRUE
                WHERE run_id = %s AND generation = %s AND track_id = %s""",
                (step, self.run_id, generation, track_id),
            )

    def _publish(self, result: AttributeResult) -> None:
        """Retain an outcome, replacing latest only for a successful analysis."""
        payload = json.dumps(result.to_dict(), allow_nan=False)
        self._db.execute(
            "INSERT INTO boxmot_attributes.events(run_id, payload) VALUES (%s, %s::jsonb)",
            (self.run_id, payload),
        )
        self._db.execute(
            """INSERT INTO boxmot_attributes.counters AS totals(run_id, status, count) VALUES (%s, %s, 1)
            ON CONFLICT(run_id, status) DO UPDATE SET count = totals.count + 1""",
            (self.run_id, result.status),
        )
        self._db.execute(
            """UPDATE boxmot_attributes.tracks SET last_status = %s
            WHERE run_id = %s AND generation = %s AND track_id = %s""",
            (result.status, self.run_id, result.generation, result.track_id),
        )
        if result.status == "completed":
            self._db.execute(
                """UPDATE boxmot_attributes.tracks SET latest = %s::jsonb
                WHERE run_id = %s AND generation = %s AND track_id = %s""",
                (payload, self.run_id, result.generation, result.track_id),
            )

    def flush(self, generation: int | None = None, *, departed_at: int | None = None) -> None:
        """Release partial batches, recording tracks that never supplied a crop."""
        where, params = ["run_id = %s"], [self.run_id]
        if generation is not None:
            where.append("generation = %s")
            params.append(generation)
        if departed_at is not None:
            where.append("last_seen = %s")
            params.append(departed_at)
        predicate = " AND ".join(where)
        with self.transaction():
            self._db.execute(
                f"""UPDATE boxmot_attributes.jobs SET ready = TRUE
                WHERE run_id = %s AND state = 'pending' AND NOT ready
                AND (generation, track_id) IN
                (SELECT generation, track_id FROM boxmot_attributes.tracks WHERE {predicate})""",
                [self.run_id, *params],
            )
            # Page the rows to keep even a large sequence flush bounded in RAM.
            while True:
                rows = self._db.execute(
                    f"""SELECT * FROM boxmot_attributes.tracks WHERE {predicate}
                    AND NOT ever_sampled AND NOT unavailable_reported LIMIT 100""",
                    params,
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    self._publish(
                        AttributeResult(
                            run_id=self.run_id,
                            sequence_id=row["sequence_id"],
                            generation=row["generation"],
                            track_id=row["track_id"],
                            class_id=row["class_id"],
                            model=self.config.model,
                            samples=(AttributeSample(**row["observation"]),),
                            status="ineligible",
                            error="No nonempty crop with a current detection or mask was available.",
                            display_fields=self.config.display_fields,
                        )
                    )
                    self._db.execute(
                        """UPDATE boxmot_attributes.tracks SET unavailable_reported = TRUE
                        WHERE run_id = %s AND generation = %s AND track_id = %s""",
                        (self.run_id, row["generation"], row["track_id"]),
                    )

    def claim(self) -> _Job | None:
        """Atomically claim the oldest ready request in this run."""
        with self.transaction():
            row = self._db.execute(
                """SELECT j.*, t.sequence_id, t.class_id FROM boxmot_attributes.jobs j
                JOIN boxmot_attributes.tracks t USING (run_id, generation, track_id)
                WHERE j.run_id = %s AND j.available <= clock_timestamp()
                AND (j.state = 'retry' OR (j.state = 'pending' AND j.ready
                AND NOT EXISTS (SELECT 1 FROM boxmot_attributes.jobs frozen WHERE frozen.run_id = j.run_id
                AND frozen.generation = j.generation AND frozen.track_id = j.track_id
                AND frozen.state IN ('running', 'retry'))))
                ORDER BY j.ticket LIMIT 1 FOR UPDATE OF j SKIP LOCKED""",
                (self.run_id,),
            ).fetchone()
            if row is None:
                return None
            self._db.execute(
                """UPDATE boxmot_attributes.jobs SET state = 'running', attempts = attempts + 1
                WHERE run_id = %s AND id = %s""",
                (self.run_id, row["id"]),
            )
            crops = self._db.execute(
                "SELECT sample, image FROM boxmot_attributes.crops WHERE run_id = %s AND job_id = %s ORDER BY id",
                (self.run_id, row["id"]),
            )
            samples, images = [], []
            for crop in crops:
                samples.append(AttributeSample(**crop["sample"]))
                images.append(bytes(crop["image"]))
            return _Job(
                row["id"], row["generation"], row["track_id"], row["sequence_id"],
                row["class_id"], row["attempts"] + 1, tuple(samples), tuple(images),
            )

    def retry(self, job: _Job, delay: float) -> None:
        """Schedule a retry using the database clock without releasing samples."""
        with self.transaction():
            self._db.execute(
                """UPDATE boxmot_attributes.jobs
                SET state = 'retry', available = clock_timestamp() + %s * INTERVAL '1 second', ticket = %s
                WHERE run_id = %s AND id = %s""",
                (delay, self._ticket(), self.run_id, job.id),
            )

    def complete(self, job: _Job, *, values: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Atomically retain an outcome and release the completed job's crops."""
        with self.transaction():
            row = self._db.execute(
                "SELECT id FROM boxmot_attributes.jobs WHERE run_id = %s AND id = %s FOR UPDATE",
                (self.run_id, job.id),
            ).fetchone()
            if row is None:
                return  # Cancellation already owns the terminal outcome.
            self._publish(
                AttributeResult(
                    run_id=self.run_id,
                    sequence_id=job.sequence_id,
                    generation=job.generation,
                    track_id=job.track_id,
                    class_id=job.class_id,
                    model=self.config.model,
                    samples=job.samples,
                    status="completed" if error is None else "failed",
                    values=values,
                    error=error,
                    display_fields=self.config.display_fields,
                )
            )
            self._db.execute(
                "DELETE FROM boxmot_attributes.jobs WHERE run_id = %s AND id = %s", (self.run_id, job.id)
            )

    def cancel(self) -> None:
        """Record all this run's outstanding batches as cancelled, freeing crops."""
        with self.transaction():
            while True:
                row = self._db.execute(
                    """SELECT j.*, t.sequence_id, t.class_id FROM boxmot_attributes.jobs j
                    JOIN boxmot_attributes.tracks t USING (run_id, generation, track_id)
                    WHERE j.run_id = %s ORDER BY j.ticket LIMIT 1 FOR UPDATE OF j""",
                    (self.run_id,),
                ).fetchone()
                if row is None:
                    break
                samples = tuple(
                    AttributeSample(**crop["sample"])
                    for crop in self._db.execute(
                        "SELECT sample FROM boxmot_attributes.crops WHERE run_id = %s AND job_id = %s ORDER BY id",
                        (self.run_id, row["id"]),
                    )
                )
                self._publish(
                    AttributeResult(
                        run_id=self.run_id,
                        sequence_id=row["sequence_id"],
                        generation=row["generation"],
                        track_id=row["track_id"],
                        class_id=row["class_id"],
                        model=self.config.model,
                        samples=samples,
                        status="cancelled",
                        error="Attribute analysis was interrupted.",
                        display_fields=self.config.display_fields,
                    )
                )
                self._db.execute(
                    "DELETE FROM boxmot_attributes.jobs WHERE run_id = %s AND id = %s", (self.run_id, row["id"])
                )

    def poll(self, limit: int) -> list[AttributeResult]:
        """Mark a bounded event page delivered while retaining its history."""
        with self.transaction():
            rows = self._db.execute(
                """SELECT id, payload FROM boxmot_attributes.events
                WHERE run_id = %s AND delivered_at IS NULL ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED""",
                (self.run_id, limit),
            ).fetchall()
            results = [AttributeResult.from_dict(row["payload"]) for row in rows]
            if rows:
                self._db.execute(
                    """UPDATE boxmot_attributes.events SET delivered_at = clock_timestamp()
                    WHERE run_id = %s AND id = ANY(%s)""",
                    (self.run_id, [row["id"] for row in rows]),
                )
            return results

    def latest(self, generation: int, track_ids: list[int]) -> dict[int, AttributeResult]:
        """Read detached snapshots for only this run's current track rows."""
        result = {}
        with self.transaction():
            for offset in range(0, len(track_ids), 500):
                rows = self._db.execute(
                    """SELECT track_id, latest FROM boxmot_attributes.tracks
                    WHERE run_id = %s AND generation = %s AND track_id = ANY(%s) AND latest IS NOT NULL""",
                    (self.run_id, generation, track_ids[offset : offset + 500]),
                )
                for row in rows:
                    result[row["track_id"]] = AttributeResult.from_dict(row["latest"])
        return result

    @property
    def pending_count(self) -> int:
        """Count this run's pending, running, and retrying requests."""
        with self.transaction():
            return int(self._db.execute(
                "SELECT count(*) AS count FROM boxmot_attributes.jobs WHERE run_id = %s", (self.run_id,)
            ).fetchone()["count"])

    @property
    def stats(self) -> dict[str, int]:
        """Return this run's outcome counts and distinct observed/analyzed tracks."""
        with self.transaction():
            result = {
                row["status"]: int(row["count"])
                for row in self._db.execute(
                    "SELECT status, count FROM boxmot_attributes.counters WHERE run_id = %s", (self.run_id,)
                )
            }
            row = self._db.execute(
                """SELECT count(*) AS tracks, count(*) FILTER (WHERE latest IS NOT NULL) AS analyzed_tracks
                FROM boxmot_attributes.tracks WHERE run_id = %s""",
                (self.run_id,),
            ).fetchone()
            result.update(tracks=int(row["tracks"]), analyzed_tracks=int(row["analyzed_tracks"]))
            return result

    def close(self) -> None:
        """Close the connection without removing stored results or histories."""
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
