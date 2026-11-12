"""Disk-backed crop scheduling and completion storage for attribute extraction."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.models import AttributeResult, AttributeSample


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
    """Serialize short SQLite transactions across the caller and inference worker.

    Only the running request and caller-requested result pages enter memory.
    A retry retains its immutable samples while a newer pending refresh can
    continue collecting crops for the same identity.
    """

    def __init__(self, path: Path, config: AttributeConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self._lock = threading.RLock()
        self._transaction_depth = 0
        self._closed = False
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript(
            """
            CREATE TABLE tracks (
                generation INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                sequence_id TEXT,
                class_id INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                last_sample INTEGER,
                observation TEXT NOT NULL,
                ever_sampled INTEGER NOT NULL DEFAULT 0,
                unavailable_reported INTEGER NOT NULL DEFAULT 0,
                latest TEXT,
                last_status TEXT,
                PRIMARY KEY (generation, track_id)
            );
            CREATE INDEX tracks_seen ON tracks(generation, last_seen);
            CREATE TABLE tickets (value INTEGER NOT NULL);
            INSERT INTO tickets VALUES (0);
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                ready INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                available REAL NOT NULL DEFAULT 0,
                FOREIGN KEY (generation, track_id) REFERENCES tracks(generation, track_id)
            );
            CREATE UNIQUE INDEX one_pending ON jobs(generation, track_id) WHERE state = 'pending';
            CREATE UNIQUE INDEX one_frozen ON jobs(generation, track_id) WHERE state IN ('running', 'retry');
            CREATE INDEX ready_jobs ON jobs(state, ready, ticket);
            CREATE TABLE crops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                sample TEXT NOT NULL,
                image BLOB NOT NULL
            );
            CREATE INDEX job_crops ON crops(job_id, id);
            CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL);
            CREATE TABLE counters (status TEXT PRIMARY KEY, count INTEGER NOT NULL);
            """
        )

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Batch a frame's writes into one commit while allowing nested helpers."""
        with self._lock:
            outermost = self._transaction_depth == 0
            self._transaction_depth += 1
            try:
                if outermost:
                    with self._db:
                        yield
                else:
                    yield
            finally:
                self._transaction_depth -= 1

    def _ticket(self) -> int:
        self._db.execute("UPDATE tickets SET value = value + 1")
        return int(self._db.execute("SELECT value FROM tickets").fetchone()[0])

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
            self._db.execute(
                """INSERT INTO tracks(generation, track_id, sequence_id, class_id, last_seen, observation)
                VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(generation, track_id) DO UPDATE SET
                last_seen = excluded.last_seen, class_id = excluded.class_id, observation = excluded.observation""",
                (generation, track_id, sequence_id, class_id, step, json.dumps(asdict(sample))),
            )
            row = self._db.execute(
                "SELECT last_sample FROM tracks WHERE generation = ? AND track_id = ?", (generation, track_id)
            ).fetchone()
            return row[0]

    def append(self, generation: int, track_id: int, step: int, sample: AttributeSample, image: bytes) -> None:
        """Persist a sample, coalescing the pending batch without changing its rank."""
        with self.transaction():
            row = self._db.execute(
                "SELECT id FROM jobs WHERE generation = ? AND track_id = ? AND state = 'pending'",
                (generation, track_id),
            ).fetchone()
            if row is None:
                cursor = self._db.execute(
                    "INSERT INTO jobs(ticket, generation, track_id) VALUES (?, ?, ?)",
                    (self._ticket(), generation, track_id),
                )
                job_id = cursor.lastrowid
            else:
                job_id = row[0]
            self._db.execute(
                "INSERT INTO crops(job_id, sample, image) VALUES (?, ?, ?)",
                (job_id, json.dumps(asdict(sample)), image),
            )
            self._db.execute(
                """DELETE FROM crops WHERE job_id = ? AND id NOT IN
                (SELECT id FROM crops WHERE job_id = ? ORDER BY id DESC LIMIT ?)""",
                (job_id, job_id, self.config.batch_size),
            )
            count = self._db.execute("SELECT count(*) FROM crops WHERE job_id = ?", (job_id,)).fetchone()[0]
            if count >= self.config.batch_size:
                self._db.execute("UPDATE jobs SET ready = 1 WHERE id = ?", (job_id,))
            self._db.execute(
                "UPDATE tracks SET last_sample = ?, ever_sampled = 1 WHERE generation = ? AND track_id = ?",
                (step, generation, track_id),
            )

    def _publish(self, result: AttributeResult) -> None:
        payload = json.dumps(result.to_dict(), allow_nan=False)
        self._db.execute("INSERT INTO events(payload) VALUES (?)", (payload,))
        self._db.execute(
            """INSERT INTO counters(status, count) VALUES (?, 1)
            ON CONFLICT(status) DO UPDATE SET count = count + 1""",
            (result.status,),
        )
        self._db.execute(
            "UPDATE tracks SET last_status = ? WHERE generation = ? AND track_id = ?",
            (result.status, result.generation, result.track_id),
        )
        if result.status == "completed":
            self._db.execute(
                "UPDATE tracks SET latest = ? WHERE generation = ? AND track_id = ?",
                (payload, result.generation, result.track_id),
            )

    def flush(self, generation: int | None = None, *, departed_at: int | None = None) -> None:
        """Release partial batches, recording tracks that never supplied a crop."""
        where, params = [], []
        if generation is not None:
            where.append("generation = ?")
            params.append(generation)
        if departed_at is not None:
            where.append("last_seen = ?")
            params.append(departed_at)
        predicate = " AND ".join(where) or "1"
        with self.transaction():
            self._db.execute(
                f"""UPDATE jobs SET ready = 1 WHERE state = 'pending' AND ready = 0
                AND (generation, track_id) IN (SELECT generation, track_id FROM tracks WHERE {predicate})""",
                params,
            )
            # Page the rows to keep even a large sequence flush bounded in RAM.
            while True:
                rows = self._db.execute(
                    f"""SELECT * FROM tracks WHERE {predicate} AND ever_sampled = 0
                    AND unavailable_reported = 0 LIMIT 100""",
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
                            samples=(AttributeSample(**json.loads(row["observation"])),),
                            status="ineligible",
                            error="No nonempty crop with a current detection or mask was available.",
                            display_fields=self.config.display_fields,
                        )
                    )
                    self._db.execute(
                        "UPDATE tracks SET unavailable_reported = 1 WHERE generation = ? AND track_id = ?",
                        (row["generation"], row["track_id"]),
                    )

    def claim(self) -> _Job | None:
        """Claim the oldest ready request, allowing unrelated work during retries."""
        with self.transaction():
            row = self._db.execute(
                """SELECT j.*, t.sequence_id, t.class_id FROM jobs j JOIN tracks t USING (generation, track_id)
                WHERE j.available <= ? AND (j.state = 'retry' OR (j.state = 'pending' AND j.ready = 1
                AND NOT EXISTS (SELECT 1 FROM jobs frozen WHERE frozen.generation = j.generation
                AND frozen.track_id = j.track_id AND frozen.state IN ('running', 'retry'))))
                ORDER BY j.ticket LIMIT 1""",
                (time.monotonic(),),
            ).fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE jobs SET state = 'running', attempts = attempts + 1 WHERE id = ?", (row["id"],))
            crops = self._db.execute("SELECT sample, image FROM crops WHERE job_id = ? ORDER BY id", (row["id"],))
            samples, images = [], []
            for crop in crops:
                samples.append(AttributeSample(**json.loads(crop["sample"])))
                images.append(bytes(crop["image"]))
            return _Job(
                row["id"],
                row["generation"],
                row["track_id"],
                row["sequence_id"],
                row["class_id"],
                row["attempts"] + 1,
                tuple(samples),
                tuple(images),
            )

    def retry(self, job: _Job, delay: float) -> None:
        """Move a transient failure behind other work without releasing its samples."""
        with self.transaction():
            self._db.execute(
                "UPDATE jobs SET state = 'retry', available = ?, ticket = ? WHERE id = ?",
                (time.monotonic() + delay, self._ticket(), job.id),
            )

    def complete(self, job: _Job, *, values: dict[str, Any] | None = None, error: str | None = None) -> None:
        """Atomically persist an outcome and release its crop blobs."""
        with self.transaction():
            if self._db.execute("SELECT 1 FROM jobs WHERE id = ?", (job.id,)).fetchone() is None:
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
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job.id,))

    def cancel(self) -> None:
        """Record every outstanding batch as cancelled, including active inference."""
        with self.transaction():
            while True:
                row = self._db.execute(
                    "SELECT j.*, t.sequence_id, t.class_id FROM jobs j "
                    "JOIN tracks t USING (generation, track_id) LIMIT 1"
                ).fetchone()
                if row is None:
                    break
                samples = tuple(
                    AttributeSample(**json.loads(crop[0]))
                    for crop in self._db.execute("SELECT sample FROM crops WHERE job_id = ? ORDER BY id", (row["id"],))
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
                self._db.execute("DELETE FROM jobs WHERE id = ?", (row["id"],))

    def poll(self, limit: int) -> list[AttributeResult]:
        """Consume a bounded page of outcomes without modifying latest snapshots."""
        with self.transaction():
            rows = self._db.execute("SELECT id, payload FROM events ORDER BY id LIMIT ?", (limit,)).fetchall()
            results = [AttributeResult.from_dict(json.loads(row["payload"])) for row in rows]
            if rows:
                self._db.execute("DELETE FROM events WHERE id <= ?", (rows[-1]["id"],))
            return results

    def latest(self, generation: int, track_ids: list[int]) -> dict[int, AttributeResult]:
        """Read detached snapshots for only the caller's current track rows."""
        result = {}
        with self._lock:
            for offset in range(0, len(track_ids), 500):
                ids = track_ids[offset : offset + 500]
                marks = ",".join("?" for _ in ids)
                rows = self._db.execute(
                    "SELECT track_id, latest FROM tracks WHERE generation = ? "
                    f"AND track_id IN ({marks}) AND latest IS NOT NULL",
                    (generation, *ids),
                )
                for row in rows:
                    result[row["track_id"]] = AttributeResult.from_dict(json.loads(row["latest"]))
        return result

    @property
    def pending_count(self) -> int:
        """Count pending, running, and retrying requests."""
        with self._lock:
            return int(self._db.execute("SELECT count(*) FROM jobs").fetchone()[0])

    @property
    def stats(self) -> dict[str, int]:
        """Return terminal event counts and distinct observed/analyzed track counts."""
        with self._lock:
            result = {row[0]: int(row[1]) for row in self._db.execute("SELECT status, count FROM counters")}
            result["tracks"] = int(self._db.execute("SELECT count(*) FROM tracks").fetchone()[0])
            result["analyzed_tracks"] = int(
                self._db.execute("SELECT count(*) FROM tracks WHERE latest IS NOT NULL").fetchone()[0]
            )
            return result

    def close(self) -> None:
        """Release database handles after the worker has stopped."""
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
