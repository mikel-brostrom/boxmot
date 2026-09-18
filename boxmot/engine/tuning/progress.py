"""Bounded sequence-progress snapshots shared by local Tune actors and the driver."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from boxmot.engine.eval.replay import ReplayProgressEvent


def _snapshot_path(progress_dir: str | Path, trial_id: str) -> Path:
    """Keep trial identifiers inside the designated progress directory."""
    name = hashlib.sha256(trial_id.encode("utf-8")).hexdigest()
    return Path(progress_dir) / f"{name}.json"


class TrialSequenceProgressWriter:
    """Publish the latest event per sequence without reporting partial trial scores.

    Each trial owns one small, atomically replaced file. Running frame updates
    are throttled, while sequence transitions are published immediately. A new
    writer starts empty even when Ray reuses an actor or retries a trial.
    """

    def __init__(self, progress_dir: str | Path, trial_id: str, *, interval: float = 0.2) -> None:
        self._path = _snapshot_path(progress_dir, trial_id)
        self._trial_id = trial_id
        self._interval = interval
        self._latest: dict[int, ReplayProgressEvent] = {}
        self._last_write = float("-inf")
        self._dirty = True
        self.flush()

    def __call__(self, event: ReplayProgressEvent) -> None:
        """Retain one event per sequence and publish meaningful transitions."""
        previous = self._latest.get(event.ordinal)
        if previous == event:
            return
        self._latest[event.ordinal] = event
        self._dirty = True
        if (
            previous is None
            or previous.status != event.status
            or event.status in {"completed", "failed"}
            or time.monotonic() - self._last_write >= self._interval
        ):
            self.flush()

    def flush(self) -> None:
        """Publish pending events; unavailable progress storage cannot fail a trial."""
        if not self._dirty:
            return
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self._path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(
                    {
                        "trial_id": self._trial_id,
                        "sequences": [asdict(self._latest[key]) for key in sorted(self._latest)],
                    },
                    stream,
                    separators=(",", ":"),
                )
            os.replace(temporary, self._path)
            self._last_write = time.monotonic()
            self._dirty = False
        except OSError:
            # Progress is observational, matching replay's callback contract.
            return
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def read_trial_sequence_progress(progress_dir: str | Path, trial_id: str) -> tuple[ReplayProgressEvent, ...]:
    """Read one complete snapshot, tolerating missing or interrupted actor output."""
    try:
        payload = json.loads(_snapshot_path(progress_dir, trial_id).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("trial_id") != trial_id:
            return ()
        events = payload.get("sequences")
        if not isinstance(events, list):
            return ()
        return tuple(ReplayProgressEvent(**event) for event in events)
    except (OSError, ValueError, TypeError):
        return ()


def clear_trial_sequence_progress(progress_dir: str | Path, trial_id: str) -> None:
    """Discard one finished or restarted trial's snapshot without affecting peers."""
    try:
        _snapshot_path(progress_dir, trial_id).unlink(missing_ok=True)
    except OSError:
        pass
