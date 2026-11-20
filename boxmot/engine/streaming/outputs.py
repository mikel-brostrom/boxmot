"""Synchronized, rotated JSON metadata and bounded retention."""

from __future__ import annotations

import errno
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4


class RotatingJsonl:
    """Create append-only segments; syncing precedes every durable event ack."""

    def __init__(
        self, directory: Path, kind: str, *, max_bytes: int = 64 * 1024 * 1024, max_seconds: float = 3600
    ) -> None:
        self.directory = directory
        self.kind = kind
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        directory.mkdir(parents=True, exist_ok=True)
        # Persist new session/camera directories before acknowledging contents.
        for parent in (directory, *directory.parents):
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        self.path: Path | None = None
        self._handle: Any = None
        self._size = 0
        self._opened = 0.0
        self._failed = False

    def write(self, payload: dict[str, Any]) -> None:
        """Write strict JSON and rotate before crossing size/time thresholds."""
        if self._failed:
            raise OSError("Output segment is unusable after an earlier write failure.")
        try:
            self._write(payload)
        except Exception:
            # A partial write may leave a non-newline-terminated record. Never
            # append to or acknowledge this segment again, even during cleanup.
            self._failed = True
            raise

    def _write(self, payload: dict[str, Any]) -> None:
        data = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if (
            self._handle is None
            or self._size + len(data) > self.max_bytes
            or time.monotonic() - self._opened > self.max_seconds
        ):
            self.close()
            self.path = self.directory / f"{self.kind}-{time.time_ns()}-{uuid4().hex[:8]}.jsonl.open"
            self._handle = self.path.open("xb")
            # Persist the directory entry as well as file contents before ack.
            descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._size = 0
            self._opened = time.monotonic()
        self._handle.write(data)
        self._size += len(data)

    def sync(self) -> None:
        if self._failed:
            raise OSError("Output segment is unusable after an earlier write failure.")
        if self._handle is not None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except Exception:
                self._failed = True
                raise

    def close(self) -> None:
        if self._handle is not None:
            try:
                self.sync()
            finally:
                self._handle.close()
                self._handle = None
            if self.path is not None:
                closed_path = self.path.with_suffix("")
                self.path.rename(closed_path)
                self.path = closed_path
                descriptor = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)


class AttributeEventWriter:
    """Route replay to its original session while keeping only one segment open."""

    def __init__(self, camera_directory: Path, *, max_bytes: int, max_seconds: float) -> None:
        self.directory = camera_directory
        self.max_bytes = max_bytes
        self.max_seconds = max_seconds
        self._session: str | None = None
        self._writer: RotatingJsonl | None = None

    @property
    def path(self) -> Path | None:
        return self._writer.path if self._writer is not None else None

    def write(self, event: dict[str, Any]) -> None:
        session = event["session_id"]
        if not isinstance(session, str) or not session or Path(session).name != session or session in (".", ".."):
            raise ValueError("Invalid event session identity.")
        if session != self._session:
            self.close()
            self._session = session
            self._writer = RotatingJsonl(
                self.directory / session, "attributes", max_bytes=self.max_bytes, max_seconds=self.max_seconds
            )
        self._writer.write(event)

    def sync(self) -> None:
        if self._writer is not None:
            self._writer.sync()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        self._writer = None
        self._session = None


def export_events(
    journal: Any,
    writer: Any,
    *,
    limit: int = 100,
    on_export: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Export a bounded page at least once; duplicates retain their event IDs."""
    events = journal.read(limit)
    for event in events:
        writer.write(event)
    if events:
        writer.sync()
        journal.acknowledge(events)
        if on_export is not None:
            for event in events:
                on_export(event)
    return len(events)


def prune_outputs(
    root: Path,
    retention_days: int = 7,
    *,
    active: tuple[Path | None, ...] = (),
    live_sessions: frozenset[str] = frozenset(),
) -> int:
    """Delete only aged metadata segments, leaving active files untouched."""
    cutoff = time.time() - retention_days * 86400
    removed = 0
    if not root.exists():
        return removed
    for directory in root.iterdir():
        if directory.is_symlink() or not directory.is_dir():
            continue
        try:
            old_directory = directory.stat().st_mtime < cutoff
        except FileNotFoundError:
            continue
        for path in directory.glob("*.jsonl*"):
            if path in active or path.is_symlink():
                continue
            if path.suffix == ".open" and live_sessions:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                # The camera and its maintenance Job can prune the same segment.
                continue
        if old_directory and directory.name not in live_sessions:
            try:
                directory.rmdir()
            except OSError as exc:
                if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST, errno.ENOENT):
                    raise
    return removed
