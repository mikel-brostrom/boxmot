"""Crash replay, synchronization errors, rotation, and safe retention."""

import json
import os
import time

import pytest

from boxmot.engine.streaming.health import StreamHealth
from boxmot.engine.streaming.outputs import AttributeEventWriter, RotatingJsonl, export_events, prune_outputs


class Journal:
    def __init__(self):
        self.events = [{"session_id": "session", "event_id": "session:1", "status": "completed"}]
        self.crash = False

    def read(self, limit):
        return self.events[:limit]

    def acknowledge(self, events):
        if self.crash:
            raise RuntimeError("crash before acknowledgment")
        assert events == self.events
        self.events = []


def records(directory):
    return [json.loads(line) for path in directory.rglob("*.jsonl*") for line in path.read_text().splitlines()]


def test_crash_after_sync_replays_stable_duplicate_ids(tmp_path):
    journal = Journal()
    health = StreamHealth()
    writer = AttributeEventWriter(tmp_path, max_bytes=1000, max_seconds=3600)
    journal.crash = True
    with pytest.raises(RuntimeError, match="acknowledgment"):
        export_events(journal, writer, on_export=health.observe_attribute_event)
    assert health.attribute_events["completed"] == 0
    writer.close()
    assert records(tmp_path)[0]["event_id"] == "session:1"
    journal.crash = False
    replay = AttributeEventWriter(tmp_path, max_bytes=1000, max_seconds=3600)
    assert export_events(journal, replay, on_export=health.observe_attribute_event) == 1
    assert health.attribute_events["completed"] == 1
    replay.close()
    assert [e["event_id"] for e in records(tmp_path)] == ["session:1", "session:1"]
    assert export_events(journal, replay, on_export=health.observe_attribute_event) == 0
    assert health.attribute_events["completed"] == 1
    assert all(path.parent.name == "session" for path in tmp_path.rglob("*.jsonl"))


def test_sync_failure_leaves_event_unacknowledged(tmp_path, monkeypatch):
    writer = RotatingJsonl(tmp_path, "attributes")
    journal = Journal()

    def fail():
        raise OSError("storage unavailable")

    monkeypatch.setattr(writer, "sync", fail)
    with pytest.raises(OSError, match="storage"):
        export_events(journal, writer)
    assert journal.read(100)
    monkeypatch.undo()
    writer.close()


def test_rotation_size_time_nonfinite_and_retention(tmp_path):
    directory = tmp_path / "session"
    writer = RotatingJsonl(directory, "tracks", max_bytes=16, max_seconds=1)
    writer.write({"a": 1})
    writer.write({"b": 2})
    writer.write({"c": 3})  # Size rotation.
    writer._opened -= 2
    writer.write({"d": 4})  # Time rotation.
    writer.sync()
    paths = list(directory.glob("*.jsonl*"))
    assert len(paths) == 3
    old = time.time() - 8 * 86400
    for path in paths:
        os.utime(path, (old, old))
    assert prune_outputs(tmp_path, active=(writer.path,)) == 2
    assert writer.path.exists()
    assert prune_outputs(tmp_path, live_sessions=frozenset({"session"})) == 0
    writer.close()
    os.utime(writer.path, (old, old))
    assert prune_outputs(tmp_path) == 1


def test_partial_write_is_fatal_even_if_cleanup_attempts_export_again(tmp_path):
    writer = RotatingJsonl(tmp_path, "attributes")
    writer.write({"earlier": "committed"})
    writer.sync()
    original = writer._handle

    class PartialWrite:
        def write(self, data):
            original.write(data[:10])
            original.flush()
            raise OSError("disk full after partial write")

        def __getattr__(self, name):
            return getattr(original, name)

    writer._handle = PartialWrite()
    journal = Journal()
    with pytest.raises(OSError, match="partial write"):
        export_events(journal, writer)
    with pytest.raises(OSError, match="unusable"):
        export_events(journal, writer)
    with pytest.raises(OSError, match="unusable"):
        writer.close()
    assert journal.read(100) and original.closed
    replay = RotatingJsonl(tmp_path, "attributes")
    assert export_events(journal, replay) == 1
    replay.close()
    assert not journal.read(100)


def test_nonfinite_record_fails_before_persistence(tmp_path):
    writer = RotatingJsonl(tmp_path, "tracks")
    with pytest.raises(ValueError):
        writer.write({"invalid": float("inf")})
    assert not list(tmp_path.iterdir())
