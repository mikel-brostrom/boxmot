"""Replay workers must exit when a tuning actor is killed without cleanup."""

from __future__ import annotations

import os
import sys
import threading
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event

import pytest

from tests.unit.utils.test_processes import run_parent_crash_harness


def _replay_worker(busy: bool) -> tuple[int, int]:
    """Stand in for a sequence task without loading datasets or models."""
    state = os.getpid(), sum(thread.name == "boxmot-parent-watcher" for thread in threading.enumerate())
    if busy:
        from boxmot.engine.eval.replay import _WORKER_PROGRESS_QUEUE

        _WORKER_PROGRESS_QUEUE.put(state)
        threading.Event().wait(90)
    return state


def _own_replay_workers(messages: Connection, initialize: Event, state: str) -> None:
    """Keep the real session idle between trials or blocked inside a trial."""
    from boxmot.engine.eval.session import ReplaySession

    with ReplaySession(1) as session:
        if state == "busy":
            executor = session._start()
            executor.submit(_replay_worker, True)
            messages.send(session._progress_queue.get(timeout=30))
            threading.Event().wait(90)
        else:
            first = session.map(_replay_worker, [False])[0]
            assert session.map(_replay_worker, [False])[0] == first
            messages.send(first)
            threading.Event().wait(90)


@pytest.mark.skipif(sys.platform != "linux", reason="Scoped orphan reaping requires Linux subreapers.")
@pytest.mark.parametrize("state", ["idle", "busy"])
def test_replay_workers_exit_after_actor_is_killed(state: str) -> None:
    run_parent_crash_harness(f"{__name__}:_own_replay_workers", state=state)
