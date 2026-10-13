"""Exercise worker lifetime protection against real, uncatchable owner exits."""

from __future__ import annotations

import ctypes
import importlib
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event
from pathlib import Path

import pytest

from boxmot.utils.processes import guard_parent_lifetime

_TIMEOUT = 30


def _worker_state(_: object) -> tuple[int, int]:
    """Repeated initialization must retain exactly one owner watcher."""
    guard_parent_lifetime()
    guard_parent_lifetime()
    watchers = sum(thread.name == "boxmot-parent-watcher" for thread in threading.enumerate())
    return os.getpid(), watchers


def _guarded_worker(messages: Connection, initialize: Event, state: str) -> None:
    """Publish readiness before blocking either the initializer or worker body."""
    if state == "initializing":
        messages.send((os.getpid(), 0))
        if not initialize.wait(_TIMEOUT):
            raise TimeoutError("The harness did not release worker initialization.")
    guard_parent_lifetime()
    messages.send(_worker_state(None))
    if state == "busy":
        while True:
            sum(range(10_000))
    else:
        threading.Event().wait(_TIMEOUT * 3)


def _own_guarded_worker(messages: Connection, initialize: Event, state: str) -> None:
    """Represent a tuning actor with one child whose work never finishes."""
    # Fork children must replace an inherited watcher whose thread no longer exists.
    guard_parent_lifetime()
    context = multiprocessing.get_context()
    worker = context.Process(target=_guarded_worker, args=(messages, initialize, state))
    worker.start()
    threading.Event().wait(_TIMEOUT * 3)


def _cleanup_children() -> None:
    """Kill and reap only descendants of this isolated test harness."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        children = {int(pid) for pid in Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()} - {
            resource_tracker._resource_tracker._pid
        }
        if not children:
            return
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        time.sleep(0.02)
    raise AssertionError("The test harness could not reap its own descendants.")


def _crash_harness(target: str, method: str, state: str) -> None:
    """Adopt orphan test workers locally, so even a regression leaves no zombies."""
    # PR_SET_CHILD_SUBREAPER affects this subprocess only, never the pytest host.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Could not become a test child subreaper.")
    module, name = target.split(":")
    context = multiprocessing.get_context(method)
    received, sent = context.Pipe(duplex=False)
    initialize = context.Event()
    owner = context.Process(target=getattr(importlib.import_module(module), name), args=(sent, initialize, state))
    try:
        owner.start()
        sent.close()
        assert received.poll(_TIMEOUT), "Worker never reported readiness."
        worker_pid, watchers = received.recv()
        assert watchers == (0 if state == "initializing" else 1)
        os.kill(owner.pid, signal.SIGKILL)
        owner.join(timeout=10)
        assert owner.exitcode == -signal.SIGKILL
        initialize.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            reaped, status = os.waitpid(worker_pid, os.WNOHANG)
            if reaped:
                assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 1
                break
            time.sleep(0.02)
        else:
            raise AssertionError(f"Worker {worker_pid} survived the death of its owner.")
    finally:
        if owner.is_alive():
            owner.kill()
        owner.join(timeout=10)
        received.close()
        sent.close()
        del initialize
        _cleanup_children()
        # Let the tracker unlink resources abandoned by the killed actor, then reap it.
        resource_tracker._resource_tracker._stop()


def run_parent_crash_harness(target: str, *, method: str = "spawn", state: str = "idle") -> None:
    """Run deliberate owner deaths in a dedicated, bounded process group."""
    process = subprocess.Popen(
        [sys.executable, "-m", "tests.unit.utils.test_processes", target, method, state],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=_TIMEOUT + 25)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate(timeout=10)
        pytest.fail(f"Worker lifetime harness timed out.\n{stdout}\n{stderr}")
    assert process.returncode == 0, f"Worker lifetime harness failed.\n{stdout}\n{stderr}"


@pytest.mark.skipif(sys.platform != "linux", reason="Scoped orphan reaping requires Linux subreapers.")
@pytest.mark.parametrize("method", ["spawn", "fork"])
@pytest.mark.parametrize("state", ["idle", "busy"])
def test_guard_stops_workers_when_owner_is_killed(method: str, state: str) -> None:
    run_parent_crash_harness(f"{__name__}:_own_guarded_worker", method=method, state=state)


@pytest.mark.skipif(sys.platform != "linux", reason="Scoped orphan reaping requires Linux subreapers.")
def test_guard_handles_owner_death_before_initialization() -> None:
    run_parent_crash_harness(f"{__name__}:_own_guarded_worker", state="initializing")


def test_guard_preserves_normal_worker_reuse_and_shutdown() -> None:
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=guard_parent_lifetime,
    ) as executor:
        first = executor.submit(_worker_state, None).result(timeout=_TIMEOUT)
        second = executor.submit(_worker_state, None).result(timeout=_TIMEOUT)
        assert first == second
        assert first[1] == 1
        workers = tuple(executor._processes.values())
    assert all(worker.exitcode == 0 for worker in workers)


if __name__ == "__main__":
    _crash_harness(*sys.argv[1:])
