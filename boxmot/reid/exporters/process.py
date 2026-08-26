"""Process guards for memory-intensive model conversion and compilation."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessMemoryGuard:
    """Terminate a process group when aggregate resident memory crosses a cap."""

    process: subprocess.Popen[Any]
    max_memory_gb: float
    poll_interval_s: float = 0.1
    exceeded: threading.Event = field(default_factory=threading.Event, init=False)
    peak_rss_bytes: int = field(default=0, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)

    def start(self) -> "ProcessMemoryGuard":
        if self.max_memory_gb <= 0:
            return self
        self._thread = threading.Thread(
            target=self._monitor,
            name=f"boxmot-memory-guard-{self.process.pid}",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _monitor(self) -> None:
        import psutil

        limit_bytes = int(self.max_memory_gb * 1024**3)
        while not self._stop.wait(self.poll_interval_s):
            if self.process.poll() is not None:
                return
            try:
                root = psutil.Process(self.process.pid)
                processes = [root, *root.children(recursive=True)]
                rss_bytes = sum(child.memory_info().rss for child in processes if child.is_running())
            except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError):
                continue
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss_bytes)
            if rss_bytes > limit_bytes:
                self.exceeded.set()
                terminate_process_tree(self.process)
                return


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    """Terminate one isolated process group without touching the caller."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        import psutil

        try:
            root = psutil.Process(process.pid)
            children = root.children(recursive=True)
            for child in children:
                child.terminate()
            root.terminate()
            _, alive = psutil.wait_procs([*children, root], timeout=5)
            for child in alive:
                child.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess, ProcessLookupError):
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            process.kill()


def run_limited(
    command: list[str],
    *,
    timeout_s: float,
    max_memory_gb: float,
) -> subprocess.CompletedProcess[str]:
    """Run a captured subprocess with timeout, memory and process-group guards."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    memory_guard = ProcessMemoryGuard(process, max_memory_gb).start()
    try:
        stdout, _ = process.communicate(timeout=timeout_s if timeout_s > 0 else None)
    except BaseException:
        terminate_process_tree(process)
        raise
    finally:
        memory_guard.stop()
    if memory_guard.exceeded.is_set():
        peak_gb = memory_guard.peak_rss_bytes / 1024**3
        raise MemoryError(
            f"subprocess exceeded {max_memory_gb:.1f} GB RAM (observed at least {peak_gb:.1f} GB) and was terminated"
        )
    return subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=stdout,
        stderr="",
    )
