"""Smoke-test Rich screen and keyboard ownership in an actual pseudoterminal."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from tests._paths import REPO_ROOT

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Uses a Unix pseudoterminal")


@contextmanager
def _terminal_process(arguments: list[str]) -> Iterator[tuple[subprocess.Popen, int]]:
    """Launch a child attached to a terminal and assert it restores input mode."""
    import fcntl
    import struct
    import termios

    master, slave = os.openpty()
    previous = termios.tcgetattr(slave)
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 28, 110, 0, 0))
    environment = {**os.environ, "TERM": "xterm-256color"}
    environment.pop("BOXMOT_NO_ALT_SCREEN", None)
    process = subprocess.Popen(
        [sys.executable, *arguments], cwd=REPO_ROOT, env=environment,
        stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
    )
    try:
        yield process, master
        assert process.wait(timeout=10) == 0
        assert termios.tcgetattr(slave) == previous
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        os.close(slave)


def _until(master: int, expected: str, timeout: float = 10) -> str:
    """Wait for a visible state while draining output so a child cannot stall."""
    deadline = time.monotonic() + timeout
    output = b""
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master], [], [], 0.1)
        if ready:
            output += os.read(master, 65536)
            if expected.encode() in output:
                return output.decode(errors="replace")
    pytest.fail(f"Terminal did not display {expected!r}:\n{output.decode(errors='replace')[-3000:]}")


def test_launcher_review_cancel_restores_terminal() -> None:
    """Exercise the actual module entrypoint, selection preview, and cancel path."""
    with _terminal_process([
        "-m", "boxmot.engine.cli", "ui", "--command", "eval",
        "--experiment", "mot17/ablation-yolox-lmbn.yaml",
    ]) as (process, master):
        initial = _until(master, "Command preview")
        assert "Sentinel.UNSET" not in initial
        os.write(master, b"r")
        review = _until(master, "Ready to run")
        assert "--dashboard" in review
        os.write(master, b"\x1b")
        _until(master, "Configure")
        os.write(master, b"q")
        _until(master, "\x1b[?1049l")


def test_dashboard_can_navigate_during_run_and_browse_final_results() -> None:
    """A live input thread must not block progress, final results, or cleanup."""
    code = """
import time
from rich.text import Text
from boxmot.engine.ui.core.ui import WorkflowProgress
from boxmot.engine.ui.dashboard import dashboard_mode
from boxmot.engine.ui.workflow.pipeline import PipelineTracker

workflow = WorkflowProgress(
    'Terminal smoke', [('Detector', 'selected-detector')],
    steps=[('Set up', 'active'), ('Run tracker', 'todo')],
)
with dashboard_mode():
    with PipelineTracker(workflow, wire_status_fns=False) as pipeline:
        pipeline.advance('Processing')
        for frame in range(30):
            workflow.set_detail('Progress', f'Frame {frame}')
            time.sleep(0.04)
        pipeline.finish(Text('FINAL SMOKE RESULT'))
"""
    with _terminal_process(["-c", code]) as (process, master):
        _until(master, "Overview")
        os.write(master, b"4")
        _until(master, "selected-detector")
        _until(master, "Completed")
        assert process.poll() is None
        os.write(master, b"q")
        _until(master, "\x1b[?1049l")
