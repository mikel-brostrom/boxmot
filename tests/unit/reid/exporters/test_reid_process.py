from __future__ import annotations

import subprocess
import sys

import pytest

from boxmot.reid.exporters.process import run_limited


def test_run_limited_returns_captured_output():
    process = run_limited(
        [sys.executable, "-c", "print('ready')"],
        timeout_s=5,
        max_memory_gb=1,
    )

    assert process.returncode == 0
    assert process.stdout.strip() == "ready"


def test_run_limited_enforces_timeout():
    with pytest.raises(subprocess.TimeoutExpired):
        run_limited(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_s=0.1,
            max_memory_gb=1,
        )


def test_run_limited_enforces_resident_memory_cap():
    with pytest.raises(MemoryError, match="exceeded"):
        run_limited(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_s=5,
            max_memory_gb=0.005,
        )
