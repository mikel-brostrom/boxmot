from __future__ import annotations

import os
import subprocess

import pytest

from boxmot.native import _common
from boxmot.native.reid.capi import ensure_reid_capi_library


@pytest.fixture(scope="module")
def assignment_probe():
    ensure_reid_capi_library()
    build_dir = _common.tracker_build_dir("base")
    completed = subprocess.run(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "boxmot_assignment_probe",
            "--parallel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    executable_name = "boxmot_assignment_probe.exe" if os.name == "nt" else "boxmot_assignment_probe"
    candidates = (build_dir / "Release" / executable_name, build_dir / executable_name)
    executable = next((path for path in candidates if path.is_file()), None)
    assert executable is not None
    return executable


@pytest.mark.parametrize("mode", ["nonfinite", "near-tie"])
def test_native_assignment_regressions(assignment_probe, mode):
    completed = subprocess.run(
        [str(assignment_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
