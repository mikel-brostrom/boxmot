from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from boxmot.native import _common


@pytest.fixture(scope="module", params=["botsort", "occluboost"])
def cmc_estimator_probe(request: pytest.FixtureRequest) -> Path:
    """Build the real CMC estimators with each native tracker's types."""
    tracker = request.param
    build_dir = _common.tracker_build_dir(tracker)
    target = f"{tracker}_cmc_estimator_probe"
    for command in (
        [
            "cmake",
            "-S",
            str(_common.tracker_source_dir(tracker)),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["cmake", "--build", str(build_dir), "--config", "Release", "--target", target, "--parallel", "2"],
    ):
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stdout + completed.stderr

    executable_name = target + (".exe" if os.name == "nt" else "")
    candidates = (build_dir / "Release" / executable_name, build_dir / executable_name)
    executable = next((path for path in candidates if path.is_file()), None)
    assert executable is not None
    return executable


@pytest.mark.parametrize("method", ["sof", "ecc"])
def test_native_cmc_recovers_translation_and_resets_resolution(cmc_estimator_probe: Path, method: str) -> None:
    """Bounded estimators preserve image-space motion and restart across streams."""
    completed = subprocess.run(
        [str(cmc_estimator_probe), method], capture_output=True, text=True, timeout=10, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
