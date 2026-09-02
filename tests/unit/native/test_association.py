from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from boxmot.native import _common
from boxmot.native.reid.capi import ensure_reid_capi_library
from boxmot.trackers.common.association.iou import AssociationFunction


@pytest.fixture(scope="module")
def association_probe():
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
            "boxmot_association_probe",
            "--parallel",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    executable_name = "boxmot_association_probe.exe" if os.name == "nt" else "boxmot_association_probe"
    candidates = (build_dir / "Release" / executable_name, build_dir / executable_name)
    executable = next((path for path in candidates if path.is_file()), None)
    assert executable is not None
    return executable


@pytest.mark.parametrize("mode", ["iou", "giou", "diou", "ciou", "hmiou", "centroid"])
def test_native_aabb_association_matches_python(association_probe, mode):
    lhs = np.array([[0.0, 0.0, 10.0, 10.0]])
    rhs = np.array([[5.0, 2.0, 15.0, 12.0]])
    expected = AssociationFunction(w=100, h=80, asso_mode=mode).asso_func(lhs, rhs)[0, 0]

    completed = subprocess.run(
        [str(association_probe), mode],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("mode", ["iou", "diou", "centroid"])
def test_native_obb_association_matches_python(association_probe, mode):
    lhs = np.array([[10.0, 10.0, 8.0, 4.0, 0.2]])
    rhs = np.array([[12.0, 11.0, 8.0, 4.0, 0.25]])
    effective_mode = {"iou": "iou_obb", "diou": "diou_obb", "centroid": "centroid_obb"}[mode]
    expected = AssociationFunction(w=100, h=80, asso_mode=effective_mode).asso_func(lhs, rhs)[0, 0]

    completed = subprocess.run(
        [str(association_probe), mode, "obb"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout) == pytest.approx(expected, abs=1e-5)


def test_native_obb_association_rejects_unsupported_metric(association_probe):
    completed = subprocess.run(
        [str(association_probe), "giou", "obb"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert "no oriented-box implementation" in completed.stderr


def test_native_association_rejects_unknown_metric(association_probe):
    completed = subprocess.run(
        [str(association_probe), "made-up"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert "Unknown association function" in completed.stderr
