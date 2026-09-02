from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from boxmot.native import _common
from boxmot.native.reid.capi import ensure_reid_capi_library
from boxmot.trackers.common.association.iou import AssociationFunction

OBB_MODES = ("iou", "giou", "diou", "ciou", "hmiou", "centroid")
ULTRA_THIN_OBB = np.array([20.0, 30.0, 1.0e-12, 4.0, 0.3], dtype=np.float32)
EQUIVALENT_ULTRA_THIN_OBB = np.array(
    [20.0, 30.0, 4.0, 1.0e-12, 0.3 + np.pi / 2.0], dtype=np.float32
)


def _native_obb_similarity(association_probe, mode: str, lhs: np.ndarray, rhs: np.ndarray) -> float:
    completed = subprocess.run(
        [str(association_probe), mode, "obb", *map(str, lhs), *map(str, rhs)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return float(completed.stdout)


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


@pytest.mark.parametrize("mode", OBB_MODES)
@pytest.mark.parametrize(
    ("lhs", "rhs"),
    (
        (np.array([10.0, 10.0, 8.0, 4.0, 0.2]), np.array([12.0, 11.0, 8.0, 4.0, 0.25])),
        (np.array([10.0, 10.0, 8.0, 4.0, 0.2]), np.array([70.0, 60.0, 6.0, 3.0, -0.7])),
    ),
    ids=("overlapping", "disjoint"),
)
def test_native_obb_association_matches_python(association_probe, mode, lhs, rhs):
    effective_mode = f"{mode}_obb"
    expected = AssociationFunction(w=100, h=80, asso_mode=effective_mode).asso_func(
        lhs[None, :], rhs[None, :]
    )[0, 0]

    actual = _native_obb_similarity(association_probe, mode, lhs, rhs)

    assert actual == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("mode", OBB_MODES)
def test_native_obb_association_is_invariant_to_equivalent_representation(association_probe, mode):
    lhs = np.array([10.0, 10.0, 8.0, 4.0, 0.2])
    rhs = np.array([12.0, 11.0, 7.0, 3.0, 0.25])
    equivalent_rhs = rhs.copy()
    equivalent_rhs[2:4] = equivalent_rhs[[3, 2]]
    equivalent_rhs[4] += np.pi / 2.0

    direct = _native_obb_similarity(association_probe, mode, lhs, rhs)
    equivalent = _native_obb_similarity(association_probe, mode, lhs, equivalent_rhs)

    assert equivalent == pytest.approx(direct, abs=1e-5)


@pytest.mark.parametrize("mode", OBB_MODES)
@pytest.mark.parametrize(
    "rhs",
    (ULTRA_THIN_OBB, EQUIVALENT_ULTRA_THIN_OBB),
    ids=("identical", "equivalent"),
)
def test_native_ultra_thin_obb_association_matches_python(association_probe, mode, rhs):
    expected = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func(
        ULTRA_THIN_OBB[None, :], rhs[None, :]
    )[0, 0]
    actual = _native_obb_similarity(association_probe, mode, ULTRA_THIN_OBB, rhs)

    assert expected == pytest.approx(1.0, abs=1e-5)
    assert actual == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize("mode", OBB_MODES)
@pytest.mark.parametrize(
    ("geometry", "expected_shape"),
    (("obb-empty-lhs", "0 1"), ("obb-empty-rhs", "1 0")),
)
def test_native_obb_association_preserves_empty_matrix_shape(
    association_probe, mode, geometry, expected_shape
):
    completed = subprocess.run(
        [str(association_probe), mode, geometry],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == expected_shape


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
