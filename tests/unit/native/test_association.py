from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

from boxmot.native import _common
from boxmot.native.reid.capi import ensure_reid_capi_library
from boxmot.trackers.common.association.iou import AssociationFunction

OBB_MODES = ("iou", "giou", "diou", "ciou", "hmiou", "centroid")
ULTRA_THIN_OBB = np.array([20.0, 30.0, 1.0e-12, 4.0, 0.3], dtype=np.float64)
EQUIVALENT_ULTRA_THIN_OBB = np.array([20.0, 30.0, 4.0, 1.0e-12, 0.3 + np.pi / 2.0], dtype=np.float64)


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


def _native_aabb_similarity(association_probe, mode: str, lhs: np.ndarray, rhs: np.ndarray) -> float:
    completed = subprocess.run(
        [str(association_probe), mode, "aabb", *map(str, lhs), *map(str, rhs)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return float(completed.stdout)


def _rotate_obb(box: np.ndarray, angle: float) -> np.ndarray:
    rotated = np.asarray(box, dtype=float).copy()
    cosine = np.cos(angle)
    sine = np.sin(angle)
    rotation = np.array(((cosine, -sine), (sine, cosine)))
    rotated[:2] = rotation @ rotated[:2]
    rotated[4] += angle
    return rotated


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
    expected = AssociationFunction(w=100, h=80, asso_mode=effective_mode).asso_func(lhs[None, :], rhs[None, :])[0, 0]

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


@pytest.mark.parametrize(
    "invalid_box",
    (
        np.array([0.0, 0.0, 0.0, 2.0, 0.0]),
        np.array([0.0, 0.0, 2.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 2.0, 1.0, np.nan]),
        np.array([np.inf, 0.0, 2.0, 1.0, 0.0]),
    ),
)
def test_native_obb_association_rejects_nonpositive_or_nonfinite_geometry(association_probe, invalid_box):
    valid_box = np.array([0.0, 0.0, 2.0, 1.0, 0.0])

    completed = subprocess.run(
        [str(association_probe), "iou", "obb", *map(str, invalid_box), *map(str, valid_box)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert "finite" in completed.stderr or "strictly positive" in completed.stderr


@pytest.mark.parametrize(
    "candidate",
    (
        np.array([0.0, 0.0, 1.0e6, 1.0, 9.0e-7]),
        np.array([0.0, 0.0, 1.0, 1.0e6, np.pi / 2.0 + 9.0e-7]),
    ),
    ids=("direct", "swapped"),
)
def test_native_obb_iou_does_not_collapse_near_angle_extreme_box_to_identity(
    association_probe,
    candidate,
):
    reference = np.array([0.0, 0.0, 1.0e6, 1.0, 0.0])

    similarity = _native_obb_similarity(association_probe, "iou", reference, candidate)

    assert similarity == pytest.approx(0.6326530612, abs=1e-9)
    assert similarity < 1.0


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    (
        (
            np.array([0.0, 0.0, 300.0, 1.0, 0.0]),
            np.array([3.0, 1.0, 230.0, 1.25, 0.001]),
            0.05145403049909077,
        ),
        (
            np.array([0.0, 0.0, 300.0, 1.0, 0.7]),
            np.array([3.0, 1.0, 230.0, 1.25, 0.7005]),
            0.0003673997508027863,
        ),
    ),
    ids=("partial-overlap", "thin-overlap"),
)
def test_native_moderately_thin_obb_iou_matches_python(association_probe, reference, candidate, expected):
    python_similarity = AssociationFunction.iou_batch_obb(reference[None, :], candidate[None, :])[0, 0]
    native_similarity = _native_obb_similarity(association_probe, "iou", reference, candidate)

    assert python_similarity == pytest.approx(expected, abs=1e-14)
    assert native_similarity == pytest.approx(python_similarity, abs=1e-14)


def test_native_obb_iou_does_not_collapse_distinct_large_centers_to_identity(
    association_probe,
):
    reference = np.array([1.0e9, -1.0e9, 10.0, 2.0, 0.3])
    displaced = reference.copy()
    displaced[0] += 1.0e-6

    similarity = _native_obb_similarity(association_probe, "iou", reference, displaced)

    assert 0.999 < similarity < 1.0


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_native_obb_metrics_are_scale_invariant_below_former_side_floor(association_probe, mode):
    reference = np.array([0.0, 0.0, 4.0, 2.0, 0.3])
    candidate = np.array([1.2, 0.4, 3.0, 1.5, -0.2])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[:4] *= 1.0e-5
    scaled_candidate[:4] *= 1.0e-5

    baseline = _native_obb_similarity(association_probe, mode, reference, candidate)
    scaled = _native_obb_similarity(association_probe, mode, scaled_reference, scaled_candidate)

    assert scaled == pytest.approx(baseline, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
@pytest.mark.parametrize("scale", (1e-200, 1e200), ids=("subnormal-area", "overflow-area"))
def test_native_obb_metrics_keep_dimensionless_terms_across_float64_scales(association_probe, mode, scale):
    reference = np.array([0.0, 0.0, 4.0, 2.0, 0.3])
    candidate = np.array([1.2, 0.4, 3.0, 1.5, -0.2])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[:4] *= scale
    scaled_candidate[:4] *= scale

    baseline = _native_obb_similarity(association_probe, mode, reference, candidate)
    scaled = _native_obb_similarity(association_probe, mode, scaled_reference, scaled_candidate)
    python_scaled = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func(
        scaled_reference[None, :], scaled_candidate[None, :]
    )[0, 0]

    assert scaled == pytest.approx(baseline, rel=1e-12, abs=1e-12)
    assert scaled == pytest.approx(python_scaled, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_native_obb_metrics_share_huge_angle_canonicalization_with_python(association_probe, mode):
    reference = np.array([0.0, 0.0, 4.0, 2.0, 1e20])
    candidate = np.array([0.5, 0.2, 3.0, 1.5, 1e20])

    expected = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func(
        reference[None, :], candidate[None, :]
    )[0, 0]
    actual = _native_obb_similarity(association_probe, mode, reference, candidate)

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", ("iou", "hmiou"))
def test_native_obb_metrics_preserve_overlap_near_float64_limit(association_probe, mode):
    reference = np.array([-0.9, 0.0, 1.79, 1.79, np.pi / 4.0])
    candidate = np.array([0.9, 0.0, 1.79, 1.79, np.pi / 4.0])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[:4] *= 1e308
    scaled_candidate[:4] *= 1e308

    baseline = _native_obb_similarity(association_probe, mode, reference, candidate)
    scaled = _native_obb_similarity(association_probe, mode, scaled_reference, scaled_candidate)
    python_scaled = AssociationFunction(w=100, h=80, asso_mode=f"{mode}_obb").asso_func(
        scaled_reference[None, :], scaled_candidate[None, :]
    )[0, 0]

    assert scaled == pytest.approx(baseline, rel=1e-12, abs=1e-12)
    assert scaled == pytest.approx(python_scaled, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mode", ("iou", "giou", "diou", "ciou", "hmiou"))
def test_native_obb_metrics_do_not_treat_distinct_subnanometric_swapped_boxes_as_equivalent(
    association_probe,
    mode,
):
    reference = np.array([0.0, 0.0, 2.0, 1.0, 0.0])
    candidate = np.array([0.0, 0.0, 1.0, 2.0, 0.0])
    scaled_reference = reference.copy()
    scaled_candidate = candidate.copy()
    scaled_reference[2:4] *= 1e-12
    scaled_candidate[2:4] *= 1e-12

    baseline = _native_obb_similarity(association_probe, mode, reference, candidate)
    scaled = _native_obb_similarity(association_probe, mode, scaled_reference, scaled_candidate)

    assert scaled == pytest.approx(baseline, rel=1e-12, abs=1e-12)
    assert scaled < 1.0


def test_native_obb_giou_preserves_tiny_geometry_across_far_diagonal_centers(association_probe):
    reference = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    candidate = np.array([1e20, 1e20, 1.0, 1.0, 0.0])
    expected = AssociationFunction(w=100, h=80, asso_mode="giou_obb").asso_func(reference[None, :], candidate[None, :])[
        0, 0
    ]

    direct = _native_obb_similarity(association_probe, "giou", reference, candidate)
    reversed_pair = _native_obb_similarity(association_probe, "giou", candidate, reference)
    rotated = _native_obb_similarity(
        association_probe,
        "giou",
        _rotate_obb(reference, 0.73),
        _rotate_obb(candidate, 0.73),
    )

    assert direct == pytest.approx(expected, abs=1e-12)
    assert reversed_pair == pytest.approx(direct, abs=1e-12)
    assert rotated == pytest.approx(direct, abs=1e-12)
    assert direct < 1e-12


@pytest.mark.parametrize("mode", ("giou", "diou", "ciou"))
def test_native_obb_enclosure_metrics_are_invariant_to_rigid_rotation(association_probe, mode):
    reference = np.array([0.0, 0.0, 4.0, 1.0, 0.3])
    candidate = np.array([1.2, 0.4, 3.0, 2.0, -0.2])
    rotation_angle = 0.73

    baseline = _native_obb_similarity(association_probe, mode, reference, candidate)
    rotated = _native_obb_similarity(
        association_probe,
        mode,
        _rotate_obb(reference, rotation_angle),
        _rotate_obb(candidate, rotation_angle),
    )

    assert rotated == pytest.approx(baseline, rel=1e-12, abs=1e-12)


def test_native_obb_iou_resolves_near_full_extreme_elongated_overlap(association_probe):
    width = 1.0e12
    angle = 0.73
    reference = np.array([0.0, 0.0, width, 1.0, angle])
    candidate = reference.copy()
    candidate[:2] += np.array((np.cos(angle), np.sin(angle)))

    similarity = _native_obb_similarity(association_probe, "iou", reference, candidate)

    assert similarity == pytest.approx((width - 1.0) / (width + 1.0), rel=1e-14)
    assert similarity < 1.0


@pytest.mark.parametrize("mode", ("giou", "diou", "ciou"))
def test_native_normalized_obb_similarities_are_clamped_to_unit_interval(association_probe, mode):
    obb_reference = np.array([0.0, 0.0, 100.0, 1.0, 0.0])
    obb_distant = np.array([10000.0, 0.0, 1.0, 1.0, 0.0])

    similarity = _native_obb_similarity(association_probe, mode, obb_reference, obb_distant)

    assert 0.0 <= similarity <= 1.0


def test_native_aabb_ciou_similarity_is_clamped_to_unit_interval(association_probe):
    aabb_reference = np.array([-50.0, -0.5, 50.0, 0.5])
    aabb_distant = np.array([9999.5, -0.5, 10000.5, 0.5])

    aabb_similarity = _native_aabb_similarity(association_probe, "ciou", aabb_reference, aabb_distant)

    assert aabb_similarity == 0.0


@pytest.mark.parametrize("mode", OBB_MODES)
@pytest.mark.parametrize(
    ("geometry", "expected_shape"),
    (("obb-empty-lhs", "0 1"), ("obb-empty-rhs", "1 0")),
)
def test_native_obb_association_preserves_empty_matrix_shape(association_probe, mode, geometry, expected_shape):
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
