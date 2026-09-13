"""Scalar parity and geometry invariants for batched camera compensation."""

from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.geometry.obb import (
    align_obb_measurement,
    align_obb_measurements,
    transform_aabb,
    transform_aabb_kalman_state,
    transform_aabbs,
    transform_obb,
    transform_obb_kalman_state,
    transform_obbs,
    xywha_to_corners,
)
from boxmot.trackers.common.motion.cmc.state import transform_aabb_kalman_states, transform_obb_kalman_states
from boxmot.trackers.common.motion.models import create_motion_model

TRANSFORMS = [
    pytest.param(np.eye(2, 3), id="identity"),
    pytest.param(np.array([[0.96, -0.28, 12.0], [0.28, 0.96, -7.0]]), id="similarity"),
    pytest.param(np.array([[1.1, 0.15, -2.0], [0.03, 0.9, 5.0]]), id="affine"),
    pytest.param(np.array([[-1.0, 0.0, 20.0], [0.0, 1.0, -5.0]]), id="reflection"),
    pytest.param(np.array([[1.0, 0.05, 3.0], [-0.02, 1.0, -2.0], [0.0005, -0.0003, 1.0]]), id="homography"),
]


def _boxes(count: int, *, is_obb: bool) -> np.ndarray:
    """Include thin boxes and angles close to wrapping boundaries."""
    rng = np.random.default_rng(801)
    centers = rng.uniform(20.0, 200.0, (count, 2))
    sizes = rng.uniform(0.2, 80.0, (count, 2))
    if count > 1:
        sizes[-1] = [1e-8, 1e-7]
    if is_obb:
        return np.column_stack([centers, sizes, np.linspace(np.pi - 1e-4, -np.pi + 1e-4, count)])
    return np.column_stack([centers - sizes / 2.0, centers + sizes / 2.0])


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("count", [0, 1, 9])
def test_transform_aabbs_matches_scalar_and_preserves_metadata(transform, count):
    boxes = np.column_stack([_boxes(count, is_obb=False), np.arange(count), np.full(count, 0.8)])
    original = boxes.copy()
    actual = transform_aabbs(boxes, transform)
    expected = np.asarray([transform_aabb(box, transform) for box in boxes]).reshape(count, 6)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(boxes, original)
    np.testing.assert_array_equal(actual[:, 4:], boxes[:, 4:])


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("count", [0, 1, 9])
@pytest.mark.parametrize("with_reference", [False, True])
def test_transform_obbs_matches_scalar(transform, count, with_reference):
    boxes = _boxes(count, is_obb=True)
    original = boxes.copy()
    references = boxes.copy() if with_reference else None
    if references is not None:
        references[:, 4] += 2.0 * np.pi
    actual = transform_obbs(boxes, transform, reference=references)
    expected = np.asarray(
        [
            transform_obb(box, transform, reference=None if references is None else references[index])
            for index, box in enumerate(boxes)
        ]
    ).reshape(count, 5)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(boxes, original)


@pytest.mark.parametrize("kind", ["xywh", "xyah", "xysr", "xyhr"])
@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("column", [False, True])
def test_batch_kalman_transform_matches_scalar(kind, is_obb, transform, column):
    model = create_motion_model(kind, is_obb=is_obb)
    measurement_size = 5 if is_obb else 4
    velocity_indices = (0, 1, 2, 4) if is_obb and kind == "xysr" else tuple(range(measurement_size))
    if not is_obb and kind == "xysr":
        velocity_indices = (0, 1, 2)
    boxes = _boxes(7, is_obb=is_obb)
    measurements = np.asarray([model.to_measurement(box, column=False) for box in boxes])
    rng = np.random.default_rng(980)
    means = np.column_stack([measurements, rng.normal(size=(len(boxes), len(velocity_indices)))])
    factors = rng.normal(size=(len(boxes), means.shape[1], means.shape[1]))
    covariances = factors @ factors.swapaxes(1, 2) + np.eye(means.shape[1])
    if column:
        means = means[:, :, None]
    original_means = means.copy()
    original_covariances = covariances.copy()
    scalar = transform_obb_kalman_state if is_obb else transform_aabb_kalman_state
    batch = transform_obb_kalman_states if is_obb else transform_aabb_kalman_states
    expected = [
        scalar(
            mean,
            covariance,
            transform,
            measurement_to_box=model.to_box,
            box_to_measurement=lambda box: model.to_measurement(box, column=False),
            velocity_measurement_indices=velocity_indices,
        )
        for mean, covariance in zip(means, covariances)
    ]
    actual_means, actual_covariances = batch(
        means,
        covariances,
        transform,
        measurement_to_box=model.to_boxes,
        box_to_measurement=model.to_measurements,
        velocity_measurement_indices=velocity_indices,
    )
    np.testing.assert_allclose(actual_means, np.asarray([row[0] for row in expected]), rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(actual_covariances, np.asarray([row[1] for row in expected]), rtol=1e-8, atol=1e-7)
    np.testing.assert_array_equal(means, original_means)
    np.testing.assert_array_equal(covariances, original_covariances)
    np.testing.assert_array_equal(actual_covariances, actual_covariances.swapaxes(1, 2))
    assert actual_means.shape == means.shape


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("count", [0, 1])
@pytest.mark.parametrize("column", [False, True])
def test_batch_kalman_transform_empty_singleton_and_input_shape(is_obb, count, column):
    dim = 5 if is_obb else 4
    means = np.zeros((count, 2 * dim))
    means[:, :dim] = _boxes(count, is_obb=is_obb)
    if column:
        means = means[:, :, None]
    covariances = np.broadcast_to(np.eye(2 * dim), (count, 2 * dim, 2 * dim)).copy()
    batch = transform_obb_kalman_states if is_obb else transform_aabb_kalman_states
    transformed, uncertainties = batch(
        means,
        covariances,
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, -1.0]]),
        measurement_to_box=lambda rows: rows,
        box_to_measurement=lambda rows: rows,
        velocity_measurement_indices=range(dim),
    )
    assert transformed.shape == means.shape
    assert uncertainties.shape == covariances.shape
    assert np.isfinite(transformed).all()
    assert np.isfinite(uncertainties).all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_alignment_matches_scalar_across_equivalent_forms(dtype):
    boxes = _boxes(80, is_obb=True).astype(dtype)
    references = np.roll(boxes, 4, axis=0)
    references[:, 4] += 4.0 * np.pi
    expected = np.asarray([align_obb_measurement(box, ref) for box, ref in zip(boxes, references)])
    np.testing.assert_array_equal(align_obb_measurements(boxes, references), expected)


def test_batch_corners_empty_and_equivalent_forms():
    assert xywha_to_corners(np.empty((0, 5))).shape == (0, 8)
    boxes = np.array([[100, 80, 20, 40, 0.0], [100, 80, 40, 20, np.pi / 2]])
    corners = xywha_to_corners(boxes)
    np.testing.assert_allclose(corners[0], corners[1], atol=1e-5)


@pytest.mark.parametrize("transform", [np.eye(4), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])])
@pytest.mark.parametrize("is_obb", [False, True])
def test_batch_warp_rejects_invalid_transforms(transform, is_obb):
    function = transform_obbs if is_obb else transform_aabbs
    with pytest.raises(ValueError):
        function(_boxes(3, is_obb=is_obb), transform)
