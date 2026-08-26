from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.geometry.obb import (
    smooth_display_angle,
    smooth_obb_corners,
    transform_obb,
    transform_obb_kalman_state,
    xywha_to_corners,
    xywha_to_xyxy,
)


def test_xywha_to_corners_canonicalizes_equivalent_forms():
    base = np.array([640.0, 512.0, 320.0, 160.0, 0.45], dtype=np.float32)
    equivalent = np.array(
        [640.0, 512.0, 160.0, 320.0, 0.45 + (np.pi / 2.0)],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        xywha_to_corners(base),
        xywha_to_corners(equivalent),
        atol=1e-4,
    )


def test_xywha_to_xyxy_returns_enclosing_aabb():
    boxes = np.array([[10.0, 20.0, 8.0, 4.0, np.pi / 2.0]], dtype=np.float32)

    xyxy = xywha_to_xyxy(boxes)

    np.testing.assert_allclose(
        xyxy,
        np.array([[8.0, 16.0, 12.0, 24.0]], dtype=np.float32),
        atol=1e-5,
    )


def test_smooth_display_angle_keeps_equivalent_obb_continuous():
    prev_angle = 0.45
    equivalent = np.array(
        [640.0, 512.0, 160.0, 320.0, 0.45 + (np.pi / 2.0)],
        dtype=np.float32,
    )

    angle, display_box = smooth_display_angle(prev_angle, equivalent)

    np.testing.assert_allclose(angle, prev_angle, atol=1e-6)
    np.testing.assert_allclose(display_box[2:4], np.array([320.0, 160.0]))
    np.testing.assert_allclose(display_box[4], prev_angle, atol=1e-6)


def test_smooth_obb_corners_returns_flat_corners_and_next_angle():
    box = np.array([50.0, 40.0, 20.0, 10.0, 0.2], dtype=np.float32)

    corners, angle = smooth_obb_corners(box, None)

    assert corners.shape == (8,)
    np.testing.assert_allclose(angle, 0.2, atol=1e-6)
    np.testing.assert_allclose(corners.reshape(4, 2).mean(axis=0), box[:2])


def test_transform_obb_kalman_state_updates_geometry_velocity_and_covariance():
    mean = np.array([50, 40, 20, 10, 0.3, 2, 1, 0.5, 0.2, 0.1], dtype=np.float64)
    covariance = np.diag(np.arange(1, 11, dtype=np.float64))
    angle = 0.25
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    linear = rotation @ np.diag([1.2, 0.8])
    transform = np.column_stack([linear, np.array([7.0, -4.0])])

    transformed_mean, transformed_covariance = transform_obb_kalman_state(
        mean,
        covariance,
        transform,
        measurement_to_box=lambda values: values,
        box_to_measurement=lambda box: box,
        velocity_measurement_indices=(0, 1, 2, 3, 4),
    )

    np.testing.assert_allclose(transformed_mean[:5], transform_obb(mean[:5], transform), atol=1e-5)
    np.testing.assert_allclose(transformed_mean[5:7], linear @ mean[5:7], atol=2e-2)
    np.testing.assert_allclose(transformed_covariance, transformed_covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(transformed_covariance).min() >= -1e-8
    assert not np.allclose(transformed_covariance, covariance)


def test_repeated_rigid_obb_state_transforms_preserve_size_and_covariance_conditioning():
    mean = np.array([100, 110, 1960, 2.5, -0.55, 4, 1.5, 0, 0.055], dtype=np.float64)
    covariance = np.eye(9, dtype=np.float64)
    angle = 1e-3
    transform = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.1],
            [np.sin(angle), np.cos(angle), -0.05],
        ],
        dtype=np.float64,
    )

    for _ in range(100):
        mean, covariance = transform_obb_kalman_state(
            mean,
            covariance,
            transform,
            measurement_to_box=lambda values: np.array(
                [
                    values[0],
                    values[1],
                    np.sqrt(values[2] * values[3]),
                    np.sqrt(values[2] / values[3]),
                    values[4],
                ]
            ),
            box_to_measurement=lambda box: np.array([box[0], box[1], box[2] * box[3], box[2] / box[3], box[4]]),
            velocity_measurement_indices=(0, 1, 2, 4),
        )

    width = np.sqrt(mean[2] * mean[3])
    height = np.sqrt(mean[2] / mean[3])
    np.testing.assert_allclose([width, height], [70.0, 28.0], rtol=1e-6, atol=1e-6)
    assert np.linalg.cond(covariance) < 2.0


def test_transform_obb_supports_homography():
    box = np.array([50, 40, 20, 10, 0.3], dtype=np.float64)
    homography = np.array(
        [[1.0, 0.05, 3.0], [-0.02, 1.0, -2.0], [0.0005, -0.0003, 1.0]],
        dtype=np.float64,
    )

    transformed = transform_obb(box, homography)

    assert transformed.shape == (5,)
    assert np.isfinite(transformed).all()
    assert np.all(transformed[2:4] > 0)


@pytest.mark.parametrize("angle", [np.pi / 4, 3 * np.pi / 4, (np.pi / 4) + 1e-7])
def test_square_obb_corner_order_has_four_unique_vertices_and_area(angle):
    corners = xywha_to_corners(np.array([50, 50, 20, 20, angle], dtype=np.float32)).reshape(4, 2)

    assert len(np.unique(np.round(corners, decimals=5), axis=0)) == 4
    twice_area = np.dot(corners[:, 0], np.roll(corners[:, 1], -1)) - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))
    assert twice_area > 0
    assert np.isclose(abs(twice_area) / 2.0, 400.0, rtol=1e-5)
    assert tuple(corners[0]) == min(map(tuple, corners), key=lambda point: (point[1], point[0]))


def test_square_obb_identity_transform_does_not_collapse_geometry():
    box = np.array([50, 50, 20, 20, np.pi / 4], dtype=np.float64)

    transformed = transform_obb(box, np.eye(2, 3, dtype=np.float64))

    np.testing.assert_allclose(transformed[:4], box[:4], atol=1e-4)
