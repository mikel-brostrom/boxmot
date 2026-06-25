from __future__ import annotations

import numpy as np

from boxmot.trackers.common.geometry.obb import (
    smooth_display_angle,
    smooth_obb_corners,
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
