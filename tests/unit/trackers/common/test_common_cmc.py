from __future__ import annotations

import numpy as np
import pytest

from boxmot.trackers.common.detections.layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.trackers.common.motion.cmc import (
    apply_cmc_to_tracks,
    cmc_detection_boxes,
    create_cmc,
    reset_cmc,
)


class DummyCMC:
    def __init__(self):
        self.calls: list[np.ndarray] = []
        self.reset_called = False
        self.prev_img = np.ones((2, 2), dtype=np.uint8)
        self.prev_img_aligned = np.ones((2, 2), dtype=np.uint8)
        self.prev_keypoints = ["kp"]
        self.prev_descriptors = np.ones((1, 2), dtype=np.float32)

    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        del img
        self.calls.append(np.asarray(dets, dtype=np.float32).copy())
        return np.array([[1.0, 0.0, 5.0], [0.0, 1.0, -2.0]], dtype=np.float32)

    def reset(self) -> None:
        self.reset_called = True


class DummyTrack:
    def __init__(self):
        self.warps: list[np.ndarray] = []

    def camera_update(self, warp: np.ndarray) -> None:
        self.warps.append(np.asarray(warp, dtype=np.float32).copy())


def test_create_cmc_disabled_returns_none():
    assert create_cmc("ecc", enabled=False) is None
    assert create_cmc(None) is None


def test_create_cmc_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown cmc_method"):
        create_cmc("not-a-cmc-method")


def test_cmc_detection_boxes_aabb_passthrough():
    dets = np.array([[10, 20, 30, 50, 0.9, 2]], dtype=np.float32)

    boxes = cmc_detection_boxes(dets, AABB_DETECTIONS)

    np.testing.assert_allclose(boxes, dets[:, :4])


def test_cmc_detection_boxes_obb_preserves_oriented_geometry():
    dets = np.array([[10, 20, 8, 4, np.pi / 2.0, 0.9, 2]], dtype=np.float32)

    boxes = cmc_detection_boxes(dets, OBB_DETECTIONS)

    np.testing.assert_allclose(boxes, dets[:, :5], atol=1e-5)


def test_apply_cmc_to_tracks_noops_when_disabled():
    track = DummyTrack()

    warp = apply_cmc_to_tracks(
        None,
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.empty((0, 6), dtype=np.float32),
        AABB_DETECTIONS,
        [track],
    )

    assert warp is None
    assert track.warps == []


def test_apply_cmc_to_tracks_uses_layout_boxes_and_updates_tracks():
    cmc = DummyCMC()
    track = DummyTrack()
    dets = np.array([[10, 20, 8, 4, np.pi / 2.0, 0.9, 2]], dtype=np.float32)

    warp = apply_cmc_to_tracks(
        cmc,
        np.zeros((8, 8, 3), dtype=np.uint8),
        dets,
        OBB_DETECTIONS,
        [track],
    )

    np.testing.assert_allclose(warp, track.warps[0])
    np.testing.assert_allclose(cmc.calls[0], dets[:, :5])


def test_cmc_mask_uses_obb_polygon_instead_of_enclosing_aabb():
    from boxmot.motion.cmc.sof import SOF

    cmc = SOF(scale=1.0)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cmc.preprocess(image)
    obb = np.array([[50, 50, 60, 10, np.pi / 4]], dtype=np.float32)

    mask = cmc.generate_mask(gray, obb, cmc.scale)

    assert mask[50, 50] == 0
    assert mask[30, 70] == 255  # inside the enclosing AABB, outside the OBB


def test_cmc_mask_scales_obb_corners_for_nonuniform_resize():
    import cv2

    from boxmot.motion.cmc.sof import SOF

    cmc = SOF()
    cmc.scale = (200, 100)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cmc.preprocess(image)
    obb = np.array([[50, 50, 60, 10, np.pi / 4]], dtype=np.float32)

    mask = cmc.generate_mask(gray, obb, cmc.scale)
    expected = np.zeros_like(mask)
    expected[2:98, 4:196] = 255
    polygon = cv2.boxPoints(((50.0, 50.0), (60.0, 10.0), 45.0))
    polygon[:, 0] *= 2.0
    cv2.fillConvexPoly(expected, np.rint(polygon).astype(np.int32), 0)

    np.testing.assert_array_equal(mask, expected)


def test_cmc_mask_does_not_mutate_aabb_detections():
    from boxmot.motion.cmc.sof import SOF

    cmc = SOF(scale=0.15)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cmc.preprocess(image)
    detections = np.array([[10, 20, 60, 80]], dtype=np.float32)
    expected = detections.copy()

    cmc.generate_mask(gray, detections, cmc.scale)

    np.testing.assert_array_equal(detections, expected)


def test_reset_cmc_calls_reset_and_clears_legacy_state():
    cmc = DummyCMC()

    reset_cmc(cmc)

    assert cmc.reset_called is True
    assert cmc.prev_img is None
    assert cmc.prev_img_aligned is None
    assert cmc.prev_keypoints is None
    assert cmc.prev_descriptors is None


def test_reset_cmc_clears_sof_sequence_state():
    from boxmot.motion.cmc.sof import SOF

    cmc = SOF(scale=1.0)
    cmc.prev_frame = np.ones((8, 8), dtype=np.uint8)
    cmc.prev_keypoints = np.ones((4, 1, 2), dtype=np.float32)
    cmc.initialized = True

    reset_cmc(cmc)

    assert cmc.prev_frame is None
    assert cmc.prev_keypoints is None
    assert cmc.initialized is False
