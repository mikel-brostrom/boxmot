from __future__ import annotations

import numpy as np
import pytest

from boxmot.detectors.base import Detections


def _image() -> np.ndarray:
    return np.zeros((64, 96, 3), dtype=np.uint8)


def test_aabb_detections_normalize_rows_and_expose_array_accessors():
    detections = Detections(
        dets=[10, 20, 30, 45, 0.75, 2],
        orig_img=_image(),
        path="frame.jpg",
        names={2: "car"},
    )

    assert detections.dets.dtype == np.float32
    assert detections.shape == (1, 6)
    assert len(detections) == 1
    assert detections.is_obb is False
    np.testing.assert_array_equal(detections.boxes, [[10, 20, 30, 45]])
    np.testing.assert_array_equal(detections.xyxy, detections.boxes)
    np.testing.assert_array_equal(detections.conf, [0.75])
    np.testing.assert_array_equal(detections.classes, [2])
    np.testing.assert_array_equal(detections.cls, detections.classes)
    np.testing.assert_array_equal(np.asarray(detections), detections.dets)
    np.testing.assert_array_equal(detections[0], detections.dets[0])


def test_obb_detections_preserve_empty_width_and_expose_enclosing_xyxy():
    row = np.array([[48, 32, 20, 10, np.pi / 2, 0.9, 1]], dtype=np.float32)
    detections = Detections(row, _image())

    assert detections.is_obb is True
    np.testing.assert_array_equal(detections.boxes, row[:, :5])
    np.testing.assert_array_equal(detections.xywha, row[:, :5])
    np.testing.assert_allclose(detections.xyxy, [[43, 22, 53, 42]], atol=1e-5)
    np.testing.assert_allclose(detections.conf, [0.9], atol=1e-6)
    np.testing.assert_array_equal(detections.classes, [1])

    empty = Detections(np.empty((0, 7), dtype=np.float64), _image())
    assert empty.dets.dtype == np.float32
    assert empty.shape == (0, 7)
    assert empty.is_obb is True
    assert empty.xyxy.shape == (0, 4)


@pytest.mark.parametrize("columns", [0, 1, 5, 8])
def test_detections_reject_unsupported_column_layouts(columns):
    with pytest.raises(ValueError):
        Detections(np.empty((1, columns), dtype=np.float32), _image())


def test_detection_masks_must_remain_row_aligned():
    rows = np.array(
        [
            [1, 2, 10, 12, 0.9, 0],
            [4, 5, 14, 16, 0.8, 1],
        ],
        dtype=np.float32,
    )
    masks = np.stack(
        [
            np.ones((8, 12), dtype=np.uint8),
            np.full((8, 12), 2, dtype=np.uint8),
        ]
    )

    detections = Detections(rows, _image(), masks=masks)
    np.testing.assert_array_equal(detections.masks, masks)

    with pytest.raises(ValueError):
        Detections(rows, _image(), masks=masks[:1])
