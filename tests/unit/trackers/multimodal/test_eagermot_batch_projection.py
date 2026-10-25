"""Batched camera projection retains the scalar visibility and pixel contract."""

import numpy as np
import pytest

from boxmot.trackers.eagermot.geometry import project_box3d, project_boxes3d


@pytest.mark.parametrize("count", [0, 1, 200])
@pytest.mark.parametrize("translated_depth", [False, True])
def test_batch_projection_matches_scalar(count, translated_depth):
    """Include clipped boxes, camera crossings, rear boxes and depth translation."""
    rng = np.random.default_rng(918)
    boxes = np.column_stack((
        rng.uniform(-20, 20, (count, 3)),
        rng.uniform(-4 * np.pi, 4 * np.pi, count),
        rng.uniform(0.1, 8, (count, 3)),
    ))
    projection = np.array([[250, 0, 100, 3], [0, 250, 50, -2], [0, 0, 1, 3 if translated_depth else 0]])
    expected_boxes = np.zeros((count, 4))
    expected_valid = np.zeros(count, dtype=bool)
    for index, box in enumerate(boxes):
        projected = project_box3d(box, projection, (100, 200))
        if projected is not None:
            expected_boxes[index] = projected
            expected_valid[index] = True
    original = boxes.copy()
    actual_boxes, actual_valid = project_boxes3d(boxes, projection, (100, 200))
    np.testing.assert_array_equal(actual_boxes, expected_boxes)
    np.testing.assert_array_equal(actual_valid, expected_valid)
    np.testing.assert_array_equal(boxes, original)


def test_batch_projection_retains_four_forward_corner_rule():
    """Depth-zero corners are excluded even when their image coordinates are finite."""
    projection = np.array([[100, 0, 100, 0], [0, 100, 50, 0], [0, 0, 1, 0]])
    boxes = np.array([[0, 2, z, 0, 4, 2, 2] for z in (-1, 0, 1)])
    bounds, valid = project_boxes3d(boxes, projection, (100, 200))
    np.testing.assert_array_equal(valid, [False, True, True])
    np.testing.assert_array_equal(bounds[0], 0)
    for index in (1, 2):
        np.testing.assert_array_equal(bounds[index], project_box3d(boxes[index], projection, (100, 200)))
