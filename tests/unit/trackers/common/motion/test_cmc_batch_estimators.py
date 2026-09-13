from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxmot.trackers.common.motion.cmc.keypoints import draw_keypoint_matches, filter_keypoint_matches
from boxmot.trackers.common.motion.cmc.orb import ORB
from boxmot.trackers.common.motion.cmc.sift import SIFT
from boxmot.trackers.common.motion.cmc.sof import SOF


def _scalar_mask(shape: tuple[int, int], detections: np.ndarray, scales: tuple[float, float]) -> np.ndarray:
    """Reference the original per-detection mask, including pixel boundaries."""
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    mask[int(0.02 * height) : int(0.98 * height), int(0.02 * width) : int(0.98 * width)] = 255
    for detection in detections:
        if len(detection) == 5:
            cx, cy, box_width, box_height, angle = map(float, detection)
            polygon = cv2.boxPoints(((cx, cy), (max(box_width, 1e-4), max(box_height, 1e-4)), float(np.degrees(angle))))
            polygon[:, 0] *= scales[0]
            polygon[:, 1] *= scales[1]
            cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 0)
        else:
            bounds = np.array(detection[:4], dtype=np.float32, copy=True)
            bounds[[0, 2]] *= scales[0]
            bounds[[1, 3]] *= scales[1]
            x1, y1, x2, y2 = bounds.astype(int).tolist()
            x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
            y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
    return mask


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scales", [(0.15, 0.15), (1.0, 1.0), (1.7, 0.63)])
@pytest.mark.parametrize("oriented", [False, True])
def test_batched_masks_preserve_pixels_and_inputs(dtype, scales, oriented):
    rng = np.random.default_rng(873)
    centers = rng.uniform(-20, 260, (60, 2))
    sizes = rng.uniform(-1, 50, (60, 2))
    if oriented:
        detections = np.column_stack((centers, sizes, rng.uniform(-np.pi, np.pi, len(centers))))
        # Coincident/overlapping contours must form a union, never holes.
        detections[:3] = [[60, 80, 20, 10, 0.5], [60, 80, 20, 10, 0.5], [67, 85, 20, 10, -0.5]]
    else:
        detections = np.column_stack((centers - sizes / 2, centers + sizes / 2))
        # Include reversed, clipped, zero-width and fractional pixel boundaries.
        detections[:5] = [[-1, -1, 3, 3], [11, 12, 9, 10], [3, 1, 3, 8], [3.3, 5.9, 7.1, 8.8], [0, 0, 0, 0]]
    detections = detections.astype(dtype)
    original = detections.copy()
    cmc = SOF()
    cmc._preprocess_scale = scales
    image = np.zeros((180, 270), dtype=np.uint8)

    actual = cmc.generate_mask(image, detections)

    np.testing.assert_array_equal(actual, _scalar_mask(image.shape, detections, scales))
    np.testing.assert_array_equal(detections, original)


@pytest.mark.parametrize("detections", [None, np.empty((0, 4)), np.empty((0, 5)), np.zeros((3, 3))])
def test_batched_mask_empty_or_short_detections(detections):
    cmc = SOF()
    cmc._preprocess_scale = (1.0, 1.0)
    image = np.zeros((40, 50), dtype=np.uint8)
    np.testing.assert_array_equal(cmc.generate_mask(image, detections), cmc.generate_mask(image, None))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rows", [2, 3])
def test_broadcast_transform_scale_matches_diagonal_conjugation(dtype, rows):
    rng = np.random.default_rng(843)
    cmc = SOF()
    for scales in [(0.15, 0.15), (1.0, 1.0), (1.3, 0.37)]:
        matrix = rng.normal(size=(rows, 3)).astype(dtype)
        cmc._preprocess_scale = scales
        homogeneous = np.eye(3, dtype=dtype)
        homogeneous[:rows] = matrix
        diagonal = np.diag([*scales, 1.0]).astype(dtype)
        expected = (np.linalg.inv(diagonal) @ homogeneous @ diagonal)[:rows]

        actual = cmc.restore_transform_scale(matrix)

        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == dtype


def _scalar_matches(knn, previous_keypoints, current_keypoints, image_size):
    """Reference the original scalar matching gates and float32 arithmetic."""
    matches, distances = [], []
    maximum = 0.25 * np.asarray(image_size, dtype=np.float32)
    for pair in knn:
        if len(pair) != 2:
            continue
        match, second = pair
        if match.distance >= 0.9 * second.distance:
            continue
        previous = np.asarray(previous_keypoints[match.queryIdx].pt, dtype=np.float32)
        current = np.asarray(current_keypoints[match.trainIdx].pt, dtype=np.float32)
        distance = previous - current
        if abs(distance[0]) < maximum[0] and abs(distance[1]) < maximum[1]:
            matches.append(match)
            distances.append(distance)
    if len(matches) >= 4:
        distances = np.asarray(distances, dtype=np.float32)
        mean, std = distances.mean(axis=0), distances.std(axis=0) + 1e-6
        keep = np.all(np.abs(distances - mean) < 2.5 * std, axis=1)
        matches = [match for match, accepted in zip(matches, keep) if accepted]
    previous = np.asarray([previous_keypoints[match.queryIdx].pt for match in matches], dtype=np.float32).reshape(-1, 2)
    current = np.asarray([current_keypoints[match.trainIdx].pt for match in matches], dtype=np.float32).reshape(-1, 2)
    return matches, previous, current


def _keypoints(points: np.ndarray) -> list[cv2.KeyPoint]:
    return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in points]


@pytest.mark.parametrize("count", [0, 1, 3, 4, 7, 101])
def test_vectorized_keypoint_gates_preserve_selection_order_and_points(count):
    rng = np.random.default_rng(844)
    previous = rng.uniform(0, 100, (count, 2)).astype(np.float32)
    current = previous + rng.normal(0, 7, previous.shape).astype(np.float32)
    query_indices, train_indices = rng.permutation(count), rng.permutation(count)
    previous_keypoints = _keypoints(previous)
    current_keypoints = _keypoints(current)
    knn = [
        [cv2.DMatch(int(query), int(train), float(rng.uniform(0, 11))), cv2.DMatch(int(query), int(train), 10.0)]
        for query, train in zip(query_indices, train_indices)
    ]
    knn.extend([[], [cv2.DMatch(0, 0, 1.0)]])

    actual = filter_keypoint_matches(knn, previous_keypoints, current_keypoints, (400, 300))
    expected = _scalar_matches(knn, previous_keypoints, current_keypoints, (400, 300))

    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


def test_keypoint_gates_preserve_strict_boundaries_and_nonfinite_rejection():
    previous = _keypoints(np.zeros((8, 2)))
    current = _keypoints(np.array([[0, 0], [25, 0], [0, 20], [24.9, 19.9], [np.nan, 1], [1, 1], [2, 2], [3, 3]]))
    distances = [9.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.inf, 0.0]
    knn = [
        [cv2.DMatch(index, index, distance), cv2.DMatch(index, index, 10.0)] for index, distance in enumerate(distances)
    ]

    actual = filter_keypoint_matches(knn, previous, current, (100, 80))
    expected = _scalar_matches(knn, previous, current, (100, 80))

    assert [match.queryIdx for match in actual[0]] == [3, 5, 7]
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


@pytest.mark.parametrize("cmc_class", [ORB, SIFT])
def test_descriptor_estimators_keep_transform_and_debug_state(cmc_class):
    rng = np.random.default_rng(845)
    previous_points = rng.uniform(20, 80, (15, 2)).astype(np.float32)
    current_points = previous_points + np.asarray([5, -3], dtype=np.float32)
    previous, current = _keypoints(previous_points), _keypoints(current_points)
    descriptors = rng.integers(0, 255, (15, 32), dtype=np.uint8)
    matches = [[cv2.DMatch(index, index, 1.0), cv2.DMatch(index, index, 10.0)] for index in range(15)]
    estimator = cmc_class(scale=1.0, align=True, draw_keypoint_matches=True)
    estimator.prev_img = np.zeros((100, 100), dtype=np.uint8)
    estimator.prev_keypoints = previous
    estimator.prev_descriptors = descriptors.copy()
    estimator.detector = SimpleNamespace(detect=lambda image, mask: current)
    estimator.extractor = SimpleNamespace(compute=lambda image, points: (points, descriptors))
    estimator.matcher = SimpleNamespace(knnMatch=lambda *args, **kwargs: matches)

    result = estimator.apply(np.zeros((100, 100, 3), dtype=np.uint8), np.empty((0, 4)))

    np.testing.assert_allclose(result, [[1, 0, 5], [0, 1, -3]], atol=1e-5)
    assert estimator.matches_img.shape == (100, 200, 3)
    assert estimator.prev_img_aligned.shape == (100, 100)
    assert estimator.prev_keypoints == current


def test_batched_debug_coordinates_preserve_overlapping_draw_order():
    previous_image = np.zeros((40, 50), dtype=np.uint8)
    current_image = np.zeros_like(previous_image)
    previous = _keypoints(np.array([[0.5, 3.1], [42.2, 11.8], [21.8, 38.8]]))
    current = _keypoints(np.array([[20.3, 17.7], [8.2, 22.5], [1.8, 2.2]]))
    matches = [cv2.DMatch(index, index, 1.0) for index in range(3)]
    detections = np.array([[2.4, 1.7, 19.5, 28.5], [0, 0, 15, 15]])
    expected = cv2.cvtColor(np.hstack((previous_image, current_image)), cv2.COLOR_GRAY2BGR)
    for match in matches:
        left = np.asarray(previous[match.queryIdx].pt, dtype=np.int32)
        right = np.asarray(current[match.trainIdx].pt, dtype=np.int32)
        right[0] += 50
        cv2.line(expected, tuple(left), tuple(right), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(expected, tuple(left), 2, (255, 255, 255), -1)
        cv2.circle(expected, tuple(right), 2, (255, 255, 255), -1)
    for detection in detections:
        x1, y1, x2, y2 = detection.astype(int).tolist()
        cv2.rectangle(expected, (x1 + 50, y1), (x2 + 50, y2), (0, 0, 255), 2)

    actual = draw_keypoint_matches(previous_image, current_image, previous, current, matches, detections)

    np.testing.assert_array_equal(actual, expected)
