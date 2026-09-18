# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def filter_keypoint_matches(
    knn: Sequence[Sequence[cv2.DMatch]],
    previous_keypoints: Sequence[cv2.KeyPoint],
    current_keypoints: Sequence[cv2.KeyPoint],
    image_size: tuple[int, int],
) -> tuple[list[cv2.DMatch], np.ndarray, np.ndarray]:
    """Apply ORB/SIFT ratio and spatial gates in batches, preserving match order.

    OpenCV exposes descriptor matches as objects, so their attributes must be
    collected once. Keypoint conversion, coordinate arithmetic and all numeric
    gates then operate on arrays. Returned points are float32 RANSAC inputs.
    """
    pairs = [pair for pair in knn if len(pair) == 2]
    if not pairs:
        empty = np.empty((0, 2), dtype=np.float32)
        return [], empty, empty.copy()

    attributes = np.asarray(
        [(m.queryIdx, m.trainIdx, m.distance, n.distance) for m, n in pairs],
        dtype=np.float64,
    )
    # Preserve the scalar rejection comparison, including non-finite distances.
    selected = np.flatnonzero(~(attributes[:, 2] >= 0.9 * attributes[:, 3]))
    if not len(selected):
        empty = np.empty((0, 2), dtype=np.float32)
        return [], empty, empty.copy()

    indices = attributes[selected, :2].astype(np.int32)
    previous_points = cv2.KeyPoint_convert(previous_keypoints, indices[:, 0])
    current_points = cv2.KeyPoint_convert(current_keypoints, indices[:, 1])
    distances = previous_points - current_points
    max_distance = 0.25 * np.asarray(image_size, dtype=np.float32)
    spatial = np.all(np.abs(distances) < max_distance, axis=1)
    selected = selected[spatial]
    previous_points = previous_points[spatial]
    current_points = current_points[spatial]
    distances = distances[spatial]

    if len(selected) >= 4:
        mean = distances.mean(axis=0)
        std = distances.std(axis=0) + 1e-6
        inliers = np.all(np.abs(distances - mean) < 2.5 * std, axis=1)
        selected = selected[inliers]
        previous_points = previous_points[inliers]
        current_points = current_points[inliers]

    return [pairs[index][0] for index in selected], previous_points, current_points


def draw_keypoint_matches(
    previous: np.ndarray,
    current: np.ndarray,
    previous_keypoints: Sequence[cv2.KeyPoint],
    current_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[cv2.DMatch],
    detections: np.ndarray | None,
) -> np.ndarray:
    """Draw matching points with batched conversion and unchanged draw order."""
    canvas = cv2.cvtColor(np.hstack((previous, current)), cv2.COLOR_GRAY2BGR)
    width = previous.shape[1]
    if matches:
        indices = np.asarray([(match.queryIdx, match.trainIdx) for match in matches], dtype=np.int32)
        previous_points = cv2.KeyPoint_convert(previous_keypoints, indices[:, 0]).astype(np.int32)
        current_points = cv2.KeyPoint_convert(current_keypoints, indices[:, 1]).astype(np.int32)
        current_points[:, 0] += width
        # These bindings draw one primitive at a time. Preserve the interleaved
        # order because antialiased lines and point markers can overlap.
        for previous_point, current_point in zip(previous_points, current_points):
            cv2.line(canvas, tuple(previous_point), tuple(current_point), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(previous_point), 2, (255, 255, 255), -1)
            cv2.circle(canvas, tuple(current_point), 2, (255, 255, 255), -1)

    if detections is not None:
        boxes = np.asarray(detections)
        if boxes.size and boxes.shape[1] >= 4:
            bounds = boxes[:, :4].astype(int)
            bounds[:, [0, 2]] += width
            for x1, y1, x2, y2 in bounds:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return canvas
