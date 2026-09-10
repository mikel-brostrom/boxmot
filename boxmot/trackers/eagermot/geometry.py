"""Upright 3D geometry for the EagerMOT sensor-fusion tracker.

Adapted from EagerMOT (MIT License), Copyright (c) 2021 Aleksandr Kim.
See LICENSE in this directory. Boxes are ``[x, y, z, yaw, l, w, h]``:
bottom center, downward-positive y, and yaw about the positive y axis.
"""

from __future__ import annotations

import numpy as np


def _boxes_array(boxes: np.ndarray) -> np.ndarray:
    """Validate a batch of upright source-format 3D boxes without modifying it."""
    values = np.asarray(boxes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("3D boxes must have shape (N, 7) with rows [x, y, z, yaw, l, w, h].")
    if not np.isfinite(values).all() or np.any(values[:, 4:] <= 0):
        raise ValueError("3D boxes must be finite with strictly positive length, width, and height.")
    return values


def yaw_difference(current: float | np.ndarray, observed: float | np.ndarray) -> np.ndarray:
    """Return the closest rectangular-box yaw difference in [-pi/2, pi/2].

    Opposite headings describe the same box. Unlike the source's recursive
    correction, this also handles states that have rotated beyond a full turn.
    """
    difference = np.asarray(observed) - np.asarray(current)
    difference = np.arctan2(np.sin(difference), np.cos(difference))
    difference = np.where(difference > np.pi / 2, difference - np.pi, difference)
    return np.where(difference < -np.pi / 2, difference + np.pi, difference)


def boxes3d_corners(boxes: np.ndarray) -> np.ndarray:
    """Return (N, 8, 3) corners in the source's bottom-face then top-face order."""
    values = _boxes_array(boxes)
    x_signs = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * 0.5
    z_signs = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * 0.5
    x = values[:, 4, None] * x_signs
    z = values[:, 5, None] * z_signs
    cosine, sine = np.cos(values[:, 3, None]), np.sin(values[:, 3, None])
    corners = np.empty((len(values), 8, 3), dtype=np.float64)
    corners[:, :, 0] = cosine * x + sine * z + values[:, 0, None]
    corners[:, :, 1] = values[:, 1, None] - values[:, 6, None] * np.array([0, 0, 0, 0, 1, 1, 1, 1])
    corners[:, :, 2] = -sine * x + cosine * z + values[:, 2, None]
    return corners


def transform_boxes3d(boxes: np.ndarray, pose: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    """Transform centers rigidly and approximate orientation using a +y yaw.

    ``pose`` maps camera coordinates into world coordinates. As in EagerMOT,
    the complete rotation and translation act on bottom centers, dimensions
    stay fixed, and the pose's XYZ Euler y angle is added to object yaw. Its
    full-range extension uses the sign of R00 to avoid folding camera turns
    beyond 90 degrees; it agrees with the source on its usual Euler branch
    and preserves exact geometry for upright poses. Roll and pitch are not
    retained as cuboid orientation in the seven-coordinate tracking state.

    ``inverse=True`` maps back through the original pose and subtracts that
    same yaw. Extracting a new yaw from the inverse rotation would not undo
    the forward approximation when the pose includes roll and pitch.
    """
    values = _boxes_array(boxes)
    pose = np.asarray(pose, dtype=np.float64)
    if not isinstance(inverse, bool):
        raise TypeError("inverse must be bool.")
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("pose must be a finite 4x4 rigid transformation.")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0.0)
    ):
        raise ValueError("pose must be a homogeneous rigid transform without reflection.")
    # The signed cosine retains the full yaw range instead of Euler's +/-pi/2 fold.
    cosine = np.copysign(np.hypot(rotation[0, 0], rotation[1, 0]), rotation[0, 0])
    angle = np.arctan2(-rotation[2, 0], cosine)
    if inverse:
        pose = np.linalg.inv(pose)
        angle = -angle
    result = values.copy()
    result[:, :3] = values[:, :3] @ pose[:3, :3].T + pose[:3, 3]
    result[:, 3] += angle
    return result


def project_box3d(box: np.ndarray, projection: np.ndarray, image_size: tuple[int, int]) -> np.ndarray | None:
    """Project an upright box through a 3x4 tracking-to-image camera matrix.

    ``image_size`` is (height, width). The third projected coordinate is
    camera depth, including any translation in the projection's final row.
    Pixel rounding and image clipping follow the source. Its NuScenes path
    requires four forward corners; this port applies that visibility rule
    to every camera, also rejecting behind-camera boxes in KITTI-style input.
    Invisible or zero-area projections return None.
    """
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (7,):
        raise ValueError("A projected 3D box must have shape (7,).")
    projection = np.asarray(projection, dtype=np.float64)
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError("projection must be a finite 3x4 matrix.")
    if len(image_size) != 2 or any(not np.isfinite(value) or value <= 0 for value in image_size):
        raise ValueError("image_size must contain positive height and width.")
    corners = boxes3d_corners(box[None])[0]
    projected = corners @ projection[:, :3].T + projection[:, 3]
    projected = projected[projected[:, 2] > 0]
    if len(projected) < 4:
        return None
    pixels = np.rint(projected[:, :2] / projected[:, 2, None])
    bounds = np.r_[pixels.min(axis=0), pixels.max(axis=0)]
    bounds[[0, 2]] = np.clip(bounds[[0, 2]], 0, image_size[1])
    bounds[[1, 3]] = np.clip(bounds[[1, 3]], 0, image_size[0])
    return bounds if np.all(bounds[2:] > bounds[:2]) else None


def iou2d_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return pairwise xyxy IoU, treating nonfinite/zero-area rows as invisible."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or first.shape[1] != 4 or second.ndim != 2 or second.shape[1] != 4:
        raise ValueError("2D box batches must have shape (N, 4).")
    first_valid = np.isfinite(first).all(axis=1) & np.all(first[:, 2:] > first[:, :2], axis=1)
    second_valid = np.isfinite(second).all(axis=1) & np.all(second[:, 2:] > second[:, :2], axis=1)
    safe_first = np.where(first_valid[:, None], first, 0)
    safe_second = np.where(second_valid[:, None], second, 0)
    lengths = np.maximum(
        np.minimum(safe_first[:, None, 2:], safe_second[None, :, 2:])
        - np.maximum(safe_first[:, None, :2], safe_second[None, :, :2]),
        0,
    )
    intersection = lengths.prod(axis=2)
    area_first = (safe_first[:, 2:] - safe_first[:, :2]).prod(axis=1)
    area_second = (safe_second[:, 2:] - safe_second[:, :2]).prod(axis=1)
    union = area_first[:, None] + area_second[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Evaluate a two-dimensional cross product without deprecated NumPy behavior."""
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    """Clip one counterclockwise convex footprint against another in float64."""
    # Translation keeps the polygon arithmetic stable in world coordinates.
    origin = first[0].copy()
    polygon = first - origin
    clip = second - origin
    for edge_start, edge_end in zip(clip, np.roll(clip, -1, axis=0)):
        if not len(polygon):
            return 0.0
        distance = _cross(edge_end - edge_start, polygon - edge_start)
        inside = distance >= 0
        result = []
        for index, point in enumerate(polygon):
            previous = index - 1
            if inside[index] != inside[previous]:
                fraction = distance[previous] / (distance[previous] - distance[index])
                result.append(polygon[previous] + fraction * (point - polygon[previous]))
            if inside[index]:
                result.append(point)
        polygon = np.asarray(result, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return 0.0
    return float(abs(_cross(polygon, np.roll(polygon, -1, axis=0)).sum()) * 0.5)


def iou3d_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return volumetric IoU of yawed boxes, including vertical overlap.

    Corners are rounded to four decimals as upstream. Convex polygon
    clipping replaces the source's Shapely dependency for the x-z footprint.
    """
    first, second = _boxes_array(first), _boxes_array(second)
    corners_first = boxes3d_corners(first).round(4)
    corners_second = boxes3d_corners(second).round(4)
    volume_first, volume_second = first[:, 4:].prod(axis=1), second[:, 4:].prod(axis=1)
    similarity = np.zeros((len(first), len(second)), dtype=np.float64)
    for row, corners_a in enumerate(corners_first):
        footprint_a = corners_a[3::-1][:, [0, 2]]
        for col, corners_b in enumerate(corners_second):
            height = min(corners_a[0, 1], corners_b[0, 1]) - max(corners_a[4, 1], corners_b[4, 1])
            if height <= 0:
                continue
            footprint_b = corners_b[3::-1][:, [0, 2]]
            if np.any(
                np.minimum(footprint_a.max(axis=0), footprint_b.max(axis=0))
                <= np.maximum(footprint_a.min(axis=0), footprint_b.min(axis=0))
            ):
                continue
            intersection = _intersection_area(footprint_a, footprint_b) * height
            union = volume_first[row] + volume_second[col] - intersection
            if union > 0:
                similarity[row, col] = np.clip(intersection / union, 0, 1)
    return similarity
