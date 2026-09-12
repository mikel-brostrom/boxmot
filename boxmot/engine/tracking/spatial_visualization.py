"""Render camera-space 3D tracking boxes without changing their coordinate frame."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import cv2
import numpy as np

from boxmot.trackers.eagermot.geometry import boxes3d_corners

if TYPE_CHECKING:
    from boxmot.structures import CameraModel, Tracks3D

_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
_NEAR_DEPTH = 0.1


def _visible_edge(
    start: np.ndarray, end: np.ndarray, image_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Clip a homogeneous image segment against depth and viewport before rounding."""
    if max(start[2], end[2]) <= _NEAR_DEPTH:
        return None
    # Homogeneous line coefficients avoid subtracting enormous pixel positions
    # when intersecting the viewport. Normalize before the cross product.
    line = np.cross(start / max(1.0, float(np.abs(start).max())), end / max(1.0, float(np.abs(end).max())))
    if start[2] <= _NEAR_DEPTH:
        fraction = (_NEAR_DEPTH - start[2]) / (end[2] - start[2])
        start = (1 - fraction) * start + fraction * end
        start[2] = _NEAR_DEPTH
    elif end[2] <= _NEAR_DEPTH:
        fraction = (_NEAR_DEPTH - end[2]) / (start[2] - end[2])
        end = (1 - fraction) * end + fraction * start
        end[2] = _NEAR_DEPTH
    first, second = start[:2] / start[2], end[:2] / end[2]
    height, width = image_size
    maximum_x, maximum_y = width - 1, height - 1

    def outside(point: np.ndarray) -> int:
        """Return the four viewport half-planes excluding an image point."""
        return (
            int(point[0] < 0)
            | (int(point[0] > maximum_x) << 1)
            | (int(point[1] < 0) << 2)
            | (int(point[1] > maximum_y) << 3)
        )

    for _ in range(8):
        first_code, second_code = outside(first), outside(second)
        if not (first_code | second_code):
            return tuple(map(int, np.rint(first))), tuple(map(int, np.rint(second)))
        if first_code & second_code:
            return None
        code = first_code or second_code
        if code & (4 | 8):
            if line[0] == 0:
                return None
            y = maximum_y if code & 8 else 0
            point = np.array([-(line[1] * y + line[2]) / line[0], y])
        else:
            if line[1] == 0:
                return None
            x = maximum_x if code & 2 else 0
            point = np.array([x, -(line[0] * x + line[2]) / line[1]])
        if not np.isfinite(point).all():
            return None
        if first_code:
            first = point
        else:
            second = point
    return None


def draw_spatial_tracks(
    image: np.ndarray,
    tracks: Tracks3D,
    camera: CameraModel,
    *,
    class_names: Mapping[int, str] | None = None,
    image_track_ids: frozenset[int] = frozenset(),
    line_width: int = 2,
) -> np.ndarray:
    """Draw visible cuboid edges onto a BGR image and return that same image.

    Spatial tracks already use the current camera coordinates. Project through
    all twelve camera matrix entries without applying ``camera_to_world``.
    Only spatial identities lacking an image-track label receive an extra label.
    """
    from boxmot.engine.tracking.sinks import _track_color

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Spatial overlays require an HWC BGR image.")
    if image.shape[:2] != camera.image_size:
        raise ValueError("Spatial overlay dimensions must match CameraModel.image_size.")
    if isinstance(line_width, bool) or not isinstance(line_width, int) or line_width <= 0:
        raise ValueError("Spatial overlay line_width must be a positive integer.")
    corners = boxes3d_corners(tracks.geometry.values.detach().numpy())
    projection = camera.projection.detach().numpy().astype(np.float64)
    projected = corners @ projection[:, :3].T + projection[:, 3]
    for points, track_id, class_id in zip(projected, tracks.track_ids.tolist(), tracks.class_ids.tolist(), strict=True):
        color = _track_color(track_id)
        visible = []
        for first, second in _EDGES:
            edge = _visible_edge(points[first], points[second], camera.image_size)
            if edge is not None:
                cv2.line(image, *edge, color, line_width, cv2.LINE_AA)
                visible.extend(edge)
        if visible and track_id not in image_track_ids:
            class_label = str(class_id) if class_names is None else class_names.get(class_id, str(class_id))
            anchor = (
                min(point[0] for point in visible),
                min(camera.image_size[0] - 1, max(12, min(point[1] for point in visible) - 4)),
            )
            cv2.putText(
                image, f"{class_label} #{track_id} 3D", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
            )
    return image


__all__ = ("draw_spatial_tracks",)
