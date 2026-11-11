"""Immutable user prompts in original-image pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


def _coordinate(value: object) -> float:
    """Own one finite, nonnegative coordinate without coercing strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Prompt coordinates must be real numbers.")
    coordinate = float(value)
    if not np.isfinite(coordinate) or coordinate < 0:
        raise ValueError("Prompt coordinates must be finite and nonnegative.")
    return coordinate


@dataclass(frozen=True, slots=True)
class EdgeTAMPrompt:
    """Select one object with an XYXY box and/or foreground/background points.

    Labels are 1 for foreground and 0 for background. Sequences are copied to
    immutable tuples; image bounds are checked when the prompt is processed.
    """

    object_id: int
    box: tuple[float, float, float, float] | None = None
    points: tuple[tuple[float, float], ...] = ()
    labels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate prompt structure and detach all caller-owned containers."""
        if isinstance(self.object_id, bool) or not isinstance(self.object_id, Integral) or self.object_id < 0:
            raise ValueError("Prompt object_id must be a nonnegative integer.")
        object.__setattr__(self, "object_id", int(self.object_id))
        points = tuple(tuple(_coordinate(value) for value in point) for point in self.points)
        if any(len(point) != 2 for point in points):
            raise ValueError("Prompt points must contain two coordinates each.")
        labels = tuple(self.labels)
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1) for value in labels):
            raise ValueError("Prompt labels must be integers 0 (background) or 1 (foreground).")
        if len(labels) != len(points):
            raise ValueError("Prompt labels must align with points.")
        if self.box is None and not points:
            raise ValueError("A prompt must contain a box or a nonempty point set.")
        if self.box is not None:
            box = tuple(_coordinate(value) for value in self.box)
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Prompt boxes must be XYXY coordinates with positive width and height.")
            object.__setattr__(self, "box", box)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", tuple(int(value) for value in labels))

    def validate_bounds(self, image_size: tuple[int, int]) -> None:
        """Reject coordinates outside an image before changing temporal state."""
        height, width = image_size
        if self.box is not None and (self.box[2] > width or self.box[3] > height):
            raise ValueError("Prompt box coordinates must lie within the frame dimensions.")
        if any(x >= width or y >= height for x, y in self.points):
            raise ValueError("Prompt points must lie within the frame dimensions.")


__all__ = ("EdgeTAMPrompt",)
