"""Adapters for convenient public tracker inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NumPyDetectionRows:
    """Validated arrays split from one packed detection row matrix."""

    geometry: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.scores.shape[0])


def parse_numpy_detection_rows(detections: np.ndarray, *, is_obb: bool) -> NumPyDetectionRows:
    """Validate and split packed AABB6 or OBB7 rows for a configured tracker."""

    if type(detections) is not np.ndarray:
        raise TypeError(f"detections must be Detections or a plain numpy.ndarray, got {type(detections).__name__}.")

    geometry_columns = 5 if is_obb else 4
    expected_columns = geometry_columns + 2
    geometry_name = "OBB" if is_obb else "AABB"
    if detections.ndim != 2 or detections.shape[1] != expected_columns:
        raise ValueError(
            f"{geometry_name} detection rows must have shape [N, {expected_columns}], got {detections.shape}."
        )
    if not (np.issubdtype(detections.dtype, np.integer) or np.issubdtype(detections.dtype, np.floating)):
        raise TypeError(f"Detection rows must use a real numeric dtype, got {detections.dtype}.")
    if detections.size and not np.isfinite(detections).all():
        raise ValueError("Detection rows must contain only finite values.")
    raw_scores = detections[:, geometry_columns]
    if raw_scores.size and (np.any(raw_scores < 0) or np.any(raw_scores > 1)):
        raise ValueError("Detection scores must be in the inclusive range [0, 1].")

    parsed_class_ids: list[int] = []
    for value in detections[:, -1].tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Detection class IDs must be non-negative integers.")
        class_id = int(value)
        if class_id != value or class_id < 0 or class_id > np.iinfo(np.int64).max:
            raise ValueError("Detection class IDs must be non-negative integers.")
        parsed_class_ids.append(class_id)

    with np.errstate(over="ignore", invalid="ignore"):
        geometry = np.ascontiguousarray(detections[:, :geometry_columns], dtype=np.float32)
        scores = np.ascontiguousarray(detections[:, geometry_columns], dtype=np.float32)
    if geometry.size and not np.isfinite(geometry).all():
        raise ValueError("Detection geometry must be representable as finite float32 values.")
    if scores.size and not np.isfinite(scores).all():
        raise ValueError("Detection scores must be representable as finite float32 values.")
    if len(geometry):
        if is_obb and np.any(geometry[:, 2:4] <= 0):
            raise ValueError("OBB detection rows must have positive width and height.")
        if not is_obb and (np.any(geometry[:, 2] <= geometry[:, 0]) or np.any(geometry[:, 3] <= geometry[:, 1])):
            raise ValueError("AABB detection rows must satisfy x2 > x1 and y2 > y1.")

    return NumPyDetectionRows(
        geometry=geometry,
        scores=scores,
        class_ids=np.asarray(parsed_class_ids, dtype=np.int64),
    )


__all__ = ("NumPyDetectionRows", "parse_numpy_detection_rows")
