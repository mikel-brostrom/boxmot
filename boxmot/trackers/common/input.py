"""Adapters for packed NumPy tracker inputs and outputs."""

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

    max_exact_float64_integer = 2**53
    parsed_class_ids: list[int] = []
    for value in detections[:, -1].tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Detection class IDs must be non-negative integers.")
        class_id = int(value)
        if class_id != value or class_id < 0 or class_id > np.iinfo(np.int64).max:
            raise ValueError("Detection class IDs must be non-negative integers.")
        if class_id > max_exact_float64_integer:
            raise ValueError(
                "Detection class IDs must be within the exact float64 integer range; use Detections input for Tracks."
            )
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


def pack_numpy_track_rows(
    *,
    geometry: np.ndarray,
    track_ids: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    detection_indices: np.ndarray,
    is_obb: bool,
    detection_count: int,
    owner: str,
) -> np.ndarray:
    """Validate split tracker output and pack float64 AABB8 or OBB9 rows."""

    geometry_columns = 5 if is_obb else 4
    geometry_name = "OBB" if is_obb else "AABB"
    geometry_values = np.asarray(geometry)
    if geometry_values.ndim != 2 or geometry_values.shape[1] != geometry_columns:
        raise ValueError(
            f"{owner} returned {geometry_name} geometry with shape {geometry_values.shape}; "
            f"expected [M, {geometry_columns}]."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        geometry_values = np.ascontiguousarray(geometry_values, dtype=np.float32)
        score_values = np.ascontiguousarray(scores, dtype=np.float32)
    if geometry_values.size and not np.isfinite(geometry_values).all():
        raise ValueError(f"{owner} returned non-finite track geometry.")
    if len(geometry_values):
        if is_obb and np.any(geometry_values[:, 2:4] <= 0):
            raise ValueError(f"{owner} returned OBB tracks without positive width and height.")
        if not is_obb and (
            np.any(geometry_values[:, 2] <= geometry_values[:, 0])
            or np.any(geometry_values[:, 3] <= geometry_values[:, 1])
        ):
            raise ValueError(f"{owner} returned AABB tracks that do not satisfy x2 > x1 and y2 > y1.")

    count = len(geometry_values)
    if score_values.ndim != 1:
        raise ValueError(f"{owner} returned scores with shape {score_values.shape}; expected [M].")
    if len(score_values) != count:
        raise ValueError(f"{owner} returned {len(score_values)} scores for {count} track rows.")
    if score_values.size and not np.isfinite(score_values).all():
        raise ValueError(f"{owner} returned non-finite track scores.")
    if score_values.size and (np.any(score_values < 0) or np.any(score_values > 1)):
        raise ValueError(f"{owner} returned track scores outside the inclusive range [0, 1].")

    integer_columns: dict[str, np.ndarray] = {}
    for name, values in (
        ("track IDs", track_ids),
        ("class IDs", class_ids),
        ("detection indices", detection_indices),
    ):
        array = np.asarray(values)
        if array.ndim != 1:
            raise ValueError(f"{owner} returned {name} with shape {array.shape}; expected [M].")
        if len(array) != count:
            raise ValueError(f"{owner} returned {len(array)} {name} for {count} track rows.")
        if array.size and not np.isfinite(array).all():
            raise ValueError(f"{owner} returned non-finite {name}.")
        if array.size and not np.equal(array, np.floor(array)).all():
            raise ValueError(f"{owner} returned non-integer {name}.")
        if array.size and (np.any(array < np.iinfo(np.int64).min) or np.any(array > np.iinfo(np.int64).max)):
            raise ValueError(f"{owner} returned {name} outside the int64 range.")
        integer_columns[name] = np.ascontiguousarray(array, dtype=np.int64)

    track_id_values = integer_columns["track IDs"]
    class_id_values = integer_columns["class IDs"]
    detection_index_values = integer_columns["detection indices"]
    if track_id_values.size and np.any(track_id_values < 0):
        raise ValueError(f"{owner} returned negative track IDs.")
    if track_id_values.size and len(np.unique(track_id_values)) != count:
        raise ValueError(f"{owner} returned duplicate track IDs.")
    if class_id_values.size and np.any(class_id_values < 0):
        raise ValueError(f"{owner} returned negative class IDs.")
    if detection_index_values.size and np.any(detection_index_values < -1):
        raise ValueError(f"{owner} returned detection indices below -1.")
    if detection_index_values.size and np.any(detection_index_values >= detection_count):
        raise ValueError(f"{owner} returned a detection index outside the current batch.")

    max_exact_float64_integer = 2**53
    for name, values in integer_columns.items():
        if values.size and (np.any(values < -max_exact_float64_integer) or np.any(values > max_exact_float64_integer)):
            raise ValueError(
                f"{owner} returned {name} outside the exact float64 integer range; use Detections input for Tracks."
            )

    rows = np.empty((count, geometry_columns + 4), dtype=np.float64)
    rows[:, :geometry_columns] = geometry_values
    rows[:, geometry_columns] = track_id_values
    rows[:, geometry_columns + 1] = score_values
    rows[:, geometry_columns + 2] = class_id_values
    rows[:, geometry_columns + 3] = detection_index_values
    return rows


__all__ = ("NumPyDetectionRows", "pack_numpy_track_rows", "parse_numpy_detection_rows")
