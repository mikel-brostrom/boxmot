from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BoxType(str, Enum):
    """Geometry modes supported by the detector/tracker pipeline."""

    AABB = "aabb"
    OBB = "obb"


@dataclass(frozen=True)
class BoxSchema:
    """Column contract shared by detection, tracking, cache, and MOT paths."""

    box_type: BoxType
    geometry_cols: int
    detection_cols: int
    track_cols: int
    cache_cols: int
    mot_cols: int

    @property
    def is_obb(self) -> bool:
        return self.box_type is BoxType.OBB

    @property
    def detection_conf_index(self) -> int:
        return self.geometry_cols

    @property
    def detection_class_index(self) -> int:
        return self.geometry_cols + 1

    @property
    def indexed_detection_cols(self) -> int:
        return self.detection_cols + 1

    @property
    def frame_tagged_track_cols(self) -> int:
        """Columns in an internal ``[frame, *tracker_output]`` row."""
        return self.track_cols + 1

    @property
    def track_id_index(self) -> int:
        return self.geometry_cols

    @property
    def track_conf_index(self) -> int:
        return self.geometry_cols + 1

    @property
    def track_class_index(self) -> int:
        return self.geometry_cols + 2

    @property
    def track_detection_index(self) -> int:
        return self.geometry_cols + 3

    def empty_detections(self, dtype=np.float32) -> np.ndarray:
        return np.empty((0, self.detection_cols), dtype=dtype)

    def empty_tracks(self, dtype=np.float32) -> np.ndarray:
        return np.empty((0, self.track_cols), dtype=dtype)

    def empty_cache(self, dtype=np.float32) -> np.ndarray:
        return np.empty((0, self.cache_cols), dtype=dtype)

    def empty_mot(self, dtype=np.float32) -> np.ndarray:
        return np.empty((0, self.mot_cols), dtype=dtype)


AABB_SCHEMA = BoxSchema(
    box_type=BoxType.AABB,
    geometry_cols=4,
    detection_cols=6,
    track_cols=8,
    cache_cols=7,
    mot_cols=9,
)
OBB_SCHEMA = BoxSchema(
    box_type=BoxType.OBB,
    geometry_cols=5,
    detection_cols=7,
    track_cols=9,
    cache_cols=8,
    mot_cols=13,
)
BOX_SCHEMAS = (AABB_SCHEMA, OBB_SCHEMA)


def normalize_box_type(value: BoxType | str | None, *, default: BoxType | str | None = None) -> BoxType:
    """Normalize a box type and reject values that would otherwise fall back silently."""
    resolved = default if value in (None, "") else value
    if isinstance(resolved, BoxType):
        return resolved
    if resolved in (None, ""):
        raise ValueError("A box type is required; expected 'aabb' or 'obb'.")
    try:
        return BoxType(str(resolved).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported box type {value!r}; expected 'aabb' or 'obb'.") from exc


def get_box_schema(box_type: BoxType | str) -> BoxSchema:
    return OBB_SCHEMA if normalize_box_type(box_type) is BoxType.OBB else AABB_SCHEMA


def get_box_schema_for_mode(is_obb: bool) -> BoxSchema:
    return OBB_SCHEMA if bool(is_obb) else AABB_SCHEMA


def _schema_from_columns(columns: int, attribute: str, label: str) -> BoxSchema:
    matches = [schema for schema in BOX_SCHEMAS if getattr(schema, attribute) == int(columns)]
    if len(matches) == 1:
        return matches[0]
    expected = ", ".join(str(getattr(schema, attribute)) for schema in BOX_SCHEMAS)
    raise ValueError(f"Unsupported {label} column count {columns}; expected {expected}.")


def schema_from_detection_columns(columns: int) -> BoxSchema:
    return _schema_from_columns(columns, "detection_cols", "detection")


def schema_from_track_columns(columns: int) -> BoxSchema:
    return _schema_from_columns(columns, "track_cols", "tracker output")


def schema_from_cache_columns(columns: int) -> BoxSchema:
    return _schema_from_columns(columns, "cache_cols", "detection-cache")


def schema_from_frame_tagged_track_columns(columns: int) -> BoxSchema:
    return _schema_from_columns(columns, "frame_tagged_track_cols", "frame-tagged tracker output")


def schema_from_mot_columns(columns: int) -> BoxSchema:
    return _schema_from_columns(columns, "mot_cols", "MOT output")


__all__ = (
    "AABB_SCHEMA",
    "BOX_SCHEMAS",
    "BoxSchema",
    "BoxType",
    "OBB_SCHEMA",
    "get_box_schema",
    "get_box_schema_for_mode",
    "normalize_box_type",
    "schema_from_cache_columns",
    "schema_from_detection_columns",
    "schema_from_frame_tagged_track_columns",
    "schema_from_mot_columns",
    "schema_from_track_columns",
)
