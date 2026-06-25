from __future__ import annotations

import numpy as np

from boxmot.trackers.common.detections.layout import DetectionLayout
from boxmot.trackers.common.geometry.obb import normalize_angle


def empty_output(layout: DetectionLayout, dtype=float) -> np.ndarray:
    """Return an empty public output array for a detection layout."""
    return layout.empty_output(dtype=dtype)


def format_output_row(
    layout: DetectionLayout,
    box: np.ndarray,
    track_id: int,
    conf: float,
    cls: int,
    det_ind: int,
    dtype=np.float32,
) -> np.ndarray:
    """Format one track row using the public tracker output contract."""
    box = np.asarray(box, dtype=dtype).reshape(-1)[: layout.box_cols]
    if box.shape[0] != layout.box_cols:
        raise ValueError(f"Expected {layout.box_cols} output box values, got {box.shape[0]}")
    if layout.is_obb:
        box = box.copy()
        box[4] = normalize_angle(float(box[4]))
    return np.asarray([*box, track_id, conf, cls, det_ind], dtype=dtype)


def format_output_rows(
    layout: DetectionLayout,
    rows,
    dtype=np.float32,
) -> np.ndarray:
    """Return formatted rows with the correct empty shape when no rows exist."""
    return np.asarray(rows, dtype=dtype) if len(rows) else empty_output(layout, dtype=dtype)
