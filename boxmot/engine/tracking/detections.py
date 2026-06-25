from __future__ import annotations

from typing import Any

import numpy as np

from boxmot.detectors.base import Detections


def as_2d_array(values: Any, empty_cols: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        cols = arr.shape[1] if arr.ndim == 2 else empty_cols
        return np.empty((0, cols), dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def extract_detection_array(output: Any) -> np.ndarray:
    if isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if isinstance(output, Detections):
        cols = output.dets.shape[1] if output.dets.ndim == 2 else (7 if output.is_obb else 6)
        return as_2d_array(output.dets, empty_cols=cols)
    if hasattr(output, "dets"):
        dets = getattr(output, "dets")
        cols = dets.shape[1] if isinstance(dets, np.ndarray) and dets.ndim == 2 else 6
        return as_2d_array(dets, empty_cols=cols)
    if output is None:
        return np.empty((0, 6), dtype=np.float32)
    return as_2d_array(output, empty_cols=6)


def extract_masks(output: Any) -> np.ndarray | None:
    if isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if isinstance(output, Detections) and output.masks is not None:
        return output.masks
    if hasattr(output, "masks"):
        return getattr(output, "masks")
    return None


def prepare_detections(result: Detections) -> np.ndarray:
    """
    Extract detections from a result and sanitize them for downstream use.

    For AABB (N, 6) - [x1, y1, x2, y2, conf, cls]:
      removes boxes where x2 <= x1, y2 <= y1, or area < 10 px2.

    For OBB (N, 7) - [cx, cy, w, h, angle, conf, cls]:
      removes boxes where w <= 0, h <= 0, or w*h < 10 px2.

    Returns filtered array of the same width, or empty (0, 6)/(0, 7) when
    no valid detections remain.
    """
    dets = result.dets
    if dets is None or len(dets) == 0:
        n_cols = dets.shape[1] if dets is not None and dets.ndim == 2 else 6
        return np.empty((0, n_cols))

    if result.is_obb:
        w, h = dets[:, 2], dets[:, 3]
        valid = (w > 0) & (h > 0) & (w * h >= 10.0)
    else:
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        valid = (x2 > x1) & (y2 > y1) & ((x2 - x1) * (y2 - y1) >= 10.0)

    return dets[valid]


__all__ = (
    "as_2d_array",
    "extract_detection_array",
    "extract_masks",
    "prepare_detections",
)
