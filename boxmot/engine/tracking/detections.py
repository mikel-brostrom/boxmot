from __future__ import annotations

from typing import Any

import numpy as np

from boxmot.box_schema import get_box_schema_for_mode, schema_from_detection_columns
from boxmot.detectors.base import Detections

MIN_DETECTION_AREA = 10.0
MAX_OBB_SIDE_IMAGE_DIAGONALS = 2.0


def as_2d_array(values: Any, empty_cols: int = 0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        cols = arr.shape[1] if arr.ndim == 2 else empty_cols
        return np.empty((0, cols), dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def extract_detection_array(output: Any, *, fallback_is_obb: bool = False) -> np.ndarray:
    if isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if isinstance(output, Detections):
        cols = output.schema.detection_cols
        return as_2d_array(output.dets, empty_cols=cols)
    if hasattr(output, "dets"):
        dets = getattr(output, "dets")
        if isinstance(dets, np.ndarray) and dets.ndim == 2:
            cols = dets.shape[1]
            schema = schema_from_detection_columns(cols)
            explicit_mode = getattr(output, "is_obb", None)
            if explicit_mode is not None and schema.is_obb != bool(explicit_mode):
                raise ValueError(
                    f"Detector result mode is_obb={bool(explicit_mode)} conflicts with its {cols}-column schema."
                )
        else:
            cols = get_box_schema_for_mode(bool(getattr(output, "is_obb", fallback_is_obb))).detection_cols
        return as_2d_array(dets, empty_cols=cols)
    if output is None:
        return get_box_schema_for_mode(fallback_is_obb).empty_detections()
    return as_2d_array(
        output,
        empty_cols=get_box_schema_for_mode(fallback_is_obb).detection_cols,
    )


def extract_masks(output: Any) -> np.ndarray | None:
    if isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if isinstance(output, Detections) and output.masks is not None:
        return output.masks
    if hasattr(output, "masks"):
        return getattr(output, "masks")
    return None


def detection_validity_mask(
    dets: np.ndarray,
    *,
    image_shape: tuple[int, ...] | None = None,
    min_area: float = MIN_DETECTION_AREA,
) -> np.ndarray:
    """Return rows safe for geometry, ReID cropping, and tracker ingestion."""
    arr = as_2d_array(dets)
    if arr.ndim != 2:
        raise ValueError(f"Detections must be a 2D array, got shape {arr.shape}")
    schema = schema_from_detection_columns(arr.shape[1])
    if len(arr) == 0:
        return np.empty((0,), dtype=bool)

    valid = np.isfinite(arr).all(axis=1)
    geometry = arr[:, : schema.geometry_cols].astype(np.float64, copy=False)
    if schema.is_obb:
        width, height = geometry[:, 2], geometry[:, 3]
        valid &= (width > 0.0) & (height > 0.0) & ((width * height) >= float(min_area))

        if image_shape is not None and len(image_shape) >= 2:
            image_height, image_width = float(image_shape[0]), float(image_shape[1])
            max_side = MAX_OBB_SIDE_IMAGE_DIAGONALS * np.hypot(image_height, image_width)
            valid &= (width <= max_side) & (height <= max_side)
            center_x, center_y = geometry[:, 0], geometry[:, 1]
            valid &= (
                (center_x >= -max_side)
                & (center_x <= image_width + max_side)
                & (center_y >= -max_side)
                & (center_y <= image_height + max_side)
            )
    else:
        x1, y1, x2, y2 = (geometry[:, index] for index in range(4))
        valid &= (x2 > x1) & (y2 > y1) & (((x2 - x1) * (y2 - y1)) >= float(min_area))
    return valid


def sanitize_detections(
    dets: np.ndarray,
    masks: np.ndarray | None = None,
    *,
    image_shape: tuple[int, ...] | None = None,
    min_area: float = MIN_DETECTION_AREA,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Filter detections and aligned masks with one shared validity mask."""
    arr = as_2d_array(dets)
    valid = detection_validity_mask(arr, image_shape=image_shape, min_area=min_area)

    filtered_masks = None
    if masks is not None:
        masks_arr = np.asarray(masks)
        if masks_arr.ndim == 0:
            raise ValueError(f"Masks must have a leading detection dimension, got {masks_arr.shape}")
        if len(masks_arr) != len(arr):
            raise ValueError(f"Masks must be aligned with detections: masks={len(masks_arr)} dets={len(arr)}")
        filtered_masks = masks_arr[valid]
    return arr[valid], filtered_masks, valid


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
    if dets is None:
        return result.schema.empty_detections()
    image_shape = getattr(getattr(result, "orig_img", None), "shape", None)
    sanitized, _, _ = sanitize_detections(dets, image_shape=image_shape)
    return sanitized


__all__ = (
    "as_2d_array",
    "detection_validity_mask",
    "extract_detection_array",
    "extract_masks",
    "prepare_detections",
    "sanitize_detections",
)
