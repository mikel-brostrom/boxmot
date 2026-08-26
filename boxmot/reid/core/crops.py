"""Shared image and detection crop preparation for ReID runtimes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from boxmot.box_schema import AABB_SCHEMA, OBB_SCHEMA

AABB_COLUMN_COUNTS = frozenset((AABB_SCHEMA.geometry_cols, AABB_SCHEMA.detection_cols, AABB_SCHEMA.track_cols))
OBB_COLUMN_COUNTS = frozenset((OBB_SCHEMA.geometry_cols, OBB_SCHEMA.detection_cols, OBB_SCHEMA.track_cols))
OBB_SQUARE_RTOL = 1e-3


def resolve_image(image: np.ndarray | str | Path) -> np.ndarray:
    """Resolve a ReID image input to a BGR numpy array."""
    if isinstance(image, (str, Path)):
        resolved = cv2.imread(str(image))
        if resolved is None:
            raise FileNotFoundError(f"Could not load image from {image}")
        return resolved
    if isinstance(image, np.ndarray):
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


def coerce_boxes(boxes: Any) -> np.ndarray:
    """Return supported AABB/OBB rows as canonical 4/5-column geometry."""
    array = np.asarray(boxes, dtype=np.float32)
    if array.ndim == 1:
        if array.size == 0:
            return np.empty((0, AABB_SCHEMA.geometry_cols), dtype=np.float32)
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"ReID boxes must be a 2D array, got shape {array.shape}")

    columns = array.shape[1]
    if columns in AABB_COLUMN_COUNTS:
        geometry_cols = AABB_SCHEMA.geometry_cols
    elif columns in OBB_COLUMN_COUNTS:
        geometry_cols = OBB_SCHEMA.geometry_cols
    else:
        raise ValueError(
            f"ReID expects AABB rows with 4/6/8 columns or OBB rows with 5/7/9 columns, got shape {array.shape}"
        )
    return np.ascontiguousarray(array[:, :geometry_cols], dtype=np.float32)


def coerce_crops(crops: Any) -> list[np.ndarray]:
    """Resolve supported crop inputs into a list of BGR arrays."""
    if isinstance(crops, (str, Path)):
        return [resolve_image(crops)]
    if isinstance(crops, np.ndarray):
        if crops.ndim == 4:
            return [np.asarray(crop) for crop in crops]
        if crops.ndim == 3:
            return [crops]
        raise ValueError(f"Unsupported crop tensor shape: {crops.shape}")
    if isinstance(crops, (list, tuple)):
        return [resolve_image(crop) if isinstance(crop, (str, Path)) else np.asarray(crop) for crop in crops]
    raise ValueError(f"Unsupported ReID input type: {type(crops)}")


def obb_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert one OBB `[cx, cy, w, h, angle]` to its enclosing AABB."""
    cx, cy, width, height, angle = np.asarray(box, dtype=np.float32).reshape(-1)[:5]
    rect = (
        (float(cx), float(cy)),
        (max(float(width), 1e-4), max(float(height), 1e-4)),
        float(np.degrees(angle)),
    )
    corners = cv2.boxPoints(rect)
    x1, y1 = corners.min(axis=0)
    x2, y2 = corners.max(axis=0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def canonicalize_obb_for_crop(
    box: np.ndarray,
    input_shape: tuple[int, int] = (256, 128),
) -> np.ndarray:
    """Return a representation-invariant OBB aligned to the model crop."""
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size < 5:
        raise ValueError("Expected an OBB with at least five values")

    canonical = values[:5].copy()
    if not np.isfinite(canonical).all():
        raise ValueError("OBB crop coordinates must be finite")

    cx, cy, width, height, angle = canonical
    if width <= 0.0 or height <= 0.0:
        raise ValueError("OBB crop width and height must be positive")

    input_height, input_width = (int(input_shape[0]), int(input_shape[1]))
    if input_height <= 0 or input_width <= 0:
        raise ValueError(f"Invalid ReID input shape: {input_shape}")

    input_is_portrait = input_height >= input_width
    if (input_is_portrait and width > height) or (not input_is_portrait and height > width):
        width, height = height, width
        angle += np.pi / 2.0

    nearly_square = np.isclose(width, height, rtol=OBB_SQUARE_RTOL, atol=1e-6)
    angle_period = np.pi / 2.0 if nearly_square else np.pi
    angle = ((angle + angle_period / 2.0) % angle_period) - angle_period / 2.0
    if np.isclose(angle, 0.0, rtol=0.0, atol=1e-6):
        angle = 0.0

    return np.array([cx, cy, width, height, angle], dtype=np.float32)


def crop_obb(
    box: np.ndarray,
    image: np.ndarray,
    input_shape: tuple[int, int] = (256, 128),
    max_output_side: int | None = None,
) -> np.ndarray:
    """Extract a canonical, rectified crop from an oriented box."""
    cx, cy, width, height, angle = canonicalize_obb_for_crop(box, input_shape)
    scale = 1.0
    if max_output_side is not None:
        bounded_side = max(int(max_output_side), 1)
        scale = min(1.0, bounded_side / max(float(width), float(height)))
    output_width = max(int(round(float(width) * scale)), 1)
    output_height = max(int(round(float(height) * scale)), 1)

    matrix = cv2.getRotationMatrix2D(
        (float(cx), float(cy)),
        float(np.degrees(angle)),
        scale,
    )
    matrix[0, 2] += output_width / 2.0 - float(cx)
    matrix[1, 2] += output_height / 2.0 - float(cy)
    return cv2.warpAffine(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def is_obb_box(box: np.ndarray) -> bool:
    """Return whether one row uses a supported OBB layout."""
    return np.asarray(box).reshape(-1).shape[0] in OBB_COLUMN_COUNTS


def boxes_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Normalize AABB/OBB detections to enclosing `[x1, y1, x2, y2]`."""
    array = coerce_boxes(boxes)
    if array.size == 0:
        return np.empty((0, AABB_SCHEMA.geometry_cols), dtype=np.float32)
    if array.shape[1] == OBB_SCHEMA.geometry_cols:
        return np.vstack([obb_to_xyxy(box[:5]) for box in array]).astype(np.float32)
    return array


def extract_crops(
    boxes: np.ndarray,
    image: np.ndarray,
    input_shape: tuple[int, int],
) -> list[np.ndarray]:
    """Extract native AABB or rectified OBB crops from an image."""
    image_height, image_width = image.shape[:2]
    coerced = coerce_boxes(boxes)
    oriented = coerced.shape[1] == OBB_SCHEMA.geometry_cols
    crops: list[np.ndarray] = []
    for box in coerced:
        if oriented:
            crop = crop_obb(
                box[:5],
                image,
                input_shape=input_shape,
                max_output_side=max(input_shape),
            )
        else:
            x1, y1, x2, y2 = boxes_to_xyxy(box.reshape(1, -1))[0].round().astype("int")
            clipped_x1, clipped_y1 = max(0, x1), max(0, y1)
            clipped_x2, clipped_y2 = min(image_width, x2), min(image_height, y2)
            if clipped_x2 > clipped_x1 and clipped_y2 > clipped_y1:
                crop = image[clipped_y1:clipped_y2, clipped_x1:clipped_x2]
            else:
                crop = np.zeros((*input_shape, 3), dtype=np.uint8)
        crops.append(crop)
    return crops


def prepare_crop_batch(
    crops: Sequence[np.ndarray],
    *,
    input_shape: tuple[int, int],
    device: torch.device,
    half: bool,
    preprocess_fn: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Resize, convert, batch, and normalize BGR ReID crops."""
    dtype = torch.float16 if half else torch.float32
    batch = torch.empty((len(crops), 3, *input_shape), dtype=dtype, device=device)
    for index, crop in enumerate(crops):
        if crop.size == 0:
            crop = np.zeros((*input_shape, 3), dtype=np.uint8)
        resized = preprocess_fn(crop, input_shape)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).to(device, dtype=dtype)
        batch[index] = tensor.permute(2, 0, 1)
    return ((batch / 255.0) - mean) / std


def build_crop_batch(
    boxes: np.ndarray,
    image: np.ndarray,
    *,
    input_shape: tuple[int, int],
    device: torch.device,
    half: bool,
    preprocess_fn: Callable[[np.ndarray, tuple[int, int]], np.ndarray],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Extract and prepare one batch of detection crops."""
    return prepare_crop_batch(
        extract_crops(boxes, image, input_shape),
        input_shape=input_shape,
        device=device,
        half=half,
        preprocess_fn=preprocess_fn,
        mean=mean,
        std=std,
    )


__all__ = (
    "AABB_COLUMN_COUNTS",
    "boxes_to_xyxy",
    "build_crop_batch",
    "canonicalize_obb_for_crop",
    "coerce_boxes",
    "coerce_crops",
    "crop_obb",
    "extract_crops",
    "is_obb_box",
    "obb_to_xyxy",
    "prepare_crop_batch",
    "resolve_image",
)
