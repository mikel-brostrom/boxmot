"""Shared canonical conversion helpers for segmentor adapters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes


def validate_inputs(
    frames: Sequence[Frame],
    detections: Sequence[Detections],
    *,
    geometry_mode: str = "auto",
) -> tuple[list[Frame], list[Detections]]:
    frame_batch = list(frames)
    detection_batch = list(detections)
    if len(frame_batch) != len(detection_batch):
        raise ValueError(
            f"frames and detections must be aligned; received {len(frame_batch)} and {len(detection_batch)}."
        )
    for frame, frame_detections in zip(frame_batch, detection_batch):
        if frame.sample_id != frame_detections.sample_id:
            raise ValueError(
                f"Frame sample_id {frame.sample_id!r} does not match detections "
                f"sample_id {frame_detections.sample_id!r}."
            )
        if geometry_mode == "aabb" and frame_detections.is_obb:
            raise ValueError("Segmentor is configured for AABB detections, but received OBB geometry.")
        if geometry_mode == "obb" and not frame_detections.is_obb:
            raise ValueError("Segmentor is configured for OBB detections, but received AABB geometry.")
    return frame_batch, detection_batch


def frame_to_bgr(frame: Frame) -> np.ndarray:
    return frame.image.permute(1, 2, 0).flip(-1).contiguous().numpy()


def enclosing_boxes(detections: Detections) -> torch.Tensor:
    geometry = detections.geometry
    if isinstance(geometry, Boxes):
        return geometry.values
    if not isinstance(geometry, OrientedBoxes):
        raise TypeError(f"Unsupported detection geometry: {type(geometry).__name__}.")
    cx, cy, width, height, angle = geometry.values.unbind(dim=1)
    cosine = angle.cos().abs()
    sine = angle.sin().abs()
    half_width = 0.5 * (width * cosine + height * sine)
    half_height = 0.5 * (width * sine + height * cosine)
    return torch.stack((cx - half_width, cy - half_height, cx + half_width, cy + half_height), dim=1)


def empty_masks(frame: Frame) -> MaskBatch:
    return MaskBatch(torch.empty((0, frame.height, frame.width), dtype=torch.bool))


def normalize_masks(values: object, *, count: int, frame: Frame, threshold: float) -> MaskBatch:
    masks = torch.as_tensor(values)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2 and count == 1:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3 or masks.shape[0] != count:
        raise ValueError(f"Segmentor returned masks with shape {tuple(masks.shape)} for {count} detections.")
    if tuple(masks.shape[-2:]) != frame.image_size:
        masks = functional.interpolate(
            masks.to(dtype=torch.float32).unsqueeze(1),
            size=frame.image_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    if masks.dtype != torch.bool:
        masks = masks > threshold
    return MaskBatch(masks.to(device="cpu", dtype=torch.bool).contiguous())


__all__ = (
    "empty_masks",
    "enclosing_boxes",
    "frame_to_bgr",
    "normalize_masks",
    "validate_inputs",
)
