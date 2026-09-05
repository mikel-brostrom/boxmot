"""Torchvision Mask R-CNN adapter for detection-aligned masks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from boxmot.segmentors.backends._common import empty_masks, enclosing_boxes, normalize_masks, validate_inputs
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.structures import Detections, Frame, MaskBatch
from boxmot.utils import logger as LOGGER


def _pairwise_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if not len(left) or not len(right):
        return torch.empty((len(left), len(right)), dtype=torch.float32, device=left.device)
    top_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=2)
    left_area = (left[:, 2:] - left[:, :2]).clamp_min(0).prod(dim=1)
    right_area = (right[:, 2:] - right[:, :2]).clamp_min(0).prod(dim=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return torch.where(union > 0, intersection / union, torch.zeros_like(intersection))


def _class_mapping(value: object) -> dict[int, int]:
    if not isinstance(value, tuple):
        raise TypeError("class_mapping must be a tuple of (detection_class, model_class) pairs.")
    mapping: dict[int, int] = {}
    for index, pair in enumerate(value):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"class_mapping[{index}] must be a two-integer tuple.")
        detection_class, model_class = pair
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in pair):
            raise ValueError(f"class_mapping[{index}] values must be non-negative integers.")
        if detection_class in mapping:
            raise ValueError(f"class_mapping contains duplicate detection class {detection_class}.")
        mapping[detection_class] = model_class
    return mapping


def _load_model(spec: SegmentorSpec) -> Any:
    if spec.artifact is None:
        raise ValueError("Segmentor backend 'maskrcnn' requires a checkpoint artifact.")

    if spec.artifact.startswith("torchvision://"):
        raise ValueError(
            "Mask R-CNN requires a resolved local checkpoint; torchvision weight policies may download implicitly."
        )

    from torchvision.models.detection import maskrcnn_resnet50_fpn

    checkpoint_path = Path(spec.artifact)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Mask R-CNN checkpoint does not exist: {checkpoint_path}")
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    return model


class MaskRCNNSegmentor:
    """Match Mask R-CNN predictions to caller-provided detections by IoU."""

    def __init__(self, spec: SegmentorSpec, *, model: Any | None = None) -> None:
        if spec.preprocessing != "default":
            raise ValueError("The built-in Mask R-CNN adapter currently requires preprocessing='default'.")
        values = spec.option_values()
        self._class_mapping = _class_mapping(values.pop("class_mapping", ()))
        self._mask_threshold = float(values.pop("mask_threshold", 0.5))
        self._match_iou = float(values.pop("match_iou", 0.5))
        self._score_threshold = float(values.pop("score_threshold", 0.0))
        for name, value in (
            ("mask_threshold", self._mask_threshold),
            ("match_iou", self._match_iou),
            ("score_threshold", self._score_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1].")
        if values:
            names = ", ".join(sorted(values))
            raise ValueError(f"Unsupported 'maskrcnn' segmentor options: {names}.")

        self.spec = spec
        self._device = torch.device(spec.device)
        self._dtype = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }[spec.precision]
        if model is None:
            model = _load_model(spec)
        self._model = model.to(device=self._device, dtype=self._dtype).eval()
        self.unmatched_count = 0
        self.last_unmatched_count = 0

    def _assignment(
        self,
        detections: Detections,
        predicted_boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> list[tuple[int, int]]:
        ious = _pairwise_iou(enclosing_boxes(detections).to(self._device), predicted_boxes)
        if not ious.numel():
            return []
        mapped_classes = torch.tensor(
            [self._class_mapping.get(class_id, class_id) for class_id in detections.class_ids.tolist()],
            dtype=torch.int64,
            device=self._device,
        )
        eligible = (mapped_classes[:, None] == labels.to(dtype=torch.int64)[None, :]) & (
            ious >= self._match_iou
        )
        costs = (1.0 - ious).to(device="cpu", dtype=torch.float64).numpy()
        costs[~eligible.cpu().numpy()] = 1_000_000.0
        costs += np.arange(costs.shape[1], dtype=np.float64)[None, :] * 1e-12
        detection_indices, prediction_indices = linear_sum_assignment(costs)
        return [
            (int(detection_index), int(prediction_index))
            for detection_index, prediction_index in zip(detection_indices, prediction_indices)
            if bool(eligible[detection_index, prediction_index])
        ]

    def _align_prediction(
        self,
        frame: Frame,
        detections: Detections,
        prediction: dict[str, torch.Tensor],
    ) -> MaskBatch:
        predicted_boxes = prediction.get("boxes", torch.empty((0, 4), device=self._device)).to(self._device)
        scores = prediction.get("scores", torch.empty((len(predicted_boxes),), device=self._device)).to(self._device)
        labels = prediction.get("labels", torch.empty((len(predicted_boxes),), device=self._device)).to(self._device)
        predicted_masks = prediction.get(
            "masks",
            torch.empty((len(predicted_boxes), 1, frame.height, frame.width), device=self._device),
        ).to(self._device)

        valid = scores >= self._score_threshold
        predicted_boxes = predicted_boxes[valid]
        labels = labels[valid]
        predicted_masks = predicted_masks[valid]
        aligned = torch.zeros(
            (len(detections), frame.height, frame.width),
            dtype=torch.float32,
            device=self._device,
        )
        matches = self._assignment(detections, predicted_boxes, labels)
        for detection_index, prediction_index in matches:
            mask = predicted_masks[prediction_index]
            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2:
                raise ValueError(f"Mask R-CNN predicted a mask with shape {tuple(mask.shape)}.")
            aligned[detection_index] = mask

        unmatched = len(detections) - len(matches)
        self.unmatched_count += unmatched
        self.last_unmatched_count += unmatched
        if unmatched:
            LOGGER.warning(
                "Mask R-CNN left %d/%d detections unmatched for sample %s; emitting all-false masks.",
                unmatched,
                len(detections),
                frame.sample_id,
            )
        return normalize_masks(
            aligned,
            count=len(detections),
            frame=frame,
            threshold=self._mask_threshold,
        )

    def segment(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
    ) -> list[MaskBatch]:
        frame_batch, detection_batch = validate_inputs(
            frames,
            detections,
            geometry_mode=self.spec.geometry_mode,
        )
        self.last_unmatched_count = 0
        outputs: list[MaskBatch | None] = [None] * len(frame_batch)
        active_indices = [index for index, item in enumerate(detection_batch) if len(item)]
        for index, frame in enumerate(frame_batch):
            if index not in active_indices:
                outputs[index] = empty_masks(frame)
        if active_indices:
            images = [
                frame_batch[index].image.to(device=self._device, dtype=self._dtype).div(255.0)
                for index in active_indices
            ]
            with torch.inference_mode():
                predictions = self._model(images)
            if not isinstance(predictions, (list, tuple)) or len(predictions) != len(active_indices):
                count = len(predictions) if isinstance(predictions, (list, tuple)) else type(predictions).__name__
                raise ValueError(f"Mask R-CNN returned {count} results for {len(active_indices)} frames.")
            for index, prediction in zip(active_indices, predictions):
                outputs[index] = self._align_prediction(frame_batch[index], detection_batch[index], prediction)
        return [output for output in outputs if output is not None]


def create_maskrcnn_segmentor(spec: SegmentorSpec) -> MaskRCNNSegmentor:
    """Build a Torchvision Mask R-CNN segmentor."""
    return MaskRCNNSegmentor(spec)


__all__ = ("MaskRCNNSegmentor", "create_maskrcnn_segmentor")
