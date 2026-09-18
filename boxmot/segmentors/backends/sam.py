"""Ultralytics SAM adapter for detection-aligned masks."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Any

from boxmot.segmentors.backends._common import (
    empty_masks,
    enclosing_boxes,
    frame_to_bgr,
    normalize_masks,
    validate_inputs,
)
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.structures import Detections, Frame, MaskBatch
from boxmot.utils.devices import resolve_device


class SamSegmentor:
    """Run SAM with each detection box as an instance prompt."""

    def __init__(self, spec: SegmentorSpec, *, model: Any | None = None) -> None:
        if spec.artifact is None:
            raise ValueError("Segmentor backend 'sam' requires an artifact.")
        if spec.preprocessing != "default":
            raise ValueError("The built-in SAM adapter currently requires preprocessing='default'.")
        values = spec.option_values()
        self._mask_threshold = float(values.pop("mask_threshold", 0.5))
        if not 0.0 <= self._mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be within [0, 1].")
        if values:
            names = ", ".join(sorted(values))
            raise ValueError(f"Unsupported 'sam' segmentor options: {names}.")
        if spec.precision == "bf16":
            raise ValueError("The built-in SAM adapter does not currently support bf16 inference.")

        self.spec = spec
        self.device = resolve_device(spec.device)
        if model is None:
            sam_class = getattr(import_module("ultralytics"), "SAM")
            model = sam_class(spec.artifact)
        self._model = model

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
        outputs: list[MaskBatch] = []
        for frame, frame_detections in zip(frame_batch, detection_batch):
            if len(frame_detections) == 0:
                outputs.append(empty_masks(frame))
                continue

            results = self._model.predict(
                source=frame_to_bgr(frame),
                bboxes=enclosing_boxes(frame_detections).numpy(),
                device=self.device,
                half=self.spec.precision == "fp16",
                retina_masks=True,
                verbose=False,
            )
            if not isinstance(results, (list, tuple)) or len(results) != 1:
                count = len(results) if isinstance(results, (list, tuple)) else type(results).__name__
                raise ValueError(f"SAM returned {count} results for one frame.")
            masks = getattr(results[0], "masks", None)
            mask_values = None if masks is None else getattr(masks, "data", masks)
            if mask_values is None:
                raise ValueError("SAM returned no masks for non-empty detections.")
            outputs.append(
                normalize_masks(
                    mask_values,
                    count=len(frame_detections),
                    frame=frame,
                    threshold=self._mask_threshold,
                )
            )
        return outputs


def create_sam_segmentor(spec: SegmentorSpec) -> SamSegmentor:
    """Build an Ultralytics SAM segmentor."""
    return SamSegmentor(spec)


__all__ = ("SamSegmentor", "create_sam_segmentor")
