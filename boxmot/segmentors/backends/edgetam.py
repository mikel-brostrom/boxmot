"""Official EdgeTAM image prediction with detection-aligned box prompts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from importlib import import_module
from typing import Any

import torch

from boxmot.segmentors.backends._common import (
    empty_masks,
    enclosing_boxes,
    normalize_masks,
    validate_inputs,
)
from boxmot.segmentors.propagation.model import build_edgetam_predictor, inference_context
from boxmot.segmentors.specs import SegmentorSpec
from boxmot.structures import Detections, Frame, MaskBatch
from boxmot.utils.devices import resolve_device


class EdgeTAMSegmentor:
    """Generate one mask per detection while retaining only one image embedding.

    ``model`` may be the official video predictor used by mask propagation.
    The image predictor owns its own temporary inference state, so sharing the
    model does not retain or modify temporal propagation history.
    """

    def __init__(self, spec: SegmentorSpec, *, model: Any | None = None) -> None:
        if spec.artifact is None:
            raise ValueError("Segmentor backend 'edgetam' requires an artifact.")
        if spec.preprocessing != "default":
            raise ValueError("The EdgeTAM adapter requires preprocessing='default'.")
        options = spec.option_values()
        mask_threshold = float(options.pop("mask_threshold", 0.0))
        if not math.isfinite(mask_threshold):
            raise ValueError("mask_threshold must be a finite logit threshold.")
        if options:
            names = ", ".join(sorted(options))
            raise ValueError(f"Unsupported 'edgetam' segmentor options: {names}.")

        self.spec = spec
        self.device = resolve_device(spec.device)
        # Validate requested precision before loading checkpoint weights.
        with inference_context(self.device, spec.precision):
            pass
        try:
            predictor_class = getattr(import_module("sam2.sam2_image_predictor"), "SAM2ImagePredictor")
        except ModuleNotFoundError as exc:
            # EdgeTAM is optional; explain how to install it without masking
            # a missing transitive dependency in an existing installation.
            if exc.name not in {"sam2", "sam2.sam2_image_predictor"}:
                raise
            raise ImportError(
                "EdgeTAM segmentation requires the optional mask-guidance dependencies. "
                "Install with 'uv sync --extra cpu --group mask-guidance' "
                "(use --extra cu130 instead of --extra cpu on CUDA hosts)."
            ) from exc
        self.model = model if model is not None else build_edgetam_predictor(spec.artifact, self.device)
        self._predictor = predictor_class(self.model, mask_threshold=mask_threshold)

    def segment(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
    ) -> list[MaskBatch]:
        """Segment sequential boxes in RGB frames and release each image's state."""
        frame_batch, detection_batch = validate_inputs(
            frames, detections, geometry_mode=self.spec.geometry_mode
        )
        outputs: list[MaskBatch] = []
        with torch.inference_mode(), inference_context(self.device, self.spec.precision):
            for frame, frame_detections in zip(frame_batch, detection_batch):
                if len(frame_detections) == 0:
                    outputs.append(empty_masks(frame))
                    continue
                values = torch.empty((len(frame_detections), *frame.image_size), dtype=torch.bool)
                try:
                    image = frame.image.permute(1, 2, 0).contiguous().numpy()
                    self._predictor.set_image(image)
                    del image
                    for index, box in enumerate(enclosing_boxes(frame_detections).numpy()):
                        prediction = self._predictor.predict(
                            box=box, multimask_output=False, return_logits=False
                        )
                        # Upstream applies the configured logit threshold and
                        # returns binary masks as float arrays from predict().
                        values[index] = normalize_masks(
                            prediction[0], count=1, frame=frame, threshold=0.5
                        ).values[0]
                        del prediction
                finally:
                    self._predictor.reset_predictor()
                outputs.append(MaskBatch(values))
        return outputs


def create_edgetam_segmentor(spec: SegmentorSpec, *, model: Any | None = None) -> EdgeTAMSegmentor:
    """Build an official EdgeTAM segmentor, optionally sharing its model."""
    return EdgeTAMSegmentor(spec, model=model)


__all__ = ("EdgeTAMSegmentor", "create_edgetam_segmentor")
