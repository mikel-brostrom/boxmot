# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Canonical RT-DETR detector backend."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from boxmot.components.timing import timed_component_phase
from boxmot.detectors._capabilities import capabilities_from_spec
from boxmot.detectors.backends.base import (
    _as_numpy,
    _canonical_detections,
    _filter_rows,
    _validate_backend_spec,
    _validate_results,
    _validated_frames,
)
from boxmot.detectors.specs import DetectorSpec
from boxmot.structures import Detections, Frame
from boxmot.utils import logger as LOGGER
from boxmot.utils.devices import resolve_device


def _transformers_classes():
    """Import optional RT-DETR dependencies only when constructing the backend."""

    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    return RTDetrImageProcessor, RTDetrV2ForObjectDetection


class RTDetrDetector:
    """Offline Hugging Face RT-DETR v2 implementation of the detector API."""

    def __init__(self, spec: DetectorSpec) -> None:
        values = _validate_backend_spec(spec, backend="rtdetr", supports_obb=False)
        snapshot = Path(spec.artifact).expanduser().resolve()
        if not snapshot.is_dir():
            raise ValueError(
                "RT-DETR requires a resolved local Hugging Face snapshot directory "
                "containing both processor configuration and model weights."
            )

        self.spec = spec
        self.capabilities = capabilities_from_spec(spec)
        self.device = resolve_device(spec.device)
        self.imgsz = values.get("image_size")
        self._confidence = float(values.get("confidence", 0.25))
        self._classes = values.get("classes")
        self.model_id = str(snapshot)

        LOGGER.info(f"Loading RT-DETR model snapshot: {self.model_id}")
        processor_class, model_class = _transformers_classes()
        self.image_processor = processor_class.from_pretrained(self.model_id, local_files_only=True)
        self.model = model_class.from_pretrained(self.model_id, local_files_only=True).to(self.device).eval()
        self.names = dict(self.model.config.id2label)

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        """Detect one ordered canonical result for every input frame."""

        batch = _validated_frames(frames)
        if not batch:
            return []
        with timed_component_phase("detector", "preprocess", device=self.device):
            pil_images = [
                Image.fromarray(frame.image.permute(1, 2, 0).contiguous().numpy(), mode="RGB") for frame in batch
            ]
            target_sizes = torch.tensor(
                [(image.height, image.width) for image in pil_images],
                device=self.device,
            )
            model_inputs = self.image_processor(images=pil_images, return_tensors="pt").to(self.device)
        with timed_component_phase("detector", "process", device=self.device):
            with torch.inference_mode():
                predictions = self.model(**model_inputs)
        with timed_component_phase("detector", "postprocess", device=self.device):
            decoded = self.image_processor.post_process_object_detection(
                predictions,
                target_sizes=target_sizes,
                threshold=0.0,
            )
            if len(decoded) != len(batch):
                raise ValueError(f"RT-DETR decoded {len(decoded)} results for {len(batch)} frames.")

            results = [
                self._decode_result(frame, result, classes=self._classes) for frame, result in zip(batch, decoded)
            ]
            return _validate_results(batch, results, self.capabilities)

    def _decode_result(
        self,
        frame: Frame,
        result: Any,
        *,
        classes: int | Iterable[int] | None,
    ) -> Detections:
        boxes = _as_numpy(result["boxes"]).reshape(-1, 4)
        scores = _as_numpy(result["scores"]).reshape(-1, 1)
        labels = _as_numpy(result["labels"]).reshape(-1, 1)
        rows = np.concatenate((boxes, scores, labels), axis=1)
        rows = _filter_rows(rows, confidence=self._confidence, classes=classes)
        return _canonical_detections(frame, rows)


__all__ = ("RTDetrDetector",)
