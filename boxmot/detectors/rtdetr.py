# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from boxmot.detectors.base import BaseDetectorBackend, Detections, as_numpy, ensure_image_batch, filter_detections
from boxmot.utils import logger as LOGGER


def _transformers_classes():
    """Import the optional RT-DETR dependencies only when this backend is built."""
    from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

    return RTDetrImageProcessor, RTDetrV2ForObjectDetection


class RTDetrDetector(BaseDetectorBackend):
    """Hugging Face RT-DETR v2 backend with a batch-preserving staged pipeline."""

    ch = 3

    def __init__(self, model: str | Path, device: str | torch.device, imgsz: Any = None) -> None:
        self.device = device
        self.imgsz = imgsz  # RT-DETR's image processor owns resize policy.
        self.model_id = self._model_id(model)

        LOGGER.info(f"Loading RT-DETR model: {self.model_id}")
        processor_class, model_class = _transformers_classes()
        self.image_processor = processor_class.from_pretrained(self.model_id)
        self.model = model_class.from_pretrained(self.model_id).to(device).eval()
        self.names = dict(self.model.config.id2label)
        self._images: list[np.ndarray] = []
        self._target_sizes: torch.Tensor | None = None

    @staticmethod
    def _model_id(model: str | Path) -> str:
        model_reference = str(model)
        while model_reference.lower().endswith(".pt"):
            model_reference = model_reference[:-3]
        if model_reference.startswith("PekingU/"):
            return model_reference
        return f"PekingU/{Path(model_reference).name}"

    def preprocess(self, images: list[np.ndarray], **kwargs: Any) -> Any:
        """Convert BGR arrays to RGB PIL images and build processor inputs."""
        self._images = ensure_image_batch(images)
        pil_images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in self._images]
        self._target_sizes = torch.tensor(
            [(image.height, image.width) for image in pil_images],
            device=self.device,
        )
        return self.image_processor(images=pil_images, return_tensors="pt").to(self.device)

    @torch.inference_mode()
    def process(self, preprocessed: Any, **kwargs: Any) -> Any:
        """Run the model forward pass without mixing images in the batch."""
        return self.model(**preprocessed)

    def postprocess(
        self,
        predictions: Any,
        conf: float = 0.25,
        iou: float = 0.7,
        classes: int | Iterable[int] | None = None,
        agnostic_nms: bool = False,
        **kwargs: Any,
    ) -> list[Detections]:
        """Decode, filter, and preserve one result for every input image."""
        if self._target_sizes is None:
            raise RuntimeError("RT-DETR postprocess called before preprocess.")

        decoded = self.image_processor.post_process_object_detection(
            predictions,
            target_sizes=self._target_sizes,
            threshold=0.0,
        )
        if len(decoded) != len(self._images):
            raise ValueError(f"RT-DETR decoded {len(decoded)} results for {len(self._images)} input images.")

        results: list[Detections] = []
        for image, result in zip(self._images, decoded):
            boxes = as_numpy(result["boxes"]).reshape(-1, 4)
            scores = as_numpy(result["scores"]).reshape(-1, 1)
            labels = as_numpy(result["labels"]).reshape(-1, 1)
            detections = np.concatenate((boxes, scores, labels), axis=1)
            detections = filter_detections(detections, confidence=conf, classes=classes)
            results.append(Detections(dets=detections, orig_img=image, names=self.names))
        return results


__all__ = ("RTDetrDetector",)
