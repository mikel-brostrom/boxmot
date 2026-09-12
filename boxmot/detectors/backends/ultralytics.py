# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Canonical Ultralytics detector backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import ops

from boxmot.components.timing import timed_component_phase
from boxmot.detectors._capabilities import capabilities_from_spec
from boxmot.detectors.backends.base import (
    _as_numpy,
    _canonical_detections,
    _empty_rows,
    _frame_to_bgr,
    _validate_backend_spec,
    _validate_results,
    _validated_frames,
)
from boxmot.detectors.specs import DetectorSpec
from boxmot.resources.paths import resolve_model_path
from boxmot.structures import Detections, Frame
from boxmot.utils.devices import resolve_device


class UltralyticsDetector:
    """Ultralytics YOLO detector exposing only the canonical detector API."""

    def __init__(self, spec: DetectorSpec) -> None:
        values = _validate_backend_spec(spec, backend="ultralytics", supports_obb=True)
        model_path = resolve_model_path(spec.artifact)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Ultralytics detector artifact does not exist: {model_path}. "
                "Resolve downloads before constructing the detector."
            )

        self.spec = spec
        self.capabilities = capabilities_from_spec(spec)
        self.device = resolve_device(spec.device)
        self.imgsz = values.get("image_size")
        self._prediction_options = {
            "conf": values.get("confidence", 0.25),
            "iou": values.get("iou", 0.7),
            "classes": values.get("classes"),
            "agnostic_nms": values.get("agnostic_nms", False),
        }
        self._yolo = YOLO(str(model_path))
        self._is_obb = str(getattr(self._yolo, "task", "")).lower() == "obb"
        self.names = self._yolo.names or {}
        self._predictor = None

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        """Detect one ordered canonical result for every input frame."""

        batch = _validated_frames(frames)
        if not batch:
            return []

        device = getattr(self, "device", "cpu")
        with timed_component_phase("detector", "preprocess", device=device):
            # Predictor construction performs Ultralytics' lazy initialization
            # and warm-up, so its first-call cost belongs to preprocessing.
            self._ensure_predictor(**self._prediction_options)
            images = [_frame_to_bgr(frame) for frame in batch]
            self._predictor.batch = ([frame.source_uri or "" for frame in batch], images, None)
            preprocessed = self._predictor.preprocess(images)
        with timed_component_phase("detector", "process", device=device):
            raw_predictions = self._predictor.inference(preprocessed)
        with timed_component_phase("detector", "postprocess", device=device):
            raw_results = self._predictor.postprocess(raw_predictions, preprocessed, images)
            if len(raw_results) != len(batch):
                raise ValueError(f"Ultralytics returned {len(raw_results)} results for {len(batch)} frames.")

            results: list[Detections] = []
            for frame, result in zip(batch, raw_results):
                rows, masks = self._extract_rows(result)
                results.append(
                    _canonical_detections(
                        frame,
                        rows,
                        masks=masks,
                        empty_is_obb=self._is_obb,
                    )
                )
            return _validate_results(batch, results, self.capabilities)

    def _ensure_predictor(
        self,
        conf: float | None = None,
        iou: float | None = None,
        classes: Any = None,
        agnostic_nms: bool | None = None,
    ) -> None:
        """Create and configure Ultralytics' private stateful predictor."""

        if self._predictor is None:
            dummy = np.zeros((32, 32, 3), dtype=np.uint8)
            predictor_options = {
                "source": dummy,
                "conf": 0.25 if conf is None else float(conf),
                "iou": 0.7 if iou is None else float(iou),
                "classes": classes,
                "agnostic_nms": False if agnostic_nms is None else bool(agnostic_nms),
                "device": self.device,
                "verbose": False,
                "save": False,
                "stream": False,
            }
            if self.imgsz is not None:
                predictor_options["imgsz"] = self.imgsz
            self._yolo.predict(**predictor_options)
            self._predictor = self._yolo.predictor

        if conf is not None:
            self._predictor.args.conf = float(conf)
        if iou is not None:
            self._predictor.args.iou = float(iou)
        self._predictor.args.classes = classes
        self._predictor.args.agnostic_nms = bool(agnostic_nms) if agnostic_nms is not None else False

    def _extract_rows(self, result: Any) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract private AABB/OBB rows and optional original-size masks."""

        if getattr(result, "obb", None) is not None:
            if len(result.obb) == 0:
                return _empty_rows(is_obb=True), None
            xywhr = _as_numpy(result.obb.xywhr)
            confidence = _as_numpy(result.obb.conf).reshape(-1, 1)
            class_ids = _as_numpy(result.obb.cls).reshape(-1, 1)
            return np.concatenate((xywhr, confidence, class_ids), axis=1), None

        if getattr(result, "boxes", None) is not None:
            if len(result.boxes) == 0:
                return _empty_rows(is_obb=self._is_obb), None
            xyxy = _as_numpy(result.boxes.xyxy)
            confidence = _as_numpy(result.boxes.conf).reshape(-1, 1)
            class_ids = _as_numpy(result.boxes.cls).reshape(-1, 1)
            rows = np.concatenate((xyxy, confidence, class_ids), axis=1)
            masks = None
            if getattr(result, "masks", None) is not None and len(result.masks) > 0:
                masks = self._extract_original_shape_masks(result)
            return rows, masks

        return _empty_rows(is_obb=self._is_obb), None

    @staticmethod
    def _extract_original_shape_masks(result: Any) -> np.ndarray:
        """Threshold masks at 0.5 after inverse-resizing to original pixels."""

        mask_data = result.masks.data
        if isinstance(mask_data, np.ndarray):
            mask_tensor = torch.from_numpy(mask_data)
        else:
            mask_tensor = mask_data.detach() if hasattr(mask_data, "detach") else torch.as_tensor(mask_data)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor[None]

        orig_shape = getattr(result.masks, "orig_shape", None)
        if orig_shape is None:
            orig_shape = result.orig_img.shape[:2]
        orig_shape = tuple(int(dimension) for dimension in orig_shape[:2])
        if tuple(mask_tensor.shape[-2:]) != orig_shape:
            mask_tensor = ops.scale_masks(mask_tensor[None].float(), orig_shape)[0]
        return (mask_tensor > 0.5).to(torch.bool).cpu().numpy()


__all__ = ("UltralyticsDetector",)
