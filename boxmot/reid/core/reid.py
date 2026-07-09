from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from boxmot.detectors.base import resolve_image
from boxmot.reid.backends.onnx_backend import ONNXBackend
from boxmot.reid.backends.openvino_backend import OpenVinoBackend
from boxmot.reid.backends.pytorch_backend import PyTorchBackend
from boxmot.reid.backends.tensorrt_backend import TensorRTBackend
from boxmot.reid.backends.tflite_backend import TFLiteBackend
from boxmot.reid.backends.torchscript_backend import TorchscriptBackend
from boxmot.reid.core.config import REID_EXPORT_SUFFIXES
from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS, get_preprocess_fn
from boxmot.utils import WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import select_device

OPENVINO_FILE_SUFFIXES = (".xml", ".bin")


class ReID:
    """Unified ReID runtime that also exposes overrideable public stage hooks."""

    def __init__(
        self,
        path: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        *,
        weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        device: str | torch.device = "cpu",
        half: bool = False,
        preprocess_name: str | None = None,
    ) -> None:
        model_ref = path if path is not None else weights
        if model_ref is None:
            model_ref = WEIGHTS / "osnet_x0_25_msmt17.pt"

        primary_weight = model_ref[0] if isinstance(model_ref, (list, tuple)) else model_ref
        self.path = Path(primary_weight)
        self.weights = model_ref
        self.device = device if isinstance(device, torch.device) else select_device(device)
        self.half = bool(half)
        # Honour the caller-provided preprocessing choice. ReID models in the
        # zoo (OSNet, LMBN, etc.) are trained with plain ``cv2.resize`` to the
        # input shape, so we fall back to the registry default (``"resize"``)
        # rather than letterbox-padding the crop. Hardcoding ``"resize_pad"``
        # here was a regression that silently changed embedding distributions
        # and degraded IDF1 versus the v17 baseline.
        self.preprocess_name = preprocess_name or DEFAULT_PREPROCESS
        (
            self.pt,
            self.jit,
            self.onnx,
            self.xml,
            self.engine,
            self.tflite,
        ) = self.model_type(self.path)
        self.backend = self
        self.model = self.get_backend()

    @classmethod
    def from_backend(cls, backend: Any) -> "ReID":
        """Build a ``ReID`` runtime around an already-instantiated backend.

        Useful when a tracker has already loaded a ReID backend and we want to
        reuse it (rather than reloading the weights) while still exposing the
        public ``preprocess`` / ``process`` / ``postprocess`` stage hooks.
        """
        instance = cls.__new__(cls)
        instance.path = Path(getattr(backend, "weights", "") or "")
        instance.weights = instance.path
        instance.device = getattr(backend, "device", torch.device("cpu"))
        instance.half = bool(getattr(backend, "half", False))
        instance.preprocess_name = DEFAULT_PREPROCESS
        instance.pt = instance.jit = instance.onnx = False
        instance.xml = instance.engine = instance.tflite = False
        instance.backend = instance
        instance.model = backend
        return instance

    def get_backend(self):
        if hasattr(self, "_backend_model"):
            return self._backend_model

        backend_map = (
            (self.pt, PyTorchBackend),
            (self.jit, TorchscriptBackend),
            (self.onnx, ONNXBackend),
            (self.engine, TensorRTBackend),
            (self.xml, OpenVinoBackend),
            (self.tflite, TFLiteBackend),
        )

        for enabled, backend_class in backend_map:
            if enabled:
                self._backend_model = backend_class(
                    self.weights, self.device, self.half, preprocess=self.preprocess_name
                )
                return self._backend_model

        LOGGER.error("This model framework is not supported yet!")
        raise SystemExit(1)

    def check_suffix(
        self,
        file: Path | str = "osnet_x0_25_msmt17.pt",
        suffix: str | tuple[str, ...] = (".pt",),
        msg: str = "",
    ) -> None:
        suffixes = [suffix] if isinstance(suffix, str) else list(suffix)
        files = [file] if isinstance(file, (str, Path)) else list(file)

        for candidate in files:
            file_suffix = Path(candidate).suffix.lower()
            if file_suffix and file_suffix not in suffixes:
                LOGGER.error(
                    f"File {candidate} does not have an acceptable suffix. Expected: {suffixes}{msg}"
                )

    @staticmethod
    def _matches_export_suffix(path: Path, suffix: str) -> bool:
        name = path.name.lower()
        suffix = suffix.lower()
        if suffix.startswith("."):
            return path.suffix.lower() == suffix
        return name.endswith(suffix)

    def model_type(self, path: Path) -> tuple[bool, ...]:
        suffixes = REID_EXPORT_SUFFIXES
        self.check_suffix(path, suffixes + OPENVINO_FILE_SUFFIXES)
        types = [self._matches_export_suffix(path, suffix) for suffix in suffixes]

        if path.suffix.lower() in OPENVINO_FILE_SUFFIXES:
            try:
                openvino_index = suffixes.index("_openvino_model")
                types[openvino_index] = True
            except ValueError:
                pass

        return tuple(types)

    @staticmethod
    def _coerce_boxes(boxes: Any) -> np.ndarray:
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            cols = arr.shape[1] if arr.ndim == 2 else 4
            return np.empty((0, cols), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _coerce_crops(crops: Any) -> list[np.ndarray]:
        if isinstance(crops, (str, Path)):
            return [resolve_image(crops)]

        if isinstance(crops, np.ndarray):
            if crops.ndim == 4:
                return [np.asarray(crop) for crop in crops]
            if crops.ndim == 3:
                return [crops]
            raise ValueError(f"Unsupported crop tensor shape: {crops.shape}")

        if isinstance(crops, (list, tuple)):
            return [
                resolve_image(crop) if isinstance(crop, (str, Path)) else np.asarray(crop)
                for crop in crops
            ]

        raise ValueError(f"Unsupported ReID input type: {type(crops)}")

    def _prepare_crop_batch(self, crops: list[np.ndarray]) -> torch.Tensor:
        if not crops:
            return torch.empty(
                (0, 3, *self.model.input_shape),
                dtype=torch.float32,
                device=self.model.device,
            )

        preprocess_fn = get_preprocess_fn(self.preprocess_name)

        batch = torch.empty(
            (len(crops), 3, *self.model.input_shape),
            dtype=torch.float16 if self.model.half else torch.float32,
            device=self.model.device,
        )

        for index, crop in enumerate(crops):
            if crop.size == 0:
                crop = np.zeros((*self.model.input_shape, 3), dtype=np.uint8)

            resized = preprocess_fn(crop, self.model.input_shape)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(resized).to(batch.device, dtype=batch.dtype)
            batch[index] = tensor.permute(2, 0, 1)

        batch = batch / 255.0
        batch = (batch - self.model.mean_array) / self.model.std_array
        return batch

    def preprocess(self, inputs, boxes=None, **kwargs):
        """Build the model-ready input batch (cropping + standardization)."""
        if boxes is not None:
            image = resolve_image(inputs)
            coerced = self._coerce_boxes(boxes)
            if coerced.size == 0:
                empty = torch.empty(
                    (0, 3, *self.model.input_shape),
                    dtype=torch.float16 if self.model.half else torch.float32,
                    device=self.model.device,
                )
                batch = self.model.inference_preprocess(empty)
                return {"mode": "image_boxes", "batch": batch, "empty": True}
            if not hasattr(self.model, "get_crops"):
                return {"mode": "image_boxes", "image": image, "boxes": coerced, "fallback": True}
            batch = self.model.get_crops(coerced, image)
            batch = self.model.inference_preprocess(batch)
            return {"mode": "image_boxes", "batch": batch, "empty": False}

        crops = self._coerce_crops(inputs)
        if not crops:
            empty = torch.empty(
                (0, 3, *self.model.input_shape),
                dtype=torch.float16 if self.model.half else torch.float32,
                device=self.model.device,
            )
            batch = self.model.inference_preprocess(empty)
            return {"mode": "crops", "batch": batch, "empty": True}

        batch = self._prepare_crop_batch(crops)
        batch = self.model.inference_preprocess(batch)
        return {"mode": "crops", "batch": batch, "empty": False}

    def process(self, payload, **kwargs):
        """Run the ReID model forward pass."""
        if payload.get("fallback", False):
            return {"_features": self.model.get_features(payload["boxes"], payload["image"])}
        if payload.get("empty", False):
            return None
        with torch.no_grad():
            return self.model.forward(payload["batch"])

    def postprocess(self, features, **kwargs) -> np.ndarray:
        """Move features to numpy and L2-normalize them."""
        if features is None:
            return np.empty((0, 0), dtype=np.float32)
        if isinstance(features, dict) and "_features" in features:
            return np.asarray(features["_features"], dtype=np.float32)
        if not hasattr(self.model, "inference_postprocess"):
            return np.asarray(features, dtype=np.float32)
        features = np.asarray(self.model.inference_postprocess(features), dtype=np.float32)
        if features.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return features / norms

    def __call__(self, inputs, boxes=None, **kwargs) -> np.ndarray:
        payload = self.preprocess(inputs, boxes=boxes, **kwargs)
        features = self.process(payload, boxes=boxes, **kwargs)
        return self.postprocess(features, boxes=boxes, **kwargs)


__all__ = ("ReID",)
