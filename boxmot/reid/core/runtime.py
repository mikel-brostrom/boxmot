from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot.reid.backends.registry import get_backend_class
from boxmot.reid.core.crops import coerce_boxes, coerce_crops, prepare_crop_batch, resolve_image
from boxmot.reid.core.formats import ReIDFormat, resolve_reid_format
from boxmot.reid.core.preprocessing import DEFAULT_PREPROCESS, get_preprocess_fn
from boxmot.utils import WEIGHTS
from boxmot.utils.torch_utils import select_device


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
        self.preprocess_name = preprocess_name or DEFAULT_PREPROCESS
        self.format = resolve_reid_format(self.path)
        self.backend = self
        self.model = self.get_backend()

    @classmethod
    def from_backend(cls, backend: Any) -> "ReID":
        """Build a ReID runtime around an already-instantiated backend."""
        instance = cls.__new__(cls)
        instance.path = Path(getattr(backend, "weights", "") or "")
        instance.weights = instance.path
        instance.device = getattr(backend, "device", torch.device("cpu"))
        instance.half = bool(getattr(backend, "half", False))
        instance.preprocess_name = DEFAULT_PREPROCESS
        instance.format = None
        instance.backend = instance
        instance.model = backend
        return instance

    def get_backend(self):
        if hasattr(self, "_backend_model"):
            return self._backend_model

        if not isinstance(self.format, ReIDFormat):
            raise RuntimeError("Cannot select a backend for a wrapped ReID runtime")
        backend_class = get_backend_class(self.format)
        self._backend_model = backend_class(
            self.weights, self.device, self.half, preprocess=self.preprocess_name
        )
        return self._backend_model

    def preprocess(self, inputs, boxes=None, **kwargs):
        """Build the model-ready input batch (cropping + standardization)."""
        if boxes is not None:
            image = resolve_image(inputs)
            coerced = coerce_boxes(boxes)
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

        crops = coerce_crops(inputs)
        if not crops:
            empty = torch.empty(
                (0, 3, *self.model.input_shape),
                dtype=torch.float16 if self.model.half else torch.float32,
                device=self.model.device,
            )
            batch = self.model.inference_preprocess(empty)
            return {"mode": "crops", "batch": batch, "empty": True}

        batch = prepare_crop_batch(
            crops,
            input_shape=self.model.input_shape,
            device=self.model.device,
            half=self.model.half,
            preprocess_fn=get_preprocess_fn(self.preprocess_name),
            mean=self.model.mean_array,
            std=self.model.std_array,
        )
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
