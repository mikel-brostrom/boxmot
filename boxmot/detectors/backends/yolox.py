# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Canonical YOLOX detector backend."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from yolox.exp import get_exp
from yolox.utils import postprocess as yolox_postprocess
from yolox.utils.model_utils import fuse_model

from boxmot.components.timing import timed_component_phase
from boxmot.detectors._capabilities import capabilities_from_spec
from boxmot.detectors.backends.base import (
    _canonical_detections,
    _empty_rows,
    _frame_to_bgr,
    _validate_backend_spec,
    _validate_results,
    _validated_frames,
)
from boxmot.detectors.config import load_detector_artifact_profile
from boxmot.detectors.specs import DetectorSpec
from boxmot.resources.paths import resolve_model_path
from boxmot.structures import Detections, Frame
from boxmot.utils import logger as LOGGER
from boxmot.utils.torch_utils import canonical_torch_device

YOLOX_MODELS = ("yolox_n", "yolox_s", "yolox_m", "yolox_l", "yolox_x")


def _coerce_torch_dtype(dtype: Any, fallback: torch.Tensor) -> torch.dtype:
    """Map YOLOX dtype strings to real torch dtypes for MPS."""

    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        lowered = dtype.lower()
        if "bfloat16" in lowered:
            return torch.bfloat16
        if "float16" in lowered or "half" in lowered:
            return torch.float16
    return fallback.dtype if isinstance(fallback, torch.Tensor) else torch.float32


def _patch_yolox_head_decode_outputs_for_mps() -> None:
    """Patch upstream YOLOX decoding to avoid dtype-string MPS transfers."""

    try:
        from yolox.models.yolo_head import YOLOXHead
        from yolox.utils import meshgrid
    except Exception:
        return

    if getattr(YOLOXHead, "_boxmot_mps_patched", False):
        return

    def decode_outputs(self, outputs, dtype):
        dtype = _coerce_torch_dtype(dtype, outputs)
        device = outputs.device
        grids = []
        strides = []
        for (height, width), stride in zip(self.hw, self.strides):
            y_coordinates, x_coordinates = meshgrid(
                [
                    torch.arange(height, device=device),
                    torch.arange(width, device=device),
                ]
            )
            grid = torch.stack((x_coordinates, y_coordinates), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, device=device, dtype=grid.dtype))

        grids = torch.cat(grids, dim=1).to(device=device, dtype=dtype)
        strides = torch.cat(strides, dim=1).to(device=device, dtype=dtype)
        outputs = outputs.clone()
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    YOLOXHead.decode_outputs = decode_outputs
    YOLOXHead._boxmot_mps_patched = True


_patch_yolox_head_decode_outputs_for_mps()


class YoloXDetector:
    """YOLOX implementation of ``predict(Sequence[Frame])``."""

    names = {0: "person"}

    def __init__(self, spec: DetectorSpec) -> None:
        values = _validate_backend_spec(spec, backend="yolox", supports_obb=False)
        model_path = resolve_model_path(spec.artifact)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLOX detector artifact does not exist: {model_path}. "
                "Resolve downloads before constructing the detector."
            )

        raw_size = values.get("image_size") or 640
        image_size = list(raw_size) if isinstance(raw_size, (list, tuple)) else [raw_size]
        if len(image_size) == 1:
            image_size *= 2
        if len(image_size) != 2:
            raise ValueError("YOLOX image_size must contain one or two dimensions.")

        self.spec = spec
        self.capabilities = capabilities_from_spec(spec)
        self.device = canonical_torch_device(spec.device)
        self.imgsz = [int(image_size[0]), int(image_size[1])]
        self._prediction_options = {
            "conf": float(values.get("confidence", 0.25)),
            "iou": float(values.get("iou", 0.7)),
            "classes": values.get("classes"),
            "agnostic_nms": bool(values.get("agnostic_nms", False)),
        }

        detector_config = load_detector_artifact_profile(model_path)
        configured_names = detector_config.get("classes", {})
        self.num_classes = len(configured_names) if configured_names else 1
        self.names = dict(configured_names) if configured_names else {0: "person"}
        model_type = self._get_model_type(YOLOX_MODELS, model_path)
        experiment = get_exp(None, "yolox_nano" if model_type == "yolox_n" else model_type)
        experiment.num_classes = self.num_classes

        LOGGER.info(f"Loading {model_type} with {model_path}")
        checkpoint = torch.load(str(model_path), map_location=torch.device("cpu"))
        self.model = experiment.get_model()
        self.model.eval()
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model = fuse_model(self.model)

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        """Detect one ordered canonical result for every input frame."""

        batch = _validated_frames(frames)
        if not batch:
            return []
        with timed_component_phase("detector", "preprocess", device=self.device):
            images = [_frame_to_bgr(frame) for frame in batch]
            preprocessed, resize_ratios = self._preprocess_images(images)
        with timed_component_phase("detector", "process", device=self.device):
            with torch.inference_mode():
                raw_predictions = self.model(preprocessed)
        with timed_component_phase("detector", "postprocess", device=self.device):
            results = self._decode(batch, raw_predictions, resize_ratios)
            return _validate_results(batch, results, self.capabilities)

    @staticmethod
    def _get_model_type(model_names: Sequence[str], weight_path: str | Path) -> str:
        weight_name = Path(str(weight_path)).name.lower()
        return next((name for name in model_names if name.lower() in weight_name), "yolox_s")

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        input_size: Sequence[int],
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> tuple[np.ndarray, float]:
        """Apply the private ByteTrack-compatible YOLOX letterbox transform."""

        if image.ndim == 3:
            padded = np.full((input_size[0], input_size[1], 3), 114.0, dtype=np.float32)
        else:
            padded = np.full(input_size, 114.0, dtype=np.float32)
        ratio = min(input_size[0] / image.shape[0], input_size[1] / image.shape[1])
        resized = cv2.resize(
            image,
            (int(image.shape[1] * ratio), int(image.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded[: int(image.shape[0] * ratio), : int(image.shape[1] * ratio)] = resized
        padded = padded[:, :, ::-1] / 255.0
        padded = (padded - mean) / std
        return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio

    def _preprocess_images(self, images: Sequence[np.ndarray]) -> tuple[torch.Tensor, list[float]]:
        tensors: list[torch.Tensor] = []
        ratios: list[float] = []
        for image in images:
            transformed, ratio = self._letterbox(image, input_size=self.imgsz)
            tensors.append(torch.from_numpy(transformed).unsqueeze(0).to(self.device))
            ratios.append(ratio)
        return torch.vstack(tensors), ratios

    def _decode(
        self,
        frames: Sequence[Frame],
        predictions: Any,
        resize_ratios: Sequence[float],
    ) -> list[Detections]:
        results: list[Detections] = []
        for frame, prediction, ratio in zip(frames, predictions, resize_ratios):
            # YOLOX postprocessing converts center-size boxes to corners with
            # an in-place assignment. Model outputs created by
            # ``torch.inference_mode`` cannot be mutated after that context
            # exits, so materialize a regular tensor before handing it to the
            # upstream implementation. This is required by recent PyTorch
            # releases and applies equally to CPU, CUDA, and MPS tensors.
            prediction = prediction.clone()
            filtered = yolox_postprocess(
                prediction.unsqueeze(0),
                self.num_classes,
                conf_thre=self._prediction_options["conf"],
                nms_thre=self._prediction_options["iou"],
                class_agnostic=self._prediction_options["agnostic_nms"],
            )[0]
            if filtered is None:
                rows = _empty_rows(is_obb=False)
            else:
                filtered = filtered.clone()
                filtered[:, :4] /= ratio
                filtered[:, 4] *= filtered[:, 5]
                filtered = filtered[:, [0, 1, 2, 3, 4, 6]]
                classes = self._prediction_options["classes"]
                if classes is not None:
                    allowed = torch.as_tensor(classes, dtype=filtered.dtype, device=filtered.device).reshape(-1)
                    filtered = filtered[torch.isin(filtered[:, -1], allowed)]
                rows = filtered.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            results.append(_canonical_detections(frame, rows))
        return results


__all__ = ("YoloXDetector",)
