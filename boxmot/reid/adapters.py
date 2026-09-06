"""Canonical appearance encoder adapters around existing ReID runtimes."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from math import ceil, sqrt
from typing import Any

import numpy as np
import torch

from boxmot.components.timing import timed_component_phase
from boxmot.reid.protocols import EncoderRequirements
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes
from boxmot.utils.torch_utils import canonical_torch_device

_DEFAULT_INFERENCE_BATCH_SIZE = 64


def _frame_to_bgr(frame: Frame) -> np.ndarray:
    return frame.image.permute(1, 2, 0).flip(-1).contiguous().numpy()


def _crop_geometry(detections: Detections) -> np.ndarray:
    geometry = detections.geometry
    if not isinstance(geometry, (Boxes, OrientedBoxes)):
        raise TypeError(f"Unsupported detection geometry: {type(geometry).__name__}.")
    return geometry.values.contiguous().numpy()


def _runtime_input_shape(runtime: Any) -> tuple[int, int]:
    candidates = (runtime, getattr(runtime, "model", None))
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "input_shape", None)
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            height, width = value[-2:]
            if (
                isinstance(height, int)
                and not isinstance(height, bool)
                and isinstance(width, int)
                and not isinstance(width, bool)
                and height > 0
                and width > 0
            ):
                return height, width
    return 256, 128


def _extract_crops(
    frame: Frame,
    detections: Detections,
    *,
    input_shape: tuple[int, int],
) -> list[np.ndarray]:
    from boxmot.reid.core.crops import extract_crops

    image = _frame_to_bgr(frame)
    return extract_crops(_crop_geometry(detections), image, input_shape)


def _pack_crops(crops: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pack independent BGR crops into one image for one backend batch call."""
    if not crops:
        raise ValueError("At least one crop is required to construct a ReID batch.")
    normalized: list[np.ndarray] = []
    for crop in crops:
        array = np.asarray(crop)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"ReID crop must have shape [H, W, 3], got {array.shape}.")
        if array.shape[0] == 0 or array.shape[1] == 0:
            array = np.zeros((1, 1, 3), dtype=np.uint8)
        normalized.append(np.ascontiguousarray(array, dtype=np.uint8))

    columns = max(1, ceil(sqrt(len(normalized))))
    rows = ceil(len(normalized) / columns)
    cell_height = max(crop.shape[0] for crop in normalized)
    cell_width = max(crop.shape[1] for crop in normalized)
    mosaic = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    boxes = np.empty((len(normalized), 4), dtype=np.float32)
    for index, crop in enumerate(normalized):
        y1 = (index // columns) * cell_height
        x1 = (index % columns) * cell_width
        height, width = crop.shape[:2]
        mosaic[y1 : y1 + height, x1 : x1 + width] = crop
        boxes[index] = (x1, y1, x1 + width, y1 + height)
    return boxes, mosaic


def _canonical_features(features: Any, *, expected_rows: int) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        tensor = features.detach().to(device="cpu", dtype=torch.float32)
    else:
        tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if tensor.ndim == 1 and expected_rows == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != expected_rows:
        raise ValueError(
            f"ReID backend returned embeddings with shape {tuple(tensor.shape)} for {expected_rows} detections."
        )
    if tensor.shape[1] <= 0:
        raise ValueError("ReID backend returned embeddings with no feature columns.")
    if tensor.numel() and not bool(torch.isfinite(tensor).all()):
        raise ValueError("ReID backend returned non-finite embeddings.")
    norms = torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
    if norms.numel() and bool((norms <= 1e-12).any()):
        raise ValueError("ReID backend returned a zero-norm embedding.")
    tensor = tensor / norms
    return tensor.contiguous()


def _declared_embedding_dim(runtime: Any, explicit: object) -> int | None:
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit <= 0:
            raise ValueError("embedding_dim must be a positive integer.")
        return explicit
    candidates = (
        runtime,
        getattr(runtime, "model", None),
        getattr(getattr(runtime, "model", None), "model", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("embedding_dim", "feature_dim", "num_features"):
            try:
                value = getattr(candidate, attribute, None)
            except (AttributeError, RuntimeError):
                continue
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


class RuntimeAppearanceEncoder:
    """Adapt a runtime exposing ``get_features(boxes, image)`` to batches."""

    def __init__(self, spec: ReIDEncoderSpec, runtime: Any) -> None:
        values = spec.option_values()
        explicit_dim = values.pop("embedding_dim", None)
        explicit_shape = values.pop("image_size", None)
        batch_size = values.pop("batch_size", _DEFAULT_INFERENCE_BATCH_SIZE)
        if values:
            names = ", ".join(sorted(values))
            raise ValueError(f"Unsupported {spec.backend!r} ReID options: {names}.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        self.spec = spec
        self._runtime = runtime
        self._device = getattr(runtime, "device", spec.device)
        if explicit_shape is None:
            self._input_shape = _runtime_input_shape(runtime)
        else:
            if (
                not isinstance(explicit_shape, tuple)
                or len(explicit_shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in explicit_shape)
            ):
                raise ValueError("image_size must be a two-integer (height, width) tuple.")
            self._input_shape = explicit_shape
        self._embedding_dim = _declared_embedding_dim(runtime, explicit_dim)
        self._batch_size = batch_size
        self.requirements = EncoderRequirements()

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            raise RuntimeError(
                "The ReID backend does not declare its embedding dimension; "
                "set the 'embedding_dim' option in ReIDEncoderSpec."
            )
        return self._embedding_dim

    def encode(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
    ) -> list[torch.Tensor]:
        frame_batch = list(frames)
        detection_batch = list(detections)
        counts: list[int] = []
        crops: list[np.ndarray] = []
        with timed_component_phase("reid", "preprocess", device=self._device):
            if len(frame_batch) != len(detection_batch):
                raise ValueError(
                    f"frames and detections must be aligned; received {len(frame_batch)} and {len(detection_batch)}."
                )

            for frame, frame_detections in zip(frame_batch, detection_batch):
                if not isinstance(frame, Frame):
                    raise TypeError(f"frames entries must be Frame objects, not {type(frame).__name__}.")
                if not isinstance(frame_detections, Detections):
                    raise TypeError(
                        f"detections entries must be Detections objects, not {type(frame_detections).__name__}."
                    )
                if frame.sample_id != frame_detections.sample_id:
                    raise ValueError(
                        f"Frame sample_id {frame.sample_id!r} does not match detections "
                        f"sample_id {frame_detections.sample_id!r}."
                    )
                count = len(frame_detections)
                counts.append(count)
                if count:
                    frame_crops = _extract_crops(
                        frame,
                        frame_detections,
                        input_shape=self._input_shape,
                    )
                    if len(frame_crops) != count:
                        raise RuntimeError(f"Extracted {len(frame_crops)} ReID crops for {count} detections.")
                    crops.extend(frame_crops)

        total = sum(counts)
        if total == 0:
            if not counts:
                return []
            width = self.embedding_dim
            return [torch.empty((0, width), dtype=torch.float32) for _ in counts]

        encoded: list[torch.Tensor] = []
        width = self._embedding_dim
        staged_runtime = all(
            callable(getattr(self._runtime, name, None))
            for name in ("get_crops", "inference_preprocess", "forward", "inference_postprocess")
        )
        for start in range(0, total, self._batch_size):
            with timed_component_phase("reid", "preprocess", device=self._device):
                crop_boxes, mosaic = _pack_crops(crops[start : start + self._batch_size])
                if staged_runtime:
                    payload = self._runtime.get_crops(crop_boxes, mosaic)
                    payload = self._runtime.inference_preprocess(payload)

            if staged_runtime:
                with timed_component_phase("reid", "process", device=self._device):
                    with torch.inference_mode():
                        raw_features = self._runtime.forward(payload)
                with timed_component_phase("reid", "postprocess", device=self._device):
                    raw_features = self._runtime.inference_postprocess(raw_features)
                    part = _canonical_features(raw_features, expected_rows=len(crop_boxes))
            else:
                # Third-party runtimes may expose only the established
                # get_features() operation. Its opaque work is inference;
                # canonical conversion and normalization remain postprocess.
                with timed_component_phase("reid", "process", device=self._device):
                    raw_features = self._runtime.get_features(crop_boxes, mosaic)
                with timed_component_phase("reid", "postprocess", device=self._device):
                    part = _canonical_features(raw_features, expected_rows=len(crop_boxes))

            with timed_component_phase("reid", "postprocess", device=self._device):
                part_width = int(part.shape[1])
                if width is None:
                    width = part_width
                elif part_width != width:
                    raise ValueError(f"ReID embedding width changed from {width} to {part_width}.")
                encoded.append(part)

        assert width is not None
        self._embedding_dim = width
        with timed_component_phase("reid", "postprocess", device=self._device):
            tensor = torch.cat(encoded, dim=0).contiguous()
            return [part.contiguous() for part in tensor.split(counts)]


def create_python_reid_encoder(spec: ReIDEncoderSpec) -> RuntimeAppearanceEncoder:
    """Build an encoder using one of BoxMOT's Python ReID backends."""
    if spec.artifact is None:
        raise ValueError(f"ReID backend {spec.backend!r} requires an artifact.")
    if spec.precision == "bf16":
        raise ValueError("Built-in ReID backends do not currently support bf16 inference.")
    runtime_class = getattr(import_module("boxmot.reid.core.runtime"), "ReID")
    preprocessing = None if spec.preprocessing == "default" else spec.preprocessing
    runtime = runtime_class(
        weights=spec.artifact,
        device=canonical_torch_device(spec.device),
        half=spec.precision == "fp16",
        preprocess_name=preprocessing,
    )
    resolved_backend = getattr(getattr(runtime, "format", None), "id", None)
    if resolved_backend != spec.backend:
        raise ValueError(
            f"Artifact {spec.artifact!r} resolves to ReID backend {resolved_backend!r}, "
            f"not requested backend {spec.backend!r}."
        )
    return RuntimeAppearanceEncoder(spec, runtime.model)


def create_native_reid_encoder(spec: ReIDEncoderSpec) -> RuntimeAppearanceEncoder:
    """Build an encoder using the native ONNX ReID C API."""
    if spec.artifact is None:
        raise ValueError("ReID backend 'native' requires an artifact.")
    if spec.device != "cpu" or spec.precision != "fp32":
        raise ValueError("Native ReID currently requires device='cpu' and precision='fp32'.")
    runtime_class = getattr(import_module("boxmot.reid.backends.native"), "CppOnnxReID")
    preprocessing = None if spec.preprocessing == "default" else spec.preprocessing
    runtime = runtime_class(spec.artifact, preprocess_name=preprocessing)
    return RuntimeAppearanceEncoder(spec, runtime)


__all__ = (
    "RuntimeAppearanceEncoder",
    "create_native_reid_encoder",
    "create_python_reid_encoder",
)
