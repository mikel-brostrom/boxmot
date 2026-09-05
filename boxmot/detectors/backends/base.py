"""Private boundary helpers shared by built-in detector backends.

The public detector boundary is deliberately structured: implementations accept
``Sequence[Frame]`` and return ``list[Detections]``. Tensor/NumPy conversion is
kept in this private support module so no staged raw-array API leaks out of a
backend class.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

from boxmot.detectors.protocols import DetectorCapabilities
from boxmot.detectors.specs import DetectorSpec
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes

_AABB_COLUMNS = 6
_OBB_COLUMNS = 7
_PREDICTION_OPTIONS = frozenset(("agnostic_nms", "classes", "confidence", "iou"))
_CONSTRUCTION_OPTIONS = frozenset(("image_size",))


def _validate_backend_spec(
    spec: DetectorSpec,
    *,
    backend: str,
    supports_obb: bool,
) -> dict[str, Any]:
    """Validate one built-in backend spec and return its normalized options."""

    if not isinstance(spec, DetectorSpec):
        raise TypeError(f"spec must be a DetectorSpec, not {type(spec).__name__}.")
    if spec.backend != backend:
        raise ValueError(f"{backend!r} backend cannot construct detector spec {spec.backend!r}.")
    if spec.artifact is None:
        raise ValueError(f"Detector backend {backend!r} requires an artifact.")
    if spec.preprocessing != "default":
        raise ValueError(f"Detector backend {backend!r} requires preprocessing='default'.")
    if spec.geometry_mode == "obb" and not supports_obb:
        raise ValueError(f"Detector backend {backend!r} does not support OBB geometry.")

    values = spec.option_values()
    unsupported = set(values) - _PREDICTION_OPTIONS - _CONSTRUCTION_OPTIONS
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"Unsupported {backend!r} detector options: {names}.")
    return values


def _validated_frames(frames: Sequence[Frame]) -> tuple[Frame, ...]:
    """Validate the canonical batch without accepting raw arrays or iterators."""

    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise TypeError("frames must be a Sequence[Frame].")
    batch = tuple(frames)
    for index, frame in enumerate(batch):
        if not isinstance(frame, Frame):
            raise TypeError(f"frames[{index}] must be a Frame, not {type(frame).__name__}.")
        frame.validate()
    return batch


def _frame_to_bgr(frame: Frame) -> np.ndarray:
    """Explicitly convert a canonical CPU RGB frame to private OpenCV BGR."""

    rgb = frame.image.permute(1, 2, 0)
    return rgb.flip(-1).contiguous().numpy()


def _as_numpy(values: Any) -> np.ndarray:
    """Move a tensor-like backend value to CPU as a private float32 array."""

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def _empty_rows(*, is_obb: bool) -> np.ndarray:
    return np.empty((0, _OBB_COLUMNS if is_obb else _AABB_COLUMNS), dtype=np.float32)


def _filter_rows(
    rows: Any,
    *,
    confidence: float | None,
    classes: int | Iterable[int] | None,
) -> np.ndarray:
    """Apply configured score and class filters to private backend rows."""

    filtered = np.asarray(rows, dtype=np.float32)
    if filtered.size == 0:
        columns = filtered.shape[1] if filtered.ndim == 2 else _AABB_COLUMNS
        return np.empty((0, columns), dtype=np.float32)
    if filtered.ndim == 1:
        filtered = filtered.reshape(1, -1)
    if filtered.ndim != 2 or filtered.shape[1] not in (_AABB_COLUMNS, _OBB_COLUMNS):
        raise ValueError(
            "Detector rows must have shape [N, 6] for AABB or [N, 7] for OBB; "
            f"received {tuple(filtered.shape)}."
        )

    keep = np.ones(len(filtered), dtype=bool)
    if confidence is not None:
        keep &= filtered[:, -2] >= float(confidence)
    if classes is not None:
        if isinstance(classes, (int, np.integer)):
            class_ids = np.asarray([classes], dtype=np.int64)
        else:
            class_ids = np.asarray(list(classes), dtype=np.int64)
        keep &= np.isin(filtered[:, -1].astype(np.int64), class_ids)
    return np.ascontiguousarray(filtered[keep], dtype=np.float32)


def _canonical_detections(
    frame: Frame,
    rows: Any,
    *,
    masks: Any | None = None,
    empty_is_obb: bool = False,
) -> Detections:
    """Convert private backend rows to the strict CPU Torch contract."""

    raw_rows = np.asarray(rows)
    if raw_rows.size == 0:
        columns = raw_rows.shape[1] if raw_rows.ndim == 2 else (_OBB_COLUMNS if empty_is_obb else _AABB_COLUMNS)
        raw_rows = np.empty((0, columns), dtype=np.float32)
    elif raw_rows.ndim == 1:
        raw_rows = raw_rows.reshape(1, -1)
    if raw_rows.ndim != 2 or raw_rows.shape[1] not in (_AABB_COLUMNS, _OBB_COLUMNS):
        raise ValueError(
            "Detector backend returned rows with invalid shape "
            f"{tuple(raw_rows.shape)}; expected [N, 6] or [N, 7]."
        )

    parsed_class_ids: list[int] = []
    for value in raw_rows[:, -1].tolist():
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Detector backend class IDs must be finite non-negative integers.")
        try:
            numeric = float(value)
            class_id = int(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("Detector backend class IDs must be finite non-negative integers.") from exc
        if (
            not np.isfinite(numeric)
            or numeric != class_id
            or class_id < 0
            or class_id > np.iinfo(np.int64).max
        ):
            raise ValueError("Detector backend class IDs must be finite non-negative integers.")
        parsed_class_ids.append(class_id)

    tensor = torch.as_tensor(raw_rows, dtype=torch.float32).contiguous()
    geometry = (
        Boxes(tensor[:, :4].contiguous())
        if tensor.shape[1] == _AABB_COLUMNS
        else OrientedBoxes(tensor[:, :5].contiguous())
    )
    canonical_masks = None
    if masks is not None:
        canonical_masks = MaskBatch(torch.as_tensor(np.asarray(masks), dtype=torch.bool).contiguous())
        if canonical_masks.image_size != frame.image_size:
            raise ValueError(
                "Detector masks must use original-frame coordinates "
                f"{frame.image_size}, got {canonical_masks.image_size}."
            )
    return Detections(
        geometry=geometry,
        scores=tensor[:, -2].contiguous(),
        class_ids=torch.tensor(parsed_class_ids, dtype=torch.int64),
        sample_id=frame.sample_id,
        masks=canonical_masks,
    )


def _validate_results(
    frames: Sequence[Frame],
    results: list[Detections],
    capabilities: DetectorCapabilities,
) -> list[Detections]:
    """Validate result cardinality, identity, and declared geometry."""

    if not isinstance(results, list):
        raise TypeError(f"Detector predict() must return list[Detections], not {type(results).__name__}.")
    if len(results) != len(frames):
        raise ValueError(f"Detector returned {len(results)} results for {len(frames)} frames.")
    for index, (frame, result) in enumerate(zip(frames, results)):
        if not isinstance(result, Detections):
            raise TypeError(f"Detector result {index} must be Detections, not {type(result).__name__}.")
        result.validate()
        if result.sample_id != frame.sample_id:
            raise ValueError(
                f"Detector result {index} sample_id {result.sample_id!r} does not match frame {frame.sample_id!r}."
            )
        if result.is_obb and not capabilities.supports_obb:
            raise ValueError("Detector returned OBB geometry contrary to its configured geometry_mode.")
        if not result.is_obb and not capabilities.supports_aabb:
            raise ValueError("Detector returned AABB geometry contrary to its configured geometry_mode.")
        if result.masks is not None and result.masks.image_size != frame.image_size:
            raise ValueError(
                f"Detector result {index} masks use {result.masks.image_size}, expected {frame.image_size}."
            )
    return results


__all__: tuple[str, ...] = ()
