from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

AABB_COLUMNS = 6
OBB_COLUMNS = 7
DETECTION_COLUMNS = (AABB_COLUMNS, OBB_COLUMNS)


def as_detection_array(values: Any, *, empty_columns: int = AABB_COLUMNS) -> np.ndarray:
    """Return a validated ``float32`` AABB or OBB detection matrix."""
    detections = np.asarray(values, dtype=np.float32)
    if detections.size == 0:
        columns = detections.shape[1] if detections.ndim == 2 else empty_columns
        detections = np.empty((0, columns), dtype=np.float32)
    elif detections.ndim == 1:
        detections = detections.reshape(1, -1)

    if detections.ndim != 2 or detections.shape[1] not in DETECTION_COLUMNS:
        raise ValueError(f"Detections must have shape (N, 6) for AABB or (N, 7) for OBB; received {detections.shape}.")
    return detections


def empty_detections(*, is_obb: bool = False) -> np.ndarray:
    """Create an empty detection matrix with the canonical schema."""
    columns = OBB_COLUMNS if is_obb else AABB_COLUMNS
    return np.empty((0, columns), dtype=np.float32)


def as_numpy(values: Any) -> np.ndarray:
    """Move a tensor-like value to CPU and return a ``float32`` array."""
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=np.float32)


def filter_detections(
    detections: Any,
    *,
    confidence: float | None,
    classes: int | Iterable[int] | None,
) -> np.ndarray:
    """Apply the shared confidence and class filters to AABB or OBB rows."""
    filtered = as_detection_array(detections)
    if len(filtered) == 0:
        return filtered

    keep = np.ones(len(filtered), dtype=bool)
    if confidence is not None:
        keep &= filtered[:, -2] >= float(confidence)
    if classes is not None:
        if isinstance(classes, (int, np.integer)):
            class_ids = np.asarray([classes], dtype=np.int64)
        else:
            class_ids = np.asarray(list(classes), dtype=np.int64)
        keep &= np.isin(filtered[:, -1].astype(np.int64), class_ids)
    return filtered[keep]


def ensure_image_batch(images: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    """Normalize one image or an image sequence to a validated image list."""
    if isinstance(images, np.ndarray):
        if images.ndim == 3:
            batch = [images]
        elif images.ndim == 4:
            batch = list(images)
        else:
            raise ValueError(f"Images must have 3 or 4 dimensions; received {images.shape}.")
    elif isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
        batch = list(images)
    else:
        raise TypeError(f"Images must be a numpy array or a sequence of arrays, not {type(images).__name__}.")

    if not batch:
        raise ValueError("At least one image is required for detector inference.")
    for index, image in enumerate(batch):
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            shape = getattr(image, "shape", None)
            raise ValueError(f"Image {index} must be a 3D numpy array; received {shape}.")
    return batch


@dataclass
class Detections:
    """One image's detections in the canonical BoxMOT schema.

    Axis-aligned rows have shape ``(N, 6)`` and contain
    ``[x1, y1, x2, y2, confidence, class]``. Oriented rows have shape
    ``(N, 7)`` and contain ``[cx, cy, width, height, angle, confidence, class]``.
    Angles are expressed in radians.
    """

    dets: np.ndarray
    orig_img: np.ndarray | None
    path: str | Path = ""
    names: Mapping[int, str] = field(default_factory=dict)
    masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.dets = as_detection_array(self.dets)
        self.path = str(self.path)
        self.names = dict(self.names)
        if self.masks is not None:
            self.masks = np.asarray(self.masks)
            if self.masks.ndim < 1 or len(self.masks) != len(self.dets):
                raise ValueError(
                    "Masks must have one entry per detection; "
                    f"received {len(self.dets)} detections and masks with shape {self.masks.shape}."
                )

    @classmethod
    def empty(
        cls,
        orig_img: np.ndarray | None,
        *,
        is_obb: bool = False,
        path: str | Path = "",
        names: Mapping[int, str] | None = None,
    ) -> Detections:
        """Build an empty result while preserving its AABB or OBB schema."""
        return cls(
            dets=empty_detections(is_obb=is_obb),
            orig_img=orig_img,
            path=path,
            names={} if names is None else names,
        )

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if copy is None:
            return np.asarray(self.dets, dtype=dtype)
        return np.array(self.dets, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        return int(self.dets.shape[0])

    def __getitem__(self, item):
        return self.dets[item]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.dets.shape

    @property
    def is_obb(self) -> bool:
        return self.dets.shape[1] == OBB_COLUMNS

    @property
    def boxes(self) -> np.ndarray:
        """Return native box geometry: ``xyxy`` for AABB or ``xywha`` for OBB."""
        return self.dets[:, :5] if self.is_obb else self.dets[:, :4]

    @property
    def xyxy(self) -> np.ndarray:
        """Return axis-aligned boxes, enclosing each oriented box when needed."""
        if not self.is_obb:
            return self.dets[:, :4]
        if len(self) == 0:
            return np.empty((0, 4), dtype=np.float32)

        cx, cy, width, height, angle = self.dets[:, :5].T.astype(np.float64, copy=False)
        cos_angle = np.abs(np.cos(angle))
        sin_angle = np.abs(np.sin(angle))
        half_width = 0.5 * ((width * cos_angle) + (height * sin_angle))
        half_height = 0.5 * ((width * sin_angle) + (height * cos_angle))
        return np.column_stack((cx - half_width, cy - half_height, cx + half_width, cy + half_height)).astype(
            np.float32
        )

    @property
    def xywha(self) -> np.ndarray:
        if self.is_obb:
            return self.dets[:, :5]
        return np.empty((len(self), 0), dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        return self.dets[:, -2]

    @property
    def classes(self) -> np.ndarray:
        return self.dets[:, -1].astype(int)

    @property
    def cls(self) -> np.ndarray:
        return self.classes


def resolve_image(image: np.ndarray | str | Path) -> np.ndarray:
    """Resolve an image input to a numpy array in OpenCV BGR format."""
    if isinstance(image, (str, Path)):
        resolved = cv2.imread(str(image))
        if resolved is None:
            raise FileNotFoundError(f"Could not load image from {image}")
        return resolved
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def load_weights(path: str | Path) -> Any:
    """Load a PyTorch checkpoint from an explicit local path."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Weights file not found: {resolved}")
    return torch.load(str(resolved), map_location="cpu")


class BaseDetectorBackend:
    """Staged detector backend contract implemented by concrete integrations."""

    names: Mapping[int, str] = {}
    pt = False
    stride = 32
    fp16 = False
    triton = False

    def preprocess(self, images: list[np.ndarray], **kwargs: Any) -> Any:
        raise NotImplementedError

    def process(self, preprocessed: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def postprocess(self, predictions: Any, **kwargs: Any) -> list[Detections]:
        raise NotImplementedError

    def __call__(
        self,
        images: np.ndarray | Sequence[np.ndarray],
        *,
        conf: float = 0.25,
        iou: float = 0.7,
        classes: int | Iterable[int] | None = None,
        agnostic_nms: bool = False,
    ) -> list[Detections]:
        batch = ensure_image_batch(images)
        preprocessed = self.preprocess(batch)
        predictions = self.process(preprocessed)
        results = self.postprocess(
            predictions,
            conf=float(conf),
            iou=float(iou),
            classes=classes,
            agnostic_nms=bool(agnostic_nms),
        )
        if not isinstance(results, list) or not all(isinstance(result, Detections) for result in results):
            raise TypeError("Detector postprocess must return a list of Detections objects.")
        if len(results) != len(batch):
            raise ValueError(f"Detector returned {len(results)} results for a batch of {len(batch)} images.")
        return results


__all__ = (
    "AABB_COLUMNS",
    "BaseDetectorBackend",
    "DETECTION_COLUMNS",
    "Detections",
    "OBB_COLUMNS",
    "as_detection_array",
    "as_numpy",
    "empty_detections",
    "ensure_image_batch",
    "filter_detections",
    "load_weights",
    "resolve_image",
)
