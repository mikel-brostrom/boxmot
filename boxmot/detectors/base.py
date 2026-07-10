from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
import torch


@dataclass
class Detections:
    """
    Unified detection result returned by all BoxMOT detectors.

    AABB format — dets shape (N, 6): [x1, y1, x2, y2, conf, cls]
    OBB  format — dets shape (N, 7): [cx, cy, w, h, angle, conf, cls]

    Fields:
        dets:     numpy array of shape (N, 6) or (N, 7).
                  Empty (0, 6) or (0, 7) when no detections.
        orig_img: Original BGR image as numpy array.
        path:     Source image/video path (empty string when unavailable).
        names:    Class name mapping {class_id: name}.
        masks:    Optional segmentation masks, shape (N, H, W) uint8.
                  None when the model does not produce masks.
    """

    dets: np.ndarray
    orig_img: np.ndarray
    path: str = ""
    names: dict = field(default_factory=dict)
    masks: np.ndarray | None = None

    def __post_init__(self) -> None:
        dets = np.asarray(self.dets, dtype=np.float32)
        if dets.ndim == 1 and dets.size > 0:
            dets = dets.reshape(1, -1)
        elif dets.size == 0:
            cols = dets.shape[1] if dets.ndim == 2 else 6
            dets = dets.reshape(0, cols)
        self.dets = dets

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
        return self.dets.ndim == 2 and self.dets.shape[1] == 7

    @property
    def boxes(self) -> np.ndarray:
        return self.dets[:, :4]

    @property
    def xyxy(self) -> np.ndarray:
        return self.dets[:, :4]

    @property
    def xywha(self) -> np.ndarray:
        if self.is_obb:
            return self.dets[:, :5]
        return np.empty((len(self), 0), dtype=np.float32)

    @property
    def conf(self) -> np.ndarray:
        return self.dets[:, 5] if self.is_obb else self.dets[:, 4]

    @property
    def classes(self) -> np.ndarray:
        return (self.dets[:, 6] if self.is_obb else self.dets[:, 5]).astype(int)

    @property
    def cls(self) -> np.ndarray:
        return self.classes


def resolve_image(image: Union[np.ndarray, str]) -> np.ndarray:
    """Resolve an image input to a numpy array in cv2 BGR format."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image}")
        return img
    if isinstance(image, np.ndarray):
        return image
    raise ValueError(f"Unsupported image type: {type(image)}")


def load_weights(path: str) -> Any:
    """Generic weight loader using torch.load."""
    if isinstance(path, str) and not Path(path).exists():
        raise FileNotFoundError(f"Weights file not found: {path}")
    return torch.load(path, map_location="cpu")


class BaseDetectorBackend:
    """Abstract detector backend contract implemented by concrete detector integrations."""

    def preprocess(self, images, **kwargs):
        raise NotImplementedError()

    def process(self, preprocessed, **kwargs):
        raise NotImplementedError()

    def postprocess(self, detections, **kwargs):
        raise NotImplementedError()

    def __call__(self, images, **kwargs) -> Detections:
        preprocessed = self.preprocess(images)
        detections = self.process(preprocessed)
        return self.postprocess(detections, **kwargs)


Detector = BaseDetectorBackend

__all__ = ("BaseDetectorBackend", "Detector", "Detections", "load_weights", "resolve_image")
