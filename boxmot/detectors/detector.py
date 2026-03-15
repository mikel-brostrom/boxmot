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
    """
    dets: np.ndarray
    orig_img: np.ndarray
    path: str = ""
    names: dict = field(default_factory=dict)

    @property
    def is_obb(self) -> bool:
        return self.dets.ndim == 2 and self.dets.shape[1] == 7

    @property
    def boxes(self) -> np.ndarray:
        return self.dets[:, :4]

    @property
    def conf(self) -> np.ndarray:
        return self.dets[:, 5] if self.is_obb else self.dets[:, 4]

    @property
    def classes(self) -> np.ndarray:
        return (self.dets[:, 6] if self.is_obb else self.dets[:, 5]).astype(int)


def resolve_image(image: Union[np.ndarray, str]) -> np.ndarray:
    """Resolves an image input to a numpy array (cv2 BGR format)."""
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
    return torch.load(path, map_location='cpu')


class Detector:
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
