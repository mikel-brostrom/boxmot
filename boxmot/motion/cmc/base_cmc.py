# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import cv2
import numpy as np


Scale = Union[float, Tuple[int, int], None]


class BaseCMC(ABC):
    """
    Base class for camera motion compensation (CMC) modules.

    Contract:
      - `apply(img, dets)` returns an affine warp matrix (2x3) or homography (3x3),
        depending on the method and configuration.
      - `dets` is expected in tlbr format (x1, y1, x2, y2) in *original image scale*.
    """

    grayscale: bool = True
    scale: Scale = 0.15

    @abstractmethod
    def apply(self, img: np.ndarray, dets: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Convert BGR->GRAY (optional) and resize (optional).
        Supports:
          - scale as float (fx, fy)
          - scale as (W, H) target size
          - None => no resize
        """
        if img is None or not hasattr(img, "shape"):
            raise ValueError("Expected img to be a valid numpy array.")

        out = img
        if getattr(self, "grayscale", True):
            # assume BGR input
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        sc = getattr(self, "scale", None)
        if sc is None:
            return out

        if isinstance(sc, (int, float)):
            if sc <= 0:
                raise ValueError(f"scale must be > 0, got {sc}")
            out = cv2.resize(out, (0, 0), fx=float(sc), fy=float(sc), interpolation=cv2.INTER_LINEAR)
        else:
            # treat as explicit size (W, H)
            w, h = int(sc[0]), int(sc[1])
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid target size for scale: {(w, h)}")
            out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

        return out

    def generate_mask(self, img_gray: np.ndarray, dets: Optional[np.ndarray], scale: float) -> np.ndarray:
        """
        Create a mask that:
          - keeps a central safe region
          - removes detected dynamic objects (dets)
        `img_gray` must be a 2D grayscale image (after preprocess).
        """
        if img_gray.ndim != 2:
            raise ValueError("generate_mask expects a 2D grayscale image.")

        h, w = img_gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Keep most of the image, drop extreme borders (often noisy for motion estimation).
        y1, y2 = int(0.02 * h), int(0.98 * h)
        x1, x2 = int(0.02 * w), int(0.98 * w)
        mask[y1:y2, x1:x2] = 255

        if dets is None:
            return mask

        dets = np.asarray(dets)
        if dets.size == 0:
            return mask

        # dets in original scale -> map to preprocessed scale
        for det in dets:
            # guard shape issues
            if len(det) < 4:
                continue
            tlbr = (np.asarray(det[:4], dtype=np.float32) * float(scale)).astype(int)

            x1b, y1b, x2b, y2b = tlbr.tolist()
            x1b = max(0, min(w, x1b))
            x2b = max(0, min(w, x2b))
            y1b = max(0, min(h, y1b))
            y2b = max(0, min(h, y2b))

            if x2b > x1b and y2b > y1b:
                mask[y1b:y2b, x1b:x2b] = 0

        return mask
