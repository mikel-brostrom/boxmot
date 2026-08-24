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
      - `dets` contains geometry in original image scale: AABB ``xyxy`` rows
        (4 columns) or OBB ``xywha`` rows (5 columns).
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
        original_h, original_w = out.shape[:2]
        if getattr(self, "grayscale", True):
            # assume BGR input
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        sc = getattr(self, "scale", None)
        if sc is None:
            self._preprocess_scale = (1.0, 1.0)
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

        self._preprocess_scale = (out.shape[1] / original_w, out.shape[0] / original_h)
        return out

    def restore_transform_scale(self, transform: np.ndarray) -> np.ndarray:
        """Map a transform estimated on a resized image back to image coordinates.

        If preprocessing maps image points with ``p_scaled = S @ p_image``, a
        transform estimated in scaled coordinates must be conjugated as
        ``S^-1 @ H_scaled @ S``. This is required for homographies and for
        non-uniform resize factors; scaling only the translation terms is not
        generally correct.
        """
        matrix = np.asarray(transform)
        original_shape = matrix.shape
        if original_shape == (2, 3):
            homogeneous = np.vstack([matrix, np.array([0.0, 0.0, 1.0], dtype=matrix.dtype)])
        elif original_shape == (3, 3):
            homogeneous = matrix.copy()
        else:
            raise ValueError(f"Expected a 2x3 affine or 3x3 homography, got {original_shape}")

        scale_x, scale_y = getattr(self, "_preprocess_scale", (1.0, 1.0))
        if not np.isfinite((scale_x, scale_y)).all() or scale_x <= 0.0 or scale_y <= 0.0:
            raise ValueError(f"Invalid preprocessing scale {(scale_x, scale_y)}")

        scale_matrix = np.diag([scale_x, scale_y, 1.0]).astype(homogeneous.dtype, copy=False)
        restored = np.linalg.inv(scale_matrix) @ homogeneous @ scale_matrix
        if original_shape == (2, 3):
            restored = restored[:2]
        return restored.astype(matrix.dtype, copy=False)

    @staticmethod
    def is_valid_transform(
        transform: np.ndarray,
        *,
        min_abs_determinant: float = 1e-6,
        max_abs_determinant: float = 1e6,
    ) -> bool:
        """Return whether an affine/homography is finite and non-degenerate."""
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape == (2, 3):
            determinant = float(np.linalg.det(matrix[:, :2]))
        elif matrix.shape == (3, 3):
            determinant = float(np.linalg.det(matrix))
        else:
            return False
        abs_determinant = abs(determinant)
        return bool(
            np.isfinite(matrix).all()
            and np.isfinite(determinant)
            and min_abs_determinant <= abs_determinant <= max_abs_determinant
        )

    @staticmethod
    def has_enough_inliers(
        inliers: Optional[np.ndarray],
        match_count: int,
        *,
        min_inliers: int,
        min_inlier_ratio: float,
    ) -> bool:
        """Validate a RANSAC mask against absolute and relative thresholds."""
        if inliers is None or match_count <= 0:
            return False
        inlier_count = int(np.count_nonzero(inliers))
        return inlier_count >= min_inliers and inlier_count / match_count >= min_inlier_ratio

    def generate_mask(self, img_gray: np.ndarray, dets: Optional[np.ndarray], scale: Scale) -> np.ndarray:
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

        scale_x, scale_y = getattr(
            self,
            "_preprocess_scale",
            (float(scale), float(scale)) if isinstance(scale, (int, float)) else (1.0, 1.0),
        )

        # Boxes are either AABB ``xyxy`` rows or OBB ``xywha`` rows in the
        # original image scale. Mask the actual oriented polygon for OBBs so
        # static background inside an enclosing AABB remains available to CMC.
        is_obb = dets.ndim == 2 and dets.shape[1] == 5
        for det in dets:
            if len(det) < 4:
                continue

            if is_obb:
                cx, cy, bw, bh, angle = (float(value) for value in det[:5])
                rect = (
                    (cx, cy),
                    (max(bw, 1e-4), max(bh, 1e-4)),
                    float(np.degrees(angle)),
                )
                polygon = cv2.boxPoints(rect)
                polygon[:, 0] *= scale_x
                polygon[:, 1] *= scale_y
                polygon = np.rint(polygon).astype(np.int32)
                cv2.fillConvexPoly(mask, polygon, 0)
                continue

            # ``det`` can be a view into the caller's float32 detection
            # array. Copy before scaling so mask generation never mutates the
            # detections that will subsequently be associated.
            tlbr = np.array(det[:4], dtype=np.float32, copy=True)
            tlbr[[0, 2]] *= scale_x
            tlbr[[1, 3]] *= scale_y
            tlbr = tlbr.astype(int)

            x1b, y1b, x2b, y2b = tlbr.tolist()
            x1b = max(0, min(w, x1b))
            x2b = max(0, min(w, x2b))
            y1b = max(0, min(h, y1b))
            y2b = max(0, min(h, y2b))

            if x2b > x1b and y2b > y1b:
                mask[y1b:y2b, x1b:x2b] = 0

        return mask
