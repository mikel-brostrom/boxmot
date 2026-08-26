# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from boxmot.motion.cmc.base_cmc import BaseCMC
from boxmot.utils import logger as LOGGER


class ECC(BaseCMC):
    """
    OpenCV ECC-based motion estimation using cv2.findTransformECC.
    Produces:
      - 2x3 affine-like matrix for TRANSLATION/EUCLIDEAN/AFFINE
      - 3x3 homography matrix for HOMOGRAPHY
    """

    def __init__(
        self,
        warp_mode: int = cv2.MOTION_TRANSLATION,
        eps: float = 1e-5,
        max_iter: int = 100,
        scale: float = 0.15,
        align: bool = False,
        grayscale: bool = True,
        min_correlation: float = 0.5,
    ) -> None:
        self.align = bool(align)
        self.grayscale = bool(grayscale)
        self.scale = float(scale)
        self.warp_mode = int(warp_mode)
        self.min_correlation = float(min_correlation)
        if not 0.0 <= self.min_correlation <= 1.0:
            raise ValueError("min_correlation must be in [0, 1]")

        self.termination_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(max_iter),
            float(eps),
        )

        self.prev_img: Optional[np.ndarray] = None
        self.prev_img_aligned: Optional[np.ndarray] = None

    def apply(self, img: np.ndarray, dets: Optional[np.ndarray] = None) -> np.ndarray:
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            identity = np.eye(3, 3, dtype=np.float32)
        else:
            identity = np.eye(2, 3, dtype=np.float32)

        if self.prev_img is None:
            self.prev_img = self.preprocess(img)
            self.prev_img_aligned = None
            return identity

        curr = self.preprocess(img)
        mask = self.generate_mask(curr, dets, self.scale)
        scaled_warp = identity.copy()

        try:
            correlation, scaled_warp = cv2.findTransformECC(
                self.prev_img,
                curr,
                scaled_warp,
                self.warp_mode,
                self.termination_criteria,
                mask,
                1,
            )
        except cv2.error as e:
            # StsNoConv => ECC did not converge; return identity (common in practice).
            try:
                if e.code == cv2.Error.StsNoConv:
                    LOGGER.warning("ECC did not converge; returning identity warp.")
                    self.prev_img = curr
                    self.prev_img_aligned = None
                    return identity
            except Exception:
                pass
            raise

        if not np.isfinite(correlation) or correlation < self.min_correlation:
            LOGGER.warning(
                f"ECC correlation {correlation:.3f} is below {self.min_correlation:.3f}; "
                "returning identity warp."
            )
            self.prev_img = curr
            self.prev_img_aligned = None
            return identity

        if not self.is_valid_transform(scaled_warp):
            LOGGER.warning("ECC produced a non-finite or degenerate transform; returning identity warp.")
            self.prev_img = curr
            self.prev_img_aligned = None
            return identity

        if self.align:
            h, w = self.prev_img.shape[:2]
            if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
                self.prev_img_aligned = cv2.warpPerspective(
                    self.prev_img,
                    scaled_warp,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                )
            else:
                self.prev_img_aligned = cv2.warpAffine(
                    self.prev_img,
                    scaled_warp,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                )
        else:
            self.prev_img_aligned = None

        self.prev_img = curr
        return self.restore_transform_scale(scaled_warp)
