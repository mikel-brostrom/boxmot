# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from boxmot.motion.cmc.base_cmc import BaseCMC
from boxmot.utils import logger as LOGGER


class SOF(BaseCMC):
    """
    Sparse Optical Flow tracker estimating a 2x3 affine partial transform between frames.
    Uses:
      - goodFeaturesToTrack for keypoints
      - calcOpticalFlowPyrLK for tracking
      - estimateAffinePartial2D (RANSAC) for robust motion estimation
    """

    def __init__(self, scale: float = 0.15) -> None:
        self.scale = float(scale)
        self.grayscale = True

        self.feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.01,
            minDistance=1,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04,
        )

        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[np.ndarray] = None
        self.initialized: bool = False

    def apply(self, img: np.ndarray, dets: Optional[np.ndarray] = None) -> np.ndarray:
        frame_gray = self.preprocess(img)
        H = np.eye(2, 3, dtype=np.float32)

        # First frame init
        if not self.initialized or self.prev_frame is None or self.prev_keypoints is None:
            kps = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
            if kps is None or len(kps) < 4:
                # can't initialize reliably; keep trying on next frame
                self.prev_frame = frame_gray.copy()
                self.prev_keypoints = kps
                self.initialized = False
                return H

            # optional refinement for stability
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            cv2.cornerSubPix(frame_gray, kps, winSize=(5, 5), zeroZone=(-1, -1), criteria=term_crit)

            self.prev_frame = frame_gray.copy()
            self.prev_keypoints = kps.copy()
            self.initialized = True
            return H

        # Track keypoints
        next_kps, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame, frame_gray, self.prev_keypoints, None, **self.lk_params
        )

        if next_kps is None or status is None:
            self._reset(frame_gray)
            return H

        status = status.reshape(-1)
        prev_valid = self.prev_keypoints[status == 1]
        next_valid = next_kps[status == 1]

        if prev_valid is None or next_valid is None or len(prev_valid) < 4:
            # not enough matches -> re-detect
            self._reset(frame_gray)
            return H

        # estimate transform
        H_est, inliers = cv2.estimateAffinePartial2D(prev_valid, next_valid, method=cv2.RANSAC)
        if H_est is None:
            H_est = H
        else:
            H_est = H_est.astype(np.float32, copy=False)
            if self.scale < 1.0:
                H_est = H_est.copy()
                H_est[0, 2] /= self.scale
                H_est[1, 2] /= self.scale

        # refresh keypoints each frame (more stable long-term than purely tracking)
        new_kps = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        if new_kps is None or len(new_kps) < 4:
            # fallback: keep tracked points
            new_kps = next_valid

        self.prev_frame = frame_gray.copy()
        self.prev_keypoints = new_kps.copy()
        self.initialized = True

        return H_est

    def _reset(self, frame_gray: np.ndarray) -> None:
        kps = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        self.prev_frame = frame_gray.copy()
        self.prev_keypoints = kps
        self.initialized = kps is not None and len(kps) >= 4
