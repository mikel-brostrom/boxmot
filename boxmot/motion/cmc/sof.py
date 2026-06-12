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

    def __init__(
        self,
        scale: float = 0.15,
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.2,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        self.scale = float(scale)
        self.grayscale = True
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)

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
            kps = self._detect_keypoints(frame_gray, dets)
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
            self._reset(frame_gray, dets)
            return H

        status = status.reshape(-1)
        prev_valid = self.prev_keypoints[status == 1]
        next_valid = next_kps[status == 1]

        if prev_valid is None or next_valid is None or len(prev_valid) < 4:
            # not enough matches -> re-detect
            self._reset(frame_gray, dets)
            return H

        # estimate transform
        H_est, inliers = cv2.estimateAffinePartial2D(
            prev_valid,
            next_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )
        if H_est is None or not self._has_enough_inliers(inliers, len(prev_valid)):
            if H_est is not None:
                LOGGER.debug(
                    "SOF rejected weak affine estimate: "
                    f"inliers={0 if inliers is None else int(np.count_nonzero(inliers))}/"
                    f"{len(prev_valid)}"
                )
            H_est = H
        else:
            H_est = H_est.astype(np.float32, copy=False)
            if self.scale < 1.0:
                H_est = H_est.copy()
                H_est[0, 2] /= self.scale
                H_est[1, 2] /= self.scale

        # refresh keypoints each frame (more stable long-term than purely tracking)
        new_kps = self._detect_keypoints(frame_gray, dets)
        if new_kps is None or len(new_kps) < 4:
            # fallback: keep tracked points
            new_kps = next_valid

        self.prev_frame = frame_gray.copy()
        self.prev_keypoints = new_kps.copy()
        self.initialized = True

        return H_est

    def _detect_keypoints(self, frame_gray: np.ndarray, dets: Optional[np.ndarray]) -> Optional[np.ndarray]:
        mask = self.generate_mask(frame_gray, dets, self.scale)
        return cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)

    def _has_enough_inliers(self, inliers: Optional[np.ndarray], match_count: int) -> bool:
        if inliers is None or match_count <= 0:
            return False
        inlier_count = int(np.count_nonzero(inliers))
        return (
            inlier_count >= self.min_inliers
            and inlier_count / match_count >= self.min_inlier_ratio
        )

    def _reset(self, frame_gray: np.ndarray, dets: Optional[np.ndarray] = None) -> None:
        kps = self._detect_keypoints(frame_gray, dets)
        self.prev_frame = frame_gray.copy()
        self.prev_keypoints = kps
        self.initialized = kps is not None and len(kps) >= 4
