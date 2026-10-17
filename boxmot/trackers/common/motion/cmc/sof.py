# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from boxmot.trackers.common.motion.cmc.base import BaseCMC
from boxmot.utils import logger as LOGGER

SOF_DEFAULT_TARGET_WIDTH = 192
SOF_DEFAULT_RANSAC_REPROJ_THRESHOLD = 3.0


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
        scale: float | None = None,
        min_inliers: int = 8,
        min_inlier_ratio: float = 0.2,
        ransac_reproj_threshold: float = SOF_DEFAULT_RANSAC_REPROJ_THRESHOLD,
        resize_before_grayscale: bool = True,
        target_width: int | None = SOF_DEFAULT_TARGET_WIDTH,
    ) -> None:
        """Estimate motion with bounded image, feature and optical-flow budgets."""
        self.configure_preprocessing(scale, target_width)
        self.grayscale = True
        self.resize_before_grayscale = bool(resize_before_grayscale)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.ransac_reproj_threshold = float(ransac_reproj_threshold)
        self.feature_refresh_interval = 5

        self.feature_params = dict(
            maxCorners=48,
            qualityLevel=0.01,
            minDistance=4,
            blockSize=3,
            useHarrisDetector=False,
            k=0.04,
        )

        self.lk_params = dict(
            winSize=(11, 11),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_keypoints: Optional[np.ndarray] = None
        self.initialized: bool = False
        self._frames_since_detection = 0

    def reset(self) -> None:
        """Clear the optical-flow sequence state."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.initialized = False
        self._frames_since_detection = 0

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

            # OpenCV needs 2 * window + 5 pixels per dimension for refinement.
            # Smaller frames retain their valid, unrefined feature locations.
            if min(frame_gray.shape[:2]) >= 15:
                term_crit = self.lk_params["criteria"]
                cv2.cornerSubPix(frame_gray, kps, winSize=(5, 5), zeroZone=(-1, -1), criteria=term_crit)

            self.prev_frame = frame_gray.copy()
            self.prev_keypoints = kps.copy()
            self.initialized = True
            self._frames_since_detection = 0
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

        if len(prev_valid) and len(next_valid):
            finite = np.isfinite(prev_valid.reshape(len(prev_valid), -1)).all(axis=1)
            finite &= np.isfinite(next_valid.reshape(len(next_valid), -1)).all(axis=1)
            prev_valid = prev_valid[finite]
            next_valid = next_valid[finite]

        if prev_valid is None or next_valid is None or len(prev_valid) < 4:
            # not enough matches -> re-detect
            self._reset(frame_gray, dets)
            return H

        # Reused points must still belong to the current background. Moving
        # detections can cover points that were static in the previous frame.
        mask = self.generate_mask(frame_gray, dets)
        coordinates = np.rint(next_valid.reshape(-1, 2)).astype(np.int32)
        height, width = frame_gray.shape
        inside = ((coordinates >= 0) & (coordinates < (width, height))).all(axis=1)
        keep = np.flatnonzero(inside)
        keep = keep[mask[coordinates[keep, 1], coordinates[keep, 0]] != 0]
        prev_valid, next_valid = prev_valid[keep], next_valid[keep]
        if len(prev_valid) < 4:
            self._reset(frame_gray, dets)
            return H

        # estimate transform
        H_est, inliers = cv2.estimateAffinePartial2D(
            prev_valid,
            next_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )
        accepted = not (
            H_est is None
            or not self._has_enough_inliers(inliers, len(prev_valid))
            or not self.is_valid_transform(H_est)
        )
        if not accepted:
            if H_est is not None:
                LOGGER.debug(
                    "SOF rejected weak or invalid affine estimate: "
                    f"inliers={0 if inliers is None else int(np.count_nonzero(inliers))}/"
                    f"{len(prev_valid)}"
                )
            H_est = H
        else:
            H_est = H_est.astype(np.float32, copy=False)
            H_est = self.restore_transform_scale(H_est)

        # Keep subpixel inlier locations between refreshes; rebuilding the
        # detector every frame costs time and discards that tracking precision.
        # Refresh within five frames to bound drift and restore spatial cover.
        self._frames_since_detection += 1
        new_kps = next_valid[inliers.reshape(-1) != 0] if accepted else None
        if (
            new_kps is None
            or len(new_kps) < max(self.min_inliers, self.feature_params["maxCorners"] // 2)
            or self._frames_since_detection >= self.feature_refresh_interval
        ):
            new_kps = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            self._frames_since_detection = 0

        self.prev_frame = frame_gray.copy()
        self.prev_keypoints = None if new_kps is None else new_kps.copy()
        self.initialized = new_kps is not None and len(new_kps) >= 4

        return H_est

    def _detect_keypoints(self, frame_gray: np.ndarray, dets: Optional[np.ndarray]) -> Optional[np.ndarray]:
        mask = self.generate_mask(frame_gray, dets)
        return cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)

    def _has_enough_inliers(self, inliers: Optional[np.ndarray], match_count: int) -> bool:
        return self.has_enough_inliers(
            inliers,
            match_count,
            min_inliers=self.min_inliers,
            min_inlier_ratio=self.min_inlier_ratio,
        )

    def _reset(self, frame_gray: np.ndarray, dets: Optional[np.ndarray] = None) -> None:
        kps = self._detect_keypoints(frame_gray, dets)
        self.prev_frame = frame_gray.copy()
        self.prev_keypoints = kps
        self.initialized = kps is not None and len(kps) >= 4
        self._frames_since_detection = 0
