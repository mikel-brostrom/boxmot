# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import copy
from typing import Optional

import cv2
import numpy as np

from boxmot.trackers.common.motion.cmc.base import BaseCMC
from boxmot.trackers.common.motion.cmc.keypoints import draw_keypoint_matches, filter_keypoint_matches


class ORB(BaseCMC):
    """
    FAST + ORB descriptors + BFMatcher (KNN) to estimate a 2x3 affine partial transform.
    """

    def __init__(
        self,
        feature_detector_threshold: int = 20,
        matcher_norm_type: int = cv2.NORM_HAMMING,
        scale: float = 0.15,
        grayscale: bool = True,
        draw_keypoint_matches: bool = False,
        align: bool = False,
    ) -> None:
        self.grayscale = bool(grayscale)
        self.scale = float(scale)

        self.detector = cv2.FastFeatureDetector_create(threshold=int(feature_detector_threshold))
        self.extractor = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(int(matcher_norm_type))

        self.prev_img: Optional[np.ndarray] = None
        self.prev_keypoints = None
        self.prev_descriptors: Optional[np.ndarray] = None
        self.prev_dets: Optional[np.ndarray] = None

        self.draw_keypoint_matches = bool(draw_keypoint_matches)
        self.align = bool(align)

        self.prev_img_aligned: Optional[np.ndarray] = None
        self.matches_img: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clear the previous-frame and debug state."""
        self.prev_img = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_dets = None
        self.prev_img_aligned = None
        self.matches_img = None

    def apply(self, img: np.ndarray, dets: Optional[np.ndarray] = None) -> np.ndarray:
        H = np.eye(2, 3, dtype=np.float32)

        img_p = self.preprocess(img)
        h, w = img_p.shape[:2]

        # dynamic object mask
        mask = self.generate_mask(img_p, dets)

        # detect/describe
        keypoints = self.detector.detect(img_p, mask)
        keypoints, descriptors = self.extractor.compute(img_p, keypoints)

        if descriptors is None or len(keypoints) < 4:
            # Not enough features; just update prev and return identity
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        # first frame init
        if self.prev_img is None or self.prev_descriptors is None or self.prev_keypoints is None:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        # match descriptors
        knn = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        if not knn:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        good_matches, prev_pts, curr_pts = filter_keypoint_matches(knn, self.prev_keypoints, keypoints, (w, h))
        if len(good_matches) < 4:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        H_est, ransac_inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
        if (
            H_est is None
            or not self.has_enough_inliers(
                ransac_inliers,
                len(good_matches),
                min_inliers=4,
                min_inlier_ratio=0.2,
            )
            or not self.is_valid_transform(H_est)
        ):
            H_est = H
            self.prev_img_aligned = None
        else:
            H_est = H_est.astype(np.float32, copy=False)

            if self.align:
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, H_est, (w, h), flags=cv2.INTER_LINEAR)
            else:
                self.prev_img_aligned = None
            H_est = self.restore_transform_scale(H_est)

        # optional debug visualization
        if self.draw_keypoint_matches:
            self.matches_img = draw_keypoint_matches(
                self.prev_img,
                img_p,
                self.prev_keypoints,
                keypoints,
                good_matches,
                dets,
            )
        else:
            self.matches_img = None

        # store for next iteration
        self._store_state(img_p, keypoints, descriptors, dets)
        return H_est

    def _store_state(self, img_p: np.ndarray, keypoints, descriptors, dets) -> None:
        self.prev_img = img_p.copy()
        self.prev_keypoints = copy.copy(keypoints)
        self.prev_descriptors = None if descriptors is None else descriptors.copy()
        self.prev_dets = None if dets is None else np.asarray(dets).copy()
