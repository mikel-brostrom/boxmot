# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import copy
from typing import Optional

import cv2
import numpy as np

from boxmot.motion.cmc.base_cmc import BaseCMC


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

    def apply(self, img: np.ndarray, dets: Optional[np.ndarray] = None) -> np.ndarray:
        H = np.eye(2, 3, dtype=np.float32)

        img_p = self.preprocess(img)
        h, w = img_p.shape[:2]

        # dynamic object mask
        mask = self.generate_mask(img_p, dets, self.scale)

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

        # Lowe ratio + spatial gating
        matches = []
        spatial_distances = []
        max_spatial_distance = 0.25 * np.array([w, h], dtype=np.float32)

        for pair in knn:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance >= 0.9 * n.distance:
                continue

            prev_pt = np.array(self.prev_keypoints[m.queryIdx].pt, dtype=np.float32)
            curr_pt = np.array(keypoints[m.trainIdx].pt, dtype=np.float32)
            dxy = prev_pt - curr_pt

            if (abs(dxy[0]) < max_spatial_distance[0]) and (abs(dxy[1]) < max_spatial_distance[1]):
                matches.append(m)
                spatial_distances.append(dxy)

        if len(matches) < 4:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        spatial_distances = np.asarray(spatial_distances, dtype=np.float32)
        mean = spatial_distances.mean(axis=0)
        std = spatial_distances.std(axis=0) + 1e-6
        inliers_spatial = np.all((spatial_distances - mean) < 2.5 * std, axis=1)

        good_matches = [matches[i] for i in range(len(matches)) if inliers_spatial[i]]
        if len(good_matches) < 4:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        prev_pts = np.array([self.prev_keypoints[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        curr_pts = np.array([keypoints[m.trainIdx].pt for m in good_matches], dtype=np.float32)

        H_est, ransac_inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
        if H_est is None:
            H_est = H
        else:
            H_est = H_est.astype(np.float32, copy=False)

            # upscale translation back to original image coordinates
            if self.scale < 1.0:
                H_est = H_est.copy()
                H_est[0, 2] /= self.scale
                H_est[1, 2] /= self.scale

            if self.align:
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, H_est, (w, h), flags=cv2.INTER_LINEAR)
            else:
                self.prev_img_aligned = None

        # optional debug visualization
        if self.draw_keypoint_matches:
            self.matches_img = self._draw_matches(self.prev_img, img_p, self.prev_keypoints, keypoints, good_matches, dets)
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

    @staticmethod
    def _draw_matches(prev_img: np.ndarray, curr_img: np.ndarray, prev_kp, curr_kp, matches, dets):
        # prev_img/curr_img are grayscale
        canvas = np.hstack((prev_img, curr_img))
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        W = prev_img.shape[1]
        for m in matches:
            p = np.array(prev_kp[m.queryIdx].pt, dtype=np.int32)
            c = np.array(curr_kp[m.trainIdx].pt, dtype=np.int32)
            c[0] += W
            cv2.line(canvas, tuple(p), tuple(c), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p), 2, (255, 255, 255), -1)
            cv2.circle(canvas, tuple(c), 2, (255, 255, 255), -1)

        if dets is not None:
            # draw detections on the right image for context
            h, w = curr_img.shape[:2]
            for det in np.asarray(dets):
                if len(det) < 4:
                    continue
                x1, y1, x2, y2 = det[:4].astype(int).tolist()
                cv2.rectangle(canvas, (x1 + W, y1), (x2 + W, y2), (0, 0, 255), 2)

        return canvas
