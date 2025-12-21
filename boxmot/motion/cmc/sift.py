# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import copy
from typing import Optional

import cv2
import numpy as np

from boxmot.motion.cmc.base_cmc import BaseCMC


class SIFT(BaseCMC):
    """
    SIFT keypoints + BFMatcher(L2) to estimate a 2x3 affine partial transform.
    """

    def __init__(
        self,
        warp_mode: int = cv2.MOTION_EUCLIDEAN,  # kept for API compatibility; we still output 2x3
        eps: float = 1e-5,
        max_iter: int = 100,
        scale: float = 0.15,
        grayscale: bool = True,
        draw_keypoint_matches: bool = False,
        align: bool = False,
    ) -> None:
        self.grayscale = bool(grayscale)
        self.scale = float(scale)
        self.warp_mode = int(warp_mode)  # not strictly used (kept as parameter)

        self.detector = cv2.SIFT_create(nOctaveLayers=2, contrastThreshold=0.5, edgeThreshold=10)
        self.extractor = cv2.SIFT_create(nOctaveLayers=2, contrastThreshold=0.5, edgeThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

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

        mask = self.generate_mask(img_p, dets, self.scale)

        keypoints = self.detector.detect(img_p, mask)
        keypoints, descriptors = self.extractor.compute(img_p, keypoints)

        if descriptors is None or len(keypoints) < 4:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        if self.prev_img is None or self.prev_descriptors is None or self.prev_keypoints is None:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

        knn = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)
        if not knn:
            self._store_state(img_p, keypoints, descriptors, dets)
            self.prev_img_aligned = None
            self.matches_img = None
            return H

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
                spatial_distances.append(dxy)
                matches.append(m)

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

            if self.scale < 1.0:
                H_est = H_est.copy()
                H_est[0, 2] /= self.scale
                H_est[1, 2] /= self.scale

            if self.align:
                self.prev_img_aligned = cv2.warpAffine(self.prev_img, H_est, (w, h), flags=cv2.INTER_LINEAR)
            else:
                self.prev_img_aligned = None

        if self.draw_keypoint_matches:
            self.matches_img = ORBLikeDraw.draw(prev=self.prev_img, curr=img_p, prev_kp=self.prev_keypoints, curr_kp=keypoints, matches=good_matches, dets=dets)
        else:
            self.matches_img = None

        self._store_state(img_p, keypoints, descriptors, dets)
        return H_est

    def _store_state(self, img_p: np.ndarray, keypoints, descriptors, dets) -> None:
        self.prev_img = img_p.copy()
        self.prev_keypoints = copy.copy(keypoints)
        self.prev_descriptors = None if descriptors is None else descriptors.copy()
        self.prev_dets = None if dets is None else np.asarray(dets).copy()


class ORBLikeDraw:
    @staticmethod
    def draw(prev: np.ndarray, curr: np.ndarray, prev_kp, curr_kp, matches, dets):
        canvas = np.hstack((prev, curr))
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

        W = prev.shape[1]
        for m in matches:
            p = np.array(prev_kp[m.queryIdx].pt, dtype=np.int32)
            c = np.array(curr_kp[m.trainIdx].pt, dtype=np.int32)
            c[0] += W
            cv2.line(canvas, tuple(p), tuple(c), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p), 2, (255, 255, 255), -1)
            cv2.circle(canvas, tuple(c), 2, (255, 255, 255), -1)

        if dets is not None:
            for det in np.asarray(dets):
                if len(det) < 4:
                    continue
                x1, y1, x2, y2 = det[:4].astype(int).tolist()
                cv2.rectangle(canvas, (x1 + W, y1), (x2 + W, y2), (0, 0, 255), 2)

        return canvas
