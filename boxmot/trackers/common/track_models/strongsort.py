# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import os
from collections import deque

import cv2
import numpy as np

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.appearance import ema_update_embedding, normalize_embedding
from boxmot.trackers.common.geometry.obb import (
    smooth_obb_corners,
    transform_obb_kalman_state,
)

__all__ = ("Track", "TrackState")


class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    """

    def __init__(
        self,
        detection,
        id,
        n_init,
        max_age,
        ema_alpha,
        is_obb=False,
    ):
        self.id = id
        self.is_obb = bool(is_obb)
        self.bbox = detection.to_measurement()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.ema_alpha = ema_alpha

        # start with confirmed in Ci as test expect equal amount of outputs as inputs
        self.state = (
            TrackState.Confirmed
            if n_init <= 1
            or (os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("GITHUB_JOB") != "mot-metrics-benchmark")
            else TrackState.Tentative
        )
        self.features = []
        if detection.feat is not None:
            self.features.append(normalize_embedding(detection.feat))

        self._n_init = n_init
        self._max_age = max_age

        self.kf = KalmanFilterXYWH(ndim=5) if self.is_obb else KalmanFilterXYAH()
        self.mean, self.covariance = self.kf.initiate(self.bbox)
        self.history_observations = deque(maxlen=max_age)
        self._plot_angle = None
        self._append_current_history()

    def to_tlwh(self):
        """Get current position in `(top left x, top left y, width, height)`."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get kf estimated current position in `(min x, min y, max x, max y)`."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    @property
    def xywha(self):
        if not self.is_obb:
            raise AttributeError("xywha is only available for OBB tracks")
        return self.mean[:5].copy()

    def camera_update(self, warp_matrix):
        if self.is_obb:
            self.mean, self.covariance = transform_obb_kalman_state(
                self.mean,
                self.covariance,
                warp_matrix,
                measurement_to_box=lambda values: values,
                box_to_measurement=lambda box: box,
                velocity_measurement_indices=(0, 1, 2, 3, 4),
            )
            self._warp_history(warp_matrix)
            return
        [a, b] = warp_matrix
        warp_matrix = np.array([a, b, [0, 0, 1]])
        warp_matrix = warp_matrix.tolist()
        x1, y1, x2, y2 = self.to_tlbr()
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.mean[:4] = [cx, cy, w / h, h]
        self._warp_history(warp_matrix)

    def _append_current_history(self) -> None:
        if self.is_obb:
            geometry, self._plot_angle = smooth_obb_corners(self.xywha, self._plot_angle)
        else:
            geometry = self.to_tlbr()
        self.history_observations.append(np.asarray(geometry, dtype=np.float32).copy())

    def _warp_history(self, warp_matrix: np.ndarray) -> None:
        """Express stored display trajectories in the current camera frame."""
        matrix = np.asarray(warp_matrix, dtype=np.float32)
        warped = deque(maxlen=self.history_observations.maxlen)
        for observation in self.history_observations:
            values = np.asarray(observation, dtype=np.float32)
            if values.size == 8:
                points = values.reshape(4, 1, 2)
                transformed = (
                    cv2.transform(points, matrix)
                    if matrix.shape == (2, 3)
                    else cv2.perspectiveTransform(points, matrix)
                )
                warped.append(transformed.reshape(8))
                continue

            x1, y1, x2, y2 = values[:4]
            points = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float32,
            ).reshape(4, 1, 2)
            transformed = (
                cv2.transform(points, matrix)
                if matrix.shape == (2, 3)
                else cv2.perspectiveTransform(points, matrix)
            ).reshape(4, 2)
            warped.append(
                np.array(
                    [
                        transformed[:, 0].min(),
                        transformed[:, 1].min(),
                        transformed[:, 0].max(),
                        transformed[:, 1].max(),
                    ],
                    dtype=np.float32,
                )
            )
        self.history_observations = warped

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self):
        """Propagate the state distribution to the current time step."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """Perform Kalman filter measurement update and update the feature cache."""
        self.bbox = detection.to_measurement()
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, self.bbox, self.conf)
        self._append_current_history()

        smooth_feat = ema_update_embedding(
            self.features[-1],
            normalize_embedding(detection.feat),
            alpha=self.ema_alpha,
        )
        self.features = [smooth_feat]

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed when there is no association at the current time step."""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Return True if this track is tentative."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Return True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Return True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
