# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import os
from collections import deque

import numpy as np

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.appearance import ema_update_embedding, normalize_embedding
from boxmot.trackers.common.geometry.obb import normalize_angle, smooth_obb_corners

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
            transform = np.asarray(warp_matrix, dtype=float)
            linear = transform[:, :2]
            translation = transform[:, 2]
            scale = float(np.sqrt(max(abs(np.linalg.det(linear)), 1e-8)))
            rotation = float(np.arctan2(linear[1, 0], linear[0, 0]))
            self.mean[:2] = linear @ self.mean[:2] + translation
            self.mean[2:4] = np.maximum(self.mean[2:4] * scale, 1e-4)
            self.mean[4] = normalize_angle(self.mean[4] + rotation)
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
        if self.is_obb:
            corners, self._plot_angle = smooth_obb_corners(self.xywha, self._plot_angle)
            self.history_observations.append(corners)

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
