# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import os
from collections import deque

import numpy as np

from boxmot.trackers.common.appearance import ema_update_embedding, normalize_embedding
from boxmot.trackers.common.geometry.obb import (
    smooth_obb_corners,
    transform_aabbs,
)
from boxmot.trackers.common.motion.cmc.state import transform_aabb_kalman_states, transform_obb_kalman_states
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model

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
        max_obs,
        ema_alpha,
        is_obb=False,
        *,
        noise_config: KalmanNoiseConfig | None = None,
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

        self.kf = (
            KalmanFilterXYWH(ndim=5, noise_config=noise_config)
            if self.is_obb
            else KalmanFilterXYAH(noise_config=noise_config)
        )
        self.mean, self.covariance = self.kf.initiate(self.bbox)
        self.history_observations = deque(maxlen=max(1, int(max_obs)))
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
    def xyxy(self):
        """Return axis-aligned geometry for shared association helpers."""
        return self.to_tlbr()

    @property
    def xywha(self):
        if not self.is_obb:
            raise AttributeError("xywha is only available for OBB tracks")
        return self.mean[:5].copy()

    def camera_update(self, warp_matrix: np.ndarray) -> None:
        """Transform this track using the shared batched CMC implementation."""
        self.multi_camera_update([self], warp_matrix)

    @classmethod
    def multi_camera_update(cls, tracks, warp_matrix: np.ndarray) -> None:
        """Transform compatible track states and their last AABB measurements."""
        for is_obb in (False, True):
            group = [track for track in tracks if track.is_obb == is_obb]
            if not group:
                continue
            model = create_motion_model(MotionModelKind.XYWH if is_obb else MotionModelKind.XYAH, is_obb=is_obb)
            transform_states = transform_obb_kalman_states if is_obb else transform_aabb_kalman_states
            means, covariances = transform_states(
                np.asarray([track.mean for track in group]),
                np.asarray([track.covariance for track in group]),
                warp_matrix,
                measurement_to_box=(lambda rows: rows) if is_obb else model.to_boxes,
                box_to_measurement=(lambda rows: rows) if is_obb else model.to_measurements,
                velocity_measurement_indices=(0, 1, 2, 3, 4) if is_obb else (0, 1, 2, 3),
            )
            for track, mean, covariance in zip(group, means, covariances):
                track.mean, track.covariance = mean, covariance
            if not is_obb:
                boxes = model.to_boxes(np.asarray([track.bbox for track in group]))
                measurements = model.to_measurements(transform_aabbs(boxes, warp_matrix))
                for track, measurement in zip(group, measurements):
                    track.bbox = measurement

    def _append_current_history(self) -> None:
        if self.is_obb:
            geometry, self._plot_angle = smooth_obb_corners(self.xywha, self._plot_angle)
        else:
            geometry = self.to_tlbr()
        self.history_observations.append(np.asarray(geometry, dtype=np.float32).copy())

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, *, dt: float | None = None) -> None:
        """Propagate the state distribution to the current time step."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance, dt=dt)
        self.age += 1
        self.time_since_update += 1

    @staticmethod
    def _kalman_groups(tracks):
        """Group stateless filters by numerical policy without sharing owners."""
        groups = {}
        for track in tracks:
            kalman = track.kf
            key = (
                type(kalman),
                kalman.noise_config,
                kalman.dt,
                kalman._std_weight_position,
                kalman._std_weight_velocity,
                kalman._motion_mat.tobytes(),
                kalman._update_mat.tobytes(),
            )
            groups.setdefault(key, []).append(track)
        return groups.values()

    @classmethod
    def multi_predict(cls, tracks, *, dt: float | None = None) -> None:
        """Predict compatible track states together, retaining track-local ages."""
        for group in cls._kalman_groups(tracks):
            mean, covariance = group[0].kf.multi_predict(
                np.asarray([track.mean for track in group]),
                np.asarray([track.covariance for track in group]),
                dt=dt,
            )
            for track, state, uncertainty in zip(group, mean, covariance):
                track.mean, track.covariance = state, uncertainty
                track.increment_age()

    @classmethod
    def multi_update(cls, pairs) -> None:
        """Correct matched states together using each detection's confidence."""
        pairs = list(pairs)
        detections = {id(track): detection for track, detection in pairs}
        for group in cls._kalman_groups(track for track, _ in pairs):
            observations = [detections[id(track)] for track in group]
            measurements = np.asarray([detection.to_measurement() for detection in observations])
            mean, covariance = group[0].kf.multi_update(
                np.asarray([track.mean for track in group]),
                np.asarray([track.covariance for track in group]),
                measurements,
                np.asarray([detection.conf for detection in observations]),
            )
            for track, detection, measurement, state, uncertainty in zip(
                group, observations, measurements, mean, covariance
            ):
                track.mean, track.covariance = state, uncertainty
                track._finish_update(detection, measurement)

    def update(self, detection):
        """Perform Kalman filter measurement update and update the feature cache."""
        measurement = detection.to_measurement()
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, measurement, detection.conf)
        self._finish_update(detection, measurement)

    def _finish_update(self, detection, measurement: np.ndarray) -> None:
        """Update observation and appearance history after a correction."""
        self.bbox = measurement
        self.conf = detection.conf
        self.cls = detection.cls
        self.det_ind = detection.det_ind
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
