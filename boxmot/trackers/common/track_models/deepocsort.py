# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections import deque

import numpy as np

from boxmot.trackers.common.appearance import (
    ema_update_embedding,
)
from boxmot.trackers.common.geometry.obb import (
    transform_aabb,
    transform_aabb_kalman_state,
    transform_obb,
    transform_obb_kalman_state,
    transform_points,
)
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.common.track_models.base import SortBoxTrack
from boxmot.trackers.common.track_models.ocsort import KalmanBoxTracker as OBBKalmanBoxTracker
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(SortBoxTrack):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    def __init__(
        self,
        det,
        delta_t=3,
        emb=None,
        alpha=0,
        max_obs=50,
        Q_xy_scaling=0.01,
        Q_s_scaling=0.0001,
        id_allocator: TrackIdAllocator | None = None,
    ):
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        self.max_obs = max_obs
        bbox = det[0:5]
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

        self.Q_xy_scaling = Q_xy_scaling
        self.Q_s_scaling = Q_s_scaling

        self.motion_model = create_motion_model(MotionModelKind.XYSR, is_obb=False)
        self.kf = self.motion_model.create_filter()
        self.kf.F = np.array(
            [
                # x  y  s  r  x' y' s'
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[4:6, 4:6] *= self.Q_xy_scaling
        self.kf.Q[-1, -1] *= self.Q_s_scaling

        self.bbox_to_z_func = self.motion_model.to_measurement
        self.x_to_bbox_func = self.motion_model.to_box

        self.kf.x[:4] = self.bbox_to_z_func(bbox)

        self._assign_sort_id(id_allocator=id_allocator)
        self._init_sort_counters(max_obs=max_obs)
        self.history = deque([], maxlen=self.max_obs)
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        # Used for OCR
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        # Used to output track after min_hits reached
        self.features = deque([], maxlen=self.max_obs)
        # Used for velocity
        self.observations = dict()
        self.velocity = None
        self.delta_t = delta_t
        self.history_observations = deque([], maxlen=self.max_obs)

        self.emb = emb

        self.frozen = False
        self._append_current_history()
        self._sync_initial_sort_meta()

    def _append_current_history(self) -> None:
        geometry = self.get_state()[0, :4]
        self.history_observations.append(np.asarray(geometry, dtype=np.float32).copy())

    def update(self, det):
        """
        Updates the state vector with observed bbox.
        """

        if det is not None:
            bbox = np.asarray(det[0:5]).copy()
            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
            self.frozen = False

            if self.last_observation[-1] >= 0:  # no previous observation
                previous_box = None
                for dt in range(self.delta_t, 0, -1):
                    if self.age - dt in self.observations:
                        previous_box = self.observations[self.age - dt]
                        break
                if previous_box is None:
                    previous_box = self.last_observation
                # Estimate the track speed direction with observations Δt steps away
                self.velocity = speed_direction(previous_box, bbox)
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox.copy()
            self.observations[self.age] = bbox.copy()
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1

            self.kf.update(self.bbox_to_z_func(bbox))
            self._append_current_history()
            sync_track_meta(self, TrackState.TRACKED)
        else:
            self.kf.update(det)
            self.frozen = True
            sync_track_meta(self)

    def update_emb(self, emb, alpha=0.9):
        self.emb = ema_update_embedding(self.emb, emb, alpha=alpha)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        transform = np.asarray(affine, dtype=float)
        source_center = self.get_state()[0, :2].copy()
        warped_observations: dict[int, np.ndarray] = {}

        def warp_observation_once(observation):
            identity = id(observation)
            if identity not in warped_observations:
                warped_observations[identity] = transform_aabb(observation, transform)
            return warped_observations[identity]

        # For OCR
        if self.last_observation[-1] >= 0:
            self.last_observation[:] = warp_observation_once(self.last_observation)

        # Apply to each box in the range of velocity computation
        for age, observation in self.observations.items():
            self.observations[age][:] = warp_observation_once(observation)

        if self.velocity is not None:
            velocity_xy = np.asarray(self.velocity, dtype=np.float64).reshape(2)[::-1]
            mapped = transform_points(
                np.stack((source_center, source_center + velocity_xy)),
                transform,
            )
            transformed_xy = mapped[1] - mapped[0]
            norm = float(np.linalg.norm(transformed_xy))
            if np.isfinite(transformed_xy).all() and np.isfinite(norm) and norm > 1e-12:
                self.velocity = (transformed_xy / norm)[::-1]

        def transform_state(mean, covariance):
            return transform_aabb_kalman_state(
                mean,
                covariance,
                transform,
                measurement_to_box=lambda values: self.motion_model.to_box(values)[0],
                box_to_measurement=lambda box: self.motion_model.to_measurement(box, column=False),
                velocity_measurement_indices=(0, 1, 2),
            )

        def warp_measurement(measurement):
            if measurement is None:
                return None
            original_shape = np.asarray(measurement).shape
            box = self.motion_model.to_box(measurement)[0]
            warped = self.motion_model.to_measurement(
                transform_aabb(box, transform),
                column=False,
            )
            return np.asarray(warped).reshape(original_shape)

        self.kf.x, self.kf.P = transform_state(self.kf.x, self.kf.P)
        self.kf.history_obs = deque(
            (warp_measurement(item) for item in self.kf.history_obs),
            maxlen=self.kf.history_obs.maxlen,
        )
        self.kf.last_measurement = warp_measurement(self.kf.last_measurement)
        if not self.kf.observed and self.kf.attr_saved is not None:
            saved = self.kf.attr_saved
            saved["x"], saved["P"] = transform_state(saved["x"], saved["P"])
            saved["history_obs"] = deque(
                (warp_measurement(item) for item in saved["history_obs"]),
                maxlen=saved["history_obs"].maxlen,
            )
            saved["last_measurement"] = warp_measurement(saved["last_measurement"])

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        # Don't allow negative bounding boxes
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        Q = None

        self.kf.predict(Q=Q)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.x_to_bbox_func(self.kf.x))
        sync_track_meta(self)
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))


class DeepOBBKalmanBoxTracker(OBBKalmanBoxTracker):
    """OC-SORT oriented motion state extended with DeepOCSORT appearance state."""

    def __init__(self, det, *, emb, alpha, delta_t, max_obs, Q_xy_scaling, Q_s_scaling, id_allocator):
        super().__init__(
            det[:6],
            det[6],
            det[7],
            delta_t=delta_t,
            max_obs=max_obs,
            Q_xy_scaling=Q_xy_scaling,
            Q_s_scaling=Q_s_scaling,
            Q_a_scaling=Q_s_scaling,
            is_obb=True,
            id_allocator=id_allocator,
        )
        self.emb = emb
        self.alpha = alpha
        self.frozen = False

    def update(self, det):
        if det is None:
            super().update(None, None, None)
            self.frozen = True
            return
        super().update(det[:6], det[6], det[7])
        self.frozen = False

    def update_emb(self, emb, alpha=0.9):
        self.emb = ema_update_embedding(self.emb, emb, alpha=alpha)

    def get_emb(self):
        return self.emb

    def apply_affine_correction(self, affine):
        """Warp OBB observations plus the complete XYSR state and covariance."""
        transform = np.asarray(affine, dtype=float)
        source_center = self.get_state()[0, :2].copy()

        def warp_box(box):
            return transform_obb(np.asarray(box, dtype=float)[:5], transform)

        def warp_measurement(measurement):
            if measurement is None:
                return None
            box = self.motion_model.to_box(measurement)[0]
            return self.motion_model.to_measurement(warp_box(box))

        warped_observations: dict[int, np.ndarray] = {}

        def warp_observation_once(observation):
            identity = id(observation)
            if identity not in warped_observations:
                warped_observations[identity] = warp_box(observation)
            return warped_observations[identity]

        if self.last_observation[-1] >= 0:
            self.last_observation[:5] = warp_observation_once(self.last_observation)
        for age, observation in self.observations.items():
            self.observations[age][:5] = warp_observation_once(observation)

        self._transform_cached_velocity(transform, source_center)

        def transform_state(mean, covariance):
            return transform_obb_kalman_state(
                mean,
                covariance,
                transform,
                measurement_to_box=lambda values: self.motion_model.to_box(values)[0],
                box_to_measurement=lambda box: self.motion_model.to_measurement(box, column=False),
                velocity_measurement_indices=(0, 1, 2, 4),
            )

        self.kf.x, self.kf.P = transform_state(self.kf.x, self.kf.P)
        self.kf.history_obs = deque(
            (warp_measurement(item) for item in self.kf.history_obs),
            maxlen=self.kf.history_obs.maxlen,
        )
        self.kf.last_measurement = warp_measurement(self.kf.last_measurement)

        if not self.kf.observed and self.kf.attr_saved is not None:
            saved = self.kf.attr_saved
            saved["x"], saved["P"] = transform_state(saved["x"], saved["P"])
            saved["history_obs"] = deque(
                (warp_measurement(item) for item in saved["history_obs"]),
                maxlen=saved["history_obs"].maxlen,
            )
            saved["last_measurement"] = warp_measurement(saved["last_measurement"])
