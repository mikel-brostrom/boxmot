# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections import deque

import numpy as np

from boxmot.trackers.common.appearance import (
    ema_update_embedding,
)
from boxmot.trackers.common.geometry.obb import xywha_to_xyxy
from boxmot.trackers.common.motion.cmc.batching import transform_ocsort_tracks
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model
from boxmot.trackers.common.track_state import SortBoxTrack
from boxmot.trackers.common.tracking.observations import speed_direction
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta
from boxmot.trackers.ocsort.track import KalmanBoxTracker as OBBKalmanBoxTracker


class KalmanBoxTracker(SortBoxTrack):
    """One DeepOCSORT identity with observation-centric motion and appearance."""

    observation_centric = True

    def __init__(
        self,
        det,
        delta_t=3,
        emb=None,
        alpha=0,
        max_obs=50,
        id_allocator: TrackIdAllocator | None = None,
        *,
        noise_config: KalmanNoiseConfig | None = None,
        kalman_type: str | None = None,
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

        self.motion_model = create_motion_model(
            kalman_type or MotionModelKind.XYSR,
            is_obb=False,
            cls_id=int(self.cls),
        )
        self.kf = self.motion_model.create_filter(noise_config=noise_config)
        if self.motion_model.kind is MotionModelKind.XYSR:
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

        self.bbox_to_z_func = self.motion_model.to_measurement
        self.x_to_bbox_func = self.motion_model.to_box

        if self.motion_model.kind is MotionModelKind.XYSR:
            self.kf.x[:4] = self.bbox_to_z_func(bbox)
        else:
            self.motion_model.initialize_filter(self.kf, self.bbox_to_z_func(bbox), confidence=float(self.conf))

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

        measurement = self._prepare_update(det)
        self.motion_model.update_filter(self.kf, measurement, observation_centric=True, confidence=float(self.conf))
        self._finish_update(measurement)

    def _prepare_update(self, det) -> np.ndarray | None:
        """Prepare appearance and motion observation history for correction."""
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

            return self.bbox_to_z_func(bbox)
        self.frozen = True
        return None

    def _finish_update(self, measurement: np.ndarray | None) -> None:
        """Record display geometry after scalar or batched Kalman correction."""
        if measurement is not None:
            self._append_current_history()
            sync_track_meta(self, TrackState.TRACKED)
        else:
            sync_track_meta(self)

    def update_emb(self, emb, alpha=0.9):
        self.emb = ema_update_embedding(self.emb, emb, alpha=alpha)

    def get_emb(self):
        return self.emb

    def camera_update(self, affine: np.ndarray) -> None:
        """Warp this track's current state and observation histories."""
        self.multi_camera_update([self], affine)

    @classmethod
    def multi_camera_update(cls, tracks, affine: np.ndarray) -> None:
        """Correct camera motion using the selected model and its recovery histories."""
        if tracks:
            transform_ocsort_tracks(tracks, affine, model=tracks[0].motion_model)

    def predict(self, *, dt: float | None = None) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self._prepare_prediction(dt=dt)
        self.motion_model.predict_filter(self.kf, dt=dt, observation_centric=self.observation_centric)
        return self._finish_prediction()

    def _prepare_prediction(self, *, dt: float | None = None) -> None:
        """Prevent the selected model from extrapolating nonpositive sizes."""
        self.motion_model.prevent_invalid_size(self.kf, dt=dt)

    def _finish_prediction(self) -> np.ndarray:
        """Advance counters after scalar or batched Kalman prediction."""
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

    @property
    def xyxy(self) -> np.ndarray:
        """Return canonical predicted AABB geometry for shared association gates."""
        return np.asarray(self.get_state()[0], dtype=float)

    def mahalanobis(self, bbox):
        """Should be run after a predict() call for accuracy."""
        if self.motion_model.kind is MotionModelKind.XYSR:
            return self.kf.md_for_measurement(self.bbox_to_z_func(bbox))
        distances = self.motion_model.gating_distance(
            self.kf,
            np.asarray(self.kf.x).reshape(-1),
            self.kf.P,
            self.bbox_to_z_func(bbox).reshape(1, -1),
        )
        return float(np.sqrt(distances[0]))


class DeepOBBKalmanBoxTracker(OBBKalmanBoxTracker):
    """OcSort oriented motion state extended with DeepOcSort appearance state."""

    def __init__(
        self,
        det,
        *,
        emb,
        alpha,
        delta_t,
        max_obs,
        id_allocator,
        noise_config: KalmanNoiseConfig | None = None,
        kalman_type: str | None = None,
    ):
        super().__init__(
            det[:6],
            det[6],
            det[7],
            delta_t=delta_t,
            max_obs=max_obs,
            is_obb=True,
            id_allocator=id_allocator,
            noise_config=noise_config,
            kalman_type=kalman_type,
        )
        self.emb = emb
        self.alpha = alpha
        self.frozen = False

    def update(self, det):
        measurement = self._prepare_update(det)
        self.motion_model.update_filter(self.kf, measurement, observation_centric=True, confidence=float(self.conf))
        self._finish_update(measurement)

    @property
    def xywha(self) -> np.ndarray:
        """Return native predicted OBB geometry for shared association gates."""
        return np.asarray(self.get_state()[0], dtype=float)

    @property
    def xyxy(self) -> np.ndarray:
        """Return the enclosing AABB when a consumer needs axis-aligned geometry."""
        return xywha_to_xyxy(self.xywha.reshape(1, 5))[0]

    def _prepare_update(self, det) -> np.ndarray | None:
        """Adapt packed oriented detections to the shared OC-SORT lifecycle."""
        self.frozen = det is None
        if det is None:
            return super()._prepare_update(None, None, None)
        return super()._prepare_update(det[:6], det[6], det[7])

    def update_emb(self, emb, alpha=0.9):
        self.emb = ema_update_embedding(self.emb, emb, alpha=alpha)

    def get_emb(self):
        return self.emb
