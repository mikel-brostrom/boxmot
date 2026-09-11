from __future__ import annotations

# Hybrid-SORT-ReID with ECC + ReID (explicit config, BaseTracker-style)
# - Assumes detection input is M x [x1, y1, x2, y2, conf, cls]
# - ECC via shared CMC factory and BaseTracker.apply_cmc(...)
# - ReID consumes embeddings supplied by the canonical detection structure
# - Used by the private NumPy kernel behind the canonical tracker update path
# - Emits rows: [x1,y1,x2,y2, track_id, conf, cls, det_ind]
# - Preserves detector class IDs and det_ind; guards out-of-range indices
from collections import deque
from typing import Optional

import numpy as np

from boxmot.trackers.common.appearance import (
    blend_embeddings,
    ema_update_embedding,
    normalize_embedding,
)
from boxmot.trackers.common.geometry.obb import transform_aabbs, transform_points
from boxmot.trackers.common.motion.cmc.batching import transform_directions, transform_filter_histories
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model
from boxmot.trackers.common.track_state import SortBoxTrack
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta


def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(SortBoxTrack):
    """
    Single-object tracker with 9D custom KF (u,v,s,c,r, du,dv,ds,dc) by default.
    Stores `cls` and `det_ind` metadata from the most recent matched detection.
    """

    def __init__(
        self,
        bbox,
        temp_feat,
        *,
        delta_t: int = 3,
        longterm_bank_length: int = 30,
        max_obs: int = 50,
        alpha: float = 0.9,
        adapfs: bool = False,
        track_thresh: float = 0.5,
        cls: int = 0,
        det_ind: int = -1,
        id_allocator: TrackIdAllocator | None = None,
        noise_config: KalmanNoiseConfig | None = None,
    ):
        self.motion_model = create_motion_model(MotionModelKind.XYSCR, max_obs=max_obs)
        self.kf = self.motion_model.create_filter(noise_config=noise_config)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[5:, 5:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01
        self.kf.x[:5] = self.motion_model.to_measurement(bbox)

        # tracker state
        self._assign_sort_id(id_allocator=id_allocator)
        self._init_sort_counters(max_obs=max(1, int(max_obs)))
        self.history = deque([], maxlen=self.max_obs)

        # observations
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)

        # velocity aids
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None

        # parameters
        self.delta_t = int(delta_t)
        self.confidence_pre = None
        self.conf = float(bbox[-1])

        # ReID buffers
        self.smooth_feat = None
        self.features = deque([], maxlen=int(longterm_bank_length))
        self.alpha = float(alpha)
        self.adapfs = bool(adapfs)
        self.track_thresh = float(track_thresh)

        # metadata
        self.cls = int(cls)
        self.det_ind = int(det_ind)

        # first feature update
        self.update_features(temp_feat)
        self._append_current_history()
        self._sync_initial_sort_meta()

    def _append_current_history(self) -> None:
        geometry = self.motion_model.to_box(self.kf.x)[0, :4]
        self.history_observations.append(np.asarray(geometry, dtype=np.float32).copy())

    def _prune_observations(self) -> None:
        cutoff = self.age - self.max_obs + 1
        for obs_age in list(self.observations):
            if obs_age < cutoff:
                self.observations.pop(obs_age, None)

    def update_features(self, feat, score: float = -1.0):
        feat = normalize_embedding(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if self.adapfs:
                assert score > 0, "score must be > 0 when adapfs=True"
                pre_w = self.alpha * (self.conf / (self.conf + score))
                cur_w = (1.0 - self.alpha) * (score / (self.conf + score))
                s = pre_w + cur_w
                pre_w /= s
                cur_w /= s
                self.smooth_feat = blend_embeddings(self.smooth_feat, feat, pre_w, cur_w)
            else:
                self.smooth_feat = ema_update_embedding(
                    self.smooth_feat,
                    feat,
                    alpha=self.alpha,
                )
        self.features.append(feat)

    @staticmethod
    def _map_camera_states(states: np.ndarray, transform: np.ndarray, model) -> np.ndarray:
        """Map XYSCR state rows, retaining confidence and velocity conventions."""
        values = np.asarray(states, dtype=float)
        boxes = model.to_boxes(values, include_score=True)
        measurements = model.to_measurements(transform_aabbs(boxes, transform))
        mapped = values.copy()
        mapped[:, :5] = measurements
        points = np.stack((values[:, :2], values[:, :2] + values[:, 5:7]), axis=1)
        warped = transform_points(points, transform).reshape(-1, 2, 2)
        mapped[:, 5:7] = warped[:, 1] - warped[:, 0]
        mapped[:, 7] = values[:, 7] * (measurements[:, 2] / np.maximum(values[:, 2], 1e-6))
        return mapped

    @classmethod
    def _map_camera_states_and_covariances(
        cls,
        states: np.ndarray,
        covariances: np.ndarray,
        transform: np.ndarray,
        model,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the full forward-difference XYSCR Jacobians in one batch."""
        values = np.asarray(states, dtype=float)
        count, dimension = values.shape
        steps = 1e-5 * np.maximum(np.abs(values), 1.0)
        samples = np.repeat(values[:, None, :], dimension + 1, axis=1)
        indices = np.arange(dimension)
        samples[:, indices + 1, indices] += steps
        mapped = cls._map_camera_states(samples.reshape(-1, dimension), transform, model).reshape(
            count, dimension + 1, dimension
        )
        jacobians = ((mapped[:, 1:] - mapped[:, :1]) / steps[:, :, None]).swapaxes(1, 2)
        covariance = jacobians @ np.asarray(covariances, dtype=float) @ jacobians.swapaxes(1, 2)
        covariance = 0.5 * (covariance + covariance.swapaxes(1, 2))
        return mapped[:, 0].copy(), covariance

    def camera_update(self, warp_matrix: np.ndarray) -> None:
        """Move this track's motion and association state to the new frame."""
        self.multi_camera_update([self], warp_matrix)

    @classmethod
    def multi_camera_update(cls, tracks, warp_matrix: np.ndarray) -> None:
        """Transform all XYSCR states and observation histories in batches."""
        transform = np.asarray(warp_matrix, dtype=float)
        if transform.shape not in ((2, 3), (3, 3)):
            raise ValueError(f"Expected a 2x3 affine or 3x3 homography, got {transform.shape}.")
        if not tracks:
            return
        model = tracks[0].motion_model
        centers = np.asarray([np.asarray(track.kf.x).reshape(-1)[:2] for track in tracks])
        transform_filter_histories(
            [track.kf for track in tracks],
            lambda means, covariances: cls._map_camera_states_and_covariances(means, covariances, transform, model),
            lambda measurements: model.to_measurements(
                transform_aabbs(model.to_boxes(measurements, include_score=True), transform)
            ),
        )
        # HybridSORT replaces observation arrays and containers, unlike the
        # in-place observation updates used by OC-SORT and DeepOC-SORT.
        records = []
        for track in tracks:
            for name in ("last_observation", "last_observation_save"):
                value = getattr(track, name)
                if value[-1] >= 0:
                    records.append((track, name, value))
            records.append((track, "observations", track.observations))
            records.append((track, "history", track.history))
        observations = []
        for _, _, value in records:
            if isinstance(value, dict):
                items = value.values()
            elif isinstance(value, deque):
                items = value
            else:
                items = (value,)
            observations.extend(np.asarray(item).reshape(-1) for item in items)
        if observations:
            # History rows may omit confidence; warp geometry in one batch and
            # restore each row's metadata and original shape during scattering.
            geometry = transform_aabbs(np.asarray([row[:4] for row in observations]), transform)
            warped = iter(geometry)
            for track, name, value in records:

                def replace(row, *, preserve_shape=False):
                    shape = np.asarray(row).shape
                    result = np.asarray(row, dtype=float).reshape(-1).copy()
                    result[:4] = next(warped)
                    return result.reshape(shape) if preserve_shape else result

                if isinstance(value, dict):
                    result = {age: replace(row) for age, row in value.items()}
                elif isinstance(value, deque):
                    result = deque((replace(row, preserve_shape=True) for row in value), maxlen=value.maxlen)
                else:
                    result = replace(value)
                setattr(track, name, result)
        transform_directions(
            tracks,
            centers,
            transform,
            attributes=("velocity_lt", "velocity_rt", "velocity_lb", "velocity_rb"),
            invalid_to_zero=True,
        )

    def update(
        self,
        bbox,
        id_feature,
        update_feature: bool = True,
        *,
        cls: Optional[int] = None,
        det_ind: Optional[int] = None,
    ):
        """Correct a box and its appearance using the track's own history."""
        measurement = self._prepare_update(bbox, id_feature, update_feature=update_feature, cls=cls, det_ind=det_ind)
        self.kf.update(measurement)
        self._finish_update(measurement)

    def _prepare_update(
        self,
        bbox,
        id_feature,
        update_feature: bool = True,
        *,
        cls: Optional[int] = None,
        det_ind: Optional[int] = None,
    ) -> np.ndarray | None:
        """Prepare observation and appearance state for a grouped correction."""
        vlt = vrt = vlb = vrb = None
        if bbox is not None:
            if self.last_observation[-1] >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if vlt is not None:
                            vlt += speed_direction_lt(previous_box, bbox)
                            vrt += speed_direction_rt(previous_box, bbox)
                            vlb += speed_direction_lb(previous_box, bbox)
                            vrb += speed_direction_rb(previous_box, bbox)
                        else:
                            vlt = speed_direction_lt(previous_box, bbox)
                            vrt = speed_direction_rt(previous_box, bbox)
                            vlb = speed_direction_lb(previous_box, bbox)
                            vrb = speed_direction_rb(previous_box, bbox)
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity_lt = speed_direction_lt(previous_box, bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, bbox)
                else:
                    self.velocity_lt, self.velocity_rt = vlt, vrt
                    self.velocity_lb, self.velocity_rb = vlb, vrb

            self.last_observation = np.asarray(bbox).copy()
            self.last_observation_save = np.asarray(bbox).copy()
            self.observations[self.age] = np.asarray(bbox).copy()
            self._prune_observations()

            self.time_since_update = 0
            self.history.clear()
            self.hits += 1
            self.hit_streak += 1
            # update metadata
            if cls is not None:
                self.cls = int(cls)
            if det_ind is not None:
                self.det_ind = int(det_ind)

            if update_feature:
                if self.adapfs:
                    self.update_features(id_feature, score=bbox[-1])
                else:
                    self.update_features(id_feature)
            self.confidence_pre = self.conf
            self.conf = float(bbox[-1])
            return self.motion_model.to_measurement(bbox)
        self.confidence_pre = None
        return None

    def _finish_update(self, measurement: np.ndarray | None) -> None:
        """Record corrected geometry after scalar or batched Kalman updates."""
        if measurement is not None:
            self._append_current_history()
            sync_track_meta(self, TrackState.TRACKED)
        else:
            sync_track_meta(self)

    def predict(self, *, dt: float | None = None) -> tuple[np.ndarray, float, float]:
        """Predict box and score over an optional elapsed time interval."""
        self._prepare_prediction(dt=dt)
        self.kf.predict(dt=dt)
        return self._finish_prediction()

    def _prepare_prediction(self, *, dt: float | None = None) -> None:
        """Prevent the predicted area from becoming negative."""
        interval = 1.0 if dt is None else dt
        if (interval * self.kf.x[7] + self.kf.x[2]) <= 0:
            self.kf.x[7] *= 0.0

    def _finish_prediction(self) -> tuple[np.ndarray, float, float]:
        """Advance counters and expose box and confidence predictions."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self.motion_model.to_box(self.kf.x))
        sync_track_meta(self)

        # --- make scalars robustly ---
        x3 = self.kf.x[3, 0] if self.kf.x.ndim == 2 else self.kf.x[3]
        kalman_score = float(np.clip(x3, self.track_thresh, 1.0))

        if not self.confidence_pre:
            simple_score = float(np.clip(self.conf, 0.1, self.track_thresh))
        else:
            simple_score = float(
                np.clip(
                    self.conf - (self.confidence_pre - self.conf),
                    0.1,
                    self.track_thresh,
                )
            )

        return self.history[-1], kalman_score, simple_score
