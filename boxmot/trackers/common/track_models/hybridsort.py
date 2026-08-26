from __future__ import annotations

# Hybrid-SORT-ReID with ECC + ReID (explicit config, BaseTracker-style)
# - Assumes detection input is M x [x1, y1, x2, y2, conf, cls]
# - ECC via shared CMC factory and BaseTracker.apply_cmc(...)
# - ReID via pre-built backend passed as ``reid_model``
# - update(dets, img, embs=None) signature compatible with BoxMOT trackers
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
from boxmot.trackers.common.geometry.obb import transform_points
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.common.track_models.base import SortBoxTrack
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
    ):
        self.motion_model = create_motion_model(MotionModelKind.XYSCR, max_obs=max_obs)
        self.kf = self.motion_model.create_filter()
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
    def _warp_aabb_row(box: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Warp an ``xyxy`` row, preserving any trailing score metadata."""
        values = np.asarray(box, dtype=float).reshape(-1)
        x1, y1, x2, y2 = values[:4]
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
        warped = transform_points(corners, transform)
        result = values.copy()
        result[:4] = [
            warped[:, 0].min(),
            warped[:, 1].min(),
            warped[:, 0].max(),
            warped[:, 1].max(),
        ]
        return result

    def _map_camera_state(self, state: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Map one XYSCR state, including translational and scale velocities."""
        original_shape = np.asarray(state).shape
        values = np.asarray(state, dtype=float).reshape(-1)
        score = float(values[3])
        box = self.motion_model.to_box(values, score=score)[0]
        warped_box = self._warp_aabb_row(box, transform)
        measurement = self.motion_model.to_measurement(warped_box, column=False)

        mapped = values.copy()
        mapped[:5] = measurement

        center = values[:2]
        velocity = values[5:7]
        mapped_points = transform_points(np.stack((center, center + velocity)), transform)
        mapped[5:7] = mapped_points[1] - mapped_points[0]
        area_scale = float(measurement[2]) / max(float(values[2]), 1e-6)
        mapped[7] = values[7] * area_scale
        mapped[8] = values[8]
        return mapped.reshape(original_shape)

    def _map_camera_state_and_covariance(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        transform: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map XYSCR state/covariance with a local numerical Jacobian."""
        values = np.asarray(state, dtype=float).reshape(-1)
        mapped = self._map_camera_state(values, transform).reshape(-1)
        jacobian = np.empty((len(values), len(values)), dtype=float)
        for index in range(len(values)):
            step = 1e-5 * max(abs(float(values[index])), 1.0)
            shifted = values.copy()
            shifted[index] += step
            jacobian[:, index] = (self._map_camera_state(shifted, transform).reshape(-1) - mapped) / step
        mapped_covariance = jacobian @ np.asarray(covariance, dtype=float) @ jacobian.T
        mapped_covariance = 0.5 * (mapped_covariance + mapped_covariance.T)
        return mapped.reshape(np.asarray(state).shape), mapped_covariance

    def _warp_camera_measurement(self, measurement: np.ndarray | None, transform: np.ndarray):
        if measurement is None:
            return None
        values = np.asarray(measurement, dtype=float)
        box = self.motion_model.to_box(values, score=float(values.reshape(-1)[3]))[0]
        warped = self.motion_model.to_measurement(self._warp_aabb_row(box, transform))
        return warped.reshape(values.shape)

    def _warp_direction(self, direction: np.ndarray | None, transform: np.ndarray, center: np.ndarray):
        if direction is None:
            return None
        velocity_xy = np.asarray(direction, dtype=float).reshape(2)[::-1]
        mapped = transform_points(np.stack((center, center + velocity_xy)), transform)
        transformed = mapped[1] - mapped[0]
        norm = float(np.linalg.norm(transformed))
        return (transformed / norm)[::-1] if np.isfinite(norm) and norm > 1e-12 else np.zeros(2)

    def camera_update(self, warp_matrix):
        """Move all AABB motion and association state to the current camera frame."""
        transform = np.asarray(warp_matrix, dtype=float)
        if transform.shape not in ((2, 3), (3, 3)):
            raise ValueError(f"Expected a 2x3 affine or 3x3 homography, got {transform.shape}.")

        source_center = np.asarray(self.kf.x, dtype=float).reshape(-1)[:2].copy()
        self.kf.x, self.kf.P = self._map_camera_state_and_covariance(
            self.kf.x,
            self.kf.P,
            transform,
        )

        if self.last_observation[-1] >= 0:
            self.last_observation = self._warp_aabb_row(self.last_observation, transform)
        if self.last_observation_save[-1] >= 0:
            self.last_observation_save = self._warp_aabb_row(self.last_observation_save, transform)
        self.observations = {
            age: self._warp_aabb_row(observation, transform) for age, observation in self.observations.items()
        }
        self.history = deque(
            (
                self._warp_aabb_row(observation, transform).reshape(np.asarray(observation).shape)
                for observation in self.history
            ),
            maxlen=self.history.maxlen,
        )

        for attr_name in ("velocity_lt", "velocity_rt", "velocity_lb", "velocity_rb"):
            setattr(
                self,
                attr_name,
                self._warp_direction(getattr(self, attr_name), transform, source_center),
            )

        if hasattr(self.kf, "history_obs"):
            self.kf.history_obs = deque(
                (self._warp_camera_measurement(item, transform) for item in self.kf.history_obs),
                maxlen=self.kf.history_obs.maxlen,
            )
        if getattr(self.kf, "last_measurement", None) is not None:
            self.kf.last_measurement = self._warp_camera_measurement(self.kf.last_measurement, transform)
        if not getattr(self.kf, "observed", True) and getattr(self.kf, "attr_saved", None) is not None:
            saved = self.kf.attr_saved
            saved["x"], saved["P"] = self._map_camera_state_and_covariance(
                saved["x"],
                saved["P"],
                transform,
            )
            saved["history_obs"] = deque(
                (self._warp_camera_measurement(item, transform) for item in saved["history_obs"]),
                maxlen=saved["history_obs"].maxlen,
            )
            saved["last_measurement"] = self._warp_camera_measurement(saved["last_measurement"], transform)

    def update(
        self,
        bbox,
        id_feature,
        update_feature: bool = True,
        *,
        cls: Optional[int] = None,
        det_ind: Optional[int] = None,
    ):
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
            self.kf.update(self.motion_model.to_measurement(bbox))
            self._append_current_history()

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
            sync_track_meta(self, TrackState.TRACKED)
        else:
            self.kf.update(bbox)
            self.confidence_pre = None
            sync_track_meta(self)

    def predict(self):
        if (self.kf.x[7] + self.kf.x[2]) <= 0:
            self.kf.x[7] *= 0.0

        self.kf.predict()
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
