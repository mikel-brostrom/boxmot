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
from boxmot.trackers.common.geometry.obb import (
    smooth_obb_corners,
    xywha_to_xyxy,
)
from boxmot.trackers.common.motion.cmc.batching import transform_directions, transform_filter_histories
from boxmot.trackers.common.motion.confidence import ConfidencePrediction
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
    One HybridSORT identity with confidence state and OBB angular motion.
    Stores `cls` and `det_ind` metadata from the most recent matched detection.
    """

    observation_centric = True

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
        is_obb: bool = False,
        cls: int = 0,
        det_ind: int = -1,
        id_allocator: TrackIdAllocator | None = None,
        noise_config: KalmanNoiseConfig | None = None,
        kalman_type: str | None = None,
    ):
        self.is_obb = bool(is_obb)
        self.motion_model = create_motion_model(
            kalman_type or MotionModelKind.XYSCR, is_obb=self.is_obb, max_obs=max_obs, cls_id=int(cls)
        )
        self.kf = self.motion_model.create_filter(noise_config=noise_config)
        self.confidence_filter = None
        if self.motion_model.kind is MotionModelKind.XYSCR:
            measurement_size = self.motion_model.dim_z
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[measurement_size:, measurement_size:] *= 1000.0
            self.kf.P *= 10.0
            self.kf.Q[measurement_size + 3, measurement_size + 3] *= 0.01
            self.kf.Q[measurement_size + 2, measurement_size + 2] *= 0.01
            self.kf.Q[measurement_size:, measurement_size:] *= 0.01
            if self.is_obb:
                self.kf.Q[-1, -1] *= 0.01
            self.kf.x[:measurement_size] = self.motion_model.to_measurement(bbox)
        else:
            self.motion_model.initialize_filter(
                self.kf, self.motion_model.to_measurement(bbox), confidence=float(bbox[-1])
            )
            self.confidence_filter = ConfidencePrediction(float(bbox[-1]), noise_config=noise_config, max_obs=max_obs)

        # tracker state
        self._assign_sort_id(id_allocator=id_allocator)
        self._init_sort_counters(max_obs=max(1, int(max_obs)))
        self.history = deque([], maxlen=self.max_obs)

        # observations
        self.last_observation = np.full(6 if self.is_obb else 5, -1)
        self.last_observation_save = self.last_observation.copy()
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
        self._plot_angle = None
        self._append_current_history()
        self._sync_initial_sort_meta()

    def _append_current_history(self) -> None:
        geometry = self.get_state()[0]
        if self.is_obb:
            geometry, self._plot_angle = smooth_obb_corners(geometry, self._plot_angle)
        self.history_observations.append(np.asarray(geometry, dtype=np.float32).copy())

    def get_state(self) -> np.ndarray:
        """Return current geometry in the configured detection coordinates."""
        return self.motion_model.to_box(self.kf.x)

    @property
    def xywha(self) -> np.ndarray:
        """Expose oriented geometry to shared plotting and lifecycle consumers."""
        if self.is_obb:
            return self.get_state()[0]
        x1, y1, x2, y2 = self.get_state()[0]
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0.0])

    @property
    def xyxy(self) -> np.ndarray:
        """Return the enclosing axis-aligned box for shared consumers."""
        return xywha_to_xyxy(self.get_state())[0] if self.is_obb else self.get_state()[0]

    def _motion_box(self, observation: np.ndarray) -> np.ndarray:
        """Express corner-direction cues as enclosing image-space corners."""
        if self.is_obb:
            return xywha_to_xyxy(np.asarray(observation[:5]).reshape(1, 5))[0]
        return observation

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
        geometry_width = 5 if model.is_obb else 4
        centers = np.asarray([np.asarray(track.kf.x).reshape(-1)[:2] for track in tracks])
        transform_filter_histories(
            [track.kf for track in tracks],
            lambda means, covariances: model.transform_states(means, covariances, transform),
            lambda measurements: model.transform_measurements(measurements, transform),
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
            geometry = model.transform_boxes(np.asarray([row[:geometry_width] for row in observations]), transform)
            warped = iter(geometry)
            for track, name, value in records:

                def replace(row, *, preserve_shape=False):
                    shape = np.asarray(row).shape
                    result = np.asarray(row, dtype=float).reshape(-1).copy()
                    result[:geometry_width] = next(warped)
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
            preserve_magnitude=True,
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
        self.motion_model.update_filter(self.kf, measurement, observation_centric=True, confidence=float(self.conf))
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
            motion_box = self._motion_box(bbox)
            if self.last_observation[-1] >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    if self.age - i - 1 in self.observations:
                        previous_box = self._motion_box(self.observations[self.age - i - 1])
                        if vlt is not None:
                            vlt += speed_direction_lt(previous_box, motion_box)
                            vrt += speed_direction_rt(previous_box, motion_box)
                            vlb += speed_direction_lb(previous_box, motion_box)
                            vrb += speed_direction_rb(previous_box, motion_box)
                        else:
                            vlt = speed_direction_lt(previous_box, motion_box)
                            vrt = speed_direction_rt(previous_box, motion_box)
                            vlb = speed_direction_lb(previous_box, motion_box)
                            vrb = speed_direction_rb(previous_box, motion_box)
                if previous_box is None:
                    previous_box = self._motion_box(self.last_observation)
                    self.velocity_lt = speed_direction_lt(previous_box, motion_box)
                    self.velocity_rt = speed_direction_rt(previous_box, motion_box)
                    self.velocity_lb = speed_direction_lb(previous_box, motion_box)
                    self.velocity_rb = speed_direction_rb(previous_box, motion_box)
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
            if self.confidence_filter is not None:
                self.confidence_filter.update(self.conf)
            return self.motion_model.to_measurement(bbox)
        self.confidence_pre = None
        if self.confidence_filter is not None:
            self.confidence_filter.update(None)
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
        self.motion_model.predict_filter(self.kf, dt=dt, observation_centric=self.observation_centric)
        return self._finish_prediction()

    def _prepare_prediction(self, *, dt: float | None = None) -> None:
        """Predict confidence and prevent nonpositive box-size extrapolation."""
        self.motion_model.prevent_invalid_size(self.kf, dt=dt)
        if self.confidence_filter is not None:
            self.confidence_filter.predict(dt=dt)

    def _finish_prediction(self) -> tuple[np.ndarray, float, float]:
        """Advance counters and expose box and confidence predictions."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(self.motion_model.to_box(self.kf.x))
        sync_track_meta(self)

        # --- make scalars robustly ---
        x3 = (
            self.confidence_filter.value if self.confidence_filter is not None else np.asarray(self.kf.x).reshape(-1)[3]
        )
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
