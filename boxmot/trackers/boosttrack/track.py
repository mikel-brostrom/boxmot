from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from boxmot.trackers.common.appearance import (
    ema_update_embedding,
)
from boxmot.trackers.common.geometry.obb import (
    align_obb_measurement,
    smooth_obb_corners,
    xywha_to_xyxy,
)
from boxmot.trackers.common.motion.cmc.batching import transform_filter_histories
from boxmot.trackers.common.motion.kalman_filters.noise import KalmanNoiseConfig
from boxmot.trackers.common.motion.models import MotionModelKind, create_motion_model
from boxmot.trackers.common.track_state import SortBoxTrack
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta


class KalmanBoxTracker(SortBoxTrack):
    """
    One BoostTrack or OccluBoost identity with Kalman motion and appearance.

    Supports both axis-aligned (default) and oriented (OBB) bounding boxes.
    When ``is_obb=True`` the track stores ``(cx, cy, w, h, angle)`` state and
    expects detections in the layout
    ``(cx, cy, w, h, angle, conf, cls, det_ind)``.
    """

    def __init__(
        self,
        det,
        max_obs,
        emb: Optional[np.ndarray] = None,
        is_obb: bool = False,
        adaptive_kf: bool = False,
        id_allocator: TrackIdAllocator | None = None,
        track_id: int | None = None,
        *,
        noise_config: KalmanNoiseConfig | None = None,
        kalman_type: str | None = None,
        confidence: float | None = None,
    ):
        self.is_obb = bool(is_obb)
        self._assign_sort_id(id_allocator=id_allocator, track_id=track_id)
        if self.is_obb:
            # det = (cx, cy, w, h, angle, conf, cls, det_ind)
            self.conf = float(det[5])
            self.cls = int(det[6])
            self.det_ind = int(det[7])
        else:
            self.conf = float(det[4])
            self.cls = int(det[5])
            self.det_ind = int(det[6])
        self.motion_model = create_motion_model(
            kalman_type or MotionModelKind.XYHR,
            is_obb=self.is_obb,
            adaptive_kf=adaptive_kf,
            cls_id=int(self.cls),
        )
        self.kf = self.motion_model.create_filter(
            self.motion_model.to_measurement(det[:6] if self.is_obb else det[:5], column=False),
            noise_config=noise_config,
        )
        if self.motion_model.kind is not MotionModelKind.XYHR:
            self.motion_model.initialize_filter(
                self.kf,
                self.motion_model.to_measurement(det[:6] if self.is_obb else det[:5]),
                confidence=float(self.conf if confidence is None else confidence),
            )
        self.emb = emb
        self._init_sort_counters(max_obs=max_obs)
        self.history_observations = deque([], maxlen=self.max_obs)
        self._plot_angle = None
        self._append_current_history()
        self._sync_initial_sort_meta()

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7
        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update - 1)

    def update(self, det: np.ndarray, *, confidence: float | None = None) -> None:
        """Correct geometry, optionally using the original detector's trust score."""
        measurement = self._prepare_update(det)
        self.motion_model.update_filter(
            self.kf, measurement, confidence=float(self.conf if confidence is None else confidence)
        )
        self._finish_update(measurement)

    def _prepare_update(self, det: np.ndarray) -> np.ndarray:
        """Align the observation before the scalar or batched correction."""
        self.time_since_update = 0
        self.hit_streak += 1
        if self.is_obb:
            aligned = align_obb_measurement(det[:5], self.get_state()[0])
            measurement = self.motion_model.to_measurement(np.r_[aligned, det[5]], column=False)
            self.conf = float(det[5])
            self.cls = int(det[6])
            self.det_ind = int(det[7])
        else:
            measurement = self.motion_model.to_measurement(det[:5], column=False)
            self.conf = float(det[4])
            self.cls = int(det[5])
            self.det_ind = int(det[6])
        return measurement

    def _finish_update(self, measurement: np.ndarray) -> None:
        """Record the corrected geometry after Kalman arithmetic completes."""
        self._append_current_history()
        sync_track_meta(self, TrackState.TRACKED)

    def _append_current_history(self) -> None:
        """Append corrected display geometry using the shared 4/8-value contract."""
        box = self.get_state()[0].astype(np.float32)
        if self.is_obb:
            box, self._plot_angle = smooth_obb_corners(box, self._plot_angle)
        else:
            box = box[:4]
        self.history_observations.append(np.asarray(box, dtype=np.float32).copy())

    def camera_update(self, transform: np.ndarray) -> None:
        """Transform this track using its AABB or OBB CMC policy."""
        self.multi_camera_update([self], transform)

    @classmethod
    def multi_camera_update(cls, tracks, transform: np.ndarray) -> None:
        """Warp state, velocity, covariance, and filter history as in calibration."""
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape == (2, 3):
            matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
        elif matrix.shape != (3, 3):
            raise ValueError(f"Expected 2×3 or 3×3 matrix, got {matrix.shape}")
        groups = {}
        for track in tracks:
            groups.setdefault((track.motion_model.kind, track.is_obb), []).append(track)
        for group in groups.values():
            model = group[0].motion_model
            transform_filter_histories(
                [track.kf for track in group],
                lambda means, covariances: model.transform_states(means, covariances, matrix),
                lambda measurements: model.transform_measurements(measurements, matrix),
            )

    def predict(self, *, dt: float | None = None) -> np.ndarray:
        """Predict geometry over an optional elapsed time interval."""
        self._prepare_prediction(dt=dt)
        self.motion_model.predict_filter(self.kf, dt=dt)
        return self._finish_prediction()

    def _prepare_prediction(self, *, dt: float | None = None) -> None:
        """Keep native XYHR prediction and guard alternate model box sizes."""
        if self.motion_model.kind is not MotionModelKind.XYHR:
            self.motion_model.prevent_invalid_size(self.kf, dt=dt)

    def _finish_prediction(self) -> np.ndarray:
        """Advance lifecycle counters after scalar or batched prediction."""
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        sync_track_meta(self)
        return self.get_state()

    def get_state(self):
        return self.motion_model.to_box(self.kf.x)

    @property
    def xyxy(self) -> np.ndarray:
        """Return canonical AABB geometry, enclosing native OBB state if needed."""
        state = np.asarray(self.get_state()[0], dtype=float)
        return xywha_to_xyxy(state.reshape(1, 5))[0] if self.is_obb else state

    @property
    def xywha(self) -> np.ndarray:
        """Return the current OBB state as ``[cx, cy, w, h, angle]``.

        Available for both AABB and OBB tracks; AABB tracks return ``angle=0``.
        """
        if self.is_obb:
            return self.get_state()[0].astype(float)
        x1, y1, x2, y2 = self.get_state()[0]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(float(x2 - x1), 1e-4)
        h = max(float(y2 - y1), 1e-4)
        return np.array([cx, cy, w, h, 0.0], dtype=float)

    def update_emb(self, emb, alpha=0.9):
        self.emb = ema_update_embedding(self.emb, emb, alpha=alpha)

    def get_emb(self):
        return self.emb
