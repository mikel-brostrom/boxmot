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
    transform_obb_kalman_state,
)
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.common.track_models.base import SortBoxTrack
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta


class KalmanBoxTracker(SortBoxTrack):
    """
    Single object tracker using a Kalman filter.

    Supports both axis-aligned (default) and oriented (OBB) bounding boxes.
    When ``is_obb=True`` the tracker stores ``(cx, cy, w, h, angle)`` state and
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
            MotionModelKind.XYHR,
            is_obb=self.is_obb,
            adaptive_kf=adaptive_kf,
            cls_id=int(self.cls),
        )
        self.kf = self.motion_model.create_filter(
            self.motion_model.to_measurement(det[:5] if self.is_obb else det[:4], column=False)
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

    def update(self, det: np.ndarray):
        self.time_since_update = 0
        self.hit_streak += 1
        if self.is_obb:
            aligned = align_obb_measurement(det[:5], self.get_state()[0])
            self.kf.update(self.motion_model.to_measurement(aligned, column=False))
            self.conf = float(det[5])
            self.cls = int(det[6])
            self.det_ind = int(det[7])
        else:
            self.kf.update(self.motion_model.to_measurement(det[:4], column=False))
            self.conf = float(det[4])
            self.cls = int(det[5])
            self.det_ind = int(det[6])
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

    def camera_update(self, transform: np.ndarray):
        """
        Handle either a 2×3 affine or a 3×3 homography, by
        promoting the 2×3 to 3×3 [ …; 0 0 1 ].

        For OBB tracks, warps the centre and approximates the global affine
        scale on the box dimensions (rotation is folded into the angle); this
        keeps OBB CMC behaviour comparable to the AABB path while avoiding a
        full corner re-fit per track.
        """
        # ——— normalize to 3×3 —————
        wm = np.asarray(transform, dtype=float)
        if wm.shape == (2, 3):
            wm = np.vstack([wm, [0.0, 0.0, 1.0]])
        elif wm.shape != (3, 3):
            raise ValueError(f"Expected 2×3 or 3×3 matrix, got {wm.shape}")

        if self.is_obb:
            self.kf.x, self.kf.covariance = transform_obb_kalman_state(
                self.kf.x,
                self.kf.covariance,
                wm,
                measurement_to_box=lambda values: self.motion_model.to_box(values)[0],
                box_to_measurement=lambda box: self.motion_model.to_measurement(box, column=False),
                velocity_measurement_indices=(0, 1, 2, 3, 4),
            )
            return

        # Preserve the AABB CMC contract used to tune BoostTrack/OccluBoost:
        # correct only the measurement mean by warping its diagonal endpoints.
        # Transforming the full enclosure, velocity, or covariance changes the
        # subsequent association geometry and fragments established identities.
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = wm @ np.array([x1, y1, 1.0])
        x2_, y2_, _ = wm @ np.array([x2, y2, 1.0])
        width, height = x2_ - x1_, y2_ - y1_
        self.kf.x[:4] = [x1_ + (width / 2), y1_ + (height / 2), height, width / height]

    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        sync_track_meta(self)
        return self.get_state()

    def get_state(self):
        return self.motion_model.to_box(self.kf.x)

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
