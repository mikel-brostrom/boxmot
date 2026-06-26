from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from boxmot.trackers.common.appearance import (
    ema_update_embedding,
)
from boxmot.trackers.common.geometry.obb import align_obb_measurement, normalize_angle
from boxmot.trackers.common.motion import MotionModelKind, create_motion_model
from boxmot.trackers.common.tracking.track import TrackIdAllocator, TrackState, sync_track_meta
from boxmot.trackers.common.track_models.base import SortBoxTrack


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
            self.conf = det[5]
            self.cls = det[6]
            self.det_ind = det[7]
        else:
            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
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
        self._sync_initial_sort_meta()

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7
        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update - 1)

    def update(self, det: np.ndarray):
        self.time_since_update = 0
        self.hit_streak += 1
        self.history_observations.append(self.get_state()[0])
        if self.is_obb:
            aligned = align_obb_measurement(det[:5], self.get_state()[0])
            self.kf.update(self.motion_model.to_measurement(aligned, column=False))
            self.conf = det[5]
            self.cls = det[6]
            self.det_ind = det[7]
        else:
            self.kf.update(self.motion_model.to_measurement(det[:4], column=False))
            self.conf = det[4]
            self.cls = det[5]
            self.det_ind = det[6]
        sync_track_meta(self, TrackState.TRACKED)

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
            cx, cy, w, h, theta = (float(v) for v in self.get_state()[0])
            p = wm @ np.array([cx, cy, 1.0])
            cx_, cy_ = float(p[0]), float(p[1])
            # Approximate isotropic scale and rotation from the linear part
            linear = wm[:2, :2]
            scale = float(np.sqrt(max(abs(np.linalg.det(linear)), 1e-8)))
            rot = float(np.arctan2(linear[1, 0], linear[0, 0]))
            w_ = max(w * scale, 1e-4)
            h_ = max(h * scale, 1e-4)
            self.kf.x[:5] = [cx_, cy_, h_, w_ / h_, normalize_angle(theta + rot)]
            return

        # ——— warp your current bbox —————
        x1, y1, x2, y2 = self.get_state()[0]
        p1 = wm @ np.array([x1, y1, 1.0])
        p2 = wm @ np.array([x2, y2, 1.0])
        x1_, y1_, _ = p1
        x2_, y2_, _ = p2

        # ——— rebuild Kalman state —————
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h, w / h]

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
