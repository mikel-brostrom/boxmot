from collections import deque
from typing import Optional

import cv2
import numpy as np

from boxmot.trackers.common.appearance import ema_update_embedding, normalize_embedding
from boxmot.trackers.common.geometry import xywh2xyxy, xyxy2xywh
from boxmot.trackers.common.geometry.obb import (
    xywha_to_corners,
)
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.motion.models import MotionModelAdapter, MotionModelKind, create_motion_model
from boxmot.trackers.common.track_state import BoxTrack
from boxmot.trackers.common.tracking.track import TrackIdAllocator


class TrackState:
    """BotSort-local lifecycle states."""

    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(BoxTrack):
    """BotSort-local base track with shared lifecycle bookkeeping."""

    state: int = TrackState.New
    lost_state: int = TrackState.Lost
    long_lost_state: int = TrackState.LongLost
    removed_state: int = TrackState.Removed
    tracked_state: int = TrackState.Tracked

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            super().__init__(*args, **kwargs)


class STrack(BaseTrack):
    """One BoT-SORT identity with motion, appearance, and lifecycle state."""

    def __init__(
        self,
        det,
        feat=None,
        *,
        id_allocator: TrackIdAllocator,
        feat_history=50,
        max_obs=50,
        is_obb=False,
        motion_model: MotionModelAdapter | None = None,
    ):
        if motion_model is None:
            motion_model = create_motion_model(MotionModelKind.XYWH, is_obb=is_obb)
        super().__init__(det, id_allocator=id_allocator, max_obs=max_obs, is_obb=is_obb, motion_model=motion_model)

        # Classification history and feature history
        self.cls_hist = []
        self.features = deque(maxlen=feat_history)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9

        # Update initial class and features
        self.update_cls(self.cls, self.conf)
        if feat is not None:
            self.update_features(feat)

    def _init_from_aabb_detection(self, det: np.ndarray) -> None:
        self.xywh = xyxy2xywh(det[:4])
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

    def update_features(self, feat):
        """Normalize and update feature vectors."""
        feat = normalize_embedding(feat)
        self.curr_feat = feat
        self.smooth_feat = ema_update_embedding(self.smooth_feat, feat, alpha=self.alpha)
        self.features.append(feat)

    def update_cls(self, cls, conf):
        """Update class history based on detection confidence."""
        max_freq = 0
        found = False
        for c in self.cls_hist:
            if cls == c[0]:
                c[1] += conf
                found = True
            if c[1] > max_freq:
                max_freq = c[1]
                self.cls = c[0]
        if not found:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    def _measurement_for_update(self, track: "STrack") -> np.ndarray:
        if self.motion_model.kind is MotionModelKind.XYWH:
            return track.xywh
        box = track.xywh if self.is_obb else xywh2xyxy(track.xywh)
        return self.motion_model.to_measurement(np.r_[box, track.conf], column=False)

    def _current_aabb_xywh(self) -> np.ndarray:
        if self.mean is not None and self.motion_model.kind is not MotionModelKind.XYWH:
            return xyxy2xywh(self.motion_model.to_box(self.mean)[0])
        return self.mean[:4].copy() if self.mean is not None else self.xywh.copy()

    def _after_reactivate(self, new_track: "STrack") -> None:
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.update_cls(new_track.cls, new_track.conf)

    def _after_update(self, new_track: "STrack") -> None:
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.update_cls(new_track.cls, new_track.conf)

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Apply geometric motion compensation to multiple tracks."""
        if not stracks:
            return
        if getattr(stracks[0], "is_obb", False):
            return
        tracks = [track for track in stracks if track.mean is not None and track.covariance is not None]
        if not tracks:
            return
        means, covariances = tracks[0].motion_model.transform_states(
            np.asarray([track.mean for track in tracks]),
            np.asarray([track.covariance for track in tracks]),
            H,
        )
        for track, mean, covariance in zip(tracks, means, covariances):
            track.mean, track.covariance = mean, covariance


    @staticmethod
    def _warp_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Warp 2D points with either a 2x3 affine or 3x3 homography matrix."""
        matrix = np.asarray(H, dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if matrix.shape == (2, 3):
            return cv2.transform(pts, matrix).reshape(-1, 2)
        if matrix.shape == (3, 3):
            return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
        raise ValueError("Expected a 2x3 affine or 3x3 homography matrix for CMC.")

    @staticmethod
    def _affine_components(m: np.ndarray) -> tuple[float, float, float]:
        """Approximate scale and rotation terms from an affine linear component."""
        linear = np.asarray(m, dtype=np.float32).reshape(2, 2)
        u, _, vh = np.linalg.svd(linear)
        rot = u @ vh
        if np.linalg.det(rot) < 0:
            u[:, -1] *= -1.0
            rot = u @ vh
        angle = float(np.arctan2(rot[1, 0], rot[0, 0]))
        scale_x = max(float(np.linalg.norm(linear[:, 0])), 1e-6)
        scale_y = max(float(np.linalg.norm(linear[:, 1])), 1e-6)
        return scale_x, scale_y, angle

    @staticmethod
    def _xywha_to_corners(box: np.ndarray) -> np.ndarray:
        """Convert an OBB `(cx, cy, w, h, angle)` box into four corner points."""
        return xywha_to_corners(box).reshape(4, 2).astype(np.float32)

    @classmethod
    def _corners_to_xywha(cls, corners: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit an OBB to warped corners and align it to the previous state."""
        rect = cv2.minAreaRect(np.asarray(corners, dtype=np.float32))
        (cx, cy), (w, h), angle_deg = rect
        xywha = np.array(
            [cx, cy, max(w, 1e-4), max(h, 1e-4), np.deg2rad(angle_deg)],
            dtype=np.float32,
        )
        if reference is None:
            xywha[4] = float(KalmanFilterXYWH._wrap_angle(xywha[4]))
            return xywha
        return KalmanFilterXYWH._align_obb_measurement(xywha, reference).astype(np.float32)

    @classmethod
    def multi_gmc_obb(cls, stracks, H=np.eye(2, 3)):
        """Apply OBB-aware camera motion compensation to multiple tracks."""
        if not stracks:
            return

        tracks = [track for track in stracks if track.mean is not None and track.covariance is not None]
        if not tracks:
            return
        means, covariances = tracks[0].motion_model.transform_states(
            np.asarray([track.mean for track in tracks]),
            np.asarray([track.covariance for track in tracks]),
            H,
        )
        for track, mean, covariance in zip(tracks, means, covariances):
            track.mean, track.covariance = mean, covariance
