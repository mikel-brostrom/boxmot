from collections import deque
from typing import Optional

import cv2
import numpy as np

from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.appearance import ema_update_embedding, normalize_embedding
from boxmot.trackers.common.geometry import xyxy2xywh
from boxmot.trackers.common.geometry.obb import xywha_to_corners
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.common.track_models.base import BoxTrack


class TrackState:
    """BoTSORT-local lifecycle states."""

    New = 0
    Tracked = 1
    Lost = 2
    LongLost = 3
    Removed = 4


class BaseTrack(BoxTrack):
    """BoTSORT-local base track with shared lifecycle bookkeeping."""

    state: int = TrackState.New
    lost_state: int = TrackState.Lost
    long_lost_state: int = TrackState.LongLost
    removed_state: int = TrackState.Removed
    tracked_state: int = TrackState.Tracked

    def __init__(self, *args, **kwargs):
        if args or kwargs:
            super().__init__(*args, **kwargs)


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()
    shared_kalman_obb = KalmanFilterXYWH(ndim=5)

    def __init__(
        self,
        det,
        feat=None,
        *,
        id_allocator: TrackIdAllocator,
        feat_history=50,
        max_obs=50,
        is_obb=False,
    ):
        super().__init__(det, id_allocator=id_allocator, max_obs=max_obs, is_obb=is_obb)

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
        return track.xywh

    def _current_aabb_xywh(self) -> np.ndarray:
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
        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4), R)
        t = H[:2, 2]

        for st in stracks:
            mean = R8x8.dot(st.mean)
            mean[:2] += t
            st.mean = mean
            st.covariance = R8x8.dot(st.covariance).dot(R8x8.T)

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

        warp = np.asarray(H, dtype=np.float32)
        linear = warp[:2, :2]
        scale_x, scale_y, _ = cls._affine_components(linear)
        transform = np.eye(10, dtype=np.float32)
        transform[:2, :2] = linear
        transform[5:7, 5:7] = linear
        transform[2, 2] = scale_x
        transform[3, 3] = scale_y
        transform[7, 7] = scale_x
        transform[8, 8] = scale_y

        for st in stracks:
            if st.mean is None or st.covariance is None:
                continue

            reference_box = np.asarray(st.mean[:5], dtype=np.float32)
            warped_corners = cls._warp_points(cls._xywha_to_corners(reference_box), warp)
            warped_box = cls._corners_to_xywha(warped_corners, reference_box)

            warped_mean = st.mean.copy()
            warped_mean[:5] = warped_box
            warped_mean[5:7] = linear @ warped_mean[5:7]
            warped_mean[7] *= scale_x
            warped_mean[8] *= scale_y

            st.mean = warped_mean
            st.covariance = transform @ st.covariance @ transform.T
