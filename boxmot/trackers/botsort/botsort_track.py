from collections import deque
from typing import Optional

import cv2
import numpy as np

from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()
    shared_kalman_obb = KalmanFilterXYWH(ndim=5)

    def __init__(self, det, feat=None, feat_history=50, max_obs=50, is_obb=False):
        # Initialize detection parameters
        self.is_obb = is_obb
        det = np.asarray(det, dtype=np.float32)
        if self.is_obb:
            self._init_from_obb_detection(det)
        else:
            self._init_from_aabb_detection(det)
        self.max_obs = max_obs

        # Kalman filter and tracking state
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0

        # Classification history and feature history
        self.cls_hist = []
        self.history_observations = deque(maxlen=self.max_obs)
        self._plot_angle = None
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

    def _init_from_obb_detection(self, det: np.ndarray) -> None:
        self.xywh = det[:5].copy()
        self.conf = det[5]
        self.cls = det[6]
        self.det_ind = det[7]

    def update_features(self, feat):
        """Normalize and update feature vectors."""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
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

    def predict(self):
        """Predict the next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            if self.is_obb:
                mean_state[7:10] = 0  # Reset size/angle velocities
            else:
                mean_state[6:8] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        """Perform batch prediction for multiple tracks."""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        is_obb = getattr(stracks[0], "is_obb", False)
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                if is_obb:
                    multi_mean[i][7:10] = 0
                else:
                    multi_mean[i][6:8] = 0  # Reset velocities
        kalman = STrack.shared_kalman_obb if is_obb else STrack.shared_kalman
        multi_mean, multi_covariance = kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for st, mean, cov in zip(stracks, multi_mean, multi_covariance):
            st.mean, st.covariance = mean, cov

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
        cx, cy, w, h, angle = np.asarray(box, dtype=np.float32)
        rect = (
            (float(cx), float(cy)),
            (max(float(w), 1e-4), max(float(h), 1e-4)),
            float(np.degrees(angle)),
        )
        return cv2.boxPoints(rect).astype(np.float32)

    @classmethod
    def obb_to_xyxy(cls, box: np.ndarray) -> np.ndarray:
        """Return the enclosing AABB for an OBB box."""
        corners = cls._xywha_to_corners(box)
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @classmethod
    def _corners_to_xywha(
        cls, corners: np.ndarray, reference: Optional[np.ndarray] = None
    ) -> np.ndarray:
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
        return KalmanFilterXYWH._align_obb_measurement(xywha, reference).astype(
            np.float32
        )

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
            warped_corners = cls._warp_points(
                cls._xywha_to_corners(reference_box), warp
            )
            warped_box = cls._corners_to_xywha(warped_corners, reference_box)

            warped_mean = st.mean.copy()
            warped_mean[:5] = warped_box
            warped_mean[5:7] = linear @ warped_mean[5:7]
            warped_mean[7] *= scale_x
            warped_mean[8] *= scale_y

            st.mean = warped_mean
            st.covariance = transform @ st.covariance @ transform.T

    def activate(self, kalman_filter, frame_id):
        """Activate a new track."""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a track with a new detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track, frame_id):
        """Update the current track with a matched detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        if not self.is_obb:
            self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)
        if self.is_obb:
            self.history_observations.append(self._state_obb_for_plot())

    @staticmethod
    def _wrap_pi_periodic(delta: float) -> float:
        return float((delta + (np.pi / 2.0)) % np.pi - (np.pi / 2.0))

    def _state_obb_for_plot(self) -> np.ndarray:
        """Return post-update OBB state as 4 corners with state-only angle smoothing."""
        box = self.xywha.copy()
        if box[3] > box[2]:
            box[2], box[3] = box[3], box[2]
            box[4] = box[4] + (np.pi / 2.0)
        target = float((box[4] + np.pi) % (2.0 * np.pi) - np.pi)
        if self._plot_angle is None:
            self._plot_angle = target
        else:
            self._plot_angle = self._plot_angle + self._wrap_pi_periodic(
                target - self._plot_angle
            )
        box[4] = self._plot_angle
        rect = (
            (float(box[0]), float(box[1])),
            (max(float(box[2]), 1e-4), max(float(box[3]), 1e-4)),
            float(np.degrees(box[4])),
        )
        corners = cv2.boxPoints(rect).reshape(-1)
        return np.asarray(corners, dtype=np.float32)

    @property
    def xyxy(self):
        """Convert bounding box format to `(min x, min y, max x, max y)`."""
        if self.is_obb:
            return self.obb_to_xyxy(self.xywha)
        ret = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
        return xywh2xyxy(ret)

    @property
    def xywha(self):
        """Return oriented bbox format `(cx, cy, w, h, angle)` when OBB mode is enabled."""
        if not self.is_obb:
            xywh = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
            return np.array([xywh[0], xywh[1], xywh[2], xywh[3], 0.0], dtype=np.float32)
        ret = self.mean[:5].copy() if self.mean is not None else self.xywh.copy()
        return np.asarray(ret, dtype=np.float32)
