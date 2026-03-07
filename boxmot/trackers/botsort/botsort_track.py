from collections import deque

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
        self.history_observations.append(self.xywha if self.is_obb else self.xyxy)

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

    @property
    def xyxy(self):
        """Convert bounding box format to `(min x, min y, max x, max y)`."""
        if self.is_obb:
            cx, cy, w, h, angle = self.xywha
            rect = ((float(cx), float(cy)), (max(float(w), 1e-4), max(float(h), 1e-4)), float(np.degrees(angle)))
            corners = cv2.boxPoints(rect)
            x1, y1 = corners.min(axis=0)
            x2, y2 = corners.max(axis=0)
            return np.array([x1, y1, x2, y2], dtype=np.float32)
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
