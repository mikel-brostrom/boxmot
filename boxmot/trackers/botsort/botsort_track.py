import numpy as np
from collections import deque

from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, det, feat=None, feat_history=50, max_obs=50):
        # Initialize detection parameters
        self.xywh = xyxy2xywh(det[:4])  # Convert to (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
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
            mean_state[6:8] = 0  # Reset velocities
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Perform batch prediction for multiple tracks."""
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6:8] = 0  # Reset velocities
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for st, mean, cov in zip(stracks, multi_mean, multi_covariance):
            st.mean, st.covariance = mean, cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Apply geometric motion compensation to multiple tracks."""
        if not stracks:
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
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
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
        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xywh)
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
        ret = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
        return xywh2xyxy(ret)
