# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from collections import deque

import numpy as np

from boxmot.motion.kalman_filters.aabb.xywh_kf import AMSKalmanFilterXYWH
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.bytetrack.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from boxmot.utils.ops import tlwh2xyah, xywh2tlwh, xywh2xyxy, xyxy2xywh


class STrack(BaseTrack):
    # Shared instance *only* for the write-free multi_predict().
    shared_predictor = AMSKalmanFilterXYWH(alpha0=0.20, theta_v=0.20)

    # ─────────────────────────────── init ────────────────────────────────
    def __init__(self, det, feat=None, feat_history: int = 50, max_obs: int = 50):
        """
        det  = [x1, y1, x2, y2, conf, cls, det_index]
        """
        # detection → internal xywh
        self.xywh = xyxy2xywh(det[:4]).astype(float)
        self.conf, self.cls, self.det_ind = det[4:7]

        # per-track KF (created later in activate)
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        # misc state
        self.is_activated = False
        self.tracklet_len = 0
        self.max_obs = max_obs

        # history buffers
        self.cls_hist: list[list[float]] = []              # [[cls, cum_conf], …]
        self.history_observations = deque(maxlen=max_obs)
        self.features = deque(maxlen=feat_history)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9                                   # EMA factor for feats

        # initialise class / features
        self.update_cls(self.cls, self.conf)
        if feat is not None:
            self.update_features(feat)

    # ───────────────────── appearance & class helpers ────────────────────
    def update_features(self, feat: np.ndarray):
        """L2-normalise + EMA‐smooth appearance feature."""
        feat = feat / np.linalg.norm(feat)
        self.curr_feat = feat
        self.smooth_feat = (
            feat if self.smooth_feat is None
            else (self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat)
        )
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        self.features.append(feat)

    def update_cls(self, cls: float, conf: float):
        """Maintain confidence-weighted majority class."""
        for item in self.cls_hist:
            if cls == item[0]:
                item[1] += conf
                break
        else:
            self.cls_hist.append([cls, conf])

        # pick class with max cumulative confidence
        self.cls = max(self.cls_hist, key=lambda x: x[1])[0]

    # ───────────────────────────── KF predict ────────────────────────────
    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6:8] = 0.0                          # zero velocities
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks: list["STrack"]):
        """Batch-predict without touching each track’s private history."""
        if not stracks:
            return
        means  = np.asarray([st.mean.copy()        for st in stracks])
        covars = np.asarray([st.covariance.copy()  for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                means[i, 6:8] = 0.0

        means, covars = STrack.shared_predictor.multi_predict(means, covars)
        for st, m, P in zip(stracks, means, covars):
            st.mean, st.covariance = m, P

    # ───────────────────────── GMC helper (optional) ─────────────────────
    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if not stracks:
            return
        R, t = H[:2, :2], H[:2, 2]
        R8 = np.kron(np.eye(4), R)
        for st in stracks:
            st.mean = R8 @ st.mean
            st.mean[:2] += t
            st.covariance = R8 @ st.covariance @ R8.T

    # ───────────────────────── track life-cycle ──────────────────────────
    def activate(self, frame_id: int):
        """Create a private AMS-KF and start a new track."""
        self.kalman_filter = AMSKalmanFilterXYWH(alpha0=0.20, theta_v=0.20)
        self.id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xywh)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = (frame_id == 1)
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh,
            confidence=new_track.conf
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated, self.frame_id = True, frame_id
        if new_id:
            self.id = self.next_id()
        self.conf, self.cls, self.det_ind = new_track.conf, new_track.cls, new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    def update(self, new_track: "STrack", frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh,
            confidence=new_track.conf
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.conf, self.cls, self.det_ind = new_track.conf, new_track.cls, new_track.det_ind
        self.update_cls(new_track.cls, new_track.conf)

    # ───────────────────── convenience accessors ─────────────────────────
    @property
    def xyxy(self):
        """Return box as [x1, y1, x2, y2] from current mean."""
        box = self.mean[:4].copy() if self.mean is not None else self.xywh.copy()
        return xywh2xyxy(box)


class ByteTrack(BaseTracker):
    """
    BYTETracker: A tracking algorithm based on ByteTrack, which utilizes motion-based tracking.

    Args:
        min_conf (float, optional): Threshold for detection confidence. Detections below this threshold are discarded.
        track_thresh (float, optional): Threshold for detection confidence. Detections above this threshold are considered for tracking in the first association round.
        match_thresh (float, optional): Threshold for the matching step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
        track_buffer (int, optional): Number of frames to keep a track alive after it was last detected. A longer buffer allows for more robust tracking but may increase identity switches.
        frame_rate (int, optional): Frame rate of the video being processed. Used to scale the track buffer size.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
    """

    def __init__(
        self,
        min_conf: float = 0.1,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 25,
        frame_rate: int = 30,
        per_class: bool = False,
    ):
        super().__init__(per_class=per_class)
        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_buffer = track_buffer

        self.per_class = per_class
        self.min_conf = min_conf
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = AMSKalmanFilterXYWH()

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None
    ) -> np.ndarray:

        self.check_inputs(dets, img)

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        confs = dets[:, 4]

        remain_inds = confs > self.track_thresh

        inds_low = confs > self.min_conf
        inds_high = confs < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [STrack(det, max_obs=self.max_obs) for det in dets]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(det_second, max_obs=self.max_obs) for det_second in dets_second
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.det_thresh:
                continue
            track.activate(self.frame_count)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )
        # get confs of lost tracks
        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)
        outputs = np.asarray(outputs)
        return outputs


# id, class_id, conf


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
