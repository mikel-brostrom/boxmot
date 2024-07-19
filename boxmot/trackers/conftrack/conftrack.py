# Raif Olson

import numpy as np
from collections import deque

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc.sof import SOF
from boxmot.motion.kalman_filters.conf_xywh_kf import ConfKalmanFilterXYWH
from boxmot.trackers.conftrack.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from boxmot.utils.ops import xywh2xyxy, xyxy2xywh
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils import PerClassDecorator


class STrack(BaseTrack):
    shared_kalman = ConfKalmanFilterXYWH(0.6, 10)

    def __init__(self, det, feat=None, feat_history=50, max_obs=50):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs=max_obs
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.cls_hist = []  # (cls id, freq)
        self.update_cls(self.cls, self.conf)
        self.history_observations = deque([], maxlen=self.max_obs)

        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, conf):
        if len(self.cls_hist) > 0:
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
        else:
            self.cls_hist.append([cls, conf])
            self.cls = cls

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
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

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh, self.conf
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
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xywh, self.conf
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
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret


class ConfTrack(BaseTracker):
    def __init__(
        self,
        model_weights,
        device,
        fp16,
        per_class=False,
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.2,
        new_track_thresh: float = 0.1,
        tent_conf_thresh: float = 0.7,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        cov_alpha: int = 10,
        conf_thresh: float = 0.6,
        proximity_thresh: float = 0.6,
        appearance_thresh: float = 0.25,
        cmc_method: str = "sof",
        frame_rate=30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        ):
        super().__init__()
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.tent_conf_thresh = tent_conf_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.kalman_filter = ConfKalmanFilterXYWH(conf_thresh, cov_alpha)

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh

        self.with_reid = with_reid
        if self.with_reid:
            rab = ReidAutoBackend(
                weights=model_weights, device=device, half=fp16
            )
            self.model = rab.get_backend()

        self.cmc = SOF()
        self.fuse_first_associate = fuse_first_associate

    @PerClassDecorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img_numpy' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        # Remove bad detections
        confs = dets[:, 4]

        # find second round association detections
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]

        # find first round association detections
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]

        """Extract embeddings """
        # appearance descriptor extraction
        if self.with_reid:
            if embs is not None:
                features_keep = embs
            else:
                # (Ndets x X) [512, 1024, 2048]
                features_keep = self.model.get_features(dets_first[:, 0:4], img)

        if len(dets) > 0:
            """Detections"""
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_keep)]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for (det) in np.array(dets_first)]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        low_tent = []
        high_tent = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                # implement LM from ConfTrack paper
                if track.score < self.tent_conf_thresh:
                  low_tent.append(track)
                else:
                  high_tent.append(track)
            else:
                tracked_stracks.append(track)
        """ Step 2: First association, with high score detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # LM algorithm
        strack_pool = joint_stracks(strack_pool, high_tent)
        strack_u = joint_stracks(strack_pool, low_tent)

        # Fix camera motion
        warp = self.cmc.apply(img, dets_first)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(strack_u, warp)

        # Associate with high conf detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh

        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, track_conf_remain, det_high_remain = linear_assignment(
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

        """ Step 3: Second association, with low score detection boxes"""

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [STrack(det, max_obs=self.max_obs) for
                                 (det) in np.array(dets_second)]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in track_conf_remain
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, track_conf_remain, det_low_remain = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        # implement LM from ConfTrack paper
        """Step 4: low-confidence track matching with high-conf dets"""
        # Associate with high score detection boxes
        stracks_conf_remain = [r_tracked_stracks[i] for i in track_conf_remain]
        ious_dists = iou_distance(low_tent, stracks_conf_remain)
        _, low_tent_valid, _ = linear_assignment(ious_dists, thresh=1-0.7) # want to get rid of tracks with low iou costs
        stracks_low_tent_valid = [low_tent[i] for i in low_tent_valid]
        stracks_det_high_remain = [detections[i] for i in det_high_remain]
        features_det_high_remain = [features_keep[i] for i in det_high_remain]
        C_low_ious = iou_distance(stracks_low_tent_valid, stracks_det_high_remain)
        ious_dists_mask = (C_low_ious > self.proximity_thresh)

        if self.with_reid:
            emb_dists = embedding_distance(stracks_low_tent_valid, stracks_det_high_remain) / 2.0
            raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(C_low_ious, emb_dists)
        else:
            dists = C_low_ious

        matches, track_tent_remain, second_det_high_remain = linear_assignment(dists, thresh=0.3) # need to find this val in ConfTrack paper

        for itracked, idet in matches:
            low_tent[itracked].update(stracks_det_high_remain[idet], self.frame_count)
            activated_starcks.append(low_tent[itracked])


        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        for it in track_tent_remain:
            track = stracks_low_tent_valid[it]
            track.mark_removed()
            removed_stracks.append(track)
        # left over confirmed tracks get lost
        for it in track_conf_remain:
            # print(f"size of stracks_conf_remain: {len(stracks_conf_remain)}")
            # print(f"index: {it}")
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """ Step 5: Init new stracks"""
        # u_detection = [*det_high_remain, *det_low_remain]

        for inew in second_det_high_remain:
            track = stracks_det_high_remain[inew]
            # print(f"init new track {track.track_id} with score :{track.score}")
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            track.update_features(features_det_high_remain[inew])
            activated_starcks.append(track)


        if self.with_reid:
            sdet_remain = [detections_second[i] for i in det_low_remain]
            bboxes = [track.xyxy for track in sdet_remain]
            bboxes = np.array(bboxes)
            # (Ndets x X) [512, 1024, 2048]
            features_det_low = self.model.get_features(bboxes, img)

        for local_idx, inew in enumerate(det_low_remain):
            track = detections_second[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            track.update_features(features_det_low[local_idx])
            activated_starcks.append(track)

        """ Step 6: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_age:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output_stracks = [track for track in self.tracked_stracks]
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
