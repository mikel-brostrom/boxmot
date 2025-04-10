# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""

from collections import deque  # [hgx0418] deque for reid feature

import numpy as np

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.hybridsort.association import (
    associate_4_points_with_score, associate_4_points_with_score_with_reid,
    cal_score_dif_batch_two_score, embedding_distance, linear_assignment)
from boxmot.trackers.basetracker import BaseTracker


np.random.seed(0)


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h + 1e-6)
    score = bbox[4]
    if score:
        return np.array([x, y, s, score, r]).reshape((5, 1))
    else:
        return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[4])
    h = x[2] / w
    score = x[3]
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0] + bbox1[2]) / 2.0, (bbox1[1] + bbox1[3]) / 2.0
    cx2, cy2 = (bbox2[0] + bbox2[2]) / 2.0, (bbox2[1] + bbox2[3]) / 2.0
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.sqrt((cy2 - cy1)**2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(
        self,
        bbox,
        cls,
        det_ind,
        temp_feat,
        delta_t=3,
        orig=False,
        buffer_size=30,
        longterm_bank_length=30,
        alpha=0.8,
        max_obs=50
    ):     # 'temp_feat' and 'buffer_size' for reid feature
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        # if not orig and not args.kalman_GPR:
        from boxmot.motion.kalman_filters.aabb.xysr_kf import KalmanFilterXYSR
        self.kf = KalmanFilterXYSR(dim_x=9, dim_z=5, max_obs=max_obs)

        # u, v, s, c, r, ~u, ~v, ~s, ~c
        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.kf.x[:5] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.max_obs = max_obs
        self.history = deque([], maxlen=self.max_obs)
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.conf = bbox[4]
        self.cls = cls
        self.det_ind = det_ind
        self.adapfs = False
        
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]),
        let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = deque([], maxlen=self.max_obs)
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None
        self.delta_t = delta_t
        self.confidence_pre = None
        self.confidence = bbox[4]

        # add the following values and functions
        self.smooth_feat = None
        buffer_size = longterm_bank_length
        self.features = deque([], maxlen=buffer_size)
        self.update_features(temp_feat)

        # momentum of embedding update
        self.alpha = alpha

    # ReID. for update embeddings during tracking
    def update_features(self, feat, score=-1):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if self.adapfs:
                assert score > 0
                pre_w = self.alpha * (self.confidence / (self.confidence + score))
                cur_w = (1 - self.alpha) * (score / (self.confidence + score))
                sum_w = pre_w + cur_w
                pre_w = pre_w / sum_w
                cur_w = cur_w / sum_w
                self.smooth_feat = pre_w * self.smooth_feat + cur_w * feat
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def camera_update(self, warp_matrix):
        """
        update 'self.mean' of current tracklet with ecc results.
        Parameters
        ----------
        warp_matrix: warp matrix computed by ECC.
        """
        x1, y1, x2, y2, s = convert_x_to_bbox(self.kf.x)[0]
        x1_, y1_ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_ = warp_matrix @ np.array([x2, y2, 1]).T
        # w, h = x2_ - x1_, y2_ - y1_
        # cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:5] = convert_bbox_to_z([x1_, y1_, x2_, y2_, s])

    def update(self, bbox, cls, det_ind, id_feature, update_feature=True):
        """
        Updates the state vector with observed bbox.
        """
        velocity_lt = None
        velocity_rt = None
        velocity_lb = None
        velocity_rb = None
        if bbox is not None:
            self.conf = bbox[-1]
            self.cls = cls
            self.det_ind = det_ind
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    # dt = self.delta_t - i
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if velocity_lt is not None:
                            velocity_lt += speed_direction_lt(previous_box, bbox)
                            velocity_rt += speed_direction_rt(previous_box, bbox)
                            velocity_lb += speed_direction_lb(previous_box, bbox)
                            velocity_rb += speed_direction_rb(previous_box, bbox)
                        else:
                            velocity_lt = speed_direction_lt(previous_box, bbox)
                            velocity_rt = speed_direction_rt(previous_box, bbox)
                            velocity_lb = speed_direction_lb(previous_box, bbox)
                            velocity_rb = speed_direction_rb(previous_box, bbox)
                        # break
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity_lt = speed_direction_lt(previous_box, bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, bbox)
                else:
                    self.velocity_lt = velocity_lt
                    self.velocity_rt = velocity_rt
                    self.velocity_lb = velocity_lb
                    self.velocity_rb = velocity_rb
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.last_observation_save = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            # add interface for update feature or not
            if update_feature:
                if self.adapfs:
                    self.update_features(id_feature, score=bbox[4])
                else:
                    self.update_features(id_feature)
            self.confidence_pre = self.confidence
            self.confidence = bbox[4]
        else:
            self.kf.update(bbox)
            self.confidence_pre = None

    def predict(self, track_thresh=0.6):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[7] + self.kf.x[2]) <= 0):
            self.kf.x[7] *= 0.0

        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        if not self.confidence_pre:
            return (
                self.history[-1],
                np.clip(self.kf.x[3], track_thresh, 1.0),
                np.clip(self.confidence, 0.1, track_thresh)
            )
        else:
            return (
                self.history[-1],
                np.clip(self.kf.x[3], track_thresh, 1.0),
                np.clip(self.confidence - (self.confidence_pre - self.confidence), 0.1, track_thresh)
            )

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


class HybridSort(BaseTracker):
    """
    HybridSORT Tracker: A tracking algorithm that utilizes a combination of appearance and motion-based tracking
    and temporal consistency models (TCM) for improved tracking accuracy and robustness.

    Args:
        reid_weights (str): Path to the model weights for ReID (Re-Identification).
        device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
        half (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        det_thresh (float): Detection confidence threshold. Detections below this threshold will be ignored in the first association step.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        min_hits (int, optional): Minimum number of hits required to confirm a track.
        iou_threshold (float, optional): Intersection over Union (IoU) threshold for data association.
        delta_t (int, optional): Time delta for velocity estimation in Kalman Filter.
        asso_func (str, optional): Association function to use for data association. Options include "iou" for IoU-based association.
        inertia (float, optional): Weight for inertia in motion modeling. Higher values make tracks less responsive to changes.
        longterm_reid_weight (float, optional): Weight for the long-term ReID feature in the association process.
        TCM_first_step_weight (float, optional): Weight for the Temporal Consistency Model (TCM) in the first association step.
        use_byte (bool, optional): Whether to use BYTE association in the second association step.
    """
    def __init__(self, reid_weights, device, half, det_thresh, per_class=False, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, longterm_reid_weight=0, TCM_first_step_weight=0, use_byte=False):
        super().__init__(max_age=max_age, per_class=per_class, asso_func=asso_func)

        """
        Sets key parameters for SORT
        """
        self.max_age: int = max_age
        self.min_hits: int = min_hits
        self.iou_threshold: float = iou_threshold
        self.per_class: bool = per_class
        self.frame_count: int = 0
        self.det_thresh: float = det_thresh
        self.delta_t: int = delta_t
        self.inertia: float = inertia
        self.use_byte: bool = use_byte
        self.low_thresh: float = 0.1
        self.EG_weight_high_score: float = 1.3
        self.EG_weight_low_score: float = 1.2
        self.TCM_first_step: bool = True
        self.with_longterm_reid: bool = True
        self.with_longterm_reid_correction: bool = True
        self.longterm_reid_weight: float = longterm_reid_weight
        self.TCM_first_step_weight: float = TCM_first_step_weight
        self.high_score_matching_thresh: float = 0.8
        self.longterm_reid_correction_thresh: float = 0.4
        self.longterm_reid_correction_thresh_low: float = 0.4
        self.TCM_byte_step: bool = True
        self.TCM_byte_step_weight: float = 1.0
        self.dataset: str = 'dancetrack'
        self.ECC: bool = False
        KalmanBoxTracker.count = 0

        self.model = ReidAutoBackend(
            weights=reid_weights, device=device, half=half
        ).model
        self.cmc = get_cmc_method('ecc')()

    def camera_update(self, trackers, warp_matrix):
        for tracker in trackers:
            tracker.camera_update(warp_matrix)

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections
        (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        
        self.check_inputs(dets, img)
        
        if dets is None:
            return np.empty((0, 7))

        if self.ECC:
            warp_matrix = self.cmc.apply(img, dets)
            if warp_matrix is not None:
                self.camera_update(self.active_tracks, warp_matrix)

        self.frame_count += 1
        scores = dets[:, 4]
        bboxes = dets[:, :4]

        dets_embs = self.model.get_features(bboxes, img)
        dets0 = np.concatenate((dets, np.expand_dims(scores, axis=-1)), axis=1)
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > self.low_thresh
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        id_feature_keep = dets_embs[remain_inds]  # ID feature of 1st stage matching
        id_feature_second = dets_embs[inds_second]  # ID feature of 2nd stage matching

        trks = np.zeros((len(self.active_tracks), 8))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos, kalman_score, simple_score = self.active_tracks[t].predict()
            trk[:6] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3], kalman_score[0], simple_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities_lt = np.array(
            [trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.active_tracks])
        velocities_rt = np.array(
            [trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.active_tracks])
        velocities_lb = np.array(
            [trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.active_tracks])
        velocities_rb = np.array(
            [trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.active_tracks])
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.active_tracks])

        """
            First round of association
        """
        if self.EG_weight_high_score > 0 and self.TCM_first_step:
            track_features = np.asarray([track.smooth_feat for track in self.active_tracks],
                                        dtype=np.float64)
            emb_dists = embedding_distance(track_features, id_feature_keep).T
            if self.with_longterm_reid or self.with_longterm_reid_correction:
                long_track_features = np.asarray([np.vstack(list(track.features)).mean(0) for track in self.active_tracks],
                                                 dtype=np.float64)
                assert track_features.shape == long_track_features.shape
                long_emb_dists = embedding_distance(long_track_features, id_feature_keep).T
                assert emb_dists.shape == long_emb_dists.shape
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                    dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                    k_observations, self.inertia, self.TCM_first_step_weight, self.asso_func, emb_cost=emb_dists,
                    weights=(1.0, self.EG_weight_high_score), thresh=self.high_score_matching_thresh,
                    long_emb_dists=long_emb_dists, with_longterm_reid=self.with_longterm_reid,
                    longterm_reid_weight=self.longterm_reid_weight,
                    with_longterm_reid_correction=self.with_longterm_reid_correction,
                    longterm_reid_correction_thresh=self.longterm_reid_correction_thresh,
                    dataset=self.dataset)
            else:
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                    dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                    k_observations, self.inertia, self.TCM_first_step_weight, self.asso_func, emb_cost=emb_dists,
                    weights=(1.0, self.EG_weight_high_score), thresh=self.high_score_matching_thresh)
        elif self.TCM_first_step:
            matched, unmatched_dets, unmatched_trks = associate_4_points_with_score(
                dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                k_observations, self.inertia, self.TCM_first_step_weight, self.asso_func)

        # update with id feature
        for m in matched:
            self.active_tracks[m[1]].update(dets[m[0], :], dets0[m[0], 5], dets0[m[0], 6], id_feature_keep[m[0], :])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            u_tracklets = [self.active_tracks[index] for index in unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                if self.TCM_byte_step:
                    iou_left -= np.array(
                        cal_score_dif_batch_two_score(dets_second, u_trks) * self.TCM_byte_step_weight
                    )
                    iou_left_thre = iou_left
                if self.EG_weight_low_score > 0:
                    u_track_features = np.asarray([track.smooth_feat for track in u_tracklets], dtype=np.float64)
                    emb_dists_low_score = embedding_distance(u_track_features, id_feature_second).T
                    matched_indices = linear_assignment(-iou_left + self.EG_weight_low_score * emb_dists_low_score,
                                                        )
                else:
                    matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if self.with_longterm_reid_correction and self.EG_weight_low_score > 0:
                        if (iou_left_thre[m[0], m[1]] < self.iou_threshold) or \
                           (emb_dists_low_score[m[0], m[1]] > self.longterm_reid_correction_thresh_low):
                            print("correction 2nd:", emb_dists_low_score[m[0], m[1]])
                            continue
                    else:
                        if iou_left_thre[m[0], m[1]] < self.iou_threshold:
                            continue
                    self.active_tracks[trk_ind].update(
                        dets_second[det_ind, :],
                        id_feature_second[det_ind, :],
                        update_feature=False
                    )     # [hgx0523] do not update with id feature
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            # left_id_feature = id_feature_keep[unmatched_dets]       # update id feature, if needed
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)

            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.active_tracks[trk_ind].update(
                        dets[det_ind, :],
                        dets0[det_ind, 5],
                        dets0[det_ind, 6],
                        id_feature_keep[det_ind, :],
                        update_feature=False
                    )
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :], dets0[i, 5], dets0[i, 6], id_feature_keep[i, :], delta_t=self.delta_t, max_obs=self.max_obs)
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id + 1], [trk.conf], [trk.cls], [trk.det_ind])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.active_tracks.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 7))
