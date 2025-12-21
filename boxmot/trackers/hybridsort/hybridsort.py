# Hybrid-SORT-ReID with ECC + ReID (explicit config, BaseTracker-style)
# - Assumes detection input is M x [x1, y1, x2, y2, conf, cls]
# - ECC via get_cmc_method(...).apply(img, dets)
# - ReID via ReidAutoBackend(weights, device, half).model.get_features(...)
# - update(dets, img, embs=None) signature compatible with BoxMOT trackers
# - Emits rows: [x1,y1,x2,y2, track_id, conf, cls, det_ind]
# - Safe with COCO 80 classes; preserves det_ind; guards out-of-range indices

from collections import deque
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from boxmot.motion.cmc import get_cmc_method
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.trackers.basetracker import BaseTracker
# Keep your original association functions:
from boxmot.trackers.hybridsort.association import (
    associate_4_points_with_score, associate_4_points_with_score_with_reid,
    cal_score_dif_batch_two_score, ciou_batch, ct_dist, diou_batch,
    embedding_distance, giou_batch, hmiou, iou_batch, linear_assignment)


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
    # [x1,y1,x2,y2,score] -> [x,y,s,score,r] or [x,y,s,r]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    score = bbox[4]
    if score:
        return np.array([x, y, s, score, r]).reshape((5, 1))
    else:
        return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    # [x,y,s,r, ...] -> [x1,y1,x2,y2,(score_from_state)]
    w = np.sqrt(x[2] * x[4])
    h = x[2] / w
    score_val = x[3]
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score_val]).reshape((1, 5))


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
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
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
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6
    return speed / norm


class KalmanBoxTracker(object):
    """
    Single-object tracker with 9D custom KF (u,v,s,c,r, du,dv,ds,dc) by default.
    Stores `cls` and `det_ind` metadata from the most recent matched detection.
    """

    count = 0

    def __init__(
        self,
        bbox,
        temp_feat,
        *,
        delta_t: int = 3,
        use_custom_kf: bool = True,
        longterm_bank_length: int = 30,
        alpha: float = 0.9,
        adapfs: bool = False,
        track_thresh: float = 0.5,
        cls: int = 0,
        det_ind: int = -1,
    ):
        if use_custom_kf:
            from .kalmanfilter_score_new import \
                KalmanFilterNew_score_new as KalmanFilter_score_new
            self.kf = KalmanFilter_score_new(dim_x=9, dim_z=5)
            self.kf.F = np.array(
                [
                    [1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[5:, 5:] *= 1000.0
            self.kf.P *= 10.0
            self.kf.Q[-1, -1] *= 0.01
            self.kf.Q[-2, -2] *= 0.01
            self.kf.Q[5:, 5:] *= 0.01
            self.kf.x[:5] = convert_bbox_to_z(bbox)
        else:
            from filterpy.kalman import KalmanFilter
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
            self.kf.F = np.array(
                [
                    [1, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 1],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                ]
            )
            self.kf.H = np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                ]
            )
            self.kf.R[2:, 2:] *= 10.0
            self.kf.P[4:, 4:] *= 1000.0
            self.kf.P *= 10.0
            self.kf.x[:4] = convert_bbox_to_z(bbox)

        # tracker state
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[np.ndarray] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        # observations
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations: List[np.ndarray] = []

        # velocity aids
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None

        # parameters
        self.delta_t = int(delta_t)
        self.confidence_pre = None
        self.conf = float(bbox[-1])

        # ReID buffers
        self.smooth_feat = None
        self.features = deque([], maxlen=int(longterm_bank_length))
        self.alpha = float(alpha)
        self.adapfs = bool(adapfs)
        self.track_thresh = float(track_thresh)

        # metadata
        self.cls = int(cls)
        self.det_ind = int(det_ind)

        # first feature update
        self.update_features(temp_feat)

    def update_features(self, feat, score: float = -1.0):
        feat = feat.astype(np.float32)
        n = np.linalg.norm(feat) + 1e-12
        feat = feat / n
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if self.adapfs:
                assert score > 0, "score must be > 0 when adapfs=True"
                pre_w = self.alpha * (self.conf / (self.conf + score))
                cur_w = (1.0 - self.alpha) * (score / (self.conf + score))
                s = pre_w + cur_w
                pre_w /= s
                cur_w /= s
                self.smooth_feat = pre_w * self.smooth_feat + cur_w * feat
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1.0 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-12)

    def camera_update(self, warp_matrix):
        # get box + score from KF state
        x1, y1, x2, y2, score = convert_x_to_bbox(self.kf.x, score=True)[0]

        M = np.asarray(warp_matrix, dtype=float)
        # normalize to 3x3 homogeneous matrix
        if M.shape == (2, 3):
            M = np.vstack([M, [0.0, 0.0, 1.0]])
        elif M.shape != (3, 3):
            M = np.eye(3, dtype=float)

        # transform corners in homogeneous coords
        p1 = (M @ np.array([x1, y1, 1.0], dtype=float)).ravel()
        p2 = (M @ np.array([x2, y2, 1.0], dtype=float)).ravel()

        # homogeneous divide
        w1 = p1[2] if abs(p1[2]) > 1e-12 else 1.0
        w2 = p2[2] if abs(p2[2]) > 1e-12 else 1.0
        x1_, y1_ = p1[0] / w1, p1[1] / w1
        x2_, y2_ = p2[0] / w2, p2[1] / w2

        # write back to KF (keep score)
        self.kf.x[:5] = convert_bbox_to_z([x1_, y1_, x2_, y2_, float(score)])

    def update(self, bbox, id_feature, update_feature: bool = True, *, cls: Optional[int] = None, det_ind: Optional[int] = None):
        vlt = vrt = vlb = vrb = None
        if bbox is not None:
            if self.last_observation.sum() >= 0:
                previous_box = None
                for i in range(self.delta_t):
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if vlt is not None:
                            vlt += speed_direction_lt(previous_box, bbox)
                            vrt += speed_direction_rt(previous_box, bbox)
                            vlb += speed_direction_lb(previous_box, bbox)
                            vrb += speed_direction_rb(previous_box, bbox)
                        else:
                            vlt = speed_direction_lt(previous_box, bbox)
                            vrt = speed_direction_rt(previous_box, bbox)
                            vlb = speed_direction_lb(previous_box, bbox)
                            vrb = speed_direction_rb(previous_box, bbox)
                if previous_box is None:
                    previous_box = self.last_observation
                    self.velocity_lt = speed_direction_lt(previous_box, bbox)
                    self.velocity_rt = speed_direction_rt(previous_box, bbox)
                    self.velocity_lb = speed_direction_lb(previous_box, bbox)
                    self.velocity_rb = speed_direction_rb(previous_box, bbox)
                else:
                    self.velocity_lt, self.velocity_rt = vlt, vrt
                    self.velocity_lb, self.velocity_rb = vlb, vrb

            self.last_observation = bbox
            self.last_observation_save = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))

            # update metadata
            if cls is not None:
                self.cls = int(cls)
            if det_ind is not None:
                self.det_ind = int(det_ind)

            if update_feature:
                if self.adapfs:
                    self.update_features(id_feature, score=bbox[-1])
                else:
                    self.update_features(id_feature)
            self.confidence_pre = self.conf
            self.conf = float(bbox[-1])
        else:
            self.kf.update(bbox)
            self.confidence_pre = None

    def predict(self):
        if (self.kf.x[7] + self.kf.x[2]) <= 0:
            self.kf.x[7] *= 0.0

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.history.append(convert_x_to_bbox(self.kf.x))

        # --- make scalars robustly ---
        x3 = self.kf.x[3, 0] if self.kf.x.ndim == 2 else self.kf.x[3]
        kalman_score = float(np.clip(x3, self.track_thresh, 1.0))

        if not self.confidence_pre:
            simple_score = float(np.clip(self.conf, 0.1, self.track_thresh))
        else:
            simple_score = float(np.clip(
                self.conf - (self.confidence_pre - self.conf),
                0.1,
                self.track_thresh,
            ))

        return self.history[-1], kalman_score, simple_score


ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
    "hmiou": hmiou,
}


class HybridSort(BaseTracker):
    """
    Hybrid SORT + ReID with ECC CMC

    - Explicit configuration only (no self.args)
    - ReID model and ECC setup like BotSort
    - BaseTracker API: update(dets, img, embs=None) -> np.ndarray [[x1,y1,x2,y2,track_id,conf,cls,det_ind], ...]
    - Now outputs real cls & det_ind and guards det_ind bounds
    - Assumes detection input is [x1,y1,x2,y2,conf,cls]
    """

    def __init__(
        self,
        # ReID & CMC
        reid_weights: Optional[Union[Path, str]],
        device: torch.device,
        half: bool,
        cmc_method: str = "ecc",
        with_reid: bool = True,

        # Hybrid-SORT specific
        low_thresh: float = 0.1,
        delta_t: int = 3,
        inertia: float = 0.05,
        use_byte: bool = True,

        # KF / ReID
        use_custom_kf: bool = True,
        longterm_bank_length: int = 30,
        alpha: float = 0.9,
        adapfs: bool = False,
        track_thresh: float = 0.5,

        # Embedding-guided association
        EG_weight_high_score: float = 4.6,
        EG_weight_low_score: float = 1.3,

        # Two-step toggles / thresholds
        TCM_first_step: bool = True,
        TCM_byte_step: bool = True,
        TCM_byte_step_weight: float = 1.0,
        high_score_matching_thresh: float = 0.7,

        # Long-term reid
        with_longterm_reid: bool = True,
        longterm_reid_weight: float = 0.0,
        with_longterm_reid_correction: bool = True,
        longterm_reid_correction_thresh: float = 0.4,
        longterm_reid_correction_thresh_low: float = 0.4,

        # misc
        dataset: str = "",
        **kwargs,  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='HybridSort', **kwargs)

        # store core knobs
        self.low_thresh = float(low_thresh)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.use_byte = bool(use_byte)

        self.use_custom_kf = bool(use_custom_kf)
        self.longterm_bank_length = int(longterm_bank_length)
        self.alpha = float(alpha)
        self.adapfs = bool(adapfs)
        self.track_thresh = float(track_thresh)

        self.EG_weight_high_score = float(EG_weight_high_score)
        self.EG_weight_low_score = float(EG_weight_low_score)
        self.TCM_first_step = bool(TCM_first_step)
        self.TCM_byte_step = bool(TCM_byte_step)
        self.TCM_byte_step_weight = float(TCM_byte_step_weight)
        self.high_score_matching_thresh = float(high_score_matching_thresh)

        self.with_longterm_reid = bool(with_longterm_reid)
        self.longterm_reid_weight = float(longterm_reid_weight)
        self.with_longterm_reid_correction = bool(with_longterm_reid_correction)
        self.longterm_reid_correction_thresh = float(longterm_reid_correction_thresh)
        self.longterm_reid_correction_thresh_low = float(longterm_reid_correction_thresh_low)
        self.dataset = str(dataset)

        # ReID module (BotSort-style)
        self.with_reid = bool(with_reid)
        self.model = None
        if self.with_reid and reid_weights is not None:
            self.model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model

        # ECC CMC (BotSort-style)
        self.cmc = get_cmc_method(cmc_method)()

        # container
        self.active_tracks: List[KalmanBoxTracker] = []
        KalmanBoxTracker.count = 0

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        """
        dets: ndarray [N,6] -> [x1,y1,x2,y2,conf,cls]
        img: HxWxC image
        embs: optional [N,D] appearance features. If None and with_reid=True, we extract features for provided dets.
        Returns: ndarray [M,8]: [x1,y1,x2,y2,track_id,conf,cls,det_ind]
        """
        self.check_inputs(dets, img, embs)
        self.frame_count += 1

        # --- parse inputs: detections are [x1,y1,x2,y2,conf,cls] ---
        n_dets_full = int(dets.shape[0])

        # core boxes + conf
        dets_5 = dets[:, :5].copy() if n_dets_full else dets.reshape(0, 5)
        confs = dets_5[:, 4] if n_dets_full else np.array([])

        # classes (int)
        dets_cls = dets[:, 5].astype(int) if n_dets_full else np.array([], dtype=int)

        # stable det_ind column for downstream logic and output
        dets_ind = np.arange(n_dets_full, dtype=int)

        # helper guards
        def _safe_detind(x: int, n: int) -> int:
            xi = int(x)
            return xi if 0 <= xi < n else -1

        def _safe_cls(x: int, n: int) -> int:
            xi = int(x)
            return xi if 0 <= xi < n else xi  # change to -1 if you prefer strictness

        # convenience view carrying det_ind in col 6th position
        dets_idx = np.hstack([dets_5, dets_ind.reshape(-1, 1)]) if n_dets_full else dets.reshape(0, 6)

        # ECC: compute warp using all current detections (BotSort pattern)
        warp = self.cmc.apply(img, dets_idx) if len(dets_idx) else np.eye(3)

        # Apply camera motion compensation to all active tracks
        for tr in self.active_tracks:
            tr.camera_update(warp)

        # ReID: get features if not provided
        if self.with_reid:
            if embs is None and len(dets_idx):
                # ReID features extracted on [x1,y1,x2,y2] boxes
                embs = self.model.get_features(dets_idx[:, 0:4], img)
            elif embs is None:
                embs = np.zeros((0, 128), dtype=np.float32)  # safe shape

        # FIRST/SECOND stage split (Hybrid semantics)
        inds_low = confs > self.low_thresh
        inds_high = confs < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = dets_idx[inds_second]         # low-conf for BYTE
        remain_inds = confs > self.det_thresh
        dets_keep = dets_idx[remain_inds]

        # NEW: classes aligned to the two sets
        cls_keep = dets_cls[remain_inds]
        cls_second = dets_cls[inds_second]

        # slice embeddings accordingly (if present)
        if embs is None or len(embs) == 0:
            id_feature_keep = np.zeros((len(dets_keep), 1), dtype=np.float32)
            id_feature_second = np.zeros((len(dets_second), 1), dtype=np.float32)
        else:
            id_feature_keep = embs[remain_inds]
            id_feature_second = embs[inds_second]

        # Build dets arrays used by original hybrid code (no det_ind in the math)
        dets_first = dets_keep[:, :5]
        dets_low = dets_second[:, :5]

        # carry det_ind arrays aligned with above
        det_inds_keep = dets_keep[:, 5].astype(int) if len(dets_keep) else np.array([], dtype=int)
        det_inds_second = dets_second[:, 5].astype(int) if len(dets_second) else np.array([], dtype=int)

        # ---- Predict step for existing tracks
        trks = np.zeros((len(self.active_tracks), 6))
        to_del = []
        for t in range(len(trks)):
            pos, kal_score, simple_score = self.active_tracks[t].predict()
            x1, y1, x2, y2 = pos[0].tolist()
            trks[t] = [x1, y1, x2, y2, kal_score, simple_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        # Prepare motion cues
        velocities_lt = np.array([t.velocity_lt if t.velocity_lt is not None else np.array((0, 0)) for t in self.active_tracks])
        velocities_rt = np.array([t.velocity_rt if t.velocity_rt is not None else np.array((0, 0)) for t in self.active_tracks])
        velocities_lb = np.array([t.velocity_lb if t.velocity_lb is not None else np.array((0, 0)) for t in self.active_tracks])
        velocities_rb = np.array([t.velocity_rb if t.velocity_rb is not None else np.array((0, 0)) for t in self.active_tracks])
        last_boxes = np.array([t.last_observation for t in self.active_tracks])
        k_observations = np.array([k_previous_obs(t.observations, t.age, self.delta_t) for t in self.active_tracks])

        # ===== First association (optionally embedding-guided)
        if self.EG_weight_high_score > 0 and self.TCM_first_step and len(dets_first) and len(trks):
            track_features = np.asarray([t.smooth_feat for t in self.active_tracks], dtype=float)
            emb_dists = embedding_distance(track_features, id_feature_keep).T

            long_emb_dists = None
            if self.with_longterm_reid or self.with_longterm_reid_correction:
                long_track_features = np.asarray(
                    [np.vstack(list(t.features)).mean(0) if len(t.features) else t.smooth_feat for t in self.active_tracks],
                    dtype=float,
                )
                long_emb_dists = embedding_distance(long_track_features, id_feature_keep).T

            matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                dets_first,
                trks,
                self.iou_threshold,
                velocities_lt,
                velocities_rt,
                velocities_lb,
                velocities_rb,
                k_observations,
                self.inertia,
                ASSO_FUNCS[self.asso_func_name],  # from BaseTracker
                emb_cost=emb_dists,
                weights=(1.0, self.EG_weight_high_score),
                thresh=self.high_score_matching_thresh,
                long_emb_dists=long_emb_dists,
                with_longterm_reid=self.with_longterm_reid,
                longterm_reid_weight=self.longterm_reid_weight,
                with_longterm_reid_correction=self.with_longterm_reid_correction,
                longterm_reid_correction_thresh=self.longterm_reid_correction_thresh,
                dataset=self.per_class and "perclass" or self.dataset,
            )
        elif self.TCM_first_step and len(dets_first) and len(trks):
            matched, unmatched_dets, unmatched_trks = associate_4_points_with_score(
                dets_first,
                trks,
                self.iou_threshold,
                velocities_lt,
                velocities_rt,
                velocities_lb,
                velocities_rb,
                k_observations,
                self.inertia,
                ASSO_FUNCS[self.asso_func_name],
            )
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(dets_first))
            unmatched_trks = np.arange(len(trks))

        # Update matched (update features here)  —— pass cls & det_ind (safe)
        for m in matched:
            det_i = m[0]
            self.active_tracks[m[1]].update(
                dets_first[det_i, :],
                id_feature_keep[det_i, :],
                cls=_safe_cls(cls_keep[det_i], self.nr_classes),
                det_ind=_safe_detind(det_inds_keep[det_i], n_dets_full),
            )

        # ===== BYTE / low-score association (optional)
        if self.use_byte and len(dets_low) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = np.array(ASSO_FUNCS[self.asso_func_name](dets_low, u_trks))
            iou_left_thre = iou_left.copy()
            if self.TCM_byte_step:
                iou_left -= np.array(cal_score_dif_batch_two_score(dets_low, u_trks) * self.TCM_byte_step_weight)

            if iou_left.max() > self.iou_threshold:
                if self.EG_weight_low_score > 0 and self.with_reid:
                    u_tracklets = [self.active_tracks[idx] for idx in unmatched_trks]
                    u_track_features = np.asarray([t.smooth_feat for t in u_tracklets], dtype=float)
                    emb_dists_low = embedding_distance(u_track_features, id_feature_second).T
                    matched_indices = linear_assignment(-iou_left + self.EG_weight_low_score * emb_dists_low)
                else:
                    matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for mm in matched_indices:
                    det_rel, trk_rel = mm[0], mm[1]
                    trk_ind = unmatched_trks[trk_rel]
                    if self.with_longterm_reid_correction and self.EG_weight_low_score > 0 and self.with_reid:
                        if iou_left_thre[det_rel, trk_rel] < self.iou_threshold or emb_dists_low[det_rel, trk_rel] > self.longterm_reid_correction_thresh_low:
                            continue
                    else:
                        if iou_left_thre[det_rel, trk_rel] < self.iou_threshold:
                            continue
                    # do not update features in BYTE pass
                    self.active_tracks[trk_ind].update(
                        dets_low[det_rel, :],
                        id_feature_second[det_rel, :],
                        update_feature=False,
                        cls=_safe_cls(cls_second[det_rel], self.nr_classes),
                        det_ind=_safe_detind(det_inds_second[det_rel], n_dets_full),
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # ===== Final chance: IoU vs last boxes
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets_first[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = np.array(ASSO_FUNCS[self.asso_func_name](left_dets, left_trks))
            if iou_left.max() > self.iou_threshold:
                rematched = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for mm in rematched:
                    det_rel, trk_rel = mm[0], mm[1]
                    if iou_left[det_rel, trk_rel] < self.iou_threshold:
                        continue
                    det_abs = unmatched_dets[det_rel]
                    trk_abs = unmatched_trks[trk_rel]
                    self.active_tracks[trk_abs].update(
                        dets_first[det_abs, :],
                        id_feature_keep[det_abs, :],
                        update_feature=False,
                        cls=_safe_cls(cls_keep[det_abs], self.nr_classes),
                        det_ind=_safe_detind(det_inds_keep[det_abs], n_dets_full),
                    )
                    to_remove_det_indices.append(det_abs)
                    to_remove_trk_indices.append(trk_abs)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # Mark remaining unmatched tracks
        for m in unmatched_trks:
            self.active_tracks[m].update(None, None)

        # Create new trackers for unmatched high-score detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets_first[i, :],
                id_feature_keep[i, :],
                delta_t=self.delta_t,
                use_custom_kf=self.use_custom_kf,
                longterm_bank_length=self.longterm_bank_length,
                alpha=self.alpha,
                adapfs=self.adapfs,
                track_thresh=self.track_thresh,
                cls=_safe_cls(cls_keep[i], self.nr_classes),
                det_ind=_safe_detind(det_inds_keep[i], n_dets_full) if len(det_inds_keep) else -1,
            )
            self.active_tracks.append(trk)

        # Collect outputs (match BotSort/OcSort style)
        outputs = []
        for trk in self.active_tracks[::-1]:
            if trk.last_observation.sum() < 0:
                d = convert_x_to_bbox(trk.kf.x)[0][:4]
            else:
                d = trk.last_observation[:4]

            # Only output fresh tracks and valid det_ind for this frame
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append([
                    *d.tolist(),
                    trk.id + 1,                 # track id
                    float(trk.conf),      # conf
                    int(trk.cls),               # cls (from detection)
                    int(trk.det_ind),           # det index (frame-local)
                ])

        # Remove dead tracks
        i = len(self.active_tracks)
        for trk in self.active_tracks[::-1]:
            i -= 1
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)

        return np.asarray(outputs) if len(outputs) else np.zeros((0, 8), dtype=float)
