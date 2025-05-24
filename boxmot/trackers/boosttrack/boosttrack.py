from collections import deque
from typing import List, Optional

import numpy as np

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.boosttrack.assoc import (
    MhDist_similarity,
    associate,
    iou_batch,
    shape_similarity,
    soft_biou_batch,
)
from boxmot.trackers.boosttrack.kalmanfilter import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Converts a bounding box [x1,y1,x2,y2] to state vector [x, y, h, r].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    r = w / float(h + 1e-6)
    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Converts a state vector [x, y, h, r] back to bounding box [x1,y1,x2,y2].
    """
    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                         x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))
    return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                     x[0] + w / 2.0, x[1] + h / 2.0, score]).reshape((1, 5))


class KalmanBoxTracker:
    """
    Single object tracker using a Kalman filter.
    """
    count = 0

    def __init__(self, det, max_obs, emb: Optional[np.ndarray] = None):
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count 
        self.kf = KalmanFilter(convert_bbox_to_z(det[:4]))
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.emb = emb
        self.hit_streak = 0
        self.age = 0
        self.history_observations = deque([], maxlen=max_obs)

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7
        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update - 1)

    def update(self, det: np.ndarray):
        self.time_since_update = 0
        self.hit_streak += 1
        self.history_observations.append(self.get_state()[0])
        self.kf.update(convert_bbox_to_z(det))
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]

    def camera_update(self, transform: np.ndarray):
        """
        Handle either a 2×3 affine or a 3×3 homography, by
        promoting the 2×3 to 3×3 [ …; 0 0 1 ].
        """
        # ——— normalize to 3×3 —————
        wm = np.asarray(transform, dtype=float)
        if wm.shape == (2, 3):
            wm = np.vstack([wm, [0.0, 0.0, 1.0]])
        elif wm.shape != (3, 3):
            raise ValueError(f"Expected 2×3 or 3×3 matrix, got {wm.shape}")

        # ——— warp your current bbox —————
        x1, y1, x2, y2 = self.get_state()[0]
        p1 = wm @ np.array([x1, y1, 1.0])
        p2 = wm @ np.array([x2, y2, 1.0])
        x1_, y1_, _ = p1
        x2_, y2_, _ = p2

        # ——— rebuild Kalman state —————
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w/2, y1_ + h/2
        self.kf.x[:4] = [cx, cy, h, w/h]


    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb


class BoostTrack(BaseTracker):

    def __init__(
        self,
        reid_weights,
        device,
        half: bool,

        max_age: int = 60,
        min_hits: int = 3,
        det_thresh: float = 0.6,
        iou_threshold: float = 0.3,
        use_ecc: bool = True,
        min_box_area: int = 10,
        aspect_ratio_thresh: bool = 1.6,
        cmc_method: str = 'ecc',

        # BoostTrack parameters
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25,
        use_dlo_boost: bool = True,
        use_duo_boost: bool = True,
        dlo_boost_coef: float = 0.65,
        s_sim_corr: bool = False,
    
        # BoostTrack++ parameters
        use_rich_s: bool = False,
        use_sb: bool = False,
        use_vt: bool = False,

        with_reid: bool = False,
    ):
        super().__init__()
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []

        # Parameters for BoostTrack (these can be tuned as needed)
        self.max_age = max_age            # maximum allowed frames without update
        self.min_hits = min_hits          # minimum hits to output a track
        self.det_thresh = det_thresh      # detection confidence threshold
        self.iou_threshold = iou_threshold   # association IoU threshold
        self.use_ecc = use_ecc            # use ECC for camera motion compensation
        self.min_box_area = min_box_area  # minimum box area for detections
        self.aspect_ratio_thresh = aspect_ratio_thresh  # aspect ratio threshold for detections
        self.cmc_method = cmc_method

        self.lambda_iou = lambda_iou
        self.lambda_mhd = lambda_mhd
        self.lambda_shape = lambda_shape
        self.use_dlo_boost = use_dlo_boost
        self.use_duo_boost = use_duo_boost
        self.dlo_boost_coef = dlo_boost_coef
        self.s_sim_corr = s_sim_corr

        self.use_rich_s = use_rich_s
        self.use_sb = use_sb
        self.use_vt = use_vt

        self.with_reid = with_reid

        if self.with_reid:
            self.reid_model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model
        else:
            self.reid_model = None

        if self.use_ecc:
            self.cmc = get_cmc_method(cmc_method)()
        else:
            self.cmc = None

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update the tracker with detections and an image.
        
        Args:
          dets (np.ndarray): Detection boxes in the format [[x1,y1,x2,y2,score], ...]
          img (np.ndarray): The current image frame.
          embs (Optional[np.ndarray]): Optional precomputed embeddings.
          
        Returns:
          np.ndarray: Tracked objects in the format
                      [x1, y1, x2, y2, id, confidence, cls, det_ind]
                      (with cls and det_ind set to -1 if unused)
        """
        self.check_inputs(dets=dets, embs=embs, img=img)

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])

        self.frame_count += 1

        if self.cmc is not None:
            transform = self.cmc.apply(img, dets)
            for trk in self.trackers:
                trk.camera_update(transform)

        trks = []
        confs = []
        
        for trk in self.trackers:
            pos = trk.predict()[0]
            conf = trk.get_confidence()
            confs.append(conf)
            trks.append(np.concatenate([pos, [conf]]))
        trks_np = np.vstack(trks) if len(trks) > 0 else np.empty((0, 5))

        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(dets)
        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets)

        dets_embs = np.ones((dets.shape[0], 1))
        if dets.size > 0:
            remain_inds = dets[:, 4] >= self.det_thresh
            dets = dets[remain_inds]
            scores = dets[:, 4]

            if self.with_reid:
                if embs is not None:
                    dets_embs = embs[remain_inds]
                else:
                    dets_embs = self.reid_model.get_features(dets[:, :4], img)
        else:
            scores = np.empty(0)
            dets_embs = np.ones((dets.shape[0], 1))

        if self.with_reid and len(self.trackers) > 0:
            tracker_embs = np.array([trk.get_emb() for trk in self.trackers])
            if dets_embs.shape[0] == 0:
                emb_cost = np.empty((0, tracker_embs.shape[0]))
            else:
                emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ tracker_embs.reshape((tracker_embs.shape[0], -1)).T
        else:
            emb_cost = None

        mh_dist_matrix = self.get_mh_dist_matrix(dets)

        matched, unmatched_dets, unmatched_trks, _ = associate(
            dets,
            trks_np,
            self.iou_threshold,
            mahalanobis_distance=mh_dist_matrix,
            track_confidence=np.array(confs).reshape(-1, 1),
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
            s_sim_corr=self.s_sim_corr
        )

        if dets.size > 0:   
            trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
            af = 0.95
            dets_alpha = af + (1 - af) * (1 - trust)
        else:
            dets_alpha = np.empty(0)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for i in unmatched_dets:
            if dets[i, 4] >= self.det_thresh:
                self.trackers.append(KalmanBoxTracker(dets[i, :], max_obs=self.max_obs, emb=dets_embs[i]))

        outputs = []
        self.active_tracks = []
        for trk in self.trackers:
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Format: [x1, y1, x2, y2, id, confidence, cls, det_ind]
                outputs.append(np.array([d[0], d[1], d[2], d[3], trk.id + 1, trk.conf, trk.cls, trk.det_ind]))
                self.active_tracks.append(trk)
            
        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

        if len(outputs) == 0:
            return np.empty((0, 8))
        outputs = np.vstack(outputs)
        return self.filter_outputs(outputs)

    def dump_cache(self):
        if self.ecc is not None:
            self.ecc.save_cache()

    def filter_outputs(self, outputs: np.ndarray) -> np.ndarray:

        w_arr = outputs[:, 2] - outputs[:, 0]
        h_arr = outputs[:, 3] - outputs[:, 1]

        vertical_filter = w_arr / h_arr <= self.aspect_ratio_thresh
        area_filter = w_arr * h_arr > self.min_box_area

        return outputs[vertical_filter & area_filter]
    
    def get_iou_matrix(self, detections: np.ndarray, buffered: bool = False) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], self.trackers[t].get_confidence()]

        return iou_batch(detections, trackers) if not buffered else soft_biou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros((len(self.trackers), n_dims), dtype=float)

        for i in range(len(detections)):
            z[i, :n_dims] = convert_bbox_to_z(detections[i, :]).reshape(-1)[:n_dims]
        for i, trk in enumerate(self.trackers):
            x[i] = trk.kf.x[:n_dims]
            sigma_inv[i] = np.reciprocal(np.diag(trk.kf.covariance[:n_dims, :n_dims]))
        return ((z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2 *
                sigma_inv.reshape((1, -1, n_dims))).sum(axis=2)

    def duo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections
        n_dims = 4
        limit = 13.2767
        mh_dist = self.get_mh_dist_matrix(detections, n_dims)
        if mh_dist.size == 0 and self.frame_count < 2:
            return detections
        min_dists = mh_dist.min(1)
        mask = (min_dists > limit) & (detections[:, 4] < self.det_thresh)
        boost_inds = np.where(mask)[0]
        iou_limit = 0.3
        if len(boost_inds) == 0:
            return detections

        bdiou = iou_batch(detections[boost_inds], detections[boost_inds]) - np.eye(len(boost_inds))
        bdiou_max = bdiou.max(axis=1)
        remaining = boost_inds[bdiou_max <= iou_limit]
        args = np.where(bdiou_max > iou_limit)[0]
        for i in range(len(args)):
            bi = args[i]
            tmp = np.where(bdiou[bi] > iou_limit)[0]
            args_tmp = np.append(np.intersect1d(boost_inds[args], boost_inds[tmp]), boost_inds[bi])
            conf_max = np.max(detections[args_tmp, 4])
            if detections[boost_inds[bi], 4] == conf_max:
                remaining = np.concatenate([remaining, [boost_inds[bi]]])
        mask_boost = np.zeros_like(detections[:, 4], dtype=bool)
        mask_boost[remaining] = True
        detections[:, 4] = np.where(mask_boost, self.det_thresh + 1e-4, detections[:, 4])
        return detections


    def dlo_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        if len(detections) == 0:
            return detections
        
        sbiou_matrix = self.get_iou_matrix(detections, True)
        if sbiou_matrix.size == 0:
            return detections

        trackers = np.zeros((len(self.trackers), 6))
        for t, trk in enumerate(self.trackers):
            pos = trk.get_state()[0]
            trackers[t] = [pos[0], pos[1], pos[2], pos[3], 0, trk.time_since_update - 1]

        if self.use_rich_s:
            mhd_sim = MhDist_similarity(self.get_mh_dist_matrix(detections), 1)
            shape_sim = shape_similarity(detections, trackers, self.s_sim_corr)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False)

        if not self.use_sb and not self.use_vt:
            max_s = S.max(1)
            detections[:, 4] = np.maximum(detections[:, 4], max_s * self.dlo_boost_coef)
            return detections

        if self.use_sb:
            max_s = S.max(1)
            alpha = 0.65
            detections[:, 4] = np.maximum(
                detections[:, 4],
                alpha * detections[:, 4] + (1 - alpha) * max_s ** 1.5)
        if self.use_vt:
            threshold_s = 0.95
            threshold_e = 0.8
            n_steps = 20
            # alpha = (threshold_s - threshold_e) / n_steps # todo alpha is not being used probably a bug
            tmp = (S > np.maximum(
                threshold_s - np.array([trk.time_since_update - 1 for trk in self.trackers]),
                                    threshold_e)).max(1)
            scores = detections[:, 4].copy()
            scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)
            detections[:, 4] = scores
        return detections