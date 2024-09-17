# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import cv2
import numpy as np
from torch import device
from pathlib import Path
from typing import List, Tuple

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.faststrongsort.sort.detection import Detection
from boxmot.trackers.faststrongsort.sort.tracker import Tracker
from boxmot.trackers.faststrongsort.sort.iou_matching import aiou_vectorized
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh
from boxmot.trackers.basetracker import BaseTracker


class FastStrongSORT(object):
    """
    Fast-StrongSORT Tracker: StrongSORT with selective feature extraction mechanism

    Args:
        model_weights (str): Path to the model weights for ReID (Re-Identification).
        device (str): Device on which to run the model (e.g., 'cpu' or 'cuda').
        fp16 (bool): Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
        max_dist (float, optional): Maximum cosine distance for ReID feature matching in Nearest Neighbor Distance Metric.
        max_iou_dist (float, optional): Maximum Intersection over Union (IoU) distance for data association. Controls the maximum allowed distance between tracklets and detections for a match.
        max_age (int, optional): Maximum number of frames to keep a track alive without any detections.
        n_init (int, optional): Number of consecutive frames required to confirm a track.
        nn_budget (int, optional): Maximum size of the feature library for Nearest Neighbor Distance Metric. If the library size exceeds this value, the oldest features are removed.
        mc_lambda (float, optional): Weight for motion consistency in the track state estimation. Higher values give more weight to motion information.
        ema_alpha (float, optional): Alpha value for exponential moving average (EMA) update of appearance features. Controls the contribution of new and old embeddings in the ReID model.
        iou_threshold (float, optional): Threshold to determine if a tracklet can be a candidate for the detection. iou_threshold = 1.0 means that the tracker is identical to StrongSORT
        ars_threshold (float, optional): Threshold to eliminate possibly false candidates
    """
    def __init__(
        self,
        model_weights: Path,
        device: device,
        fp16: bool,
        per_class: bool = False,
        max_cos_dist=0.2,
        max_iou_dist=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.98,
        ema_alpha=0.9,
        iou_threshold=0.2,
        ars_threshold=0.6
    ):

        self.per_class = per_class
        self.model = ReidAutoBackend(
            weights=model_weights, device=device, half=fp16
        ).model

        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            iou_threshold=iou_threshold,
            ars_threshold=ars_threshold
        )
        self.cmc = get_cmc_method('ecc')()

    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        assert isinstance(
            dets, np.ndarray
        ), f"Unsupported 'dets' input format '{type(dets)}', valid format is np.ndarray"
        assert isinstance(
            img, np.ndarray
        ), f"Unsupported 'img' input format '{type(img)}', valid format is np.ndarray"
        assert (
            len(dets.shape) == 2
        ), "Unsupported 'dets' dimensions, valid number of dimensions is two"
        assert (
            dets.shape[1] == 6
        ), "Unsupported 'dets' 2nd dimension lenght, valid lenghts is 6"

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        xyxy = dets[:, 0:4]
        confs = dets[:, 4]
        clss = dets[:, 5]
        det_ind = dets[:, 6]
        tlwh = xyxy2tlwh(xyxy)
        confirmed_tracks = [trk for trk in self.tracker.tracks if trk.is_confirmed()]

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        risky_detections, safe_det_trk_pairs = self.candidates_to_detections(tlwh, confirmed_tracks)

        # extract appearance information for each detection
        if embs is not None:
            extracted_features = embs[risky_detections]
        else:
            extracted_features = self.model.get_features(xyxy[risky_detections], img)

        features = []
        safe_det_dict = dict(safe_det_trk_pairs)
        
        for i in range(len(dets)):
            if i in risky_detections:
                features.append(extracted_features[risky_detections.index(i)])
            elif i in safe_det_dict:
                candidate_track = confirmed_tracks[safe_det_dict[i]]
                features.append(candidate_track.features[-1])
            else:
                raise ValueError
            
        decay_emas = np.array([True] * len(features))
        decay_emas[risky_detections] = False
    
        detections = [
            Detection(box, conf, cls, det_ind, feat, decay_ema) for
            box, conf, cls, det_ind, feat, decay_ema in
            zip(tlwh, confs, clss, det_ind, features, decay_emas)
        ]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            x1, y1, x2, y2 = track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(
                np.concatenate(([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return np.array([])

    def candidates_to_detections(self, tlwh: np.ndarray, confirmed_tracks) -> Tuple[List[int], List[Tuple[int, int]]]:
        tracklet_bboxes = np.array([trk.to_tlwh() for trk in confirmed_tracks])

        if len(tracklet_bboxes) == 0:
            return list(range(tlwh.shape[0])), []

        ious, alphas = aiou_vectorized(tlwh, tracklet_bboxes)

        matches = ious > self.tracker.iou_threshold
        match_counts = np.sum(matches, axis=1)

        risky_detections = np.where(match_counts != 1)[0].tolist()
        safe_candidates = np.where(match_counts == 1)[0]

        safe_det_trk_pairs = []
        for i in safe_candidates:
            candidate = np.argmax(ious[i])
            if alphas[i, candidate] > self.tracker.ars_threshold:
                safe_det_trk_pairs.append((i, candidate))
            else:
                risky_detections.append(i)

        return risky_detections, safe_det_trk_pairs