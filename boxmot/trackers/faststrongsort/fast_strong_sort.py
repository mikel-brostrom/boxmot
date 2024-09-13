# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from torch import device
from pathlib import Path

from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.faststrongsort.sort.detection import Detection
from boxmot.trackers.faststrongsort.sort.tracker import Tracker
from boxmot.utils.matching import NearestNeighborDistanceMetric
from boxmot.utils.ops import xyxy2tlwh
from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.iou import iou_batch
import cv2


class FastStrongSORT(object):
    """
    FastStrongSORT Tracker: A tracking algorithm that utilizes a combination of appearance and motion-based tracking that only extracts appearence features only when they are necessary.

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
        iou_threshold (float, optional): Threshold to determine the possbile candidates for detections. If 1.0 feature extraction is performed for every track.
        ars_threshold (float, optional): Aspect ratio similarity threshold to eliminate false candidates to detection
    """
    def __init__(
        self,
        model_weights: Path,
        device: device,
        fp16: bool,
        per_class: bool = False,
        max_cos_dist=0.4,
        max_iou_dist=0.7,
        max_age=30,
        n_init=3,
        nn_budget=100,
        mc_lambda=0.98,
        ema_alpha=0.9,
        iou_threshold=0.2,
        ars_threshold=0.6,
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
        )
        self.cmc = get_cmc_method('ecc')()
        self.iou_threshold = iou_threshold
        self.ars_threshold = ars_threshold
        self.last_feature_extractions = 0

    def aspect_ratio_similarity(self, box1, box2):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2
        return 1 - (4 / (np.pi ** 2) * (np.arctan(aspect_ratio1) - np.arctan(aspect_ratio2)) ** 2)

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

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # Determine which detections need feature extraction
        risky_detections = []
        non_risky_matches = {}
        for i, det in enumerate(xyxy):
            matching_tracks = []
            for track in self.tracker.tracks:
                if track.is_confirmed():
                    iou = iou_batch(det.reshape(1, -1), track.to_tlbr().reshape(1, -1))[0][0]
                    if iou > self.iou_threshold:
                        matching_tracks.append((track, iou))

            if len(matching_tracks) == 1:
                track, iou = matching_tracks[0]
                ars = self.aspect_ratio_similarity(det, track.to_tlbr())
                v = ars
                alpha = v / ((1 - iou) + v)
                if alpha > self.ars_threshold:
                    # Non-risky detection, use track's features
                    non_risky_matches[i] = track
                    continue

            # Risky detection, needs feature extraction
            risky_detections.append(i)
        
        # Extract features only foroutputs risky detections
        if embs is not None:
            features = embs[risky_detections]
        else:
            features = self.model.get_features(xyxy[risky_detections], img)

        # Prepare detections
        tlwh = xyxy2tlwh(xyxy)
        detections = []
        for i, (box, conf, cls, ind) in enumerate(zip(tlwh, confs, clss, det_ind)):
            risky = False
            if i in risky_detections:
                feat = features[risky_detections.index(i)]
                risky = True
            else:
                # For non-risky detections, use the matching track's features
                feat = non_risky_matches[i].features[-1]  # Use the latest feature from the matching track
            detections.append(Detection(box, conf, cls, ind, feat, risky))

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # # Visualization
        # # Draw bounding boxes
        # for track in self.tracker.tracks:
        #     if not track.is_confirmed():
        #         color = (255, 0, 0)  # Blue for not confirmed tracks
        #     else:
        #         color = (0, 255, 0)  # Green for confirmed tracks
            
        #     bbox = track.to_tlbr().astype(int)
        #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

        # for i, det in enumerate(xyxy):
        #     if i in risky_detections:
        #         color = (0, 0, 255)  # Red for risky detections
        #     else:
        #         color = (0, 255, 255)  # Yellow for non-risky detections
            
        #     bbox = det.astype(int)
        #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

        # # Add legend
        # cv2.putText(img, "Blue: Not confirmed tracks", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # cv2.putText(img, "Green: Confirmed tracks", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(img, "Red: Risky detections", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(img, "Yellow: Non-risky detections", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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