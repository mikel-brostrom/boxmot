# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
from torch import device

from boxmot.motion.cmc import get_cmc_method
from boxmot.reid.core import ReID
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.linear_assignment import \
    NearestNeighborDistanceMetric
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.ops import xyxy2tlwh


class StrongSort(BaseTracker):
    """
    Initialize the StrongSort tracker with various parameters.

    Parameters:
    - reid_weights (Path): Path to the re-identification model weights.
    - device (torch.device): Device to run the model on (e.g., 'cpu', 'cuda').
    - half (bool): Whether to use half-precision (fp16) for faster inference.
    - det_thresh (float): Detection threshold for considering detections.
    - max_age (int): Maximum age (in frames) of a track before it is considered lost.
    - max_obs (int): Maximum number of historical observations stored for each track. Always greater than max_age by minimum 5.
    - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
    - iou_threshold (float): IOU threshold for determining match between detection and tracks.
    - per_class (bool): Enables class-separated tracking.
    - nr_classes (int): Total number of object classes that the tracker will handle (for per_class=True).
    - asso_func (str): Algorithm name used for data association between detections and tracks.
    - is_obb (bool): Work with Oriented Bounding Boxes (OBB) instead of standard axis-aligned bounding boxes.
    
    StrongSort-specific parameters:
    - min_conf (float): Minimum confidence threshold for detections.
    - max_cos_dist (float): Maximum cosine distance for ReID feature matching in Nearest Neighbor Distance Metric.
    - max_iou_dist (float): Maximum IoU distance for data association.
    - n_init (int): Number of consecutive frames required to confirm a track.
    - nn_budget (int): Maximum size of the feature library for Nearest Neighbor Distance Metric.
    - mc_lambda (float): Weight for motion consistency in the track state estimation.
    - ema_alpha (float): Alpha value for exponential moving average (EMA) update of appearance features.
    
    Attributes:
    - model: ReID model for appearance feature extraction.
    - tracker: StrongSort tracker instance.
    - cmc: Camera motion compensation object.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        # StrongSort-specific parameters
        min_conf: float = 0.1,
        max_cos_dist: float = 0.2,
        max_iou_dist: float = 0.7,
        n_init: int = 3,
        nn_budget: int = 100,
        mc_lambda: float = 0.98,
        ema_alpha: float = 0.9,
        **kwargs  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='StrongSort', **kwargs)
        
        # Store StrongSort-specific parameters
        self.min_conf = min_conf
        
        # Initialize ReID model
        self.model = ReID(
            weights=reid_weights, device=device, half=half
        ).model

        # Initialize StrongSort tracker
        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=self.max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )
        
        # Initialize camera motion compensation
        self.cmc = get_cmc_method("ecc")()
        
    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        dets = self.detection_layout.with_detection_indices(dets)
        remain_inds = self.detection_layout.confidences(dets) >= self.min_conf
        dets = dets[remain_inds]

        xyxy = self.detection_layout.boxes(dets)
        confs = self.detection_layout.confidences(dets)
        clss = self.detection_layout.classes(dets)
        det_ind = dets[:, self.detection_layout.det_cols]

        if len(self.tracker.tracks) >= 1:
            warp_matrix = self.cmc.apply(img, xyxy)
            for track in self.tracker.tracks:
                track.camera_update(warp_matrix)

        # extract appearance information for each detection
        if embs is not None:
            features = embs[remain_inds]
        else:
            features = self.model.get_features(xyxy, img)

        tlwh = xyxy2tlwh(xyxy)
        detections = [
            Detection(box, conf, cls, det_ind, feat)
            for box, conf, cls, det_ind, feat in zip(
                tlwh, confs, clss, det_ind, features
            )
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
                np.concatenate(
                    ([x1, y1, x2, y2], [id], [conf], [cls], [det_ind])
                ).reshape(1, -1)
            )
        if len(outputs) > 0:
            return np.concatenate(outputs)
        return self.empty_output()

    def reset(self):
        pass
