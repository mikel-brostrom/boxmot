# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from typing import Any

import numpy as np

from boxmot.motion.cmc import get_cmc_method
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.bbox.strongsort.sort.detection import Detection
from boxmot.trackers.bbox.strongsort.sort.linear_assignment import NearestNeighborDistanceMetric
from boxmot.trackers.bbox.strongsort.sort.tracker import Tracker
from boxmot.trackers.ops import xyxy2tlwh


class StrongSort(BaseTracker):
    """Initialize the StrongSort tracker.

    Args:
        reid_model: Pre-built ReID backend model (e.g. ``ReID(...).model``).
        min_conf (float): Minimum confidence threshold for detections.
        max_cos_dist (float): Maximum cosine distance accepted by the
            nearest-neighbor metric.
        max_iou_dist (float): Maximum IoU distance used during association.
        n_init (int): Number of consecutive hits required to confirm a track.
        nn_budget (int): Maximum number of appearance features stored per
            track.
        mc_lambda (float): Motion-consistency weight used by StrongSORT.
        ema_alpha (float): Exponential moving average coefficient for
            appearance features.
        **kwargs: Base tracker settings forwarded to :class:`BaseTracker`.

    Attributes:
        model: ReID model used for appearance extraction.
        tracker (Tracker): Internal StrongSORT tracker instance.
        cmc: Camera-motion compensation method.
    """

    def __init__(
        self,
        reid_model: Any | None = None,
        min_conf: float = 0.1,
        max_cos_dist: float = 0.2,
        max_iou_dist: float = 0.7,
        n_init: int = 3,
        nn_budget: int = 100,
        mc_lambda: float = 0.98,
        ema_alpha: float = 0.9,
        **kwargs: Any,
    ):
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='StrongSort', **kwargs)

        self.min_conf = min_conf
        self.model = reid_model

        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=self.max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
        )

        self.cmc = get_cmc_method("ecc")()

    def _update_impl(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None,
        masks: np.ndarray = None,
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

        self.tracker.predict()
        self.tracker.update(detections)

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
