# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from typing import Any

import numpy as np

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.appearance import resolve_batch_embeddings
from boxmot.trackers.common.association.strongsort import NearestNeighborDistanceMetric
from boxmot.trackers.common.geometry import xyxy2tlwh
from boxmot.trackers.common.geometry.obb import xywha_to_xyxy
from boxmot.trackers.common.motion.cmc import create_cmc
from boxmot.trackers.common.track_models.strongsort import Detection, Tracker


class StrongSort(BaseTracker):
    supports_obb = True

    """Initialize the StrongSort tracker.

    Args:
        reid_model (Any | None): Pre-built ReID backend model (e.g. ``ReID(...).model``).
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
        **kwargs (Any): Base tracker settings forwarded to :class:`BaseTracker`.

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
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="StrongSort", **kwargs)

        self.min_conf = min_conf
        self.model = reid_model

        self.tracker = Tracker(
            metric=NearestNeighborDistanceMetric("cosine", max_cos_dist, nn_budget),
            max_iou_dist=max_iou_dist,
            max_age=self.max_age,
            n_init=n_init,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            id_allocator=self.id_allocator,
            is_obb=self.is_obb,
        )

        self.cmc = create_cmc("ecc")

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.tracker.is_obb = self.is_obb
        batch = self.make_detection_batch(dets, embs=embs, masks=masks)
        batch = batch.select(batch.confs >= self.min_conf)
        indexed_dets = batch.as_indexed_detections(dtype=dets.dtype)

        if len(self.tracker.tracks) >= 1:
            self.apply_cmc(img, indexed_dets, self.tracker.tracks)

        features = resolve_batch_embeddings(
            batch,
            img,
            model=self.model,
            boxes=xywha_to_xyxy(batch.boxes) if self.is_obb else batch.boxes,
        )

        track_boxes = batch.boxes if self.is_obb else xyxy2tlwh(batch.boxes)
        detections = [
            Detection(box, conf, cls, det_ind, feat, is_obb=self.is_obb)
            for box, conf, cls, det_ind, feat in zip(
                track_boxes,
                batch.confs,
                batch.clss,
                batch.det_inds,
                features,
            )
        ]

        self.tracker.predict()
        self.tracker.update(detections)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            box = track.xywha if self.is_obb else track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(self.format_output_row(box, id, conf, cls, det_ind))
        return self.format_output_rows(outputs, dtype=np.float32)

    def reset(self):
        self._reset_common_state()
        self.tracker.reset()
