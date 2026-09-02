# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from typing import Any

import numpy as np

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.appearance import resolve_batch_embeddings
from boxmot.trackers.common.association.strongsort import (
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    iou_cost,
    matching_cascade,
    min_cost_matching,
)
from boxmot.trackers.common.geometry import xyxy2tlwh
from boxmot.trackers.common.motion.cmc import create_cmc
from boxmot.trackers.common.track_models.strongsort import Track


class _Detection:
    """StrongSORT-specific detection measurement used during association."""

    def __init__(
        self,
        box: np.ndarray,
        conf: float,
        cls: int,
        det_ind: int,
        feat: np.ndarray | None,
        *,
        is_obb: bool,
    ) -> None:
        self.tlwh = np.asarray(box, dtype=np.float32)
        self.conf = conf
        self.cls = cls
        self.det_ind = det_ind
        self.feat = feat
        self.is_obb = bool(is_obb)

    def to_xyah(self) -> np.ndarray:
        """Convert an AABB measurement to center-x, center-y, aspect, height."""
        measurement = self.tlwh.copy()
        measurement[:2] += measurement[2:] / 2
        measurement[2] /= measurement[3]
        return measurement

    def to_measurement(self) -> np.ndarray:
        """Return the Kalman-filter measurement for the active box mode."""
        return self.tlwh.copy() if self.is_obb else self.to_xyah()


class StrongSort(BaseTracker):
    supports_obb = True
    uses_img = True
    uses_embs = True

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
        tracks: Active StrongSORT track states.
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
        self._max_cos_dist = float(max_cos_dist)
        self._nn_budget = int(nn_budget) if nn_budget is not None else None
        self.metric = self._new_metric()
        self.max_iou_dist = max_iou_dist
        self.n_init = n_init
        self.mc_lambda = mc_lambda
        self.ema_alpha = ema_alpha
        self.tracks: list[Track] = []
        self.active_tracks = self.tracks
        self.cmc = create_cmc("ecc")

    def _new_metric(self) -> NearestNeighborDistanceMetric:
        """Create an empty appearance gallery with this tracker's settings."""
        return NearestNeighborDistanceMetric("cosine", self._max_cos_dist, self._nn_budget)

    def _load_class_track_state(self, cls_id: int) -> None:
        """Restore both motion tracks and the class-local appearance gallery."""
        super()._load_class_track_state(cls_id)
        if not self.per_class:
            return
        state = self._ensure_class_track_state(cls_id)
        metric = state.attrs.get("strongsort_metric")
        if metric is None:
            metric = self._new_metric()
            state.attrs["strongsort_metric"] = metric
        self.metric = metric

    def _save_class_track_state(self, cls_id: int) -> None:
        """Persist the gallery with its class so another class cannot prune it."""
        if self.per_class:
            self._ensure_class_track_state(cls_id).attrs["strongsort_metric"] = self.metric
        super()._save_class_track_state(cls_id)

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        batch = self.make_detection_batch(dets, embs=embs, masks=masks)
        batch = batch.select(batch.confs >= self.min_conf)
        indexed_dets = batch.as_indexed_detections(dtype=dets.dtype)

        # Advance the estimator on every frame, including initialization and
        # trackless gaps, so the next live track never receives a stale warp.
        self.apply_cmc(img, indexed_dets, self.tracks)

        features = resolve_batch_embeddings(
            batch,
            img,
            model=self.model,
            boxes=batch.boxes,
        )

        track_boxes = batch.boxes if self.is_obb else xyxy2tlwh(batch.boxes)
        detections = [
            _Detection(box, conf, cls, det_ind, feat, is_obb=self.is_obb)
            for box, conf, cls, det_ind, feat in zip(
                track_boxes,
                batch.confs,
                batch.clss,
                batch.det_inds,
                features,
            )
        ]

        self._predict_tracks()
        self._update_tracks(detections)

        outputs = []
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update >= 1:
                continue

            box = track.xywha if self.is_obb else track.to_tlbr()

            id = track.id
            conf = track.conf
            cls = track.cls
            det_ind = track.det_ind

            outputs.append(self.format_output_row(box, id, conf, cls, det_ind))
        return self.format_output_rows(outputs, dtype=np.float32)

    def requires_image(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> bool:
        """StrongSORT always advances its ECC estimator with the current frame."""
        del dets, embs, masks
        return True

    def _predict_tracks(self) -> None:
        """Propagate all active track states to the current frame."""
        for track in self.tracks:
            track.predict()

    def _update_tracks(self, detections: list[_Detection]) -> None:
        """Associate detections, update matched tracks, and manage lifecycle state."""
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [track for track in self.tracks if not track.is_deleted()]
        self.active_tracks = self.tracks

        active_targets = [track.id for track in self.tracks if track.is_confirmed()]
        features: list[np.ndarray] = []
        targets: list[int] = []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features.extend(track.features)
            targets.extend([track.id] * len(track.features))
        self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections: list[_Detection]) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Run StrongSORT appearance cascade followed by IoU association."""

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.asarray([dets[index].feat for index in detection_indices])
            targets = np.asarray([tracks[index].id for index in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            return gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )

        confirmed_tracks = [index for index, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed_tracks = [index for index, track in enumerate(self.tracks) if not track.is_confirmed()]
        matches_a, unmatched_tracks_a, unmatched_detections = matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        iou_track_candidates = unconfirmed_tracks + [
            index for index in unmatched_tracks_a if self.tracks[index].time_since_update == 1
        ]
        unmatched_tracks_a = [index for index in unmatched_tracks_a if self.tracks[index].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            iou_cost,
            self.max_iou_dist,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection: _Detection) -> None:
        """Create one track from an unmatched detection."""
        self.tracks.append(
            Track(
                detection,
                self.id_allocator.alloc(),
                self.n_init,
                self.max_age,
                self.max_obs,
                self.ema_alpha,
                is_obb=detection.is_obb,
            )
        )

    def reset(self):
        self._reset_common_state()
        self.tracks = []
        self.metric = self._new_metric()
