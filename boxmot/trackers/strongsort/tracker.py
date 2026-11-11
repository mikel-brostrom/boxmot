# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from typing_extensions import Unpack

from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.common.appearance import resolve_batch_embeddings
from boxmot.trackers.common.association.matching import (
    fuse_iou,
    fuse_score,
)
from boxmot.trackers.common.association.matching import (
    gate_cost_matrix as gate_motion_cost,
)
from boxmot.trackers.common.association.strongsort import (
    INFTY_COST,
    NearestNeighborDistanceMetric,
    gate_cost_matrix,
    min_cost_matching,
)
from boxmot.trackers.common.box.base import BoxTracker
from boxmot.trackers.common.constructor import BoxTrackerOptions, validate_runtime_options
from boxmot.trackers.common.geometry import xyxy2tlwh
from boxmot.trackers.common.motion.cmc.registry import create_cmc
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
from boxmot.trackers.common.motion.models import create_motion_model, resolve_motion_model_kind
from boxmot.trackers.strongsort.config import StrongSortConfig
from boxmot.trackers.strongsort.track import Track, _Detection


class StrongSort(BoxTracker):
    """Track AABB or OBB detections with appearance and configurable camera motion.

    Attributes:
        active_tracks: Live StrongSORT identities, including temporarily missed tracks.
        cmc: Camera-motion compensation method.
    """

    accepts_embeddings = True

    supports_variable_dt = True
    requires_embeddings = True
    use_embeddings = True

    def __init__(
        self,
        config: StrongSortConfig | None = None,
        *,
        kalman: KalmanConfig | None = None,
        reid: ReIDConfig | AppearanceEncoder | None = None,
        **kwargs: Unpack[BoxTrackerOptions],
    ) -> None:
        """Configure StrongSORT's appearance gallery, confirmation, and matching costs.

        Image frames are required when camera-motion compensation is enabled
        (ECC by default). Appearance embeddings are required and are generated
        with ReID when absent from inputs.
        Detection filtering uses min_conf, confirmation uses n_init, and geometric
        fallback matching uses max_iou_dist.

        Args:
            config: Immutable algorithm settings. None selects StrongSortConfig defaults.
            kalman: Immutable filter noise, timing, and supported behavior settings.
                None preserves tracker defaults.
            reid: Immutable encoder configuration or a canonical appearance encoder.
                Missing embeddings are generated lazily; None selects the default encoder.
            **kwargs: Runtime ``per_class``, ``is_obb``, ``class_ids``, ``class_names``,
                ``mask_guidance``, and ``edgetam`` settings.
        """
        config = StrongSortConfig.resolve(config)
        validate_runtime_options(kwargs)
        self.config = config
        super().__init__(
            det_thresh=config.det_thresh,
            max_age=config.max_age,
            max_obs=config.max_obs,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold,
            asso_func=config.asso_func,
            centroid_max_distance=config.centroid_max_distance,
            kalman=kalman,
            reid=reid,
            **kwargs,
        )

        self.motion_model = create_motion_model(
            resolve_motion_model_kind("strongsort", self.kalman_config.type, is_obb=self.is_obb),
            is_obb=self.is_obb,
            max_obs=self.max_obs,
        )
        self.min_conf = config.min_conf
        self._max_cos_dist = float(config.max_cos_dist)
        self._nn_budget = int(config.nn_budget) if config.nn_budget is not None else None
        self.metric = self._new_metric()
        self.max_iou_dist = self._geometry_cost_threshold(config.max_iou_dist)
        self.n_init = config.n_init
        self.mc_lambda = config.mc_lambda
        self.ema_alpha = config.ema_alpha
        self.gating_only_position = config.gating_only_position
        self.geometric_recovery_max_age = config.geometric_recovery_max_age
        self.active_tracks: list[Track] = []
        self.cmc = create_cmc(config.cmc_method, enabled=config.use_cmc)
        self._requires_frame = self._requires_frame or self.cmc is not None
        if self.cmc is not None:
            self._requires_frame_dimensions_only = False

    def cmc_detection_boxes(self, dets: np.ndarray) -> np.ndarray:
        """Apply the tracking confidence threshold unless CMC sets its own."""
        if self.config.cmc_min_conf is None:
            dets = dets[self.detection_layout.confidences(dets) >= self.min_conf]
        return super().cmc_detection_boxes(dets)

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
        img: np.ndarray | None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run prediction, both associations, and lifecycle updates on the owned pool."""
        batch = self._make_detection_batch(dets, embs=embs, masks=masks, copy=False)
        batch = batch.select(batch.confs >= self.min_conf)
        features = resolve_batch_embeddings(batch)

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

        Track.multi_predict(self.active_tracks, dt=self._prediction_dt)
        self.apply_cmc(img, dets, self.active_tracks)

        # Confirmed identities compete for every detection using appearance.
        confirmed_indices: list[int] = []
        unconfirmed_indices: list[int] = []
        for index, track in enumerate(self.active_tracks):
            (confirmed_indices if track.is_confirmed() else unconfirmed_indices).append(index)
        confirmed_tracks = [self.active_tracks[index] for index in confirmed_indices]
        appearance_matches, _, remaining_indices = min_cost_matching(
            self._appearance_cost(confirmed_tracks, detections),
            self.metric.matching_threshold,
        )
        pairs = [(confirmed_tracks[row], detections[col]) for row, col in appearance_matches]

        # Keep global pool indices for the established geometric tie order.
        unmatched_confirmed = list(
            set(confirmed_indices) - set(confirmed_indices[row] for row, _ in appearance_matches)
        )
        recovery_indices = unconfirmed_indices + [
            index
            for index in unmatched_confirmed
            if self.active_tracks[index].time_since_update <= self.geometric_recovery_max_age
        ]
        unmatched_indices = [
            index
            for index in unmatched_confirmed
            if self.active_tracks[index].time_since_update > self.geometric_recovery_max_age
        ]
        recovery_tracks = [self.active_tracks[index] for index in recovery_indices]
        remaining_detections = [detections[index] for index in remaining_indices]
        geometry_matches, unmatched_recovery, unmatched_detections = min_cost_matching(
            self._association_cost(recovery_tracks, remaining_detections),
            self.max_iou_dist,
        )
        pairs.extend((recovery_tracks[row], remaining_detections[col]) for row, col in geometry_matches)
        unmatched_indices.extend(recovery_indices[row] for row in unmatched_recovery)

        # Correct both passes together before handling misses and births.
        Track.multi_update(pairs)
        for index in set(unmatched_indices):
            self.active_tracks[index].mark_missed()
        for index in unmatched_detections:
            detection = remaining_detections[index]
            self.active_tracks.append(
                Track(
                    detection,
                    self.id_allocator.alloc(),
                    self.n_init,
                    self.max_age,
                    self.max_obs,
                    ema_alpha=self.ema_alpha,
                    is_obb=detection.is_obb,
                    noise_config=self.kalman_noise_config,
                    motion_model=self.motion_model,
                )
            )
        self.active_tracks = [track for track in self.active_tracks if not track.is_deleted()]

        active_targets: list[int] = []
        gallery_features: list[np.ndarray] = []
        targets: list[int] = []
        visible: list[Track] = []
        for track in self.active_tracks:
            if not track.is_confirmed():
                continue
            active_targets.append(track.id)
            # Missed identities also retain their latest feature in the gallery.
            gallery_features.extend(track.features)
            targets.extend([track.id] * len(track.features))
            if track.time_since_update < 1:
                visible.append(track)
        self.metric.partial_fit(np.asarray(gallery_features), np.asarray(targets), active_targets)

        boxes = Track.batch_geometry(visible, is_obb=self.is_obb)
        if boxes is None:
            boxes = np.asarray([track.xywha if self.is_obb else track.to_tlbr() for track in visible])
        return self.format_outputs(visible, dtype=np.float32, boxes=boxes)

    def _appearance_cost(
        self,
        tracks: Sequence[Track],
        detections: Sequence[_Detection],
    ) -> np.ndarray:
        """Combine gallery distance with optional IoU, confidence, and motion cues."""
        if not len(tracks) or not len(detections):
            return np.empty((len(tracks), len(detections)), dtype=float)
        features = np.asarray([detection.feat for detection in detections])
        targets = np.asarray([track.id for track in tracks])
        cost_matrix = self.metric.distance(features, targets)
        if self.config.fuse_iou:
            cost_matrix = fuse_iou(cost_matrix, tracks, detections, is_obb=self.is_obb)
        if self.config.fuse_detection_score:
            cost_matrix = fuse_score(cost_matrix, detections)

        gated_mask = np.zeros(cost_matrix.shape, dtype=bool)
        if self.config.motion_fusion == "blend":
            cost_matrix = gate_cost_matrix(
                cost_matrix,
                tracks,
                detections,
                self.mc_lambda,
                only_position=self.gating_only_position,
                gated_mask=gated_mask,
            )
        elif self.config.motion_fusion == "gate":
            cost_matrix = gate_motion_cost(
                cost_matrix,
                tracks,
                self._association_boxes(detections),
                only_position=self.gating_only_position,
            )
            gated_mask[:] = ~np.isfinite(cost_matrix)
        # Assignment must never recover a motion rejection by rescaling a
        # finite sentinel, changing the cost threshold, or applying mask cues.
        cost_matrix[gated_mask] = np.inf
        return self._condition_gated_association(
            cost_matrix,
            tracks,
            detections,
            threshold=self.metric.matching_threshold,
            gated_mask=gated_mask,
        )

    def _association_cost(
        self,
        tracks: Sequence[Track],
        detections: Sequence[_Detection],
    ) -> np.ndarray:
        """Build the selected geometric distance matrix for StrongSORT fallback matching."""
        if not len(tracks) or not len(detections):
            return np.empty((len(tracks), len(detections)), dtype=float)
        cost_matrix = self.association_distance(tracks, detections)
        for row, track in enumerate(tracks):
            if track.time_since_update > self.geometric_recovery_max_age:
                cost_matrix[row, :] = INFTY_COST
        return self._condition_gated_association(
            cost_matrix,
            tracks,
            detections,
            threshold=self.max_iou_dist,
            gated_mask=cost_matrix >= INFTY_COST if self._mask_guidance is not None else None,
        )

    def _condition_gated_association(
        self,
        costs: np.ndarray,
        tracks: Sequence[Track],
        detections: Sequence[_Detection],
        *,
        threshold: float,
        gated_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Apply masks while preserving motion and stale-track hard gates."""
        if self._mask_guidance is None:
            return self._gate_association(costs, tracks, detections)
        protected_costs = costs[gated_mask]
        costs[gated_mask] = np.inf
        adjusted = self._condition_association(costs, tracks, detections, threshold=threshold)
        adjusted[gated_mask] = protected_costs
        return adjusted

    def reset(self) -> None:
        """Clear active tracks and appearance galleries for a new sequence."""
        self._reset_common_state()
        self.metric = self._new_metric()
