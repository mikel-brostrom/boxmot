from __future__ import annotations

from typing import List, Optional

import numpy as np
from typing_extensions import Unpack

from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.boosttrack.config import BoostTrackConfig
from boxmot.trackers.boosttrack.track import KalmanBoxTracker
from boxmot.trackers.common.appearance import (
    confidence_aware_alpha,
    resolve_batch_embeddings,
)
from boxmot.trackers.common.association.boost import (
    MhDist_similarity,
    associate,
    shape_similarity,
    shape_similarity_obb,
    soft_biou_batch,
    soft_biou_batch_obb,
)
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.association.matching import apply_motion_fusion, fuse_iou, fuse_score
from boxmot.trackers.common.box.base import BoxTracker
from boxmot.trackers.common.constructor import BoxTrackerOptions, validate_runtime_options
from boxmot.trackers.common.geometry.obb import xywha_to_xyxy
from boxmot.trackers.common.motion.batching import current_track_boxes, predict_tracks, update_tracks
from boxmot.trackers.common.motion.cmc.registry import create_cmc, sof_options
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig


class BoostTrack(BoxTracker):
    """Track AABB or OBB detections with confidence boosting and multi-cue matching.

    Attributes:
        frame_count (int): Number of processed frames.
        active_tracks (list): Currently active tracks.
        trackers (list[KalmanBoxTracker]): Internal Kalman trackers.
        cmc: Camera-motion compensation method when enabled.
    """

    supports_variable_dt = True

    accepts_embeddings = True
    uses_frame_dimensions_for_association = True

    def __init__(
        self,
        config: BoostTrackConfig | None = None,
        *,
        kalman: KalmanConfig | None = None,
        reid: ReIDConfig | AppearanceEncoder | None = None,
        **kwargs: Unpack[BoxTrackerOptions],
    ) -> None:
        """Configure confidence boosting, association, and optional appearance features.

        Args:
            config: Immutable algorithm settings. None selects BoostTrackConfig defaults.
            kalman: Immutable filter noise, timing, and supported behavior settings.
                None preserves tracker defaults.
            reid: Immutable encoder configuration or a canonical appearance encoder.
                Missing embeddings are generated lazily; None selects the default encoder.
            **kwargs: Runtime ``per_class``, ``is_obb``, ``class_ids``, ``class_names``,
                ``mask_guidance``, and ``edgetam`` settings.
        """
        config = BoostTrackConfig.resolve(config)
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

        self.active_tracks = []
        self.frame_count = 0
        self.trackers: List[KalmanBoxTracker] = []

        # Parameters for BoostTrack (these can be tuned as needed)
        self.use_cmc = config.use_cmc  # use camera motion compensation
        self.min_box_area = config.min_box_area  # minimum box area for detections
        self.aspect_ratio_thresh = config.aspect_ratio_thresh  # aspect ratio threshold for detections
        self.cmc_method = config.cmc_method

        self.lambda_iou = config.lambda_iou
        self.lambda_mhd = config.lambda_mhd
        self.lambda_shape = config.lambda_shape
        self.lambda_emb_multiplier = config.lambda_emb_multiplier
        self.alpha_fixed_emb = config.alpha_fixed_emb
        self.appearance_thresh = config.appearance_thresh
        self.appearance_iou_fraction = config.appearance_iou_fraction
        self.use_dlo_boost = config.use_dlo_boost
        self.use_duo_boost = config.use_duo_boost
        self.dlo_boost_coef = config.dlo_boost_coef
        self.s_sim_corr = config.s_sim_corr

        self.use_rich_s = config.use_rich_s
        self.use_sb = config.use_sb
        self.use_vt = config.use_vt
        self.dlo_soft_alpha = config.dlo_soft_alpha
        self.dlo_soft_power = config.dlo_soft_power
        self.dlo_threshold_start = config.dlo_threshold_start
        self.dlo_threshold_end = config.dlo_threshold_end
        self.duo_mahalanobis_thresh = config.duo_mahalanobis_thresh
        self.duo_iou_thresh = config.duo_iou_thresh

        self.use_embeddings = config.use_embeddings

        cmc_options = sof_options(config) if config.cmc_method == "sof" else {}
        self.cmc = create_cmc(config.cmc_method, enabled=self.use_cmc, **cmc_options)
        self._requires_frame = self._requires_frame or self.cmc is not None
        if self.cmc is not None:
            self._requires_frame_dimensions_only = False

    def _fuse_first_motion(self, costs: np.ndarray, tracks: list, boxes: np.ndarray) -> np.ndarray:
        """Apply optional motion fusion to detection-by-track first-pass costs."""
        return apply_motion_fusion(
            costs.T,
            tracks,
            boxes,
            mode=self.config.motion_fusion,
            only_position=self.config.motion_fusion_only_position,
            lambda_=self.config.motion_fusion_lambda,
        ).T

    def _track_boxes(self, tracks: list[KalmanBoxTracker]) -> np.ndarray:
        """Decode current states through the shared native/custom geometry policy."""
        return current_track_boxes(tracks, is_obb=self.is_obb)

    def _predict_track_rows(self, img: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Predict once and collect native geometry and confidence for all cues."""
        predictions = predict_tracks(self.trackers, dt=self._prediction_dt)
        if self.cmc is not None:
            self.apply_cmc(img, detections, self.trackers)
            boxes = self._track_boxes(self.trackers)
        else:
            boxes = np.concatenate(predictions, axis=0) if predictions else self._track_boxes([])
        rows = np.empty((len(self.trackers), self.detection_layout.box_with_conf_cols), dtype=float)
        rows[:, :-1] = boxes
        rows[:, -1] = [track.get_confidence() for track in self.trackers]
        return rows

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: Optional[np.ndarray] = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
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
        batch = self._make_detection_batch(dets, embs=embs, masks=masks, copy=False)
        indexed_dets = batch.as_indexed_detections(dtype=dets.dtype)

        self.frame_count += 1

        tracker_rows = self._predict_track_rows(img, indexed_dets)
        mh_dist_matrix = (
            self.get_mh_dist_matrix(indexed_dets)
            if self.use_duo_boost or (self.use_dlo_boost and self.use_rich_s)
            else None
        )

        assoc_dets = self.aabb_detections_for_association(indexed_dets).copy()
        if self.use_dlo_boost:
            if self.is_obb:
                indexed_dets = self.dlo_confidence_boost_obb(
                    indexed_dets, tracker_rows=tracker_rows, motion_distances=mh_dist_matrix
                )
                batch = batch.with_confs(self.detection_layout.confidences(indexed_dets))
                assoc_dets = self.aabb_detections_for_association(indexed_dets)
            else:
                assoc_dets = self.dlo_confidence_boost(
                    assoc_dets, tracker_rows=tracker_rows, motion_distances=mh_dist_matrix
                )
        if self.use_duo_boost:
            if self.is_obb:
                indexed_dets = self.duo_confidence_boost_obb(indexed_dets, motion_distances=mh_dist_matrix)
                batch = batch.with_confs(self.detection_layout.confidences(indexed_dets))
                assoc_dets = self.aabb_detections_for_association(indexed_dets)
            else:
                assoc_dets = self.duo_confidence_boost(assoc_dets, motion_distances=mh_dist_matrix)

        keep = assoc_dets[:, 4] >= self.det_thresh
        if mh_dist_matrix is not None and mh_dist_matrix.size:
            mh_dist_matrix = mh_dist_matrix[keep]
        assoc_dets = assoc_dets[keep]
        batch = batch.select(keep).with_confs(assoc_dets[:, 4])
        dets = batch.as_indexed_detections(dtype=dets.dtype)
        scores = batch.confs
        if mh_dist_matrix is None:
            mh_dist_matrix = self.get_mh_dist_matrix(dets)

        dets_embs = resolve_batch_embeddings(
            batch,
            enabled=self.use_embeddings,
            placeholder_value=1.0,
        )

        if self.use_embeddings and len(self.trackers) > 0:
            tracker_embs = np.array([trk.get_emb() for trk in self.trackers])
            if dets_embs.shape[0] == 0:
                emb_cost = np.empty((0, tracker_embs.shape[0]))
            else:
                emb_cost = (
                    dets_embs.reshape(dets_embs.shape[0], -1) @ tracker_embs.reshape((tracker_embs.shape[0], -1)).T
                )
        else:
            emb_cost = None

        association_dets = dets[:, : self.detection_layout.box_with_conf_cols] if self.is_obb else assoc_dets
        tracker_boxes = tracker_rows[:, :-1]
        geometry_similarity = self.asso_func(batch.boxes, tracker_boxes)
        if self.config.fuse_detection_score:
            geometry_similarity = 1.0 - fuse_score((1.0 - geometry_similarity).T, batch.confs).T
        if emb_cost is not None and self.config.fuse_iou:
            emb_cost = 1.0 - fuse_iou(1.0 - emb_cost, batch.boxes, tracker_boxes, is_obb=self.is_obb)
        oriented_shape = shape_similarity_obb(batch.boxes, tracker_boxes) if self.is_obb else None
        matched, unmatched_dets, unmatched_trks, _ = associate(
            association_dets,
            tracker_rows,
            self.iou_threshold,
            mahalanobis_distance=mh_dist_matrix,
            track_confidence=tracker_rows[:, -1:],
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
            s_sim_corr=self.s_sim_corr,
            lambda_emb_multiplier=self.lambda_emb_multiplier,
            appearance_thresh=self.appearance_thresh,
            appearance_iou_fraction=self.appearance_iou_fraction,
            geometry_matrix=geometry_similarity,
            shape_matrix=oriented_shape,
            gate_before_assignment=self._asso_func_base_name == "centroid",
            geometry_conditioner=lambda similarity: self._condition_similarity(
                similarity, self.trackers, batch.boxes, threshold=self.iou_threshold
            ),
            cost_conditioner=(
                lambda costs: self._fuse_first_motion(costs, self.trackers, batch.boxes)
            ) if self.config.motion_fusion != "off" else None,
        )

        dets_alpha = confidence_aware_alpha(batch.confs, self.det_thresh, base_alpha=self.alpha_fixed_emb)

        update_tracks([self.trackers[t] for _, t in matched], [dets[d] for d, _ in matched])
        for m in matched:
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for i in unmatched_dets:
            if batch.confs[i] >= self.det_thresh:
                self.trackers.append(
                    KalmanBoxTracker(
                        dets[i, :],
                        max_obs=self.max_obs,
                        emb=dets_embs[i],
                        is_obb=self.is_obb,
                        adaptive_kf=self.kalman_config.adaptive_kf,
                        id_allocator=self.id_allocator,
                        noise_config=self.kalman_noise_config,
                        kalman_type=self.kalman_config.type,
                        track_confidence_decay=self.config.track_confidence_decay,
                        track_confidence_warmup=self.config.track_confidence_warmup,
                    )
                )

        self.active_tracks = [
            trk for trk in self.trackers
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)
        ]

        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

        outputs = self.format_outputs(self.active_tracks, dtype=np.float32, boxes=self._track_boxes(self.active_tracks))
        return self.filter_outputs(outputs)

    def filter_outputs(self, outputs: np.ndarray) -> np.ndarray:
        return self.filter_outputs_by_geometry(
            outputs,
            min_box_area=self.min_box_area,
            max_aspect_ratio=self.aspect_ratio_thresh,
        )

    def reset(self) -> None:
        self._reset_common_state()

    def _current_track_rows(self) -> np.ndarray:
        """Collect current geometry and confidence without advancing filters."""
        return np.column_stack((self._track_boxes(self.trackers), [track.get_confidence() for track in self.trackers]))

    def get_iou_matrix(
        self, detections: np.ndarray, buffered: bool = False, *, tracker_rows: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute an AABB cue from the shared predicted-track snapshot."""
        trackers = self._current_track_rows() if tracker_rows is None else tracker_rows
        if self.is_obb:
            trackers = np.column_stack((xywha_to_xyxy(trackers[:, :5]), trackers[:, 5]))

        return (
            AssociationFunction.iou_batch(detections, trackers)
            if not buffered
            else soft_biou_batch(detections, trackers)
        )

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int | None = None) -> np.ndarray:
        """Preserve host arithmetic unless an explicit policy selects aligned cues."""
        if not self.trackers:
            return np.zeros((0, 0))
        model = self.trackers[0].motion_model
        indices = np.asarray(model.geometry_measurement_indices)
        if n_dims is not None:
            indices = indices[:n_dims]
        dimensions = len(indices)
        measurements = model.to_measurements(detections)[:, indices]
        means = np.asarray([np.asarray(track.kf.x).reshape(-1)[indices] for track in self.trackers])
        inverse_variances = np.asarray([np.reciprocal(np.diag(track.kf.P)[indices]) for track in self.trackers])
        return (
            (measurements.reshape((-1, 1, dimensions)) - means.reshape((1, -1, dimensions))) ** 2
            * inverse_variances.reshape((1, -1, dimensions))
        ).sum(axis=2)

    def duo_confidence_boost(
        self, detections: np.ndarray, *, motion_distances: np.ndarray | None = None
    ) -> np.ndarray:
        """Promote isolated low-confidence observations using the shared motion cue."""
        if len(detections) == 0:
            return detections

        n_dims = 4
        limit = self.duo_mahalanobis_thresh
        mh_dist = self.get_mh_dist_matrix(detections, n_dims) if motion_distances is None else motion_distances

        # If there are no existing trackers, bail out immediately
        if mh_dist.size == 0:
            return detections

        min_dists = mh_dist.min(1)
        mask = (min_dists > limit) & (detections[:, 4] < self.det_thresh)
        boost_inds = np.where(mask)[0]
        iou_limit = self.duo_iou_thresh
        if len(boost_inds) == 0:
            return detections

        bdiou = AssociationFunction.iou_batch(detections[boost_inds], detections[boost_inds]) - np.eye(len(boost_inds))
        bdiou_max = bdiou.max(axis=1)
        remaining = boost_inds[bdiou_max <= iou_limit].tolist()
        args = np.where(bdiou_max > iou_limit)[0]
        for i in range(len(args)):
            bi = args[i]
            tmp = np.where(bdiou[bi] > iou_limit)[0]
            args_tmp = np.append(np.intersect1d(boost_inds[args], boost_inds[tmp]), boost_inds[bi])
            conf_max = np.max(detections[args_tmp, 4])
            if detections[boost_inds[bi], 4] == conf_max:
                remaining.append(int(boost_inds[bi]))

        mask_boost = np.zeros_like(detections[:, 4], dtype=bool)
        mask_boost[remaining] = True
        detections[:, 4] = np.where(mask_boost, self.det_thresh + 1e-4, detections[:, 4])
        return detections

    def dlo_confidence_boost(
        self,
        detections: np.ndarray,
        *,
        tracker_rows: np.ndarray | None = None,
        motion_distances: np.ndarray | None = None,
    ) -> np.ndarray:
        """Boost AABB confidence while keeping scores in the public [0, 1] range."""
        if len(detections) == 0 or not self.trackers:
            return detections
        trackers = self._current_track_rows() if tracker_rows is None else tracker_rows

        if self.use_rich_s:
            sbiou_matrix = self.get_iou_matrix(detections, True, tracker_rows=trackers)
            distances = self.get_mh_dist_matrix(detections) if motion_distances is None else motion_distances
            mhd_sim = MhDist_similarity(distances, 1)
            shape_sim = shape_similarity(detections, trackers, self.s_sim_corr)
            S = (mhd_sim + shape_sim + sbiou_matrix) / 3
        else:
            S = self.get_iou_matrix(detections, False, tracker_rows=trackers)

        if not self.use_sb and not self.use_vt:
            max_s = S.max(1)
            detections[:, 4] = np.clip(np.maximum(detections[:, 4], max_s * self.dlo_boost_coef), 0.0, 1.0)
            return detections

        if self.use_sb:
            max_s = S.max(1)
            alpha = self.dlo_soft_alpha
            detections[:, 4] = np.maximum(
                detections[:, 4], alpha * detections[:, 4] + (1 - alpha) * max_s**self.dlo_soft_power
            )
        if self.use_vt:
            threshold_s = self.dlo_threshold_start
            threshold_e = self.dlo_threshold_end
            tmp = (
                S
                > np.maximum(threshold_s - np.array([trk.time_since_update - 1 for trk in self.trackers]), threshold_e)
            ).max(1)
            scores = detections[:, 4].copy()
            scores[tmp] = np.maximum(scores[tmp], self.det_thresh + 1e-5)
            detections[:, 4] = scores
        return detections

    def dlo_confidence_boost_obb(
        self,
        detections: np.ndarray,
        *,
        threshold: float | None = None,
        tracker_rows: np.ndarray | None = None,
        motion_distances: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply DLO boosting with oriented, representation-invariant geometry."""
        if len(detections) == 0 or len(self.trackers) == 0:
            return detections

        boosted = detections.copy()
        score_threshold = self.det_thresh if threshold is None else float(threshold)
        trackers = np.asarray(self._current_track_rows() if tracker_rows is None else tracker_rows, dtype=np.float32)
        boxes = self.detection_layout.boxes(boosted)

        if self.use_rich_s:
            distances = self.get_mh_dist_matrix(boosted) if motion_distances is None else motion_distances
            mhd_sim = MhDist_similarity(distances, 1)
            shape_sim = shape_similarity_obb(boxes, trackers[:, :5])
            soft_iou = soft_biou_batch_obb(
                np.column_stack((boxes, self.detection_layout.confidences(boosted))),
                trackers,
            )
            similarity = (mhd_sim + shape_sim + soft_iou) / 3
        else:
            similarity = AssociationFunction.iou_batch_obb(boxes, trackers[:, :5])

        conf_idx = self.detection_layout.conf_idx
        max_similarity = similarity.max(axis=1)
        if not self.use_sb and not self.use_vt:
            boosted[:, conf_idx] = np.clip(
                np.maximum(boosted[:, conf_idx], max_similarity * self.dlo_boost_coef),
                0.0,
                1.0,
            )
            return boosted

        if self.use_sb:
            alpha = self.dlo_soft_alpha
            boosted[:, conf_idx] = np.maximum(
                boosted[:, conf_idx],
                alpha * boosted[:, conf_idx] + (1 - alpha) * max_similarity**self.dlo_soft_power,
            )
        if self.use_vt:
            visibility_thresholds = np.maximum(
                self.dlo_threshold_start - np.array([trk.time_since_update - 1 for trk in self.trackers]),
                self.dlo_threshold_end,
            )
            visible = (similarity > visibility_thresholds).max(axis=1)
            boosted[visible, conf_idx] = np.maximum(
                boosted[visible, conf_idx],
                score_threshold + 1e-5,
            )
        return boosted

    def duo_confidence_boost_obb(
        self,
        detections: np.ndarray,
        *,
        threshold: float | None = None,
        motion_distances: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply DUO boosting using native OBB motion and pairwise overlap."""
        if len(detections) == 0:
            return detections
        threshold = self.det_thresh if threshold is None else float(threshold)
        mh_dist = self.get_mh_dist_matrix(detections) if motion_distances is None else motion_distances
        if mh_dist.size == 0:
            return detections

        conf_idx = self.detection_layout.conf_idx
        boost_indices = np.flatnonzero(
            (mh_dist.min(axis=1) > self.duo_mahalanobis_thresh) & (detections[:, conf_idx] < threshold)
        )
        if not len(boost_indices):
            return detections

        candidate_boxes = self.detection_layout.boxes(detections)[boost_indices]
        pairwise_iou = AssociationFunction.iou_batch_obb(candidate_boxes, candidate_boxes)
        np.fill_diagonal(pairwise_iou, 0.0)
        max_iou = pairwise_iou.max(axis=1)
        remaining = list(boost_indices[max_iou <= self.duo_iou_thresh])
        for local_index in np.flatnonzero(max_iou > self.duo_iou_thresh):
            neighbors = np.flatnonzero(pairwise_iou[local_index] > self.duo_iou_thresh)
            group = boost_indices[np.append(neighbors, local_index)]
            best = group[np.argmax(detections[group, conf_idx])]
            remaining.append(int(best))

        boosted = detections.copy()
        boosted[np.unique(remaining), conf_idx] = threshold + 1e-4
        return boosted
