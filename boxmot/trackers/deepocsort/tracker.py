# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import numpy as np
from typing_extensions import Unpack

from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.common.appearance import (
    confidence_aware_alpha,
    resolve_batch_embeddings,
)
from boxmot.trackers.common.association import (
    AssociationStage,
    detection_track_similarity_assignment,
    run_association_stage,
    solve_assignment,
)
from boxmot.trackers.common.association.velocity import associate
from boxmot.trackers.common.box.base import BoxTracker
from boxmot.trackers.common.constructor import BoxTrackerOptions, validate_runtime_options
from boxmot.trackers.common.motion.batching import predict_tracks, update_tracks
from boxmot.trackers.common.motion.cmc.registry import create_cmc
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
from boxmot.trackers.common.tracking.observations import k_previous_obs
from boxmot.trackers.deepocsort.config import DeepOcSortConfig
from boxmot.trackers.deepocsort.track import DeepOBBKalmanBoxTracker, KalmanBoxTracker


class DeepOcSort(BoxTracker):
    """Track AABB or OBB detections with observation-centric motion and optional ReID.

    Attributes:
        cmc: Camera-motion compensation method.
    """

    accepts_embeddings = True

    supports_variable_dt = True
    uses_frame_dimensions_for_association = True

    def __init__(
        self,
        config: DeepOcSortConfig | None = None,
        *,
        kalman: KalmanConfig | None = None,
        reid: ReIDConfig | AppearanceEncoder | None = None,
        **kwargs: Unpack[BoxTrackerOptions],
    ) -> None:
        """Configure motion-direction matching and adaptive appearance weighting.

        Args:
            config: Immutable algorithm settings. None selects DeepOcSortConfig defaults.
            kalman: Immutable filter noise, timing, and supported behavior settings.
                None preserves tracker defaults.
            reid: Immutable encoder configuration or a canonical appearance encoder.
                Missing embeddings are generated lazily; None selects the default encoder.
            **kwargs: Runtime ``per_class``, ``is_obb``, ``class_ids``, ``class_names``,
                ``mask_guidance``, and ``edgetam`` settings.
        """
        config = DeepOcSortConfig.resolve(config)
        validate_runtime_options(kwargs)
        self.config = config
        super().__init__(
            det_thresh=config.det_thresh,
            max_age=config.max_age,
            max_obs=config.max_obs,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold,
            asso_func=config.asso_func,
            kalman=kalman,
            reid=reid,
            **kwargs,
        )

        self.delta_t = config.delta_t
        self.inertia = config.inertia
        self.w_association_emb = config.w_association_emb
        self.alpha_fixed_emb = config.alpha_fixed_emb
        self.aw_param = config.aw_param
        self.use_embeddings = config.use_embeddings
        self.cmc_off = config.cmc_off
        self.aw_off = config.aw_off
        # "similarity transforms using feature point extraction, optical flow, and RANSAC"
        self.cmc = create_cmc("sof", enabled=not self.cmc_off)
        self._requires_frame = self._requires_frame or self.cmc is not None
        if self.cmc is not None:
            self._requires_frame_dimensions_only = False

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        """Update tracks for one frame.

        Args:
            dets: Detection array for the current frame in the active BoxMOT
                layout.
            img: Current image frame.
            embs: Optional appearance embeddings aligned with ``dets``.

        Returns:
            Array of active tracks with the object ID in the last column.

        Notes:
            Call this once per frame, including frames with no detections.
            Pass an empty detection array with the matching layout when a frame
            has no detections. The number of returned tracks may differ from the
            number of detections provided.
        """
        # dets, s, c = dets.data
        # print(dets, s, c)
        self.frame_count += 1

        batch = self._make_detection_batch(dets, embs=embs, masks=masks)
        batch = batch.select(batch.confs > self.det_thresh)
        dets = batch.as_indexed_detections(dtype=dets.dtype)

        # appearance descriptor extraction
        dets_embs = resolve_batch_embeddings(
            batch,
            enabled=self.use_embeddings,
            placeholder_value=1.0,
        )

        # CMC
        if not self.cmc_off:
            self.apply_cmc(img, dets, self.active_tracks)

        # From [self.alpha_fixed_emb, 1], goes to 1 as detector is less confident.
        dets_alpha = confidence_aware_alpha(
            batch.confs,
            self.det_thresh,
            base_alpha=self.alpha_fixed_emb,
        )

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), self.detection_layout.box_with_conf_cols))
        trk_embs = []
        to_del = []
        ret = []
        predictions = predict_tracks(self.active_tracks, dt=self._prediction_dt)
        for t, trk in enumerate(trks):
            pos = predictions[t][0]
            trk[:] = [*pos[: self.detection_layout.box_cols], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trk_embs.append(self.active_tracks[t].get_emb())
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        if len(trk_embs) > 0:
            trk_embs = np.vstack(trk_embs)
        else:
            trk_embs = np.array(trk_embs)

        for t in reversed(to_del):
            self.active_tracks.pop(t)

        velocities = np.array(
            [trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.active_tracks]
        )
        last_boxes = np.array([trk.last_observation for trk in self.active_tracks])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t, is_obb=self.is_obb) for trk in self.active_tracks]
        )

        """
            First round of association
        """
        # (M detections X N tracks, final score)
        if not self.use_embeddings or dets.shape[0] == 0 or trk_embs.shape[0] == 0:
            stage1_emb_cost = None
        else:
            stage1_emb_cost = dets_embs @ trk_embs.T
        first_stage = AssociationStage(
            name="deepocsort_high",
            threshold=self.iou_threshold,
            matcher=lambda _tracks, _detections: associate(
                dets[:, : self.detection_layout.box_with_conf_cols],
                trks,
                self.asso_func,
                self.iou_threshold,
                velocities,
                k_observations,
                self.inertia,
                stage1_emb_cost,
                self.w_association_emb,
                self.aw_off,
                self.aw_param,
                is_obb=self.is_obb,
                similarity_conditioner=lambda similarity: self._condition_similarity(
                    similarity, _tracks, _detections, threshold=self.iou_threshold
                ),
            ),
        )
        first_result = run_association_stage(first_stage, self.active_tracks, dets)
        matched = first_result.matches
        unmatched_dets = first_result.unmatched_dets
        unmatched_trks = first_result.unmatched_tracks
        update_tracks([self.active_tracks[t] for t, _ in matched], [dets[d] for _, d in matched])
        for trk_idx, det_idx in matched:
            self.active_tracks[trk_idx].update_emb(dets_embs[det_idx], alpha=dets_alpha[det_idx])

        """
            Second round of associaton by OCR
        """
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            # New tracks retain a negative-confidence sentinel until their first
            # explicit update. Keep those rows out of strict OBB geometry while
            # preserving the mapping back to active-track indices.
            rematch_trk_indices = unmatched_trks[last_boxes[unmatched_trks, -1] >= 0]
            if rematch_trk_indices.size:
                left_trks = last_boxes[rematch_trk_indices]
                similarity = np.asarray(self.asso_func(left_dets, left_trks))
                similarity = self._condition_similarity(
                    similarity,
                    [self.active_tracks[t] for t in rematch_trk_indices],
                    left_dets,
                    threshold=self.iou_threshold,
                )
                rematch_stage = AssociationStage(
                    name="deepocsort_ocr_rematch",
                    threshold=self.iou_threshold,
                    matcher=lambda _tracks, _detections: detection_track_similarity_assignment(
                        similarity,
                        self.iou_threshold,
                        solve_assignment,
                    ),
                )
                rematch_result = run_association_stage(rematch_stage, left_trks, left_dets)
                to_remove_det_indices = [unmatched_dets[d] for _, d in rematch_result.matches]
                to_remove_trk_indices = [rematch_trk_indices[t] for t, _ in rematch_result.matches]
                update_tracks(
                    [self.active_tracks[t] for t in to_remove_trk_indices],
                    [dets[d] for d in to_remove_det_indices],
                )
                for trk_rel, det_rel in rematch_result.matches:
                    det_ind = unmatched_dets[det_rel]
                    trk_ind = rematch_trk_indices[trk_rel]
                    self.active_tracks[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.active_tracks[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            tracker_cls = DeepOBBKalmanBoxTracker if self.is_obb else KalmanBoxTracker
            trk = tracker_cls(
                dets[i],
                delta_t=self.delta_t,
                emb=dets_embs[i],
                alpha=dets_alpha[i],
                max_obs=self.max_obs,
                id_allocator=self.id_allocator,
                noise_config=self.kalman_noise_config,
            )
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation[-1] < 0:
                d = trk.get_state()[0]
            else:
                """
                this is optional to use the recent observation or the kalman filter prediction,
                we didn't notice significant difference here
                """
                d = trk.last_observation[: self.detection_layout.box_cols]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(self.format_output_row(d, trk.id, trk.conf, trk.cls, trk.det_ind))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)
        return self.format_output_rows(ret, dtype=np.float32)

    def reset(self) -> None:
        self._reset_common_state()
