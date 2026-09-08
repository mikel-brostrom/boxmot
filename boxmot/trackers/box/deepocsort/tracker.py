# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from boxmot.trackers.box.base import BoxTracker
from boxmot.trackers.box.deepocsort.track import DeepOBBKalmanBoxTracker, KalmanBoxTracker
from boxmot.trackers.box.ocsort.track import k_previous_obs
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
from boxmot.trackers.common.motion.cmc import create_cmc


class DeepOcSort(BoxTracker):
    accepts_embeddings = True

    supports_variable_dt = True
    uses_frame_dimensions_for_association = True

    """Initialize the DeepOcSort tracker.

    Args:
        delta_t (int): Time window used for motion estimation.
        inertia (float): Motion-consistency weight.
        w_association_emb (float): Weight applied to appearance distance during
            matching.
        alpha_fixed_emb (float): Fixed update rate for track embeddings.
        aw_param (float): Adaptive-weighting parameter for motion versus
            appearance.
        use_embeddings (bool): Whether to use appearance embeddings, generating
            them from the frame when absent.
        cmc_off (bool): Whether to disable camera-motion compensation.
        aw_off (bool): Whether to disable adaptive appearance weighting.
        reid_model (Any | None): Optional pre-built ReID backend used when
            embeddings are absent.
        reid_weights (str | Path | list[str | Path] | tuple[str | Path, ...] | None):
            Weights for the lazily constructed ReID backend.
        device (Any): Device used by the lazily constructed ReID backend.
        half (bool): Whether the lazy ReID backend uses FP16 inference.
        reid_preprocess (str | None): Optional ReID preprocessing profile.
        **kwargs (Any): Base tracker settings forwarded to :class:`BaseTracker`,
            including ``det_thresh``, ``max_age``, ``max_obs``, ``min_hits``,
            ``iou_threshold``, ``per_class``, ``class_ids``, ``class_names``,
            ``asso_func``, and ``is_obb``.

    Attributes:
        cmc: Camera-motion compensation method.
    """

    def __init__(
        self,
        # DeepOcSort-specific parameters
        delta_t: int = 3,
        inertia: float = 0.2,
        w_association_emb: float = 0.5,
        alpha_fixed_emb: float = 0.95,
        aw_param: float = 0.5,
        use_embeddings: bool = True,
        cmc_off: bool = False,
        aw_off: bool = False,
        *,
        reid_model: Any | None = None,
        reid_weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        device: Any = "cpu",
        half: bool = False,
        reid_preprocess: str | None = None,
        **kwargs: Any,  # BaseTracker parameters
    ):
        super().__init__(
            reid_model=reid_model,
            reid_weights=reid_weights,
            device=device,
            half=half,
            reid_preprocess=reid_preprocess,
            **kwargs,
        )

        """
        Sets key parameters for SORT
        """
        self.delta_t = delta_t
        self.inertia = inertia
        self.w_association_emb = w_association_emb
        self.alpha_fixed_emb = alpha_fixed_emb
        self.aw_param = aw_param
        if not isinstance(use_embeddings, bool):
            raise TypeError("use_embeddings must be bool.")
        self.use_embeddings = use_embeddings
        self.cmc_off = cmc_off
        self.aw_off = aw_off
        # "similarity transforms using feature point extraction, optical flow, and RANSAC"
        self.cmc = create_cmc("sof", enabled=not self.cmc_off)
        self._requires_frame = self._requires_frame or self.cmc is not None

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
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict(dt=self._prediction_dt)[0]
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
            ),
        )
        first_result = run_association_stage(first_stage, self.active_tracks, dets)
        matched = first_result.matches
        unmatched_dets = first_result.unmatched_dets
        unmatched_trks = first_result.unmatched_tracks
        for trk_idx, det_idx in matched:
            self.active_tracks[trk_idx].update(dets[det_idx, :])
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
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for trk_rel, det_rel in rematch_result.matches:
                    det_ind = unmatched_dets[det_rel]
                    trk_ind = rematch_trk_indices[trk_rel]
                    self.active_tracks[trk_ind].update(dets[det_ind, :])
                    self.active_tracks[trk_ind].update_emb(dets_embs[det_ind], alpha=dets_alpha[det_ind])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
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
