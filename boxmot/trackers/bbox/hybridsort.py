from __future__ import annotations

# Hybrid-SORT-ReID with ECC + ReID (explicit config, BaseTracker-style)
# - Assumes detection input is M x [x1, y1, x2, y2, conf, cls]
# - ECC via shared CMC factory and BaseTracker.apply_cmc(...)
# - ReID via pre-built backend passed as ``reid_model``
# - update(dets, img, embs=None) signature compatible with BoxMOT trackers
# - Emits rows: [x1,y1,x2,y2, track_id, conf, cls, det_ind]
# - Preserves detector class IDs and det_ind; guards out-of-range detection indices
from typing import Any, List

import numpy as np

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.appearance import (
    ema_update_embedding,
    resolve_batch_embeddings,
)
from boxmot.trackers.common.association.hybrid import (
    associate_4_points_with_score,
    associate_4_points_with_score_with_reid,
    cal_score_dif_batch_two_score,
    ciou_batch,
    ct_dist,
    diou_batch,
    embedding_distance,
    giou_batch,
    hmiou,
    iou_batch,
    linear_assignment,
)
from boxmot.trackers.common.motion.cmc import create_cmc
from boxmot.trackers.common.track_models.hybridsort import KalmanBoxTracker, k_previous_obs
from boxmot.trackers.common.track_models.ocsort import KalmanBoxTracker as OBBKalmanBoxTracker

ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
    "hmiou": hmiou,
}


class HybridSort(BaseTracker):
    supports_obb = True

    """Initialize the HybridSort tracker.

    Args:
        reid_model (Any | None): Pre-built ReID backend model (e.g. ``ReID(...).model``).
        cmc_method (str): Camera-motion compensation method.
        with_reid (bool): Whether to enable appearance features.
        low_thresh (float): Low-confidence threshold for second-pass matching.
        delta_t (int): Time window used for motion estimation.
        inertia (float): Motion-consistency weight.
        use_byte (bool): Whether to enable ByteTrack-style second association.
        longterm_bank_length (int): Number of appearance features to keep in
            the long-term bank.
        alpha (float): Feature update coefficient.
        adapfs (bool): Whether to enable adaptive feature smoothing.
        track_thresh (float): High-confidence threshold for the first
            association pass.
        EG_weight_high_score (float): Embedding-guided association weight for
            high-score detections.
        EG_weight_low_score (float): Embedding-guided association weight for
            low-score detections.
        TCM_first_step (bool): Whether to enable TCM in the first step.
        TCM_byte_step (bool): Whether to enable TCM in the Byte step.
        TCM_byte_step_weight (float): TCM weight in the Byte step.
        high_score_matching_thresh (float): Threshold for high-score matching.
        with_longterm_reid (bool): Whether to enable long-term ReID features.
        longterm_reid_weight (float): Weight applied to long-term ReID scores.
        with_longterm_reid_correction (bool): Whether to enable long-term ReID
            correction.
        longterm_reid_correction_thresh (float): Correction threshold for
            regular detections.
        longterm_reid_correction_thresh_low (float): Correction threshold for
            low-score detections.
        dataset (str): Dataset hint used by the association logic.
        **kwargs (Any): Base tracker settings forwarded to :class:`BaseTracker`.

    Attributes:
        with_reid (bool): Whether appearance features are enabled.
        model: ReID model used for appearance extraction when enabled.
        cmc: Camera-motion compensation method.
        active_tracks (list[KalmanBoxTracker]): Currently active tracks.
    """

    def __init__(
        self,
        # ReID & CMC
        reid_model: Any | None = None,
        cmc_method: str = "ecc",
        with_reid: bool = True,
        # Hybrid-SORT specific
        low_thresh: float = 0.1,
        delta_t: int = 3,
        inertia: float = 0.05,
        use_byte: bool = True,
        # KF / ReID
        longterm_bank_length: int = 30,
        alpha: float = 0.9,
        adapfs: bool = False,
        track_thresh: float = 0.5,
        # Embedding-guided association
        EG_weight_high_score: float = 4.6,
        EG_weight_low_score: float = 1.3,
        # Two-step toggles / thresholds
        TCM_first_step: bool = True,
        TCM_byte_step: bool = True,
        TCM_byte_step_weight: float = 1.0,
        high_score_matching_thresh: float = 0.7,
        # Long-term reid
        with_longterm_reid: bool = True,
        longterm_reid_weight: float = 0.0,
        with_longterm_reid_correction: bool = True,
        longterm_reid_correction_thresh: float = 0.4,
        longterm_reid_correction_thresh_low: float = 0.4,
        # misc
        dataset: str = "",
        **kwargs: Any,  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="HybridSort", **kwargs)

        # store core knobs
        self.low_thresh = float(low_thresh)
        self.delta_t = int(delta_t)
        self.inertia = float(inertia)
        self.use_byte = bool(use_byte)

        self.longterm_bank_length = int(longterm_bank_length)
        self.alpha = float(alpha)
        self.adapfs = bool(adapfs)
        self.track_thresh = float(track_thresh)

        self.EG_weight_high_score = float(EG_weight_high_score)
        self.EG_weight_low_score = float(EG_weight_low_score)
        self.TCM_first_step = bool(TCM_first_step)
        self.TCM_byte_step = bool(TCM_byte_step)
        self.TCM_byte_step_weight = float(TCM_byte_step_weight)
        self.high_score_matching_thresh = float(high_score_matching_thresh)

        self.with_longterm_reid = bool(with_longterm_reid)
        self.longterm_reid_weight = float(longterm_reid_weight)
        self.with_longterm_reid_correction = bool(with_longterm_reid_correction)
        self.longterm_reid_correction_thresh = float(longterm_reid_correction_thresh)
        self.longterm_reid_correction_thresh_low = float(longterm_reid_correction_thresh_low)
        self.dataset = str(dataset)

        # ReID module (BotSort-style)
        self.with_reid = bool(with_reid)
        self.model = reid_model if self.with_reid else None

        # ECC CMC (BotSort-style)
        self.cmc = create_cmc(cmc_method)

        # container
        self.active_tracks: List[KalmanBoxTracker] = []

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        """
        dets: ndarray [N,6] -> [x1,y1,x2,y2,conf,cls]
        img: HxWxC image
        embs: optional [N,D] appearance features. If None and with_reid=True, we extract features for provided dets.
        Returns: ndarray [M,8]: [x1,y1,x2,y2,track_id,conf,cls,det_ind]
        """
        self.check_inputs(dets, img, embs)
        if self.is_obb:
            return self._update_obb(dets, img, embs, masks)
        self.frame_count += 1

        batch = self.make_detection_batch(dets, embs=embs, masks=masks)
        n_dets_full = len(batch)
        dets_indexed = batch.as_indexed_detections(dtype=dets.dtype)

        # helper guards
        def _safe_detind(x: int, n: int) -> int:
            xi = int(x)
            return xi if 0 <= xi < n else -1

        def _safe_cls(x: int) -> int:
            xi = int(x)
            self.class_catalog.validate_ids([xi])
            return xi

        # ECC: compute warp using all current detections (BotSort pattern)
        if n_dets_full:
            self.apply_cmc(img, dets_indexed, self.active_tracks)

        # ReID: get features if not provided
        if self.with_reid:
            batch = batch.with_embs(
                resolve_batch_embeddings(
                    batch,
                    img,
                    model=self.model,
                    placeholder_dim=128,
                    placeholder_value=0.0,
                )
            )

        # FIRST/SECOND stage split (Hybrid semantics)
        high_batch, second_batch = batch.split_by_confidence(
            high_thresh=self.det_thresh,
            low_thresh=self.low_thresh,
        )
        cls_keep = high_batch.clss.astype(int)
        cls_second = second_batch.clss.astype(int)

        id_feature_keep = resolve_batch_embeddings(
            high_batch,
            img,
            model=self.model,
            enabled=self.with_reid,
            placeholder_value=0.0,
        )
        id_feature_second = resolve_batch_embeddings(
            second_batch,
            img,
            model=self.model,
            enabled=self.with_reid,
            placeholder_value=0.0,
        )

        # Build dets arrays used by original hybrid code (no det_ind in the math)
        dets_first = high_batch.as_box_conf_detections(dtype=dets.dtype)
        dets_low = second_batch.as_box_conf_detections(dtype=dets.dtype)

        # carry det_ind arrays aligned with above
        det_inds_keep = high_batch.det_inds.astype(int)
        det_inds_second = second_batch.det_inds.astype(int)

        # ---- Predict step for existing tracks
        trks = np.zeros((len(self.active_tracks), 6))
        to_del = []
        for t in range(len(trks)):
            pos, kal_score, simple_score = self.active_tracks[t].predict()
            x1, y1, x2, y2 = pos[0].tolist()
            trks[t] = [x1, y1, x2, y2, kal_score, simple_score]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.active_tracks.pop(t)

        # Prepare motion cues
        velocities_lt = np.array(
            [t.velocity_lt if t.velocity_lt is not None else np.array((0, 0)) for t in self.active_tracks]
        )
        velocities_rt = np.array(
            [t.velocity_rt if t.velocity_rt is not None else np.array((0, 0)) for t in self.active_tracks]
        )
        velocities_lb = np.array(
            [t.velocity_lb if t.velocity_lb is not None else np.array((0, 0)) for t in self.active_tracks]
        )
        velocities_rb = np.array(
            [t.velocity_rb if t.velocity_rb is not None else np.array((0, 0)) for t in self.active_tracks]
        )
        last_boxes = np.array([t.last_observation for t in self.active_tracks])
        k_observations = np.array([k_previous_obs(t.observations, t.age, self.delta_t) for t in self.active_tracks])

        # ===== First association (optionally embedding-guided)
        if self.with_reid and self.EG_weight_high_score > 0 and self.TCM_first_step and len(dets_first) and len(trks):
            track_features = np.asarray([t.smooth_feat for t in self.active_tracks], dtype=float)
            emb_dists = embedding_distance(track_features, id_feature_keep).T

            long_emb_dists = None
            if self.with_longterm_reid or self.with_longterm_reid_correction:
                long_track_features = np.asarray(
                    [
                        np.vstack(list(t.features)).mean(0) if len(t.features) else t.smooth_feat
                        for t in self.active_tracks
                    ],
                    dtype=float,
                )
                long_emb_dists = embedding_distance(long_track_features, id_feature_keep).T

            matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                dets_first,
                trks,
                self.iou_threshold,
                velocities_lt,
                velocities_rt,
                velocities_lb,
                velocities_rb,
                k_observations,
                self.inertia,
                ASSO_FUNCS[self.asso_func_name],  # from BaseTracker
                emb_cost=emb_dists,
                weights=(1.0, self.EG_weight_high_score),
                thresh=self.high_score_matching_thresh,
                long_emb_dists=long_emb_dists,
                with_longterm_reid=self.with_longterm_reid,
                longterm_reid_weight=self.longterm_reid_weight,
                with_longterm_reid_correction=self.with_longterm_reid_correction,
                longterm_reid_correction_thresh=self.longterm_reid_correction_thresh,
                dataset=self.per_class and "perclass" or self.dataset,
            )
        elif self.TCM_first_step and len(dets_first) and len(trks):
            matched, unmatched_dets, unmatched_trks = associate_4_points_with_score(
                dets_first,
                trks,
                self.iou_threshold,
                velocities_lt,
                velocities_rt,
                velocities_lb,
                velocities_rb,
                k_observations,
                self.inertia,
                ASSO_FUNCS[self.asso_func_name],
            )
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(len(dets_first))
            unmatched_trks = np.arange(len(trks))

        # Update matched (update features here)  —— pass cls & det_ind (safe)
        for m in matched:
            det_i = m[0]
            self.active_tracks[m[1]].update(
                dets_first[det_i, :],
                id_feature_keep[det_i, :],
                cls=_safe_cls(cls_keep[det_i]),
                det_ind=_safe_detind(det_inds_keep[det_i], n_dets_full),
            )

        # ===== BYTE / low-score association (optional)
        if self.use_byte and len(dets_low) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = np.array(ASSO_FUNCS[self.asso_func_name](dets_low, u_trks))
            iou_left_thre = iou_left.copy()
            if self.TCM_byte_step:
                iou_left -= np.array(cal_score_dif_batch_two_score(dets_low, u_trks) * self.TCM_byte_step_weight)

            if iou_left.max() > self.iou_threshold:
                if self.EG_weight_low_score > 0 and self.with_reid:
                    u_tracklets = [self.active_tracks[idx] for idx in unmatched_trks]
                    u_track_features = np.asarray([t.smooth_feat for t in u_tracklets], dtype=float)
                    emb_dists_low = embedding_distance(u_track_features, id_feature_second).T
                    matched_indices = linear_assignment(-iou_left + self.EG_weight_low_score * emb_dists_low)
                else:
                    matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for mm in matched_indices:
                    det_rel, trk_rel = mm[0], mm[1]
                    trk_ind = unmatched_trks[trk_rel]
                    if self.with_longterm_reid_correction and self.EG_weight_low_score > 0 and self.with_reid:
                        bad_iou = iou_left_thre[det_rel, trk_rel] < self.iou_threshold
                        bad_emb = emb_dists_low[det_rel, trk_rel] > self.longterm_reid_correction_thresh_low
                        if bad_iou or bad_emb:
                            continue
                    else:
                        if iou_left_thre[det_rel, trk_rel] < self.iou_threshold:
                            continue
                    # do not update features in BYTE pass
                    self.active_tracks[trk_ind].update(
                        dets_low[det_rel, :],
                        id_feature_second[det_rel, :],
                        update_feature=False,
                        cls=_safe_cls(cls_second[det_rel]),
                        det_ind=_safe_detind(det_inds_second[det_rel], n_dets_full),
                    )
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # ===== Final chance: IoU vs last boxes
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets_first[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = np.array(ASSO_FUNCS[self.asso_func_name](left_dets, left_trks))
            if iou_left.max() > self.iou_threshold:
                rematched = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for mm in rematched:
                    det_rel, trk_rel = mm[0], mm[1]
                    if iou_left[det_rel, trk_rel] < self.iou_threshold:
                        continue
                    det_abs = unmatched_dets[det_rel]
                    trk_abs = unmatched_trks[trk_rel]
                    self.active_tracks[trk_abs].update(
                        dets_first[det_abs, :],
                        id_feature_keep[det_abs, :],
                        update_feature=False,
                        cls=_safe_cls(cls_keep[det_abs]),
                        det_ind=_safe_detind(det_inds_keep[det_abs], n_dets_full),
                    )
                    to_remove_det_indices.append(det_abs)
                    to_remove_trk_indices.append(trk_abs)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        # Mark remaining unmatched tracks
        for m in unmatched_trks:
            self.active_tracks[m].update(None, None)

        # Create new trackers for unmatched high-score detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets_first[i, :],
                id_feature_keep[i, :],
                delta_t=self.delta_t,
                longterm_bank_length=self.longterm_bank_length,
                max_obs=self.max_obs,
                alpha=self.alpha,
                adapfs=self.adapfs,
                track_thresh=self.track_thresh,
                cls=_safe_cls(cls_keep[i]),
                det_ind=_safe_detind(det_inds_keep[i], n_dets_full) if len(det_inds_keep) else -1,
                id_allocator=self.id_allocator,
            )
            self.active_tracks.append(trk)

        # Collect outputs (match BotSort/OcSort style)
        outputs = []
        for trk in self.active_tracks[::-1]:
            if trk.last_observation.sum() < 0:
                d = trk.motion_model.to_box(trk.kf.x)[0][:4]
            else:
                d = trk.last_observation[:4]

            # Only output fresh tracks and valid det_ind for this frame
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(
                    [
                        *d.tolist(),
                        trk.id,  # track id
                        float(trk.conf),  # conf
                        int(trk.cls),  # cls (from detection)
                        int(trk.det_ind),  # det index (frame-local)
                    ]
                )

        # Remove dead tracks
        i = len(self.active_tracks)
        for trk in self.active_tracks[::-1]:
            i -= 1
            if trk.time_since_update > self.max_age:
                self.active_tracks.pop(i)

        return self.format_output_rows(outputs, dtype=np.float32)

    def _update_obb(self, dets, img, embs=None, masks=None):
        """HybridSORT adaptation using oriented DIoU, ReID, and XYWHA motion state."""
        self.frame_count += 1
        batch = self.make_detection_batch(dets, embs=embs, masks=masks)
        high, low = batch.split_by_confidence(high_thresh=self.det_thresh, low_thresh=self.low_thresh)
        high_dets = high.as_indexed_detections(dtype=dets.dtype)
        low_dets = low.as_indexed_detections(dtype=dets.dtype)
        high_embs = resolve_batch_embeddings(high, img, model=self.model, enabled=self.with_reid, placeholder_value=0.0)

        predicted = [track.predict()[0][:5] for track in self.active_tracks]
        predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 5)

        unmatched_dets = np.arange(len(high_dets))
        unmatched_tracks = np.arange(len(self.active_tracks))
        if len(high_dets) and len(predicted):
            similarity = self.asso_func(high.boxes, predicted)
            embedding_cost = None
            if self.with_reid:
                track_embs = np.asarray([track.smooth_feat for track in self.active_tracks])
                embedding_cost = embedding_distance(track_embs, high_embs).T
            assignment_cost = -similarity
            if embedding_cost is not None:
                assignment_cost += self.EG_weight_high_score * embedding_cost
            pairs = linear_assignment(assignment_cost)
            accepted = []
            for det_index, track_index in pairs:
                poor_geometry = similarity[det_index, track_index] < self.iou_threshold
                poor_appearance = (
                    embedding_cost is not None
                    and embedding_cost[det_index, track_index] > self.longterm_reid_correction_thresh
                )
                if not (poor_geometry and poor_appearance):
                    accepted.append((det_index, track_index))
            for det_index, track_index in accepted:
                track = self.active_tracks[track_index]
                track.update(high_dets[det_index, :6], high.clss[det_index], high.det_inds[det_index])
                track.smooth_feat = ema_update_embedding(track.smooth_feat, high_embs[det_index], alpha=self.alpha)
                track.conf = high.confs[det_index]
            unmatched_dets = np.setdiff1d(unmatched_dets, [pair[0] for pair in accepted])
            unmatched_tracks = np.setdiff1d(unmatched_tracks, [pair[1] for pair in accepted])

        if len(low_dets) and len(unmatched_tracks):
            similarity = self.asso_func(low.boxes, predicted[unmatched_tracks])
            pairs = linear_assignment(-similarity)
            accepted = [(d, t) for d, t in pairs if similarity[d, t] >= self.iou_threshold]
            for det_index, relative_track in accepted:
                track_index = unmatched_tracks[relative_track]
                self.active_tracks[track_index].update(
                    low_dets[det_index, :6], low.clss[det_index], low.det_inds[det_index]
                )
            unmatched_tracks = np.setdiff1d(unmatched_tracks, [unmatched_tracks[pair[1]] for pair in accepted])

        for track_index in unmatched_tracks:
            self.active_tracks[track_index].update(None, None, None)
        for det_index in unmatched_dets:
            track = OBBKalmanBoxTracker(
                high_dets[det_index, :6], high.clss[det_index], high.det_inds[det_index],
                delta_t=self.delta_t, max_obs=self.max_obs, is_obb=True, id_allocator=self.id_allocator,
            )
            track.smooth_feat = high_embs[det_index]
            self.active_tracks.append(track)

        outputs = []
        for track in self.active_tracks[::-1]:
            box = track.last_observation[:5] if track.last_observation.sum() >= 0 else track.get_state()[0][:5]
            if track.time_since_update < 1 and (track.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                outputs.append(self.format_output_row(box, track.id, track.conf, track.cls, track.det_ind))
        self.active_tracks = [track for track in self.active_tracks if track.time_since_update <= self.max_age]
        return self.format_output_rows(outputs, dtype=np.float32)

    def reset(self):
        self._reset_common_state()
