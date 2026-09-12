# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license
"""OccluBoost tracker.

A hybrid tracker that combines:

* BoostTrack's identity-friendly multi-cue association (IoU + Mahalanobis +
  shape similarity, optional ReID) and DLO/DUO confidence boosting — this is
  what gives strong IDF1 / AssA.
* A BotSort/StrongSort-inspired ReID-only **recovery pass** that re-attaches
  unmatched high-confidence detections to recently lost tracks when the
  appearance similarity is high. This lifts MOTA without inducing the ID
  switches an IoU-only ByteTrack second pass introduces.
* A BotSort-style **track confirmation state**: new tracks born from
  medium-confidence detections must accumulate ``confirm_hits`` consecutive
  matches before being emitted (detections above ``instant_confirm_thresh``
  skip the wait). Tentative tracks expire quickly via ``tentative_max_age``,
  cutting ghost IDs and FP from one-frame flickers.
* A safe **appearance-gated low-confidence pass** that recovers low-confidence
  detections only for already-confirmed tracks (``is_activated=True``) with
  strict IoU + appearance gates.
* Tuned defaults (longer ``max_age``) that favour identity retention.
* Optional Oriented Bounding Box (OBB) support, dispatched via a separate
  OBB-only update path that mirrors the AABB flow but uses oriented IoU
  and a 9-column output schema. AABB-only behaviour (DLO/DUO confidence
  boosting and Mahalanobis association on xyhr state) is intentionally
  disabled in OBB mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.trackers.boosttrack.track import KalmanBoxTracker
from boxmot.trackers.boosttrack.tracker import BoostTrack
from boxmot.trackers.common.appearance import (
    confidence_aware_alpha,
    resolve_batch_embeddings,
)
from boxmot.trackers.common.association.boost import associate
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.motion.batching import predict_tracks
from boxmot.trackers.common.motion.kalman_filters.xyhr import KalmanFilterXYHR
from boxmot.trackers.common.tracking.track import TrackState, sync_track_meta


class OccluBoost(BoostTrack):
    """BoostTrack augmented with an appearance-only recovery pass.

    Args:
        reid_model (Any | None): Optional pre-built ReID backend exposing
            ``get_features(boxes, image)``. When omitted, the backend is built
            lazily from ``reid_weights`` the first time live embeddings are
            needed.
        reid_weights (str | Path | list[str | Path] | tuple[str | Path, ...] | None):
            ReID weights used for live embedding extraction. The BoxMOT default
            ReID weights are used when omitted.
        device (Any): Device used by the lazily constructed ReID backend.
        half (bool): Whether the lazily constructed ReID backend uses FP16 inference.
        reid_preprocess (str | None): Optional ReID preprocessing profile.
        recovery_appearance_thresh (float): Minimum cosine similarity required
            between a detection embedding and a track embedding for the
            recovery pass to accept a match. Higher = stricter (fewer recoveries
            but safer identities).
        recovery_iou_thresh (float): Minimum IoU between detection box and the
            predicted track box (sanity gate; kept low because predicted boxes
            of long-lost tracks are inaccurate).
        recovery_max_age (int): Maximum ``time_since_update`` (after predict) of
            a tracker eligible for the recovery pass.
        feat_alpha (float): EMA factor used when updating embeddings during
            recovery (lower = slower update; preserves identity feature).
        **kwargs: Forwarded to :class:`BoostTrack`.

    Class attribute ``supports_obb = True`` advertises Oriented Bounding Box
    capability; oriented detections are dispatched to :meth:`_update_obb`.
    """

    supports_variable_dt = True

    accepts_embeddings = True

    def __init__(
        self,
        use_embeddings: bool = True,
        recovery_appearance_thresh: float = 0.99,
        recovery_iou_thresh: float = 0.1,
        recovery_max_age: int = 1,
        feat_alpha: float = 0.95,
        track_low_thresh: float = 0.1,
        second_iou_thresh: float = 0.6,
        second_appearance_thresh: float = 0.5,
        second_pass_max_age: int = 1,
        second_pass_min_hits: int = 3,
        use_second_pass: bool = False,
        new_track_thresh: float = 0.6,
        confirm_hits: int = 2,
        instant_confirm_thresh: float = 0.7,
        tentative_max_age: int = 1,
        duplicate_iou_thresh: float = 0.85,
        ams_enabled: bool = True,
        ams_alpha0: float = 0.4,
        ams_threshold: float = 0.5,
        ams_buffer_size: int = 30,
        ams_shrink_ratio: float = 0.75,
        lambda_emb_multiplier: float = 1.5,
        # ---- Adaptive KF ----
        adaptive_kf: bool = False,
        # ---- OBB-specific operating point ----
        obb_det_thresh: float = 0.2,
        obb_iou_threshold: float = 0.15,
        obb_new_track_thresh: float = 0.3,
        obb_instant_confirm_thresh: float = 0.5,
        obb_max_age: int = 30,
        obb_recovery_max_age: int = 15,
        obb_second_iou_thresh: float = 0.3,
        *,
        reid_model: Any | None = None,
        reid_weights: str | Path | list[str | Path] | tuple[str | Path, ...] | None = None,
        device: Any = "cpu",
        half: bool = False,
        reid_preprocess: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            use_embeddings=use_embeddings,
            reid_model=reid_model,
            reid_weights=reid_weights,
            device=device,
            half=half,
            reid_preprocess=reid_preprocess,
            **kwargs,
        )
        self.recovery_appearance_thresh = recovery_appearance_thresh
        self.recovery_iou_thresh = recovery_iou_thresh
        self.recovery_max_age = recovery_max_age
        self.feat_alpha = feat_alpha
        self.track_low_thresh = track_low_thresh
        self.second_iou_thresh = second_iou_thresh
        self.second_appearance_thresh = second_appearance_thresh
        self.second_pass_max_age = second_pass_max_age
        self.second_pass_min_hits = second_pass_min_hits
        self.use_second_pass = use_second_pass
        # ``new_track_thresh`` decouples new-track creation from the matching
        # det_thresh. Detections in [det_thresh, new_track_thresh) help update
        # existing tracks but do not spawn new ones.
        self.new_track_thresh = max(new_track_thresh, 0.0)
        # ---- BotSort-style track confirmation ----
        # Tracks created from low/medium-confidence detections start tentative
        # and are only emitted (and persisted past ``tentative_max_age`` frames)
        # once they accumulate ``confirm_hits`` consecutive matched updates.
        # Detections with confidence >= ``instant_confirm_thresh`` skip the
        # tentative state entirely so high-quality first detections still emit
        # immediately (preserves IDF1).
        self.confirm_hits = max(int(confirm_hits), 1)
        self.instant_confirm_thresh = instant_confirm_thresh
        self.tentative_max_age = max(int(tentative_max_age), 0)
        # The MOT-tuned AABB defaults are too restrictive for multi-class OBB
        # detections. Keep a separate OBB operating point so AABB behaviour is
        # unchanged while oriented tracks can start and recover reliably.
        self.obb_det_thresh = max(float(obb_det_thresh), 0.0)
        self.obb_iou_threshold = float(np.clip(obb_iou_threshold, 0.0, 1.0))
        self.obb_new_track_thresh = max(float(obb_new_track_thresh), self.obb_det_thresh)
        self.obb_instant_confirm_thresh = max(float(obb_instant_confirm_thresh), self.obb_new_track_thresh)
        self.obb_max_age = max(int(obb_max_age), 0)
        self.obb_recovery_max_age = max(int(obb_recovery_max_age), 0)
        self.obb_second_iou_thresh = float(np.clip(obb_second_iou_thresh, 0.0, 1.0))
        # ---- Duplicate-track suppression ----
        # IoU threshold above which two co-existing tracks are considered
        # duplicates; the younger one (lower ``age``) is dropped.
        self.duplicate_iou_thresh = duplicate_iou_thresh
        # ---- Abnormal Motion Suppression (OccluTrack AMS KF) ----
        # Detect speed spikes caused by partial occlusion (the bbox suddenly
        # shrinks/jumps because only part of the body is visible) and damp
        # the Kalman gain on the affected update so the predicted state is
        # trusted more than the abnormal observation. ``ams_threshold`` is the
        # relative-spike trigger (current speed magnitude vs. running mean),
        # ``ams_alpha0`` is the suppression factor applied to the gain when
        # an abnormal motion is detected, and ``ams_buffer_size`` is the
        # length of the per-track observation buffer used to compute the
        # mean speed. Defaults follow the paper (MOT17 setting).
        self.ams_enabled = bool(ams_enabled)
        self.ams_alpha0 = float(np.clip(ams_alpha0, 0.0, 1.0))
        self.ams_threshold = float(max(ams_threshold, 0.0))
        self.ams_buffer_size = int(max(ams_buffer_size, 2))
        self.ams_shrink_ratio = float(np.clip(ams_shrink_ratio, 0.0, 1.0))
        self.lambda_emb_multiplier = float(lambda_emb_multiplier)
        # ---- Adaptive KF ----
        self.adaptive_kf = bool(adaptive_kf)

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: Optional[np.ndarray] = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        if self.is_obb:
            return self._update_obb(dets, img, embs)

        det_dtype = dets.dtype
        batch = self._make_detection_batch(dets, embs=embs)
        dets = batch.as_indexed_detections(dtype=det_dtype)
        self.frame_count += 1

        if self.cmc is not None:
            self.apply_cmc(img, dets, self.trackers)

        trks = []
        confs = []
        predictions = predict_tracks(self.trackers, dt=self._prediction_dt)
        for trk, prediction in zip(self.trackers, predictions, strict=True):
            pos = prediction[0]
            conf = trk.get_confidence()
            confs.append(conf)
            trks.append(np.concatenate([pos, [conf]]))
        trks_np = np.vstack(trks) if len(trks) > 0 else np.empty((0, 5))

        # Capture original detection confidences before any boosting so the
        # ByteTrack-style second pass can recover the genuinely low-conf set.
        orig_confs = batch.confs.copy()

        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost(dets)
        if self.use_duo_boost:
            dets = self.duo_confidence_boost(dets)

        boosted_confs = self.detection_layout.confidences(dets)
        keep_mask = boosted_confs >= self.det_thresh
        second_mask = (
            ((~keep_mask) & (orig_confs >= self.track_low_thresh) & (orig_confs < self.det_thresh))
            if self.use_second_pass
            else np.zeros_like(keep_mask, dtype=bool)
        )

        high_batch = batch.select(keep_mask).with_confs(boosted_confs[keep_mask])
        second_batch = batch.select(second_mask).with_confs(boosted_confs[second_mask])
        dets = high_batch.as_indexed_detections(dtype=det_dtype)
        dets_second = second_batch.as_indexed_detections(dtype=det_dtype)
        scores = high_batch.confs
        dets_embs = resolve_batch_embeddings(
            high_batch,
            enabled=self.use_embeddings,
            placeholder_value=1.0,
        )
        dets_embs_second = resolve_batch_embeddings(
            second_batch,
            enabled=self.use_embeddings,
            placeholder_value=1.0,
        )

        if self.use_embeddings and len(self.trackers) > 0 and dets_embs.shape[0] > 0:
            tracker_embs = np.array([trk.get_emb() for trk in self.trackers])
            emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ tracker_embs.reshape(tracker_embs.shape[0], -1).T
        else:
            emb_cost = None

        mh_dist_matrix = self.get_mh_dist_matrix(dets)
        geometry_similarity = self.asso_func(high_batch.boxes, trks_np[:, :4])

        matched, unmatched_dets, unmatched_trks, _ = associate(
            dets,
            trks_np,
            self.iou_threshold,
            mahalanobis_distance=mh_dist_matrix,
            track_confidence=np.array(confs).reshape(-1, 1),
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
            s_sim_corr=self.s_sim_corr,
            lambda_emb_multiplier=self.lambda_emb_multiplier,
            geometry_matrix=geometry_similarity,
        )

        dets_alpha = confidence_aware_alpha(
            self.detection_layout.confidences(dets),
            self.det_thresh,
        )

        self._ams_multi_update([self.trackers[index] for index in matched[:, 1]], dets[matched[:, 0]])
        for m in matched:
            if self.use_embeddings:
                self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
            self._maybe_activate(self.trackers[m[1]])

        # ---- ReID-only recovery pass ----
        if self.use_embeddings and len(unmatched_trks) > 0 and len(unmatched_dets) > 0:
            elig = [
                int(t)
                for t in unmatched_trks
                if self.trackers[int(t)].time_since_update <= self.recovery_max_age
                and self.trackers[int(t)].get_emb() is not None
            ]
            if elig:
                u_det_idx = [int(d) for d in unmatched_dets]
                trk_e = np.stack([self.trackers[t].get_emb() for t in elig], axis=0)
                trk_e = trk_e.reshape(len(elig), -1)
                det_e = dets_embs[u_det_idx].reshape(len(u_det_idx), -1)
                sim = det_e @ trk_e.T

                trks_pos = np.zeros((len(elig), 5))
                for j, t in enumerate(elig):
                    pos = self.trackers[t].get_state()[0]
                    trks_pos[j, :4] = pos
                    trks_pos[j, 4] = self.trackers[t].get_confidence()
                ious = self.asso_func(high_batch.boxes[u_det_idx], trks_pos[:, :4])

                gated = sim.copy()
                gated[ious < self.recovery_iou_thresh] = -1.0
                gated[sim < self.recovery_appearance_thresh] = -1.0

                if (gated > 0).any():
                    row_ind, col_ind = linear_sum_assignment(-gated)
                    matched_dets_set = set()
                    matched_tracks_set = set()
                    accepted = [(u_det_idx[r], elig[c]) for r, c in zip(row_ind, col_ind) if gated[r, c] > 0]
                    self._ams_multi_update(
                        [self.trackers[t] for _, t in accepted],
                        dets[[d for d, _ in accepted]],
                    )
                    for det_global, trk_global in accepted:
                        matched_dets_set.add(det_global)
                        matched_tracks_set.add(trk_global)
                        self.trackers[trk_global].update_emb(dets_embs[det_global], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])
                    if matched_dets_set:
                        unmatched_dets = np.array(
                            [d for d in unmatched_dets if int(d) not in matched_dets_set],
                            dtype=int,
                        )
                        unmatched_trks = np.array(
                            [t for t in unmatched_trks if int(t) not in matched_tracks_set],
                            dtype=int,
                        )

        # ---- ByteTrack-style appearance-gated second pass on low-conf dets ----
        if self.use_second_pass and len(unmatched_trks) > 0 and dets_second.shape[0] > 0:
            elig_sec = [
                int(t)
                for t in unmatched_trks
                if self.trackers[int(t)].time_since_update <= self.second_pass_max_age
                and self.trackers[int(t)].hit_streak >= self.second_pass_min_hits
                and getattr(self.trackers[int(t)], "is_activated", True)
            ]
            if elig_sec:
                trks_pos = np.zeros((len(elig_sec), 5))
                for j, t in enumerate(elig_sec):
                    pos = self.trackers[t].get_state()[0]
                    trks_pos[j, :4] = pos
                    trks_pos[j, 4] = self.trackers[t].get_confidence()
                ious2 = self.asso_func(second_batch.boxes, trks_pos[:, :4])

                cost = 1.0 - ious2
                cost[ious2 < self.second_iou_thresh] = 1.0

                if (
                    self.use_embeddings
                    and dets_embs_second.shape[0] > 0
                    and self.trackers[elig_sec[0]].get_emb() is not None
                ):
                    trk_e = np.stack([self.trackers[t].get_emb() for t in elig_sec], axis=0).reshape(len(elig_sec), -1)
                    det_e = dets_embs_second.reshape(dets_embs_second.shape[0], -1)
                    sim2 = det_e @ trk_e.T
                    cost[sim2 < self.second_appearance_thresh] = 1.0

                if (cost < 1.0).any():
                    row_ind, col_ind = linear_sum_assignment(cost)
                    used = set()
                    accepted = []
                    for r, c in zip(row_ind, col_ind):
                        if cost[r, c] >= 1.0:
                            continue
                        trk_global = elig_sec[c]
                        if trk_global in used:
                            continue
                        used.add(trk_global)
                        accepted.append((r, trk_global))
                    self._ams_multi_update(
                        [self.trackers[t] for _, t in accepted],
                        dets_second[[r for r, _ in accepted]],
                    )
                    for r, trk_global in accepted:
                        if self.use_embeddings and dets_embs_second.shape[0] > 0:
                            self.trackers[trk_global].update_emb(dets_embs_second[r], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])

        for i in unmatched_dets:
            if dets[i, 4] >= self.new_track_thresh:
                det_emb = dets_embs[i] if self.use_embeddings else None
                new_trk = KalmanBoxTracker(
                    dets[i, :],
                    max_obs=self.max_obs,
                    emb=det_emb,
                    adaptive_kf=self.adaptive_kf,
                    id_allocator=self.id_allocator,
                    noise_config=self.kalman_noise_config,
                )
                # Tentative until confirmed; high-conf detections skip the
                # confirmation period so first-frame appearances still emit.
                new_trk.is_activated = bool(dets[i, 4] >= self.instant_confirm_thresh or self.confirm_hits <= 1)
                self.trackers.append(new_trk)

        outputs = []
        self.active_tracks = []
        emitted_now = []
        for trk in self.trackers:
            d = trk.get_state()[0]
            is_activated = getattr(trk, "is_activated", True)
            warmup = self.frame_count <= self.min_hits
            if (trk.time_since_update < 1) and is_activated and (trk.hit_streak >= self.min_hits or warmup):
                emitted_now.append((trk, d))

        # ---- Duplicate-track suppression on emitted tracks ----
        # When two tracks predict to nearly the same box, BotSort kills the
        # younger one. Without this step OccluBoost can emit pairs of tracks on
        # a single object after a recovery/2nd-pass pickup, hurting MOTA (FP)
        # and IDSW. We only consider currently-emitted tracks so we never
        # delete a legitimate occluded track that just happens to overlap a
        # visible one in *prediction* space.
        if len(emitted_now) > 1 and 0.0 < self.duplicate_iou_thresh < 1.0:
            emitted_now = self._suppress_duplicate_emissions(emitted_now)

        for trk, d in emitted_now:
            outputs.append(self.format_output_row(d, trk.id, trk.conf, trk.cls, trk.det_ind))
            self.active_tracks.append(trk)

        # Lifecycle: confirmed tracks live up to ``max_age`` frames; tentative
        # tracks are dropped after ``tentative_max_age`` to prevent ghost IDs
        # from spurious detections, mirroring BotSort's ``unconfirmed`` pool.
        self.trackers = [
            trk
            for trk in self.trackers
            if trk.time_since_update <= self.max_age
            and (getattr(trk, "is_activated", True) or trk.time_since_update <= self.tentative_max_age)
        ]

        outputs = self.format_output_rows(outputs, dtype=np.float32)
        return self.filter_outputs(outputs)

    def _maybe_activate(self, trk: KalmanBoxTracker) -> None:
        """Promote a tentative track to activated once it accumulates enough
        consecutive matched updates."""
        if not getattr(trk, "is_activated", True) and trk.hit_streak >= self.confirm_hits:
            trk.is_activated = True
            sync_track_meta(trk)

    @staticmethod
    def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
        """Convert ``[x1, y1, x2, y2]`` to ``[cx, cy, w, h]``."""
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)
        return np.array([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dtype=float)

    def _compute_ams_alpha(self, trk: KalmanBoxTracker, det_box: np.ndarray) -> float:
        """Compute the OccluTrack abnormal-motion suppression coefficient.

        Builds a per-track buffer of past observed ``[cx, cy, w, h]`` boxes
        (lazily attached to the tracker as ``_ams_obs_buf``). Compares the
        current speed magnitude (centre and aspect/scale separately) against
        the running mean of the previous speeds in the buffer. If either
        relative spike exceeds ``ams_threshold`` the corresponding pair of
        gain scalars is replaced with ``ams_alpha0``; the returned value is
        the mean of the four ``α_x, α_y, α_w, α_h`` per the paper.
        """
        if not self.ams_enabled or self.ams_alpha0 >= 1.0:
            return 1.0
        # OBB tracks use a different state layout (theta channel); skip AMS
        # to avoid mixing rectangular/oriented box semantics.
        if getattr(trk.kf, "_is_obb", False):
            return 1.0

        cur = self._xyxy_to_cxcywh(det_box[:4])
        buf = getattr(trk, "_ams_obs_buf", None)
        if buf is None:
            from collections import deque

            buf = deque(maxlen=self.ams_buffer_size)
            trk._ams_obs_buf = buf

        # Need at least 2 prior observations to estimate the mean speed.
        if len(buf) < 2:
            buf.append(cur)
            return 1.0

        prev = buf[-1]
        cur_v = cur - prev  # [vx, vy, vw, vh]

        # Mean speed over the (N-1) previous transitions in the buffer.
        diffs = np.diff(np.asarray(buf, dtype=float), axis=0)
        mean_v = diffs.mean(axis=0)

        eps = 1e-6
        cur_c_mag = float(np.linalg.norm(cur_v[:2]))
        mean_c_mag = float(np.linalg.norm(mean_v[:2]))
        cur_a_mag = float(np.linalg.norm(cur_v[2:]))
        mean_a_mag = float(np.linalg.norm(mean_v[2:]))

        # Relative spikes: how much faster is the current speed than the
        # running mean, normalised by the running mean magnitude.
        d_c = max(0.0, cur_c_mag - mean_c_mag) / max(mean_c_mag, eps)
        d_a = max(0.0, cur_a_mag - mean_a_mag) / max(mean_a_mag, eps)

        alpha_c = 1.0 if d_c <= self.ams_threshold else self.ams_alpha0
        alpha_a = 1.0 if d_a <= self.ams_threshold else self.ams_alpha0
        alpha = 0.5 * (alpha_c + alpha_a)

        # Physical sanity: partial occlusion specifically *shrinks* the bbox
        # (the occluder hides part of the body). Only suppress when the new
        # box area is meaningfully smaller than the running mean area;
        # otherwise the speed spike is more likely legitimate fast motion or
        # the track re-emerging from full occlusion at its true scale.
        cur_area = float(cur[2] * cur[3])
        mean_area = float(np.mean(np.asarray(buf, dtype=float)[:, 2:].prod(axis=1)))
        if cur_area >= mean_area * self.ams_shrink_ratio:
            alpha = 1.0

        buf.append(cur)
        return float(alpha)

    def _ams_update(self, trk: KalmanBoxTracker, det: np.ndarray) -> None:
        """Drop-in replacement for ``KalmanBoxTracker.update`` that also
        applies the OccluTrack abnormal-motion suppression coefficient to the
        Kalman gain.

        Mirrors :meth:`KalmanBoxTracker.update` exactly except for passing
        ``alpha`` to the underlying KF, so all bookkeeping (hit_streak,
        history_observations, conf/cls/det_ind) stays consistent across the
        first pass, ReID-only recovery, and the low-confidence second pass.
        """
        alpha = self._compute_ams_alpha(trk, det[:4])
        trk.time_since_update = 0
        trk.hit_streak += 1
        trk.kf.update(trk.motion_model.to_measurement(det[:4], column=False), alpha=alpha)
        trk.conf = float(det[4])
        trk.cls = int(det[5])
        trk.det_ind = int(det[6])
        trk._append_current_history()
        sync_track_meta(trk, TrackState.TRACKED)

    def _ams_multi_update(self, tracks: list[KalmanBoxTracker], detections: np.ndarray) -> None:
        """Batch one association stage while retaining each track's AMS and history.

        OBB observations retain track-level equivalent-form alignment. Their
        gain remains unsuppressed, as in the scalar OBB update.
        """
        if not tracks:
            return
        if len(tracks) == 1:
            if tracks[0].is_obb:
                self._ams_update_obb(tracks[0], detections[0])
            else:
                self._ams_update(tracks[0], detections[0])
            return
        alphas = [self._compute_ams_alpha(track, det[:4]) for track, det in zip(tracks, detections)]
        measurements = [track._prepare_update(det) for track, det in zip(tracks, detections)]
        KalmanFilterXYHR.update_many([track.kf for track in tracks], measurements, alpha=alphas)
        for track, measurement in zip(tracks, measurements):
            track._finish_update(measurement)

    def _suppress_duplicate_emissions(
        self, emitted: list[tuple[KalmanBoxTracker, np.ndarray]]
    ) -> list[tuple[KalmanBoxTracker, np.ndarray]]:
        """Drop duplicate emissions when two tracks predict to overlapping
        boxes. The younger track (smaller ``age``) is dropped *and* removed
        from ``self.trackers`` so it does not persist as a ghost.

        Mirrors BotSort's ``remove_duplicate_stracks``; uses ``age`` as the
        survival tiebreaker to favour the older identity.
        """
        if self.is_obb:
            # ``e[1]`` is ``[cx, cy, w, h, angle]`` in OBB mode; use oriented IoU.
            boxes = np.stack([e[1][:5] for e in emitted], axis=0)
            ious = AssociationFunction.iou_batch_obb(boxes, boxes)
        else:
            boxes = np.stack([e[1][:4] for e in emitted], axis=0)
            ious = AssociationFunction.iou_batch(boxes, boxes)
        np.fill_diagonal(ious, 0.0)
        drop = set()
        n = len(emitted)
        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j in drop:
                    continue
                if ious[i, j] >= self.duplicate_iou_thresh:
                    age_i = emitted[i][0].age
                    age_j = emitted[j][0].age
                    drop.add(j if age_i >= age_j else i)
        if not drop:
            return emitted
        # Also remove the dropped (younger) tracks from ``self.trackers`` so
        # they cannot spawn future emissions or absorb future detections.
        drop_ids = {emitted[k][0].id for k in drop}
        self.trackers = [trk for trk in self.trackers if trk.id not in drop_ids]
        return [e for k, e in enumerate(emitted) if k not in drop]

    # ------------------------------------------------------------------
    # OBB code path
    # ------------------------------------------------------------------

    def _ams_update_obb(self, trk: KalmanBoxTracker, det: np.ndarray) -> None:
        """OBB analogue of :meth:`_ams_update`.

        ``det`` is ``[cx, cy, w, h, angle, conf, cls, det_ind]``. AMS itself
        is skipped for OBB tracks (the speed-spike heuristic assumes a
        rectangular box; :meth:`_compute_ams_alpha` already returns ``1.0``
        for OBB KFs), so we just route the update through the OBB-aware KF
        and keep the same bookkeeping as :meth:`_ams_update`.
        """
        # The track-level update performs equivalent-form alignment before
        # correcting the filter and records the resulting post-update state.
        trk.update(det)

    def _update_obb(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """OBB-only update mirroring the AABB flow.

        Differences vs the AABB path:
        * Detections use the 7-col layout ``(cx, cy, w, h, angle, conf, cls)``;
          ``self.detection_layout.with_detection_indices`` appends ``det_ind``.
        * Camera-motion compensation, DLO, and DUO use native OBB geometry.
        * Association uses the selected oriented geometry, optionally fused
          with a ReID cosine-similarity term BoTSORT-style.
        * Outputs follow the OBB schema
          ``[cx, cy, w, h, angle, id, conf, cls, det_ind]`` (9 cols).
        """
        det_dtype = dets.dtype
        batch = self._make_detection_batch(dets, embs=embs)
        dets = batch.as_indexed_detections(dtype=det_dtype)
        self.frame_count += 1

        if self.cmc is not None:
            self.apply_cmc(img, dets, self.trackers)

        # Predict all current trackers
        trks_xywha = []
        confs = []
        predictions = predict_tracks(self.trackers, dt=self._prediction_dt)
        for trk, prediction in zip(self.trackers, predictions, strict=True):
            pos = prediction[0]  # [cx, cy, w, h, angle]
            trks_xywha.append(pos)
            confs.append(trk.get_confidence())
        trks_xywha = np.vstack(trks_xywha) if len(trks_xywha) > 0 else np.empty((0, 5))

        # Confidence-based detection split (high / low for second pass).
        # Preserve the detector scores so ByteTrack recovery remains a true
        # low-confidence pass even when an OBB boost promotes a row.
        orig_confs = batch.confs.copy()
        if self.use_dlo_boost:
            dets = self.dlo_confidence_boost_obb(dets, threshold=self.obb_det_thresh)
        if self.use_duo_boost:
            dets = self.duo_confidence_boost_obb(dets, threshold=self.obb_det_thresh)
        boosted_confs = self.detection_layout.confidences(dets)
        batch = batch.with_confs(boosted_confs)
        keep_mask = boosted_confs >= self.obb_det_thresh
        second_mask = (
            ((~keep_mask) & (orig_confs >= self.track_low_thresh) & (orig_confs < self.obb_det_thresh))
            if self.use_second_pass
            else np.zeros_like(keep_mask, dtype=bool)
        )

        high_batch = batch.select(keep_mask)
        second_batch = batch.select(second_mask)
        dets = high_batch.as_indexed_detections(dtype=det_dtype)
        dets_second = second_batch.as_indexed_detections(dtype=det_dtype)
        dets_embs = resolve_batch_embeddings(
            high_batch,
            enabled=self.use_embeddings,
            placeholder_value=1.0,
        )
        dets_embs_second = resolve_batch_embeddings(
            second_batch,
            enabled=self.use_embeddings,
            placeholder_value=1.0,
        )

        # First-pass association: oriented IoU (+ optional ReID fusion)
        n_dets = dets.shape[0]
        n_trks = trks_xywha.shape[0]
        if n_dets == 0 or n_trks == 0:
            matched = np.empty((0, 2), dtype=int)
            unmatched_dets = np.arange(n_dets, dtype=int)
            unmatched_trks = np.arange(n_trks, dtype=int)
        else:
            iou = self.asso_func(high_batch.boxes, trks_xywha)
            cost = 1.0 - iou
            cost[iou < self.obb_iou_threshold] = 1e6

            if self.use_embeddings and dets_embs.shape[0] > 0 and self.trackers[0].get_emb() is not None:
                tracker_embs = np.stack([trk.get_emb() for trk in self.trackers], axis=0).reshape(n_trks, -1)
                emb_sim = dets_embs.reshape(n_dets, -1) @ tracker_embs.T
                # BoTSORT-style fusion: subtract a scaled appearance term.
                lambda_emb = float(getattr(self, "lambda_iou", 0.5)) + 0.5
                cost = cost - lambda_emb * emb_sim
                # Re-apply IoU gate so good appearance can't bypass geometry.
                cost[iou < self.obb_iou_threshold] = 1e6

            row_ind, col_ind = linear_sum_assignment(cost)
            matched_pairs = []
            matched_d, matched_t = set(), set()
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] >= 1e5:
                    continue
                matched_pairs.append([r, c])
                matched_d.add(r)
                matched_t.add(c)
            matched = np.array(matched_pairs, dtype=int) if matched_pairs else np.empty((0, 2), dtype=int)
            unmatched_dets = np.array([i for i in range(n_dets) if i not in matched_d], dtype=int)
            unmatched_trks = np.array([i for i in range(n_trks) if i not in matched_t], dtype=int)

        # Apply matched updates
        self._ams_multi_update([self.trackers[index] for index in matched[:, 1]], dets[matched[:, 0]])
        for m in matched:
            if self.use_embeddings:
                alpha_emb = confidence_aware_alpha(
                    self.detection_layout.confidences(dets)[m[0] : m[0] + 1],
                    self.obb_det_thresh,
                )[0]
                self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=float(alpha_emb))
            self._maybe_activate(self.trackers[m[1]])

        # ---- ReID-only recovery pass ----
        if self.use_embeddings and len(unmatched_trks) > 0 and len(unmatched_dets) > 0:
            elig = [
                int(t)
                for t in unmatched_trks
                if self.trackers[int(t)].time_since_update <= self.obb_recovery_max_age
                and self.trackers[int(t)].get_emb() is not None
            ]
            if elig:
                u_det_idx = [int(d) for d in unmatched_dets]
                trk_e = np.stack([self.trackers[t].get_emb() for t in elig], axis=0).reshape(len(elig), -1)
                det_e = dets_embs[u_det_idx].reshape(len(u_det_idx), -1)
                sim = det_e @ trk_e.T

                trks_pos = np.stack([self.trackers[t].get_state()[0] for t in elig], axis=0)
                ious = self.asso_func(high_batch.boxes[u_det_idx], trks_pos)

                gated = sim.copy()
                gated[ious < self.recovery_iou_thresh] = -1.0
                gated[sim < self.recovery_appearance_thresh] = -1.0

                if (gated > 0).any():
                    row_ind, col_ind = linear_sum_assignment(-gated)
                    matched_dets_set = set()
                    matched_tracks_set = set()
                    accepted = [(u_det_idx[r], elig[c]) for r, c in zip(row_ind, col_ind) if gated[r, c] > 0]
                    self._ams_multi_update(
                        [self.trackers[t] for _, t in accepted],
                        dets[[d for d, _ in accepted]],
                    )
                    for det_global, trk_global in accepted:
                        matched_dets_set.add(det_global)
                        matched_tracks_set.add(trk_global)
                        self.trackers[trk_global].update_emb(dets_embs[det_global], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])
                    if matched_dets_set:
                        unmatched_dets = np.array(
                            [d for d in unmatched_dets if int(d) not in matched_dets_set],
                            dtype=int,
                        )
                        unmatched_trks = np.array(
                            [t for t in unmatched_trks if int(t) not in matched_tracks_set],
                            dtype=int,
                        )

        # ---- Appearance-gated low-confidence second pass ----
        if self.use_second_pass and len(unmatched_trks) > 0 and dets_second.shape[0] > 0:
            elig_sec = [
                int(t)
                for t in unmatched_trks
                if self.trackers[int(t)].time_since_update <= self.second_pass_max_age
                and self.trackers[int(t)].hit_streak >= self.second_pass_min_hits
                and getattr(self.trackers[int(t)], "is_activated", True)
            ]
            if elig_sec:
                trks_pos = np.stack([self.trackers[t].get_state()[0] for t in elig_sec], axis=0)
                ious2 = self.asso_func(second_batch.boxes, trks_pos)
                cost2 = 1.0 - ious2
                cost2[ious2 < self.obb_second_iou_thresh] = 1.0

                if (
                    self.use_embeddings
                    and dets_embs_second.shape[0] > 0
                    and self.trackers[elig_sec[0]].get_emb() is not None
                ):
                    trk_e = np.stack([self.trackers[t].get_emb() for t in elig_sec], axis=0).reshape(len(elig_sec), -1)
                    det_e = dets_embs_second.reshape(dets_embs_second.shape[0], -1)
                    sim2 = det_e @ trk_e.T
                    cost2[sim2 < self.second_appearance_thresh] = 1.0

                if (cost2 < 1.0).any():
                    row_ind, col_ind = linear_sum_assignment(cost2)
                    used = set()
                    accepted = []
                    for r, c in zip(row_ind, col_ind):
                        if cost2[r, c] >= 1.0:
                            continue
                        trk_global = elig_sec[c]
                        if trk_global in used:
                            continue
                        used.add(trk_global)
                        accepted.append((r, trk_global))
                    self._ams_multi_update(
                        [self.trackers[t] for _, t in accepted],
                        dets_second[[r for r, _ in accepted]],
                    )
                    for r, trk_global in accepted:
                        if self.use_embeddings and dets_embs_second.shape[0] > 0:
                            self.trackers[trk_global].update_emb(dets_embs_second[r], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])

        # ---- New tracks for remaining unmatched high-conf detections ----
        for i in unmatched_dets:
            det_conf = self.detection_layout.confidences(dets)[i]
            if det_conf >= self.obb_new_track_thresh:
                det_emb = dets_embs[i] if self.use_embeddings else None
                new_trk = KalmanBoxTracker(
                    dets[i, :],
                    max_obs=self.max_obs,
                    emb=det_emb,
                    is_obb=True,
                    adaptive_kf=self.adaptive_kf,
                    id_allocator=self.id_allocator,
                    noise_config=self.kalman_noise_config,
                )
                new_trk.is_activated = bool(det_conf >= self.obb_instant_confirm_thresh or self.confirm_hits <= 1)
                self.trackers.append(new_trk)

        # ---- Build outputs ----
        outputs = []
        self.active_tracks = []
        emitted_now = []
        for trk in self.trackers:
            d = trk.get_state()[0]  # [cx, cy, w, h, angle]
            is_activated = getattr(trk, "is_activated", True)
            warmup = self.frame_count <= self.min_hits
            if (trk.time_since_update < 1) and is_activated and (trk.hit_streak >= self.min_hits or warmup):
                emitted_now.append((trk, d))

        if len(emitted_now) > 1 and 0.0 < self.duplicate_iou_thresh < 1.0:
            emitted_now = self._suppress_duplicate_emissions(emitted_now)

        for trk, d in emitted_now:
            outputs.append(self.format_output_row(d, trk.id, trk.conf, trk.cls, trk.det_ind))
            self.active_tracks.append(trk)

        # Lifecycle
        self.trackers = [
            trk
            for trk in self.trackers
            if trk.time_since_update <= self.obb_max_age
            and (getattr(trk, "is_activated", True) or trk.time_since_update <= self.tentative_max_age)
        ]

        outputs = self.format_output_rows(outputs, dtype=np.float32)
        return self.filter_outputs(outputs)
