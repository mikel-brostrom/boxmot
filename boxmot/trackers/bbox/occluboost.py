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

from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.trackers.bbox.boosttrack import BoostTrack
from boxmot.trackers.common.appearance import (
    confidence_aware_alpha,
    resolve_batch_embeddings,
)
from boxmot.trackers.common.association.boost import associate, iou_batch
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.geometry.obb import xywha_to_xyxy
from boxmot.trackers.common.tracking.track import TrackState, sync_track_meta
from boxmot.trackers.common.track_models.boosttrack import KalmanBoxTracker


class OccluBoost(BoostTrack):
    """BoostTrack augmented with an appearance-only recovery pass.

    Args:
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

    supports_obb = True

    def __init__(
        self,
        reid_model: Any | None = None,
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
        # ---- Online GTA (Global Track Association) ----
        gta_enabled: bool = True,
        gta_appearance_thresh: float = 0.5,
        gta_min_track_length: int = 5,
        gta_smooth_tau: float = 5.0,
        gta_interpolate: bool = True,
        gta_max_gap: int = 60,
        # ---- Adaptive KF ----
        adaptive_kf: bool = False,
        **kwargs: Any,
    ):
        super().__init__(reid_model=reid_model, **kwargs)
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
        # ---- Online GTA (Global Track Association) ----
        # When a track dies it is buried in a graveyard with its EMA
        # embedding.  Before creating a new track from an unmatched
        # detection, the graveyard is searched for an appearance match.
        # If found, the new track *reuses* the dead track's ID (so
        # outputs are immediately correct — no retroactive remapping)
        # and the gap between death and resurrection is filled with
        # GP-smoothed linear interpolation.
        self.gta_enabled = bool(gta_enabled) and self.with_reid
        self.gta_appearance_thresh = float(gta_appearance_thresh)
        self.gta_min_track_length = max(int(gta_min_track_length), 1)
        self.gta_smooth_tau = float(gta_smooth_tau)
        self.gta_interpolate = bool(gta_interpolate)
        self.gta_max_gap = max(int(gta_max_gap), 1)
        # Graveyard of recently-dead tracks, keyed by track ID.
        self._gta_graveyard: dict[int, dict] = {}
        # Accumulated gap-fill rows (MOT format, 9 cols).
        self._gta_gap_entries: list[np.ndarray] = []
        # ---- Adaptive KF ----
        self.adaptive_kf = bool(adaptive_kf)

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: Optional[np.ndarray] = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        self.check_inputs(dets=dets, embs=embs, img=img)

        if self.is_obb:
            return self._update_obb(dets, img, embs)

        det_dtype = dets.dtype
        batch = self.make_detection_batch(dets, embs=embs)
        dets = batch.as_indexed_detections(dtype=det_dtype)
        self.frame_count += 1

        if self.cmc is not None:
            self.apply_cmc(img, dets, self.trackers)

        trks = []
        confs = []
        for trk in self.trackers:
            pos = trk.predict()[0]
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
            img,
            model=self.reid_model,
            enabled=self.with_reid,
            placeholder_value=1.0,
        )
        dets_embs_second = resolve_batch_embeddings(
            second_batch,
            img,
            model=self.reid_model,
            enabled=self.with_reid,
            placeholder_value=1.0,
        )

        if self.with_reid and len(self.trackers) > 0 and dets_embs.shape[0] > 0:
            tracker_embs = np.array([trk.get_emb() for trk in self.trackers])
            emb_cost = dets_embs.reshape(dets_embs.shape[0], -1) @ tracker_embs.reshape(tracker_embs.shape[0], -1).T
        else:
            emb_cost = None

        mh_dist_matrix = self.get_mh_dist_matrix(dets)

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
        )

        dets_alpha = confidence_aware_alpha(
            self.detection_layout.confidences(dets),
            self.det_thresh,
        )

        for m in matched:
            self._ams_update(self.trackers[m[1]], dets[m[0], :])
            if self.with_reid:
                self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])
            self._maybe_activate(self.trackers[m[1]])

        # ---- ReID-only recovery pass ----
        if self.with_reid and len(unmatched_trks) > 0 and len(unmatched_dets) > 0:
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
                ious = iou_batch(dets[u_det_idx], trks_pos)

                gated = sim.copy()
                gated[ious < self.recovery_iou_thresh] = -1.0
                gated[sim < self.recovery_appearance_thresh] = -1.0

                if (gated > 0).any():
                    row_ind, col_ind = linear_sum_assignment(-gated)
                    matched_dets_set = set()
                    for r, c in zip(row_ind, col_ind):
                        if gated[r, c] <= 0:
                            continue
                        det_global = u_det_idx[r]
                        trk_global = elig[c]
                        matched_dets_set.add(det_global)
                        self._ams_update(self.trackers[trk_global], dets[det_global, :])
                        self.trackers[trk_global].update_emb(dets_embs[det_global], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])
                    if matched_dets_set:
                        unmatched_dets = np.array(
                            [d for d in unmatched_dets if int(d) not in matched_dets_set],
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
                ious2 = iou_batch(dets_second, trks_pos)

                cost = 1.0 - ious2
                cost[ious2 < self.second_iou_thresh] = 1.0

                if (
                    self.with_reid
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
                    for r, c in zip(row_ind, col_ind):
                        if cost[r, c] >= 1.0:
                            continue
                        trk_global = elig_sec[c]
                        if trk_global in used:
                            continue
                        used.add(trk_global)
                        self._ams_update(self.trackers[trk_global], dets_second[r, :])
                        if self.with_reid and dets_embs_second.shape[0] > 0:
                            self.trackers[trk_global].update_emb(dets_embs_second[r], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])

        # ---- GTA: pure-appearance recovery for remaining unmatched dets ----
        # The IoU-gated recovery above can miss when the KF prediction has
        # drifted (fast-moving players). This pass matches remaining
        # unmatched detections against alive-but-unmatched tracks using
        # ONLY appearance similarity (no IoU gate), recovering the track's
        # ID without creating a new one. This is the "online windowed GTA".
        if self.gta_enabled and len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            unmatched_dets = self._gta_appearance_recovery(
                dets, dets_embs, unmatched_dets, unmatched_trks, is_obb=False
            )

        # ---- GTA: resurrect from graveyard before creating new tracks ----
        if self.gta_enabled and self.with_reid and len(unmatched_dets) > 0:
            unmatched_dets = self._gta_resurrect(dets, dets_embs, unmatched_dets, is_obb=False)

        for i in unmatched_dets:
            if dets[i, 4] >= self.new_track_thresh:
                det_emb = dets_embs[i] if self.with_reid else None
                new_trk = KalmanBoxTracker(
                    dets[i, :],
                    max_obs=self.max_obs,
                    emb=det_emb,
                    adaptive_kf=self.adaptive_kf,
                    id_allocator=self.id_allocator,
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
        surviving = []
        dead_tracks = []
        for trk in self.trackers:
            alive = trk.time_since_update <= self.max_age and (
                getattr(trk, "is_activated", True) or trk.time_since_update <= self.tentative_max_age
            )
            if alive:
                surviving.append(trk)
            else:
                dead_tracks.append(trk)
        self._gta_bury_dead(dead_tracks)
        self._gta_evict_stale()
        self.trackers = surviving

        outputs = self.format_output_rows(outputs, dtype=np.float32)
        return self.filter_outputs(outputs)

    def _maybe_activate(self, trk: KalmanBoxTracker) -> None:
        """Promote a tentative track to activated once it accumulates enough
        consecutive matched updates."""
        if not getattr(trk, "is_activated", True) and trk.hit_streak >= self.confirm_hits:
            trk.is_activated = True
            sync_track_meta(trk)

    # ------------------------------------------------------------------
    # Online GTA (Global Track Association) methods
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Online GTA: pure-appearance recovery for unmatched detections
    # ------------------------------------------------------------------

    def _gta_appearance_recovery(
        self,
        dets: np.ndarray,
        dets_embs: np.ndarray,
        unmatched_dets: np.ndarray,
        unmatched_trks: np.ndarray,
        is_obb: bool,
    ) -> np.ndarray:
        """Match remaining unmatched detections to alive-but-unmatched tracks
        using ONLY appearance similarity (no IoU gate).

        This catches cases where the KF prediction has drifted too far for
        the IoU-gated recovery to fire, but the appearance embedding is
        still a strong match.  Successfully matched detections are removed
        from *unmatched_dets* and the existing track is force-updated.

        Returns:
            Updated ``unmatched_dets`` array with recovered detections removed.
        """
        # Build eligible tracks: alive, unmatched, with embeddings,
        # within gta_max_gap frames of last match.
        elig = [
            int(t)
            for t in unmatched_trks
            if self.trackers[int(t)].time_since_update <= self.gta_max_gap
            and self.trackers[int(t)].get_emb() is not None
            and self.trackers[int(t)].age >= self.gta_min_track_length
        ]
        if not elig:
            return unmatched_dets

        u_det_idx = [int(d) for d in unmatched_dets]
        if not u_det_idx:
            return unmatched_dets

        # Filter to detections that have embeddings
        det_with_emb = [d for d in u_det_idx if dets_embs[d] is not None]
        if not det_with_emb:
            return unmatched_dets

        # Compute cosine similarity
        trk_e = np.stack([self.trackers[t].get_emb() for t in elig], axis=0).reshape(len(elig), -1)
        det_e = dets_embs[det_with_emb].reshape(len(det_with_emb), -1)
        sim = det_e @ trk_e.T

        # Gate by appearance threshold
        gated = sim.copy()
        gated[sim < self.gta_appearance_thresh] = -1.0

        if not (gated > 0).any():
            return unmatched_dets

        row_ind, col_ind = linear_sum_assignment(-gated)
        matched_dets_set: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if gated[r, c] <= 0:
                continue
            det_global = det_with_emb[r]
            trk_global = elig[c]
            matched_dets_set.add(det_global)
            # Force-update the track with this detection
            if is_obb:
                self._ams_update_obb(self.trackers[trk_global], dets[det_global, :])
            else:
                self._ams_update(self.trackers[trk_global], dets[det_global, :])
            self.trackers[trk_global].update_emb(dets_embs[det_global], alpha=self.feat_alpha)
            self._maybe_activate(self.trackers[trk_global])

        if matched_dets_set:
            unmatched_dets = np.array(
                [d for d in unmatched_dets if int(d) not in matched_dets_set],
                dtype=int,
            )
        return unmatched_dets

    def _gta_bury_dead(self, dead_tracks: list[KalmanBoxTracker]) -> None:
        """Bury recently-dead tracks in the graveyard for future resurrection.

        Only tracks with sufficient age and a valid embedding are interred.
        """
        if not self.gta_enabled:
            return
        for trk in dead_tracks:
            if trk.age < self.gta_min_track_length:
                continue
            emb = trk.get_emb()
            if emb is None:
                continue
            self._gta_graveyard[trk.id] = {
                "emb": emb.copy(),
                "last_box": trk.get_state()[0].copy(),
                "frame": self.frame_count,
                "conf": float(trk.conf),
                "cls": float(trk.cls),
                "is_obb": bool(getattr(trk, "is_obb", False)),
            }

    def _gta_evict_stale(self) -> None:
        """Remove graveyard entries older than ``gta_max_gap`` frames."""
        if not self._gta_graveyard:
            return
        stale = [gid for gid, v in self._gta_graveyard.items() if self.frame_count - v["frame"] > self.gta_max_gap]
        for gid in stale:
            del self._gta_graveyard[gid]

    def _gta_resurrect(
        self,
        dets: np.ndarray,
        dets_embs: np.ndarray,
        unmatched_dets: np.ndarray,
        is_obb: bool,
    ) -> np.ndarray:
        """Try to match unmatched detections against graveyard embeddings.

        If a strong appearance match is found the new track reuses the dead
        track's ID (so outputs are immediately correct) and the positional gap
        between death and resurrection is filled with linear interpolation
        entries stored in ``_gta_gap_entries``.

        Returns:
            Updated ``unmatched_dets`` with resurrected detections removed.
        """
        if not self.gta_enabled or not self._gta_graveyard or len(unmatched_dets) == 0:
            return unmatched_dets

        grave_ids = list(self._gta_graveyard.keys())
        grave_embs = np.stack([self._gta_graveyard[gid]["emb"] for gid in grave_ids], axis=0).reshape(
            len(grave_ids), -1
        )

        u_det_idx = [int(d) for d in unmatched_dets]
        det_e = dets_embs[u_det_idx].reshape(len(u_det_idx), -1)
        sim = det_e @ grave_embs.T

        # Gate by appearance threshold
        gated = sim.copy()
        gated[sim < self.gta_appearance_thresh] = -1.0

        if not (gated > 0).any():
            return unmatched_dets

        row_ind, col_ind = linear_sum_assignment(-gated)
        matched_dets_set: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if gated[r, c] <= 0:
                continue
            det_global = u_det_idx[r]
            grave_id = grave_ids[c]
            grave_entry = self._gta_graveyard[grave_id]

            # Determine detection confidence column index
            conf_col = 5 if is_obb else 4

            # Only resurrect if detection confidence is high enough
            if dets[det_global, conf_col] < self.new_track_thresh:
                continue

            matched_dets_set.add(det_global)

            # Create a new tracker that reuses the dead track's ID
            det_emb = dets_embs[det_global] if self.with_reid else None
            new_trk = KalmanBoxTracker(
                dets[det_global, :],
                max_obs=self.max_obs,
                emb=det_emb,
                is_obb=is_obb,
                adaptive_kf=self.adaptive_kf,
                track_id=grave_id,
            )
            new_trk.is_activated = True
            self.trackers.append(new_trk)

            # ---- Gap interpolation ----
            if self.gta_interpolate:
                death_frame = grave_entry["frame"]
                gap = self.frame_count - death_frame
                if 1 < gap <= self.gta_max_gap:
                    last_box = grave_entry["last_box"]  # [x1,y1,x2,y2] or [cx,cy,w,h,a]
                    cur_box = new_trk.get_state()[0]
                    for t in range(1, gap):
                        alpha_t = t / gap
                        interp_box = (1.0 - alpha_t) * last_box + alpha_t * cur_box
                        frame_id = death_frame + t
                        # MOT format: [frame, id, x1/cx, y1/cy, x2/w, y2/h, conf, cls, det_ind]
                        row = np.array(
                            [
                                frame_id,
                                grave_id,
                                interp_box[0],
                                interp_box[1],
                                interp_box[2],
                                interp_box[3],
                                grave_entry["conf"],
                                grave_entry["cls"],
                                -1.0,
                            ],
                            dtype=float,
                        )
                        self._gta_gap_entries.append(row)

            # Remove from graveyard
            del self._gta_graveyard[grave_id]

        if matched_dets_set:
            unmatched_dets = np.array(
                [d for d in unmatched_dets if int(d) not in matched_dets_set],
                dtype=int,
            )
        return unmatched_dets

    def flush_gta(self) -> np.ndarray:
        """Return accumulated gap-fill entries and reset state.

        Called once at the end of a sequence by the replay loop.

        Returns:
            np.ndarray: Interpolated gap entries in MOT format (9 cols).
        """
        if not self._gta_gap_entries:
            return np.empty((0, 9))

        entries = list(self._gta_gap_entries)

        # Apply GP smoothing to interpolated segments
        if self.gta_smooth_tau > 0:
            entries = self._gta_smooth_all(entries)

        self._gta_gap_entries = []
        self._gta_graveyard = {}
        return np.vstack(entries)

    def reset(self) -> None:
        super().reset()
        self._gta_graveyard = {}
        self._gta_gap_entries = []

    def _gta_smooth_all(self, entries: list[np.ndarray]) -> list[np.ndarray]:
        """Apply GP smoothing to all interpolated segments.

        Groups entries by track_id, then applies RBF-kernel GP regression
        to each segment's bounding box columns.
        """
        if len(entries) < 3:
            return entries

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor as GPR
            from sklearn.gaussian_process.kernels import RBF
        except ImportError:
            return entries

        # Group by track_id (column 1)
        from collections import defaultdict

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, row in enumerate(entries):
            groups[int(row[1])].append(idx)

        tau = self.gta_smooth_tau
        for tid, indices in groups.items():
            if len(indices) < 3:
                continue
            frames = np.array([entries[i][0] for i in indices]).reshape(-1, 1)
            boxes = np.array([entries[i][2:6] for i in indices])
            n = len(indices)
            length_scale = np.clip(tau * np.log(max(tau**3 / n, 1e-6)), tau**-1, tau**2)
            kernel = RBF(length_scale, length_scale_bounds="fixed")
            gpr = GPR(kernel)
            smoothed = gpr.fit(frames, boxes).predict(frames)
            for k, idx in enumerate(indices):
                entries[idx][2:6] = smoothed[k]

        return entries

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
        trk.history_observations.append(trk.get_state()[0])
        trk.kf.update(trk.motion_model.to_measurement(det[:4], column=False), alpha=alpha)
        trk.conf = det[4]
        trk.cls = det[5]
        trk.det_ind = det[6]
        sync_track_meta(trk, TrackState.TRACKED)

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
            ious = iou_batch(
                np.hstack([boxes, np.ones((len(boxes), 3))]),
                np.hstack([boxes, np.ones((len(boxes), 3))]),
            )
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
        trk.time_since_update = 0
        trk.hit_streak += 1
        trk.history_observations.append(trk.get_state()[0])
        trk.kf.update(trk.motion_model.to_measurement(det[:5], column=False))
        trk.conf = det[5]
        trk.cls = det[6]
        trk.det_ind = det[7]
        sync_track_meta(trk, TrackState.TRACKED)

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
        * Camera-motion compensation, DLO/DUO confidence boosting, and
          Mahalanobis association are skipped (they are tied to the xyxy/xyhr
          AABB representation).
        * Association uses oriented IoU via
          :meth:`AssociationFunction.iou_batch_obb`, optionally fused with a
          ReID cosine-similarity term BoTSORT-style.
        * Outputs follow the OBB schema
          ``[cx, cy, w, h, angle, id, conf, cls, det_ind]`` (9 cols).
        """
        det_dtype = dets.dtype
        batch = self.make_detection_batch(dets, embs=embs)
        dets = batch.as_indexed_detections(dtype=det_dtype)
        self.frame_count += 1

        # Predict all current trackers
        trks_xywha = []
        confs = []
        for trk in self.trackers:
            pos = trk.predict()[0]  # [cx, cy, w, h, angle]
            trks_xywha.append(pos)
            confs.append(trk.get_confidence())
        trks_xywha = np.vstack(trks_xywha) if len(trks_xywha) > 0 else np.empty((0, 5))

        # Confidence-based detection split (high / low for second pass)
        orig_confs = batch.confs.copy()
        keep_mask = orig_confs >= self.det_thresh
        second_mask = (
            ((~keep_mask) & (orig_confs >= self.track_low_thresh) & (orig_confs < self.det_thresh))
            if self.use_second_pass
            else np.zeros_like(keep_mask, dtype=bool)
        )

        high_batch = batch.select(keep_mask)
        second_batch = batch.select(second_mask)
        dets = high_batch.as_indexed_detections(dtype=det_dtype)
        dets_second = second_batch.as_indexed_detections(dtype=det_dtype)
        dets_embs = resolve_batch_embeddings(
            high_batch,
            img,
            model=self.reid_model,
            enabled=self.with_reid,
            boxes=xywha_to_xyxy(high_batch.boxes),
            placeholder_value=1.0,
        )
        dets_embs_second = resolve_batch_embeddings(
            second_batch,
            img,
            model=self.reid_model,
            enabled=self.with_reid,
            boxes=xywha_to_xyxy(second_batch.boxes),
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
            iou = AssociationFunction.iou_batch_obb(self.detection_layout.boxes(dets), trks_xywha)
            cost = 1.0 - iou
            cost[iou < self.iou_threshold] = 1e6

            if self.with_reid and dets_embs.shape[0] > 0 and self.trackers[0].get_emb() is not None:
                tracker_embs = np.stack([trk.get_emb() for trk in self.trackers], axis=0).reshape(n_trks, -1)
                emb_sim = dets_embs.reshape(n_dets, -1) @ tracker_embs.T
                # BoTSORT-style fusion: subtract a scaled appearance term.
                lambda_emb = float(getattr(self, "lambda_iou", 0.5)) + 0.5
                cost = cost - lambda_emb * emb_sim
                # Re-apply IoU gate so good appearance can't bypass geometry.
                cost[iou < self.iou_threshold] = 1e6

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
        for m in matched:
            self._ams_update_obb(self.trackers[m[1]], dets[m[0], :])
            if self.with_reid:
                alpha_emb = confidence_aware_alpha(
                    self.detection_layout.confidences(dets)[m[0] : m[0] + 1],
                    self.det_thresh,
                )[0]
                self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=float(alpha_emb))
            self._maybe_activate(self.trackers[m[1]])

        # ---- ReID-only recovery pass ----
        if self.with_reid and len(unmatched_trks) > 0 and len(unmatched_dets) > 0:
            elig = [
                int(t)
                for t in unmatched_trks
                if self.trackers[int(t)].time_since_update <= self.recovery_max_age
                and self.trackers[int(t)].get_emb() is not None
            ]
            if elig:
                u_det_idx = [int(d) for d in unmatched_dets]
                trk_e = np.stack([self.trackers[t].get_emb() for t in elig], axis=0).reshape(len(elig), -1)
                det_e = dets_embs[u_det_idx].reshape(len(u_det_idx), -1)
                sim = det_e @ trk_e.T

                trks_pos = np.stack([self.trackers[t].get_state()[0] for t in elig], axis=0)
                ious = AssociationFunction.iou_batch_obb(self.detection_layout.boxes(dets)[u_det_idx], trks_pos)

                gated = sim.copy()
                gated[ious < self.recovery_iou_thresh] = -1.0
                gated[sim < self.recovery_appearance_thresh] = -1.0

                if (gated > 0).any():
                    row_ind, col_ind = linear_sum_assignment(-gated)
                    matched_dets_set = set()
                    for r, c in zip(row_ind, col_ind):
                        if gated[r, c] <= 0:
                            continue
                        det_global = u_det_idx[r]
                        trk_global = elig[c]
                        matched_dets_set.add(det_global)
                        self._ams_update_obb(self.trackers[trk_global], dets[det_global, :])
                        self.trackers[trk_global].update_emb(dets_embs[det_global], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])
                    if matched_dets_set:
                        unmatched_dets = np.array(
                            [d for d in unmatched_dets if int(d) not in matched_dets_set],
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
                ious2 = AssociationFunction.iou_batch_obb(self.detection_layout.boxes(dets_second), trks_pos)
                cost2 = 1.0 - ious2
                cost2[ious2 < self.second_iou_thresh] = 1.0

                if (
                    self.with_reid
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
                    for r, c in zip(row_ind, col_ind):
                        if cost2[r, c] >= 1.0:
                            continue
                        trk_global = elig_sec[c]
                        if trk_global in used:
                            continue
                        used.add(trk_global)
                        self._ams_update_obb(self.trackers[trk_global], dets_second[r, :])
                        if self.with_reid and dets_embs_second.shape[0] > 0:
                            self.trackers[trk_global].update_emb(dets_embs_second[r], alpha=self.feat_alpha)
                        self._maybe_activate(self.trackers[trk_global])

        # ---- GTA: pure-appearance recovery for remaining unmatched dets ----
        if self.gta_enabled and len(unmatched_dets) > 0 and len(unmatched_trks) > 0:
            unmatched_dets = self._gta_appearance_recovery(dets, dets_embs, unmatched_dets, unmatched_trks, is_obb=True)

        # ---- GTA: resurrect from graveyard before creating new tracks ----
        if self.gta_enabled and self.with_reid and len(unmatched_dets) > 0:
            unmatched_dets = self._gta_resurrect(dets, dets_embs, unmatched_dets, is_obb=True)

        # ---- New tracks for remaining unmatched high-conf detections ----
        for i in unmatched_dets:
            det_conf = self.detection_layout.confidences(dets)[i]
            if det_conf >= self.new_track_thresh:
                det_emb = dets_embs[i] if self.with_reid else None
                new_trk = KalmanBoxTracker(
                    dets[i, :],
                    max_obs=self.max_obs,
                    emb=det_emb,
                    is_obb=True,
                    adaptive_kf=self.adaptive_kf,
                    id_allocator=self.id_allocator,
                )
                new_trk.is_activated = bool(det_conf >= self.instant_confirm_thresh or self.confirm_hits <= 1)
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
        surviving = []
        dead_tracks = []
        for trk in self.trackers:
            alive = trk.time_since_update <= self.max_age and (
                getattr(trk, "is_activated", True) or trk.time_since_update <= self.tentative_max_age
            )
            if alive:
                surviving.append(trk)
            else:
                dead_tracks.append(trk)
        self._gta_bury_dead(dead_tracks)
        self._gta_evict_stale()
        self.trackers = surviving

        if len(outputs) == 0:
            return self.empty_output(dtype=np.float32)
        return self.format_output_rows(outputs, dtype=np.float32)
