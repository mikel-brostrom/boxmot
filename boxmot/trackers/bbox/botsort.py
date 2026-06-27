# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.appearance import resolve_batch_embeddings
from boxmot.trackers.common.association import AssociationStage, run_association_stage
from boxmot.trackers.common.association.matching import embedding_distance, fuse_score, iou_distance
from boxmot.trackers.common.motion.cmc import create_cmc
from boxmot.trackers.common.tracking.lifecycle import joint_stracks, remove_duplicate_stracks, sub_stracks
from boxmot.trackers.common.track_models.botsort import STrack, TrackState


class BotSort(BaseTracker):
    """Initialize the BotSort tracker.

    Args:
        reid_model: Pre-built ReID backend model (e.g. ``ReID(...).model``).
            Required when ``with_reid=True``.
        track_high_thresh (float): Confidence threshold for the first
            association pass.
        track_low_thresh (float): Lower confidence bound for candidate
            detections.
        new_track_thresh (float): Threshold required to initialize a new track.
        track_buffer (int): Number of frames to keep unmatched tracks alive.
        match_thresh (float): Matching threshold used during association.
        proximity_thresh (float): IoU gate used before appearance matching.
        appearance_thresh (float): Maximum embedding distance accepted for ReID
            matching.
        use_cmc (bool): Whether to apply camera-motion compensation.
        cmc_method (str): Camera-motion compensation method.
        frame_rate (int): Frame rate used to scale the internal track buffer.
        fuse_first_associate (bool): Whether to fuse motion and appearance in
            the first association step.
        with_reid (bool): Whether to enable appearance features.
        second_match_thresh (float): Matching threshold for the second
            association pass over low-confidence detections.
        unconfirmed_match_thresh (float): Matching threshold for tentative
            tracks that have not yet been confirmed.
        unconfirmed_emb_scale (float): Divisor applied to embedding distances
            during unconfirmed-track matching.
        removed_stracks_buffer (int): Maximum number of removed tracks retained
            for duplicate bookkeeping.
        **kwargs: Base tracker settings forwarded to :class:`BaseTracker`,
            including ``det_thresh``, ``max_age``, ``max_obs``, ``min_hits``,
            ``iou_threshold``, ``per_class``, ``class_ids``, ``class_names``,
            ``asso_func``, and ``is_obb``.

    Attributes:
        lost_stracks (list[STrack]): Tracks kept in the lost state.
        removed_stracks (list[STrack]): Tracks removed from the tracker state.
        buffer_size (int): Track buffer size after frame-rate scaling.
        max_time_lost (int): Maximum number of frames a track may stay lost.
        kalman_filter (KalmanFilterXYWH): Motion model used for prediction.
        model: ReID model used for appearance extraction when enabled.
        cmc: Camera-motion compensation method.
    """

    supports_obb = True

    def __init__(
        self,
        reid_model: Any | None = None,
        # BotSort-specific parameters
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        use_cmc: bool = True,
        cmc_method: str = "ecc",
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        second_match_thresh: float = 0.5,
        unconfirmed_match_thresh: float = 0.7,
        unconfirmed_emb_scale: float = 2.0,
        removed_stracks_buffer: int = 100,
        **kwargs: Any,  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="BotSort", **kwargs)

        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = deque(maxlen=removed_stracks_buffer)  # type: deque[STrack]
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH(ndim=5 if self.is_obb else 4)

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.second_match_thresh = second_match_thresh
        self.unconfirmed_match_thresh = unconfirmed_match_thresh
        self.unconfirmed_emb_scale = unconfirmed_emb_scale
        self.with_reid = with_reid
        self.model = reid_model if self.with_reid else None

        self.cmc = create_cmc(cmc_method, enabled=use_cmc)
        self.fuse_first_associate = fuse_first_associate

    def _kalman_ndim(self) -> int:
        return self.detection_layout.box_cols

    def _detection_boxes(self, dets: np.ndarray) -> np.ndarray:
        return self.detection_layout.boxes(dets)

    def _obb_detections_to_cmc_boxes(self, dets: np.ndarray) -> np.ndarray:
        """Convert OBB detections to enclosing AABBs for CMC feature masking."""
        return self.cmc_detection_boxes(dets)

    def _apply_aabb_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Apply the legacy BoTSORT CMC path for axis-aligned tracks."""
        warp = self.cmc.apply(img, self.cmc_detection_boxes(dets))
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

    def _apply_obb_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Apply OBB-specific CMC using enclosing AABBs for estimation."""
        warp = self.cmc.apply(img, self.cmc_detection_boxes(dets))
        STrack.multi_gmc_obb(strack_pool, warp)
        STrack.multi_gmc_obb(unconfirmed, warp)

    def _apply_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Dispatch camera motion compensation without mixing AABB and OBB logic."""
        if self.cmc is None:
            return
        if self.is_obb:
            self._apply_obb_camera_motion_compensation(dets, img, strack_pool, unconfirmed)
            return
        self._apply_aabb_camera_motion_compensation(dets, img, strack_pool, unconfirmed)

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.kalman_filter = KalmanFilterXYWH(ndim=self._kalman_ndim())
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets, dets_first, first_batch, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        features_high = resolve_batch_embeddings(
            first_batch,
            img,
            model=self.model,
            enabled=self.with_reid,
            boxes=self._detection_boxes(dets_first),
            placeholder_value=1.0,
        )

        # Create detections
        detections = self._create_detections(dets_first, features_high)

        # Separate unconfirmed and active tracks
        unconfirmed, active_tracks = self._separate_tracks()

        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # First association
        matches_first, u_track_first, u_detection_first = self._first_association(
            dets,
            dets_first,
            active_tracks,
            unconfirmed,
            img,
            detections,
            activated_stracks,
            refind_stracks,
            strack_pool,
        )

        # Second association
        matches_second, u_track_second, u_detection_second = self._second_association(
            dets_second,
            activated_stracks,
            lost_stracks,
            refind_stracks,
            u_track_first,
            strack_pool,
        )

        # Handle unconfirmed tracks
        matches_unc, u_track_unc, u_detection_unc = self._handle_unconfirmed_tracks(
            u_detection_first,
            detections,
            activated_stracks,
            removed_stracks,
            unconfirmed,
        )

        # Initialize new tracks
        self._initialize_new_tracks(
            u_detection_unc,
            activated_stracks,
            [detections[i] for i in u_detection_first],
        )

        # Update lost and removed tracks
        self._update_track_states(removed_stracks)

        # Merge and prepare output
        return self._prepare_output(activated_stracks, refind_stracks, lost_stracks, removed_stracks)

    def _split_detections(self, dets, embs):
        batch = self.make_detection_batch(dets, embs=embs)
        first_batch, second_batch = batch.split_by_confidence(
            high_thresh=self.track_high_thresh,
            low_thresh=self.track_low_thresh,
        )
        dets = batch.as_indexed_detections(dtype=dets.dtype)
        dets_first = first_batch.as_indexed_detections(dtype=dets.dtype)
        dets_second = second_batch.as_indexed_detections(dtype=dets.dtype)
        return dets, dets_first, first_batch, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [
                    STrack(
                        det,
                        f,
                        id_allocator=self.id_allocator,
                        max_obs=self.max_obs,
                        is_obb=self.is_obb,
                    )
                    for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [
                    STrack(
                        det,
                        id_allocator=self.id_allocator,
                        max_obs=self.max_obs,
                        is_obb=self.is_obb,
                    )
                    for det in dets_first
                ]
        else:
            detections = []
        return detections

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(
        self,
        dets,
        dets_first,
        active_tracks,
        unconfirmed,
        img,
        detections,
        activated_stracks,
        refind_stracks,
        strack_pool,
    ):
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        self._apply_camera_motion_compensation(dets, img, strack_pool, unconfirmed)

        first_stage = AssociationStage(
            name="botsort_high",
            cost=self._first_association_cost,
            threshold=self.match_thresh,
        )
        first_result = run_association_stage(first_stage, strack_pool, detections)
        matches = first_result.matches
        u_track = first_result.unmatched_tracks
        u_detection = first_result.unmatched_dets

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        return matches, u_track, u_detection

    def _first_association_cost(self, tracks, detections) -> np.ndarray:
        ious_dists = iou_distance(tracks, detections, is_obb=self.is_obb)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if not self.with_reid:
            return ious_dists

        emb_dists = embedding_distance(tracks, detections)
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        return np.minimum(ious_dists, emb_dists)

    def _second_association(
        self,
        dets_second,
        activated_stracks,
        lost_stracks,
        refind_stracks,
        u_track_first,
        strack_pool,
    ):
        if len(dets_second) > 0:
            detections_second = [
                STrack(
                    det,
                    id_allocator=self.id_allocator,
                    max_obs=self.max_obs,
                    is_obb=self.is_obb,
                )
                for det in dets_second
            ]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track_first if strack_pool[i].state == TrackState.Tracked]

        second_stage = AssociationStage(
            name="botsort_low",
            cost=lambda tracks, dets: iou_distance(
                tracks,
                dets,
                is_obb=self.is_obb,
            ),
            threshold=self.second_match_thresh,
        )
        second_result = run_association_stage(
            second_stage,
            r_tracked_stracks,
            detections_second,
        )
        matches = second_result.matches
        u_track = second_result.unmatched_tracks
        u_detection = second_result.unmatched_dets

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        return matches, u_track, u_detection

    def _handle_unconfirmed_tracks(self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed):
        """
        Handle unconfirmed tracks (tracks with only one detection frame).

        Args:
            u_detection: Unconfirmed detection indices.
            detections: Current list of detections.
            activated_stracks: List of newly activated tracks.
            removed_stracks: List of tracks to remove.
        """
        # Only use detections that are unconfirmed (filtered by u_detection)
        detections = [detections[i] for i in u_detection]

        unconfirmed_stage = AssociationStage(
            name="botsort_unconfirmed",
            cost=self._unconfirmed_association_cost,
            threshold=self.unconfirmed_match_thresh,
        )
        unconfirmed_result = run_association_stage(
            unconfirmed_stage,
            unconfirmed,
            detections,
        )
        matches = unconfirmed_result.matches
        u_unconfirmed = unconfirmed_result.unmatched_tracks
        u_detection = unconfirmed_result.unmatched_dets

        # Update matched unconfirmed tracks
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        # Mark unmatched unconfirmed tracks as removed
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        return matches, u_unconfirmed, u_detection

    def _unconfirmed_association_cost(self, tracks, detections) -> np.ndarray:
        ious_dists = iou_distance(tracks, detections, is_obb=self.is_obb)
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)

        if not self.with_reid:
            return ious_dists

        emb_dists = embedding_distance(tracks, detections) / self.unconfirmed_emb_scale
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        return np.minimum(ious_dists, emb_dists)

    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
        for inew in u_detections:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_count)
            activated_stracks.append(track)

    def _update_tracks(
        self,
        matches,
        strack_pool,
        detections,
        activated_stracks,
        refind_stracks,
        mark_removed=False,
    ):
        # Update or reactivate matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        # Mark only unmatched tracks as removed, if mark_removed flag is True
        if mark_removed:
            unmatched_tracks = [strack_pool[i] for i in range(len(strack_pool)) if i not in [m[0] for m in matches]]
            for track in unmatched_tracks:
                track.mark_removed()

    def _update_track_states(self, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks):
        self.active_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(self.active_tracks, self.lost_stracks)

        return self.format_outputs(
            [t for t in self.active_tracks if t.is_activated],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._reset_common_state()
        self.kalman_filter = KalmanFilterXYWH(ndim=5 if self.is_obb else 4)
