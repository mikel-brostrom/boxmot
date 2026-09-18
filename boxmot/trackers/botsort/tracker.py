# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections import deque

import numpy as np
from typing_extensions import Unpack

from boxmot.reid.protocols import AppearanceEncoder
from boxmot.reid.specs import ReIDConfig
from boxmot.trackers.botsort.config import BotSortConfig
from boxmot.trackers.botsort.track import STrack, TrackState
from boxmot.trackers.common.appearance import resolve_batch_embeddings
from boxmot.trackers.common.association import AssociationStage, run_association_stage
from boxmot.trackers.common.association.matching import embedding_distance, fuse_score
from boxmot.trackers.common.box.base import BoxTracker
from boxmot.trackers.common.constructor import BoxTrackerOptions, validate_runtime_options
from boxmot.trackers.common.motion.cmc.registry import create_cmc
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig
from boxmot.trackers.common.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.common.tracking.lifecycle import joint_stracks, remove_duplicate_stracks, sub_stracks


class BotSort(BoxTracker):
    """Track AABB or OBB detections with two-stage matching, CMC, and optional ReID.

    Attributes:
        lost_stracks (list[STrack]): Tracks kept in the lost state.
        removed_stracks (list[STrack]): Tracks removed from the tracker state.
        buffer_size (int): Track buffer size after frame-rate scaling.
        max_time_lost (int): Maximum number of frames a track may stay lost.
        kalman_filter (KalmanFilterXYWH): Motion model used for prediction.
        cmc: Camera-motion compensation method.
    """

    supports_variable_dt = True

    accepts_embeddings = True

    def __init__(
        self,
        config: BotSortConfig | None = None,
        *,
        kalman: KalmanConfig | None = None,
        reid: ReIDConfig | AppearanceEncoder | None = None,
        **kwargs: Unpack[BoxTrackerOptions],
    ) -> None:
        """Configure confidence stages, lost-track retention, and appearance matching.

        Detection filtering uses track_high_thresh and track_low_thresh; lost-track
        retention uses track_buffer scaled by frame_rate.

        Args:
            config: Immutable algorithm settings. None selects BotSortConfig defaults.
            kalman: Immutable filter noise, timing, and supported behavior settings.
                None preserves tracker defaults.
            reid: Immutable encoder configuration or a canonical appearance encoder.
                Missing embeddings are generated lazily; None selects the default encoder.
            **kwargs: Runtime ``per_class``, ``is_obb``, ``class_ids``, ``class_names``,
                ``mask_guidance``, and ``edgetam`` settings.
        """
        config = BotSortConfig.resolve(config)
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

        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = deque(maxlen=config.removed_stracks_buffer)  # type: deque[STrack]
        self.track_high_thresh = config.track_high_thresh
        self.track_low_thresh = config.track_low_thresh
        self.new_track_thresh = config.new_track_thresh
        self.match_thresh = config.match_thresh

        self.buffer_size = int(config.frame_rate / 30.0 * config.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH(ndim=5 if self.is_obb else 4, noise_config=self.kalman_noise_config)

        self.proximity_thresh = config.proximity_thresh
        self.appearance_thresh = config.appearance_thresh
        self.second_match_thresh = config.second_match_thresh
        self.unconfirmed_match_thresh = config.unconfirmed_match_thresh
        self.unconfirmed_emb_scale = config.unconfirmed_emb_scale
        self.use_embeddings = config.use_embeddings

        self.cmc = create_cmc(config.cmc_method, enabled=config.use_cmc)
        self.fuse_first_associate = config.fuse_first_associate
        self._requires_frame = self._requires_frame or self.cmc is not None
        if self.cmc is not None:
            self._requires_frame_dimensions_only = False

    def _detection_boxes(self, dets: np.ndarray) -> np.ndarray:
        return self.detection_layout.boxes(dets)

    def _obb_detections_to_cmc_boxes(self, dets: np.ndarray) -> np.ndarray:
        """Return oriented detections for polygon-aware CMC masking."""
        return self.cmc_detection_boxes(dets)

    def _apply_aabb_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Apply BoT-SORT's axis-aligned camera-motion transform."""
        warp = self.cmc.apply(img, self.cmc_detection_boxes(dets))
        STrack.multi_gmc(strack_pool + unconfirmed, warp)

    def _apply_obb_camera_motion_compensation(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        strack_pool: list[STrack],
        unconfirmed: list[STrack],
    ) -> None:
        """Apply OBB-specific CMC using oriented masks and state correction."""
        warp = self.cmc.apply(img, self.cmc_detection_boxes(dets))
        STrack.multi_gmc_obb(strack_pool + unconfirmed, warp)

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

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets, dets_first, first_batch, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        features_high = resolve_batch_embeddings(
            first_batch,
            enabled=self.use_embeddings,
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
        batch = self._make_detection_batch(dets, embs=embs)
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
            if self.use_embeddings:
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
        STrack.multi_predict(strack_pool, dt=self._prediction_dt)

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

        tracked_pairs = []
        lost_pairs = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            pair = (track, detections[idet])
            if track.state == TrackState.Tracked:
                tracked_pairs.append(pair)
                activated_stracks.append(track)
            else:
                lost_pairs.append(pair)
                refind_stracks.append(track)
        STrack.multi_update(tracked_pairs, self.frame_count)
        STrack.multi_update(lost_pairs, self.frame_count, reactivate=True)

        return matches, u_track, u_detection

    def _first_association_cost(self, tracks, detections) -> np.ndarray:
        geometry_dists = self.association_distance(tracks, detections)
        geometry_dists_mask = geometry_dists > self.proximity_thresh
        if self.fuse_first_associate:
            geometry_dists = fuse_score(geometry_dists, detections)

        costs = geometry_dists
        if self.use_embeddings:
            emb_dists = embedding_distance(tracks, detections)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[geometry_dists_mask] = 1.0
            costs = np.minimum(geometry_dists, emb_dists)
        return self._condition_association(costs, tracks, detections, threshold=self.match_thresh)

    def _second_association_cost(self, tracks, detections) -> np.ndarray:
        """Apply temporal masks to the low-confidence pass at its own threshold."""
        return self._condition_association(
            self.association_distance(tracks, detections),
            tracks,
            detections,
            threshold=self.second_match_thresh,
        )

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
            cost=self._second_association_cost,
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

        tracked_pairs = []
        lost_pairs = []
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            pair = (track, detections_second[idet])
            if track.state == TrackState.Tracked:
                tracked_pairs.append(pair)
                activated_stracks.append(track)
            else:
                lost_pairs.append(pair)
                refind_stracks.append(track)
        STrack.multi_update(tracked_pairs, self.frame_count)
        STrack.multi_update(lost_pairs, self.frame_count, reactivate=True)

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
        pairs = [(unconfirmed[itracked], detections[idet]) for itracked, idet in matches]
        STrack.multi_update(pairs, self.frame_count)
        activated_stracks.extend(track for track, _ in pairs)

        # Mark unmatched unconfirmed tracks as removed
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        return matches, u_unconfirmed, u_detection

    def _unconfirmed_association_cost(self, tracks, detections) -> np.ndarray:
        geometry_dists = self.association_distance(tracks, detections)
        geometry_dists_mask = geometry_dists > self.proximity_thresh
        geometry_dists = fuse_score(geometry_dists, detections)

        costs = geometry_dists
        if self.use_embeddings:
            emb_dists = embedding_distance(tracks, detections) / self.unconfirmed_emb_scale
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[geometry_dists_mask] = 1.0
            costs = np.minimum(geometry_dists, emb_dists)
        return self._condition_association(costs, tracks, detections, threshold=self.unconfirmed_match_thresh)

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
        tracked_pairs = []
        lost_pairs = []
        for itracked, idet in matches:
            track = strack_pool[itracked]
            pair = (track, detections[idet])
            if track.state == TrackState.Tracked:
                tracked_pairs.append(pair)
                activated_stracks.append(track)
            else:
                lost_pairs.append(pair)
                refind_stracks.append(track)
        STrack.multi_update(tracked_pairs, self.frame_count)
        STrack.multi_update(lost_pairs, self.frame_count, reactivate=True)

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
        self.kalman_filter = KalmanFilterXYWH(ndim=5 if self.is_obb else 4, noise_config=self.kalman_noise_config)
