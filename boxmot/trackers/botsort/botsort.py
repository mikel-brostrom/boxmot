# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch

from boxmot.appearance.reid.auto_backend import ReidAutoBackend
from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import (
    joint_stracks,
    remove_duplicate_stracks,
    sub_stracks,
)
from boxmot.utils.matching import (
    embedding_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
)


class BotSort(BaseTracker):
    """
    BoTSORT Tracker: A tracking algorithm that combines appearance and motion-based tracking.

    Args:
        reid_weights (str): Path to the model weights for ReID.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        half (bool): Use half-precision (fp16) for faster inference.
        per_class (bool, optional): Whether to perform per-class tracking.
        track_high_thresh (float, optional): Detection confidence threshold for first association.
        track_low_thresh (float, optional): Detection confidence threshold for ignoring detections.
        new_track_thresh (float, optional): Threshold for creating a new track.
        track_buffer (int, optional): Frames to keep a track alive after last detection.
        match_thresh (float, optional): Matching threshold for data association.
        proximity_thresh (float, optional): IoU threshold for first-round association.
        appearance_thresh (float, optional): Appearance embedding distance threshold for ReID.
        cmc_method (str, optional): Method for correcting camera motion, e.g., "sof" (simple optical flow).
        frame_rate (int, optional): Video frame rate, used to scale the track buffer.
        fuse_first_associate (bool, optional): Fuse appearance and motion in the first association step.
        with_reid (bool, optional): Use ReID features for association.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        per_class: bool = False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "ecc",
        frame_rate=30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
    ):
        super().__init__(per_class=per_class)
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        if self.with_reid:
            self.model = ReidAutoBackend(
                weights=reid_weights, device=device, half=half
            ).model

        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        if self.with_reid and embs is None:
            features_high = self.model.get_features(dets_first[:, 0:4], img)
        else:
            features_high = embs_first if embs_first is not None else []

        # Create detections
        detections = self._create_detections(dets_first, features_high)

        # Separate unconfirmed and active tracks
        unconfirmed, active_tracks = self._separate_tracks()

        # Apply camera motion compensation
        warp = self.cmc.apply(img, dets)
        STrack.multi_gmc(active_tracks, warp)
        STrack.multi_gmc(unconfirmed, warp)
        STrack.multi_gmc(self.lost_stracks, warp)

        # --- STAGE 1: Associate Active Tracks (Motion + Appearance) ---
        STrack.multi_predict(active_tracks)
        dists = self._calculate_cost_matrix(active_tracks, detections, use_motion=True)
        matches_active, u_track_active, u_det_active = linear_assignment(dists, thresh=self.match_thresh)
        self._update_tracks(matches_active, active_tracks, detections, activated_stracks, refind_stracks)

        # --- STAGE 2: Associate Lost Tracks (Appearance only) ---
        remaining_dets = [detections[i] for i in u_det_active]

        if self.lost_stracks and remaining_dets and self.with_reid:
            dists_lost = self._calculate_cost_matrix(self.lost_stracks, remaining_dets, use_motion=False)
            matches_lost, u_track_lost, u_det_lost_indices = linear_assignment(dists_lost, thresh=self.appearance_thresh)
            self._update_tracks(matches_lost, self.lost_stracks, remaining_dets, activated_stracks, refind_stracks)
            final_unmatched_det_indices = [u_det_active[i] for i in u_det_lost_indices]
            unmatched_lost_tracks_indices = u_track_lost
        else:
            final_unmatched_det_indices = u_det_active
            unmatched_lost_tracks_indices = list(range(len(self.lost_stracks)))

        # --- Combine all tracks that were not matched in the first two stages ---
        unmatched_active_tracks = [active_tracks[i] for i in u_track_active]
        unmatched_lost_tracks = [self.lost_stracks[i] for i in unmatched_lost_tracks_indices]
        tracks_for_low_conf = unmatched_active_tracks + unmatched_lost_tracks

        # --- STAGE 3: Second Association (Rescue with Low-Conf Dets) ---
        if dets_second.any() and tracks_for_low_conf:
            low_conf_detections = self._create_detections(dets_second, None, with_reid=False)
            dists_low = iou_distance(tracks_for_low_conf, low_conf_detections)
            matches_second, u_track_second, _ = linear_assignment(dists_low, thresh=0.5)
            self._update_tracks(matches_second, tracks_for_low_conf, low_conf_detections, activated_stracks, refind_stracks)

            for i in u_track_second:
                track = tracks_for_low_conf[i]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
        else:
            for track in tracks_for_low_conf:
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

        # --- STAGE 4: Handle Unconfirmed and New Tracks ---
        if unconfirmed:
            remaining_high_conf_dets = [detections[i] for i in final_unmatched_det_indices]
            dists_unc = self._calculate_cost_matrix(unconfirmed, remaining_high_conf_dets, use_motion=True)
            matches_unc, u_track_unc, u_det_unc_indices = linear_assignment(dists_unc, thresh=0.7)

            for itracked, idet in matches_unc:
                unconfirmed[itracked].update(remaining_high_conf_dets[idet], self.frame_count)
                activated_stracks.append(unconfirmed[itracked])

            for it in u_track_unc:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            final_unmatched_det_indices = [final_unmatched_det_indices[i] for i in u_det_unc_indices]

        self._initialize_new_tracks(
            final_unmatched_det_indices,
            activated_stracks,
            detections,
        )

        # Update track states (e.g., remove old lost tracks)
        self._update_track_states(lost_stracks, removed_stracks)

        return self._prepare_output(
            activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )

    def _calculate_cost_matrix(self, tracks, detections, use_motion: bool):
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)))

        if use_motion:
            # Combine motion and appearance
            ious_dists = iou_distance(tracks, detections)
            if self.with_reid:
                emb_dists = embedding_distance(tracks, detections)
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                ious_dists_mask = ious_dists > self.proximity_thresh
                emb_dists[ious_dists_mask] = 1.0
                return np.minimum(ious_dists, emb_dists)
            return ious_dists
        else:  # Appearance only
            if self.with_reid:
                emb_dists = embedding_distance(tracks, detections)
                emb_dists[emb_dists > self.appearance_thresh] = 1.0
                return emb_dists
            else:  # Fallback to IoU if reid is disabled
                return iou_distance(tracks, detections)

    def _split_detections(self, dets, embs):
        # Detections with high confidence
        first_mask = dets[:, 4] >= self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else []

        # Detections with low confidence
        second_mask = (~first_mask) & (dets[:, 4] > self.track_low_thresh)
        dets_second = dets[second_mask]

        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets, features, with_reid=True):
        if len(dets) > 0:
            if self.with_reid and with_reid:
                return [
                    STrack(det, f, max_obs=self.max_obs)
                    for (det, f) in zip(dets, features)
                ]
            else:
                return [STrack(det, max_obs=self.max_obs) for det in dets]
        return []

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _initialize_new_tracks(self, u_detections_indices, activated_stracks, detections):
        for inew in u_detections_indices:
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
            unmatched_tracks = [
                strack_pool[i]
                for i in range(len(strack_pool))
                if i not in [m[0] for m in matches]
            ]
            for track in unmatched_tracks:
                track.mark_removed()

    def _update_track_states(self, lost_stracks, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(
        self, activated_stracks, refind_stracks, lost_stracks, removed_stracks
    ):
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            for t in self.active_tracks
            if t.is_activated
        ]

        return np.asarray(outputs)
