# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from pathlib import Path

import numpy as np
import torch

from boxmot.motion.cmc import get_cmc_method
from boxmot.motion.kalman_filters.aabb.xywh_kf import KalmanFilterXYWH
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.trackers.botsort.botsort_utils import (joint_stracks,
                                                   remove_duplicate_stracks,
                                                   sub_stracks)
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)


class BotSort(BaseTracker):
    """
    Initialize the BotSort tracker with various parameters.

    Parameters:
    - reid_weights (Path): Path to the re-identification model weights.
    - device (torch.device): Device to run the model on (e.g., 'cpu', 'cuda').
    - half (bool): Whether to use half-precision (fp16) for faster inference.
    - det_thresh (float): Detection threshold for considering detections.
    - max_age (int): Maximum age (in frames) of a track before it is considered lost.
    - max_obs (int): Maximum number of historical observations stored for each track. Always greater than max_age by minimum 5.
    - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
    - iou_threshold (float): IOU threshold for determining match between detection and tracks.
    - per_class (bool): Enables class-separated tracking.
    - nr_classes (int): Total number of object classes that the tracker will handle (for per_class=True).
    - asso_func (str): Algorithm name used for data association between detections and tracks.
    - is_obb (bool): Work with Oriented Bounding Boxes (OBB) instead of standard axis-aligned bounding boxes.
    
    BotSort-specific parameters:
    - track_high_thresh (float): Detection confidence threshold for first association.
    - track_low_thresh (float): Detection confidence threshold for ignoring detections.
    - new_track_thresh (float): Threshold for creating a new track.
    - track_buffer (int): Frames to keep a track alive after last detection.
    - match_thresh (float): Matching threshold for data association.
    - proximity_thresh (float): IoU threshold for first-round association.
    - appearance_thresh (float): Appearance embedding distance threshold for ReID.
    - cmc_method (str): Method for correcting camera motion, e.g., "sof" (simple optical flow).
    - frame_rate (int): Video frame rate, used to scale the track buffer.
    - fuse_first_associate (bool): Fuse appearance and motion in the first association step.
    - with_reid (bool): Use ReID features for association.
    
    Attributes:
    - frame_count (int): Counter for the frames processed.
    - active_tracks (list): List to hold active tracks.
    - lost_stracks (list[STrack]): List of lost tracks.
    - removed_stracks (list[STrack]): List of removed tracks.
    - buffer_size (int): Size of the track buffer based on frame rate.
    - max_time_lost (int): Maximum time a track can be lost.
    - kalman_filter (KalmanFilterXYWH): Kalman filter for motion prediction.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        # BotSort-specific parameters
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "ecc",
        frame_rate: int = 30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        **kwargs  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='BotSort', **kwargs)
        
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

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
        return self._prepare_output(
            activated_stracks, refind_stracks, lost_stracks, removed_stracks
        )

    def _split_detections(self, dets, embs):
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        second_mask = np.logical_and(
            confs > self.track_low_thresh, confs < self.track_high_thresh
        )
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [
                    STrack(det, f, max_obs=self.max_obs)
                    for (det, f) in zip(dets_first, features_high)
                ]
            else:
                detections = [STrack(det, max_obs=self.max_obs) for det in dets_first]
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
        warp = self.cmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high confidence detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

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
                STrack(det, max_obs=self.max_obs) for det in dets_second
            ]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Tracked
        ]

        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)

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

    def _handle_unconfirmed_tracks(
        self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed
    ):
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

        # Calculate IoU distance between unconfirmed tracks and detections
        ious_dists = iou_distance(unconfirmed, detections)

        # Apply IoU mask to filter out distances that exceed proximity threshold
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)

        # Fuse scores for IoU-based and embedding-based matching (if applicable)
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = (
                1.0  # Apply the IoU mask to embedding distances
            )
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        # Perform data association using linear assignment on the combined distances
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)

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
            unmatched_tracks = [
                strack_pool[i]
                for i in range(len(strack_pool))
                if i not in [m[0] for m in matches]
            ]
            for track in unmatched_tracks:
                track.mark_removed()

    def _update_track_states(self, removed_stracks):
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
