# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import torch
import numpy as np
from pathlib import Path

from boxmot.motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from boxmot.appearance.reid_auto_backend import ReidAutoBackend
from boxmot.motion.cmc.sof import SOF
from boxmot.trackers.botsort.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.botsort.botsort_utils import joint_stracks, sub_stracks, remove_duplicate_stracks 
from boxmot.trackers.botsort.botsort_track import STrack
from boxmot.motion.cmc import get_cmc_method



class BoTSORT(BaseTracker):
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
        cmc_method: str = "sof",
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

        self.cmc = get_cmc_method('ecc')()
        self.fuse_first_associate = fuse_first_associate

    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        self.check_inputs(dets, img)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)
        
        risky_detections, safe_det_trk_pairs = self.candidates_to_detections(dets[:, 0:4], self.active_tracks)

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
        matches_first, u_track_first, u_detection_first = self._first_association(dets, dets_first, active_tracks, unconfirmed, img, detections, activated_stracks, refind_stracks, strack_pool)

        # Second association
        matches_second, u_track_second, u_detection_second = self._second_association(dets_second, activated_stracks, lost_stracks, refind_stracks, u_track_first, strack_pool)

        # Handle unconfirmed tracks
        matches_unc, u_track_unc, u_detection_unc = self._handle_unconfirmed_tracks(u_detection_first, detections, activated_stracks, removed_stracks, unconfirmed)

        # Initialize new tracks
        self._initialize_new_tracks(u_detection_unc, activated_stracks, [detections[i] for i in u_detection_first])

        # Update lost and removed tracks
        self._update_track_states(lost_stracks, removed_stracks)

        # Merge and prepare output
        return self._prepare_output(activated_stracks, refind_stracks, lost_stracks, removed_stracks)

    def _split_detections(self, dets, embs):
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [STrack(det, f, max_obs=self.max_obs) for (det, f) in zip(dets_first, features_high)]
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

    def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img, detections, activated_stracks, refind_stracks, strack_pool):
        
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
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
                
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

    def _second_association(self, dets_second, activated_stracks, lost_stracks, refind_stracks, u_track_first, strack_pool):
        if len(dets_second) > 0:
            detections_second = [STrack(det, max_obs=self.max_obs) for det in dets_second]
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
        
        # Calculate IoU distance between unconfirmed tracks and detections
        ious_dists = iou_distance(unconfirmed, detections)
        
        # Apply IoU mask to filter out distances that exceed proximity threshold
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)
        
        # Fuse scores for IoU-based and embedding-based matching (if applicable)
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0  # Apply the IoU mask to embedding distances
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

    def _update_tracks(self, matches, strack_pool, detections, activated_stracks, refind_stracks, mark_removed=False):
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

    def _update_track_states(self, lost_stracks, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks):
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
            for t in self.active_tracks if t.is_activated
        ]

        return np.asarray(outputs)


    def aiou(self, bbox, candidates):
        """
        IoU - Aspect Ratio

        """
        candidates = np.array(candidates)
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
        candidates_tl = candidates[:, :2]
        candidates_br = candidates[:, :2] + candidates[:, 2:]

        tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
                np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
        br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
                np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
        wh = np.maximum(0., br - tl)

        area_intersection = wh.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_candidates = candidates[:, 2:].prod(axis=1)

        iou = area_intersection / (area_bbox + area_candidates - area_intersection)

        aspect_ratio = bbox[2] / bbox[3]
        candidates_aspect_ratio = candidates[:, 2] / candidates[:, 3]
        arctan = np.arctan(aspect_ratio) - np.arctan(candidates_aspect_ratio)
        v = 1 - ((4 / np.pi ** 2) * arctan ** 2)
        alpha = v / (1 - iou + v)
        
        return iou, alpha

    def aiou_vectorized(self, bboxes1, bboxes2):
        """
        Vectorized implementation of AIOU (IoU with aspect ratio consideration)
        
        Args:
        bboxes1: numpy array of shape (N, 4) in format (x, y, w, h)
        bboxes2: numpy array of shape (M, 4) in format (x, y, w, h)
        
        Returns:
        ious: numpy array of shape (N, M) containing IoU values
        alphas: numpy array of shape (N, M) containing alpha values
        """
        # Convert (x, y, w, h) to (x1, y1, x2, y2)
        bboxes1_x1y1 = bboxes1[:, :2]
        bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 2:]
        bboxes2_x1y1 = bboxes2[:, :2]
        bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 2:]
        
        # Compute intersection
        intersect_x1y1 = np.maximum(bboxes1_x1y1[:, None], bboxes2_x1y1[None, :])
        intersect_x2y2 = np.minimum(bboxes1_x2y2[:, None], bboxes2_x2y2[None, :])
        intersect_wh = np.maximum(0., intersect_x2y2 - intersect_x1y1)
        
        # Compute areas
        intersect_area = intersect_wh.prod(axis=2)
        bboxes1_area = bboxes1[:, 2:].prod(axis=1)
        bboxes2_area = bboxes2[:, 2:].prod(axis=1)
        union_area = bboxes1_area[:, None] + bboxes2_area[None, :] - intersect_area
        
        # Compute IoU
        ious = intersect_area / union_area
        
        # Compute aspect ratios
        bboxes1_aspect_ratio = bboxes1[:, 2] / bboxes1[:, 3]
        bboxes2_aspect_ratio = bboxes2[:, 2] / bboxes2[:, 3]
        
        # Compute alpha
        arctan_diff = np.arctan(bboxes1_aspect_ratio[:, None]) - np.arctan(bboxes2_aspect_ratio[None, :])
        v = 1 - ((4 / (np.pi ** 2)) * arctan_diff ** 2)
        alphas = v / (1 - ious + v)
        
        return ious, alphas


    def candidates_to_detections(self, tlwh: np.ndarray, confirmed_tracks):
        tracklet_bboxes = np.array([trk.to_xywh() for trk in confirmed_tracks])

        if len(tracklet_bboxes) == 0:
            return list(range(tlwh.shape[0])), []

        print(tlwh.shape)
        print(tracklet_bboxes.shape)

        ious, alphas = self.aiou_vectorized(tlwh, tracklet_bboxes)

        matches = ious > 0.5
        match_counts = np.sum(matches, axis=1)

        risky_detections = np.where(match_counts != 1)[0].tolist()
        safe_candidates = np.where(match_counts == 1)[0]

        safe_det_trk_pairs = []
        for i in safe_candidates:
            candidate = np.argmax(ious[i])
            if alphas[i, candidate] > 0.6:
                safe_det_trk_pairs.append((i, candidate))
            else:
                risky_detections.append(i)

        return risky_detections, safe_det_trk_pairs