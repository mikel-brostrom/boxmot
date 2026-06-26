# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from typing import Any

import numpy as np

from boxmot.motion.kalman_filters.xyah import KalmanFilterXYAH
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.association import AssociationStage, run_association_stage
from boxmot.trackers.common.association.matching import fuse_score, iou_distance
from boxmot.trackers.common.tracking.lifecycle import joint_stracks, remove_duplicate_stracks, sub_stracks
from boxmot.trackers.common.track_models.bytetrack import STrack, TrackState


class ByteTrack(BaseTracker):
    """Initialize the ByteTrack tracker.

    Args:
        min_conf (float): Minimum confidence used for the low-score association
            stage. Detections below this value are discarded.
        track_thresh (float): Confidence threshold for detections that enter the
            first association pass.
        match_thresh (float): Matching threshold used during association.
        track_buffer (int): Number of frames to keep unmatched tracks alive.
        frame_rate (int): Frame rate used to scale the internal track buffer.
        **kwargs: Base tracker settings forwarded to :class:`BaseTracker`,
            including ``det_thresh``, ``max_age``, ``max_obs``, ``min_hits``,
            ``iou_threshold``, ``per_class``, ``class_ids``, ``class_names``,
            ``asso_func``, and ``is_obb``.

    Attributes:
        frame_count (int): Number of processed frames.
        active_tracks (list[STrack]): Currently active tracks.
        lost_stracks (list[STrack]): Tracks kept in the lost state.
        removed_stracks (list[STrack]): Tracks removed from the tracker state.
        buffer_size (int): Track buffer size after frame-rate scaling.
        max_time_lost (int): Maximum number of frames a track may stay lost.
        kalman_filter (KalmanFilterXYAH): Motion model used for prediction.
    """

    supports_obb = True

    def __init__(
        self,
        # ByteTrack-specific parameters
        min_conf: float = 0.1,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 25,
        frame_rate: int = 30,
        **kwargs: Any,  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="ByteTrack", **kwargs)

        # Track lifecycle parameters
        self.frame_id = 0
        self.track_buffer = track_buffer
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

        # Detection thresholds
        self.min_conf = min_conf
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh  # Same as track_thresh

        # Motion model
        self.kalman_filter = KalmanFilterXYAH()

        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray = None,
        embs: np.ndarray = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        self.check_inputs(dets, img)

        self.kalman_filter = KalmanFilterXYWH(ndim=5) if self.is_obb else KalmanFilterXYAH()
        batch = self.make_detection_batch(dets, embs=embs, masks=masks)
        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        high_batch, second_batch = batch.split_by_confidence(
            high_thresh=self.track_thresh,
            low_thresh=self.min_conf,
        )
        dets_second = second_batch.as_indexed_detections(dtype=dets.dtype)
        dets = high_batch.as_indexed_detections(dtype=dets.dtype)

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(
                    det,
                    max_obs=self.max_obs,
                    id_allocator=self.id_allocator,
                    is_obb=self.is_obb,
                )
                for det in dets
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        first_stage = AssociationStage(
            name="bytetrack_high",
            cost=self._fused_iou_cost,
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
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(
                    det_second,
                    max_obs=self.max_obs,
                    id_allocator=self.id_allocator,
                    is_obb=self.is_obb,
                )
                for det_second in dets_second
            ]
        else:
            detections_second = []
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        second_stage = AssociationStage(
            name="bytetrack_low",
            cost=self._iou_cost,
            threshold=0.5,
        )
        second_result = run_association_stage(
            second_stage,
            r_tracked_stracks,
            detections_second,
        )
        matches = second_result.matches
        u_track = second_result.unmatched_tracks
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        unconfirmed_stage = AssociationStage(
            name="bytetrack_unconfirmed",
            cost=self._fused_iou_cost,
            threshold=0.7,
        )
        unconfirmed_result = run_association_stage(
            unconfirmed_stage,
            unconfirmed,
            detections,
        )
        matches = unconfirmed_result.matches
        u_unconfirmed = unconfirmed_result.unmatched_tracks
        u_detection = unconfirmed_result.unmatched_dets
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_count)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.active_tracks = [t for t in self.active_tracks if t.state == TrackState.Tracked]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(self.active_tracks, self.lost_stracks)
        # get confs of lost tracks
        output_stracks = [track for track in self.active_tracks if track.is_activated]
        return self.format_outputs(output_stracks, dtype=np.float32)

    def _iou_cost(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Build an IoU distance matrix using the current AABB/OBB tracker mode."""
        return iou_distance(tracks, detections, is_obb=self.is_obb)

    def _fused_iou_cost(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Build the ByteTrack score-fused IoU distance matrix."""
        return fuse_score(self._iou_cost(tracks, detections), detections)

    def reset(self) -> None:
        self._reset_common_state()
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH(ndim=5) if self.is_obb else KalmanFilterXYAH()
