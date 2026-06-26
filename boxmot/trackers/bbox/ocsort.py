# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""

from typing import Any

import numpy as np

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.association import (
    AssociationStage,
    detection_track_iou_assignment,
    detection_track_tuple_to_association_result,
    run_association_stage,
)
from boxmot.trackers.common.association.velocity import (
    associate,
)
from boxmot.trackers.common.association.velocity import (
    linear_assignment as legacy_linear_assignment,
)
from boxmot.trackers.common.track_models.ocsort import KalmanBoxTracker, k_previous_obs


class OcSort(BaseTracker):
    """Initialize the OcSort tracker.

    Args:
        min_conf (float): Minimum confidence threshold used in the second-stage
            association pass.
        delta_t (int): Time window used for motion estimation.
        inertia (float): Weight applied to the velocity-direction term during
            matching.
        use_byte (bool): Whether to enable ByteTrack-style second association.
        Q_xy_scaling (float): Process-noise scaling for position coordinates.
        Q_s_scaling (float): Process-noise scaling for scale coordinates.
        **kwargs: Base tracker settings forwarded to :class:`BaseTracker`,
            including ``det_thresh``, ``max_age``, ``max_obs``, ``min_hits``,
            ``iou_threshold``, ``per_class``, ``class_ids``, ``class_names``,
            ``asso_func``, and ``is_obb``.

    Attributes:
        frame_count (int): Number of processed frames.
        active_tracks (list): Currently active tracks.
    """

    supports_obb = True

    def __init__(
        self,
        # OcSort-specific parameters
        min_conf: float = 0.1,
        delta_t: int = 3,
        inertia: float = 0.2,
        use_byte: bool = False,
        Q_xy_scaling: float = 0.01,
        Q_s_scaling: float = 0.0001,
        **kwargs: Any,  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="OcSort", **kwargs)

        # Store OcSort-specific parameters
        self.min_conf: float = min_conf
        self.asso_threshold: float = self.iou_threshold  # Use from BaseTracker
        self.delta_t: int = delta_t
        self.inertia: float = inertia
        self.use_byte: bool = use_byte
        self.Q_xy_scaling: float = Q_xy_scaling
        self.Q_s_scaling: float = Q_s_scaling
        self.frame_count: int = 0

        # Initialize tracker collections
        self.active_tracks: list = []

    def _update_impl(
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

        self.check_inputs(dets, img)

        self.frame_count += 1
        h, w = img.shape[0:2]

        batch = self.make_detection_batch(dets, embs=embs, masks=masks)
        high_batch, second_batch = batch.split_by_confidence(
            high_thresh=self.det_thresh,
            low_thresh=self.min_conf,
        )
        dets_second = second_batch.as_box_conf_detections(dtype=dets.dtype)
        dets = high_batch.as_box_conf_detections(dtype=dets.dtype)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.active_tracks), self.detection_layout.box_with_conf_cols))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.active_tracks[t].predict()[0]
            trk[:] = [pos[i] for i in range(self.detection_layout.box_cols)] + [0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
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
        first_stage = AssociationStage(
            name="ocsort_high",
            threshold=self.asso_threshold,
            matcher=lambda _tracks, _detections: detection_track_tuple_to_association_result(
                associate(
                    dets,
                    trks,
                    self.asso_func,
                    self.asso_threshold,
                    velocities,
                    k_observations,
                    self.inertia,
                    w,
                    h,
                )
            ),
        )
        first_result = run_association_stage(first_stage, self.active_tracks, dets)
        matched = first_result.matches
        unmatched_dets = first_result.unmatched_dets
        unmatched_trks = first_result.unmatched_tracks
        for trk_idx, det_idx in matched:
            self.active_tracks[trk_idx].update(
                dets[det_idx],
                high_batch.clss[det_idx],
                high_batch.det_inds[det_idx],
            )

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)  # iou between low score detections and unmatched tracks
            iou_left = np.array(iou_left)
            low_stage = AssociationStage(
                name="ocsort_low",
                threshold=self.asso_threshold,
                matcher=lambda _tracks, _detections: detection_track_iou_assignment(
                    iou_left,
                    self.asso_threshold,
                    legacy_linear_assignment,
                ),
            )
            low_result = run_association_stage(low_stage, u_trks, dets_second)
            to_remove_trk_indices = []
            for trk_rel, det_ind in low_result.matches:
                trk_ind = unmatched_trks[trk_rel]
                self.active_tracks[trk_ind].update(
                    dets_second[det_ind],
                    second_batch.clss[det_ind],
                    second_batch.det_inds[det_ind],
                )
                to_remove_trk_indices.append(trk_ind)
            unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            rematch_stage = AssociationStage(
                name="ocsort_high_rematch",
                threshold=self.asso_threshold,
                matcher=lambda _tracks, _detections: detection_track_iou_assignment(
                    iou_left,
                    self.asso_threshold,
                    legacy_linear_assignment,
                ),
            )
            rematch_result = run_association_stage(rematch_stage, left_trks, left_dets)
            to_remove_det_indices = []
            to_remove_trk_indices = []
            for trk_rel, det_rel in rematch_result.matches:
                det_ind = unmatched_dets[det_rel]
                trk_ind = unmatched_trks[trk_rel]
                self.active_tracks[trk_ind].update(
                    dets[det_ind],
                    high_batch.clss[det_ind],
                    high_batch.det_inds[det_ind],
                )
                to_remove_det_indices.append(det_ind)
                to_remove_trk_indices.append(trk_ind)
            unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
            unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.active_tracks[m].update(None, None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(
                dets[i],
                high_batch.clss[i],
                high_batch.det_inds[i],
                delta_t=self.delta_t,
                Q_xy_scaling=self.Q_xy_scaling,
                Q_s_scaling=self.Q_s_scaling,
                Q_a_scaling=self.Q_s_scaling,
                max_obs=self.max_obs,
                is_obb=self.is_obb,
                id_allocator=self.id_allocator,
            )
            self.active_tracks.append(trk)
        i = len(self.active_tracks)
        for trk in reversed(self.active_tracks):
            if trk.last_observation.sum() < 0:
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
