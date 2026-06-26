# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license
# SFSORT implementation adapted for BoxMOT

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import cv2
import numpy as np

from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.association.matching import linear_assignment
from boxmot.trackers.common.geometry.obb import (
    align_obb_measurement,
    normalize_angle,
    smooth_obb_corners,
)
from boxmot.trackers.common.tracking.track import (
    TrackState as CommonTrackState,
)
from boxmot.trackers.common.tracking.track import (
    sync_track_meta,
)


class TrackState:
    """Enumeration of possible states of a track."""

    Active = 0
    Lost = 1


@dataclass(eq=False)
class Track:
    """Lightweight track container for SFSORT."""

    bbox: np.ndarray
    last_frame: int
    track_id: int
    conf: float
    cls: int
    det_ind: int
    state: int = TrackState.Active
    history_observations: deque = None
    time_since_update: int = 0
    lost_region: Literal["central", "marginal"] | None = None
    _plot_angle: float | None = None
    theta_damping: float = 0.8
    _theta_velocity: float = 0.0

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        self.conf = float(self.conf)
        self.cls = int(self.cls)
        self.det_ind = int(self.det_ind)
        self.theta_damping = float(np.clip(self.theta_damping, 0.0, 1.0))
        if self.bbox.shape[0] == 5:
            self.history_observations = deque([self._state_obb_for_plot()], maxlen=50)
        else:
            self.history_observations = deque([self.bbox.copy()], maxlen=50)
        self.time_since_update = 0
        self._theta_velocity = 0.0
        self._sync_meta(CommonTrackState.TRACKED)

    @property
    def id(self) -> int:
        return self.track_id

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float(normalize_angle(angle))

    @classmethod
    def _align_obb_measurement(cls, measurement: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Align equivalent OBB forms to the current track state."""
        return align_obb_measurement(measurement, reference)

    def _state_obb_for_plot(self) -> np.ndarray:
        """Return current OBB state as corners with state-only angle smoothing."""
        corners, self._plot_angle = smooth_obb_corners(self.bbox, self._plot_angle)
        return corners

    def update(self, box: np.ndarray, frame_id: int, conf: float, cls: int, det_ind: int) -> None:
        """Update a matched track with latest detection."""

        incoming_bbox = np.asarray(box, dtype=np.float32).reshape(-1)
        if self.bbox.shape[0] == 5 and incoming_bbox.shape[0] == 5:
            aligned = self._align_obb_measurement(incoming_bbox, self.bbox)
            prev_theta = float(self.bbox[4])
            theta_delta = self._wrap_angle(float(aligned[4]) - prev_theta)
            self._theta_velocity = (self.theta_damping * self._theta_velocity) + (
                (1.0 - self.theta_damping) * theta_delta
            )
            aligned[4] = self._wrap_angle(prev_theta + self._theta_velocity)
            self.bbox = aligned.astype(np.float32)
        else:
            self.bbox = incoming_bbox

        if self.bbox.shape[0] == 5:
            self.history_observations.append(self._state_obb_for_plot())
        else:
            self.history_observations.append(self.bbox.copy())
        self.state = TrackState.Active
        self.lost_region = None
        self.time_since_update = 0
        self.last_frame = frame_id
        self.conf = float(conf)
        self.cls = int(cls)
        self.det_ind = int(det_ind)
        self._sync_meta(CommonTrackState.TRACKED)

    def mark_lost(self, region: Literal["central", "marginal"]) -> None:
        """Mark the track lost while preserving SFSORT's region distinction."""
        self.state = TrackState.Lost
        self.lost_region = region
        self._sync_meta(CommonTrackState.LOST)

    def _sync_meta(self, state: CommonTrackState) -> None:
        sync_track_meta(self, state)


class SFSORT(BaseTracker):
    """Initialize the SFSORT tracker.

    Args:
        high_th (float | None): High-confidence threshold for detections.
        match_th_first (float | None): Match threshold for the first
            association pass.
        new_track_th (float | None): Confidence threshold for initializing new
            tracks.
        low_th (float | None): Low-confidence threshold for the second
            association pass.
        match_th_second (float | None): Match threshold for the second
            association pass.
        dynamic_tuning (bool): Whether to enable density-based threshold
            tuning.
        cth (float | None): Confidence threshold used by dynamic tuning.
        high_th_m (float | None): Dynamic adjustment scale for ``high_th``.
        new_track_th_m (float | None): Dynamic adjustment scale for
            ``new_track_th``.
        match_th_first_m (float | None): Dynamic adjustment scale for
            ``match_th_first``.
        obb_theta_damping (float): Damping factor applied to OBB angle updates.
        marginal_timeout (int | None): Timeout for marginally lost tracks.
        central_timeout (int | None): Timeout for centrally lost tracks.
        frame_width (int | None): Optional frame width for margin computation.
        frame_height (int | None): Optional frame height for margin
            computation.
        horizontal_margin (int | None): Horizontal margin for central-loss
            detection.
        vertical_margin (int | None): Vertical margin for central-loss
            detection.
        **kwargs: Base tracker settings forwarded to :class:`BaseTracker`,
            including ``det_thresh``, ``max_age``, ``max_obs``, ``min_hits``,
            ``iou_threshold``, ``per_class``, ``class_ids``, ``class_names``,
            ``asso_func``, and ``is_obb``.
    """

    supports_obb = True

    def __init__(
        self,
        high_th: float | None = 0.6,
        match_th_first: float | None = 0.67,
        new_track_th: float | None = 0.7,
        low_th: float | None = 0.1,
        match_th_second: float | None = 0.3,
        dynamic_tuning: bool = False,
        cth: float | None = 0.5,
        high_th_m: float | None = 0.0,
        new_track_th_m: float | None = 0.0,
        match_th_first_m: float | None = 0.0,
        obb_theta_damping: float = 0.8,
        marginal_timeout: int | None = 0,
        central_timeout: int | None = 0,
        frame_width: int | None = None,
        frame_height: int | None = None,
        horizontal_margin: int | None = None,
        vertical_margin: int | None = None,
        **kwargs: Any,
    ) -> None:
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        det_thresh = 0.6 if high_th is None else float(high_th)
        super().__init__(det_thresh=det_thresh, _tracker_name="SFSORT", **init_args, **kwargs)

        self.high_th = self._resolve_or_default(high_th, 0.6, 0.0, 1.0)
        self.match_th_first = self._resolve_or_default(match_th_first, 0.67, 0.0, 0.67)
        self.new_track_th = self._resolve_or_default(new_track_th, 0.7, self.high_th, 1.0)
        self.low_th = self._resolve_or_default(low_th, 0.1, 0.0, self.high_th)
        self.match_th_second = self._resolve_or_default(match_th_second, 0.3, 0.0, 1.0)

        self.dynamic_tuning = bool(dynamic_tuning)
        self.cth = self._resolve_or_default(cth, 0.5, self.low_th, 1.0)
        if self.dynamic_tuning:
            self.high_th_m = self._resolve_or_default(high_th_m, 0.0, 0.02, 0.1)
            self.new_track_th_m = self._resolve_or_default(new_track_th_m, 0.0, 0.02, 0.08)
            self.match_th_first_m = self._resolve_or_default(match_th_first_m, 0.0, 0.02, 0.08)
        else:
            self.high_th_m = 0.0 if high_th_m is None else float(high_th_m)
            self.new_track_th_m = 0.0 if new_track_th_m is None else float(new_track_th_m)
            self.match_th_first_m = 0.0 if match_th_first_m is None else float(match_th_first_m)
        self.obb_theta_damping = self._resolve_or_default(obb_theta_damping, 0.8, 0.0, 1.0)

        self.marginal_timeout = int(self._resolve_or_default(marginal_timeout, 0, 0, 500))
        self.central_timeout = int(self._resolve_or_default(central_timeout, 0, 0, 1000))

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.horizontal_margin = horizontal_margin
        self.vertical_margin = vertical_margin

        self.l_margin = 0.0
        self.r_margin = 0.0
        self.t_margin = 0.0
        self.b_margin = 0.0
        self._margins_ready = False
        self._maybe_set_margins(frame_width, frame_height)

        self.id_counter = self.id_allocator.next_id
        self.active_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []

    def _update_impl(
        self,
        dets: np.ndarray,
        img: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray = None,
    ) -> np.ndarray:
        self.check_inputs(dets=dets, img=img, embs=embs)

        if not self._margins_ready and hasattr(self, "w") and hasattr(self, "h"):
            self._maybe_set_margins(self.w, self.h)

        self.frame_count += 1

        batch = self.make_detection_batch(dets)

        hth, nth, mth = self._dynamic_thresholds(batch.confs)
        high_batch, second_batch = batch.split_by_confidence(
            high_thresh=hth,
            low_thresh=self.low_th,
        )

        next_active_tracks: list[Track] = []

        self._purge_stale_lost_tracks()

        track_pool = self.active_tracks + self.lost_tracks

        unmatched_tracks = np.array([], dtype=int)
        if len(high_batch):
            definite_boxes = high_batch.boxes
            definite_scores = high_batch.confs
            definite_classes = high_batch.clss
            definite_det_inds = high_batch.det_inds

            if track_pool:
                cost = self.calculate_cost(track_pool, definite_boxes, is_obb=self.is_obb)
                matches, unmatched_tracks, unmatched_detections = linear_assignment(cost, mth)
                for track_idx, detection_idx in matches:
                    track = track_pool[track_idx]
                    track.update(
                        definite_boxes[detection_idx],
                        self.frame_count,
                        definite_scores[detection_idx],
                        definite_classes[detection_idx],
                        definite_det_inds[detection_idx],
                    )
                    next_active_tracks.append(track)
                    if track in self.lost_tracks:
                        self.lost_tracks.remove(track)

                for detection_idx in unmatched_detections:
                    if definite_scores[detection_idx] > nth:
                        next_active_tracks.append(
                            self._new_track(
                                box=definite_boxes[detection_idx],
                                frame_id=self.frame_count,
                                conf=definite_scores[detection_idx],
                                cls=definite_classes[detection_idx],
                                det_ind=definite_det_inds[detection_idx],
                            )
                        )
            else:
                for detection_idx, score in enumerate(definite_scores):
                    if score > nth:
                        next_active_tracks.append(
                            self._new_track(
                                box=definite_boxes[detection_idx],
                                frame_id=self.frame_count,
                                conf=definite_scores[detection_idx],
                                cls=definite_classes[detection_idx],
                                det_ind=definite_det_inds[detection_idx],
                            )
                        )

        unmatched_track_pool = [track_pool[idx] for idx in unmatched_tracks] if len(unmatched_tracks) else []
        next_lost_tracks = unmatched_track_pool.copy()

        if len(second_batch) and len(unmatched_tracks):
            possible_boxes = second_batch.boxes
            possible_scores = second_batch.confs
            possible_classes = second_batch.clss
            possible_det_inds = second_batch.det_inds

            cost = self.calculate_cost(
                unmatched_track_pool,
                possible_boxes,
                iou_only=True,
                is_obb=self.is_obb,
            )
            matches, _, unmatched_detections = linear_assignment(cost, self.match_th_second)

            for track_idx, detection_idx in matches:
                track = unmatched_track_pool[track_idx]
                track.update(
                    possible_boxes[detection_idx],
                    self.frame_count,
                    possible_scores[detection_idx],
                    possible_classes[detection_idx],
                    possible_det_inds[detection_idx],
                )
                next_active_tracks.append(track)
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                if track in next_lost_tracks:
                    next_lost_tracks.remove(track)

        if not (len(high_batch) or len(second_batch)):
            next_lost_tracks = track_pool.copy()

        self._update_lost_tracks(next_lost_tracks)
        self.active_tracks = next_active_tracks.copy()

        return self.format_outputs(next_active_tracks, dtype=np.float32)

    def _dynamic_thresholds(self, scores: np.ndarray) -> tuple[float, float, float]:
        hth = self.high_th
        nth = self.new_track_th
        mth = self.match_th_first
        if self.dynamic_tuning:
            count = len(scores[scores > self.cth])
            if count < 1:
                count = 1
            lnc = np.log10(count)
            hth = self.clamp(hth - (self.high_th_m * lnc), 0.0, 1.0)
            nth = self.clamp(nth + (self.new_track_th_m * lnc), hth, 1.0)
            mth = self.clamp(mth - (self.match_th_first_m * lnc), 0.0, 0.67)
        return hth, nth, mth

    def _purge_stale_lost_tracks(self) -> None:
        for track in self.lost_tracks.copy():
            if track.lost_region == "central":
                if self.frame_count - track.last_frame > self.central_timeout:
                    self.lost_tracks.remove(track)
            else:
                if self.frame_count - track.last_frame > self.marginal_timeout:
                    self.lost_tracks.remove(track)

    def _update_lost_tracks(self, next_lost_tracks: Iterable[Track]) -> None:
        for track in next_lost_tracks:
            track.time_since_update = max(0, self.frame_count - track.last_frame)
            if track not in self.lost_tracks:
                self.lost_tracks.append(track)
                if track.bbox.shape[0] == 5:
                    u, v = float(track.bbox[0]), float(track.bbox[1])
                else:
                    u = track.bbox[0] + (track.bbox[2] - track.bbox[0]) / 2.0
                    v = track.bbox[1] + (track.bbox[3] - track.bbox[1]) / 2.0
                region = (
                    "central"
                    if (self.l_margin < u < self.r_margin) and (self.t_margin < v < self.b_margin)
                    else "marginal"
                )
                track.mark_lost(region)
            else:
                track._sync_meta(CommonTrackState.LOST)

    def _maybe_set_margins(self, frame_width: int | None, frame_height: int | None) -> None:
        if frame_width is None or frame_height is None:
            return

        self.l_margin = 0.0
        self.r_margin = float(frame_width)
        if self.horizontal_margin is not None:
            self.l_margin = float(self.clamp(self.horizontal_margin, 0, frame_width))
            self.r_margin = float(self.clamp(frame_width - self.horizontal_margin, 0, frame_width))

        self.t_margin = 0.0
        self.b_margin = float(frame_height)
        if self.vertical_margin is not None:
            self.t_margin = float(self.clamp(self.vertical_margin, 0, frame_height))
            self.b_margin = float(self.clamp(frame_height - self.vertical_margin, 0, frame_height))

        self._margins_ready = True

    def _new_track(self, box: np.ndarray, frame_id: int, conf: float, cls: float, det_ind: int) -> Track:
        track_id = self.id_allocator.alloc()
        self.id_counter = self.id_allocator.next_id
        track = Track(
            bbox=box,
            last_frame=frame_id,
            track_id=track_id,
            conf=float(conf),
            cls=int(cls),
            det_ind=int(det_ind),
            theta_damping=self.obb_theta_damping,
        )
        return track

    def reset(self) -> None:
        self._reset_common_state()
        self.id_counter = self.id_allocator.next_id
        self._margins_ready = False

    @staticmethod
    def _format_track(track: Track) -> list[float]:
        bbox = [float(v) for v in track.bbox.tolist()]
        return bbox + [
            float(track.track_id),
            float(track.conf),
            float(track.cls),
            float(track.det_ind),
        ]

    @staticmethod
    def clamp(value: float, min_value: float, max_value: float) -> float:
        return max(min_value, min(value, max_value))

    @staticmethod
    def _resolve_or_default(value: float | None, default: float, min_value: float, max_value: float) -> float:
        resolved = default if value is None else value
        return SFSORT.clamp(resolved, min_value, max_value)

    @staticmethod
    def _obb_to_xyxy(box: np.ndarray) -> np.ndarray:
        box = np.asarray(box, dtype=np.float32).reshape(-1)
        cx, cy, w, h, angle = box[:5]
        rect = ((float(cx), float(cy)), (max(float(w), 1e-4), max(float(h), 1e-4)), float(np.degrees(angle)))
        corners = cv2.boxPoints(rect)
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def calculate_cost(
        tracks: list[Track],
        boxes: np.ndarray,
        iou_only: bool = False,
        is_obb: bool = False,
    ) -> np.ndarray:
        """Calculates the association cost based on IoU and box similarity."""
        active_boxes = [track.bbox for track in tracks]
        if len(active_boxes) == 0 or boxes.size == 0:
            return np.empty((len(active_boxes), len(boxes)))

        active_boxes = np.asarray(active_boxes, dtype=np.float32)
        boxes = np.asarray(boxes, dtype=np.float32)

        if is_obb:
            return SFSORT._calculate_cost_obb(active_boxes, boxes, iou_only=iou_only)
        return SFSORT._calculate_cost_aabb(active_boxes, boxes, iou_only=iou_only)

    @staticmethod
    def _calculate_cost_obb(
        active_boxes: np.ndarray,
        boxes: np.ndarray,
        iou_only: bool = False,
    ) -> np.ndarray:
        eps = 1e-7
        iou = AssociationFunction.iou_batch_obb(active_boxes, boxes)
        if iou_only:
            return 1.0 - iou

        centerx1 = active_boxes[:, 0]
        centery1 = active_boxes[:, 1]
        centerx2 = boxes[:, 0]
        centery2 = boxes[:, 1]
        active_xyxy = np.vstack([SFSORT._obb_to_xyxy(box) for box in active_boxes])
        boxes_xyxy = np.vstack([SFSORT._obb_to_xyxy(box) for box in boxes])
        box1_width = active_boxes[:, 2]
        box2_width = boxes[:, 2]
        box1_height = active_boxes[:, 3]
        box2_height = boxes[:, 3]
        sw = np.minimum(box1_width[:, None], box2_width) / (np.maximum(box1_width[:, None], box2_width) + eps)
        sh = np.minimum(box1_height[:, None], box2_height) / (np.maximum(box1_height[:, None], box2_height) + eps)

        return SFSORT._combine_cost_terms(
            iou=iou,
            centerx1=centerx1,
            centery1=centery1,
            centerx2=centerx2,
            centery2=centery2,
            active_xyxy=active_xyxy,
            boxes_xyxy=boxes_xyxy,
            sw=sw,
            sh=sh,
        )

    @staticmethod
    def _calculate_cost_aabb(
        active_boxes: np.ndarray,
        boxes: np.ndarray,
        iou_only: bool = False,
    ) -> np.ndarray:
        eps = 1e-7
        b1_x1, b1_y1, b1_x2, b1_y2 = active_boxes.T
        b2_x1, b2_y1, b2_x2, b2_y2 = boxes.T

        h_intersection = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0)
        w_intersection = (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

        intersection = h_intersection * w_intersection

        box1_height = b1_x2 - b1_x1
        box2_height = b2_x2 - b2_x1
        box1_width = b1_y2 - b1_y1
        box2_width = b2_y2 - b2_y1

        box1_area = box1_height * box1_width
        box2_area = box2_height * box2_width
        union = box2_area + box1_area[:, None] - intersection + eps
        iou = intersection / union

        if iou_only:
            return 1.0 - iou

        centerx1 = (b1_x1 + b1_x2) / 2.0
        centery1 = (b1_y1 + b1_y2) / 2.0
        centerx2 = (b2_x1 + b2_x2) / 2.0
        centery2 = (b2_y1 + b2_y2) / 2.0
        delta_w = np.abs(box2_width - box1_width[:, None])
        sw = w_intersection / np.abs(w_intersection + delta_w + eps)
        delta_h = np.abs(box2_height - box1_height[:, None])
        sh = h_intersection / np.abs(h_intersection + delta_h + eps)

        return SFSORT._combine_cost_terms(
            iou=iou,
            centerx1=centerx1,
            centery1=centery1,
            centerx2=centerx2,
            centery2=centery2,
            active_xyxy=active_boxes,
            boxes_xyxy=boxes,
            sw=sw,
            sh=sh,
        )

    @staticmethod
    def _combine_cost_terms(
        iou: np.ndarray,
        centerx1: np.ndarray,
        centery1: np.ndarray,
        centerx2: np.ndarray,
        centery2: np.ndarray,
        active_xyxy: np.ndarray,
        boxes_xyxy: np.ndarray,
        sw: np.ndarray,
        sh: np.ndarray,
    ) -> np.ndarray:
        eps = 1e-7
        inner_diag = np.abs(centerx1[:, None] - centerx2) + np.abs(centery1[:, None] - centery2)

        xxc1 = np.minimum(active_xyxy[:, 0][:, None], boxes_xyxy[:, 0])
        yyc1 = np.minimum(active_xyxy[:, 1][:, None], boxes_xyxy[:, 1])
        xxc2 = np.maximum(active_xyxy[:, 2][:, None], boxes_xyxy[:, 2])
        yyc2 = np.maximum(active_xyxy[:, 3][:, None], boxes_xyxy[:, 3])
        outer_diag = np.abs(xxc2 - xxc1) + np.abs(yyc2 - yyc1)
        outer_diag = np.maximum(outer_diag, eps)

        diou = iou - (inner_diag / outer_diag)
        bbsi = diou + sh + sw
        return 1.0 - (bbsi / 3.0)
