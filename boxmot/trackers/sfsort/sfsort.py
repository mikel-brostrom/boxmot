# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license
# SFSORT implementation adapted for BoxMOT

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from boxmot.trackers.basetracker import BaseTracker
from boxmot.utils.iou import AssociationFunction
from boxmot.utils.matching import linear_assignment


class TrackState:
    """Enumeration of possible states of a track."""

    Active = 0
    Lost_Central = 1
    Lost_Marginal = 2


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
    _plot_angle: float | None = None

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        self.conf = float(self.conf)
        self.cls = int(self.cls)
        self.det_ind = int(self.det_ind)
        if self.bbox.shape[0] == 5:
            self.history_observations = deque([self._state_obb_for_plot()], maxlen=50)
        else:
            self.history_observations = deque([self.bbox.copy()], maxlen=50)
        self.time_since_update = 0

    @property
    def id(self) -> int:
        return self.track_id

    @staticmethod
    def _wrap_pi_periodic(delta: float) -> float:
        return float((delta + (np.pi / 2.0)) % np.pi - (np.pi / 2.0))

    def _state_obb_for_plot(self) -> np.ndarray:
        """Return current OBB state as corners with state-only angle smoothing."""
        box = self.bbox.copy()
        if box[3] > box[2]:
            box[2], box[3] = box[3], box[2]
            box[4] = box[4] + (np.pi / 2.0)
        target = float((box[4] + np.pi) % (2.0 * np.pi) - np.pi)
        if self._plot_angle is None:
            self._plot_angle = target
        else:
            self._plot_angle = self._plot_angle + self._wrap_pi_periodic(
                target - self._plot_angle
            )
        rect = (
            (float(box[0]), float(box[1])),
            (max(float(box[2]), 1e-4), max(float(box[3]), 1e-4)),
            float(np.degrees(self._plot_angle)),
        )
        corners = cv2.boxPoints(rect).reshape(-1)
        return np.asarray(corners, dtype=np.float32)

    def update(self, box: np.ndarray, frame_id: int, conf: float, cls: int, det_ind: int) -> None:
        """Update a matched track with latest detection."""

        self.bbox = np.asarray(box, dtype=np.float32)
        if self.bbox.shape[0] == 5:
            self.history_observations.append(self._state_obb_for_plot())
        else:
            self.history_observations.append(self.bbox.copy())
        self.state = TrackState.Active
        self.time_since_update = 0
        self.last_frame = frame_id
        self.conf = float(conf)
        self.cls = int(cls)
        self.det_ind = int(det_ind)


class SFSORT(BaseTracker):
    supports_obb = True

    """
    SFSORT tracker (v4.2) adapted for BoxMOT.

    Parameters:
    - det_thresh (float): Detection threshold for considering detections.
    - max_age (int): Maximum age (in frames) of a track before it is considered lost.
    - max_obs (int): Maximum number of historical observations stored for each track.
    - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
    - iou_threshold (float): IOU threshold for determining match between detection and tracks.
    - per_class (bool): Enables class-separated tracking.
    - nr_classes (int): Total number of object classes that the tracker will handle (for per_class=True).
    - asso_func (str): Algorithm name used for data association between detections and tracks.
    - is_obb (bool): Work with Oriented Bounding Boxes (OBB) instead of standard axis-aligned bounding boxes.

    SFSORT-specific parameters:
    - high_th (float): High confidence threshold for detections.
    - match_th_first (float): Match threshold for the first association step.
    - new_track_th (float): Confidence threshold for initializing new tracks.
    - low_th (float): Low confidence threshold for second association step.
    - match_th_second (float): Match threshold for the second association step.
    - dynamic_tuning (bool): Enable dynamic threshold tuning based on detection density.
    - cth (float): Confidence threshold for counting detections (dynamic tuning).
    - high_th_m (float): Dynamic adjustment scale for high_th.
    - new_track_th_m (float): Dynamic adjustment scale for new_track_th.
    - match_th_first_m (float): Dynamic adjustment scale for match_th_first.
    - marginal_timeout (int): Timeout for marginally lost tracks.
    - central_timeout (int): Timeout for centrally lost tracks.
    - frame_width (int | None): Optional frame width for margin computation.
    - frame_height (int | None): Optional frame height for margin computation.
    - horizontal_margin (int | None): Horizontal margin for central loss definition.
    - vertical_margin (int | None): Vertical margin for central loss definition.
    """

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
        marginal_timeout: int | None = 0,
        central_timeout: int | None = 0,
        frame_width: int | None = None,
        frame_height: int | None = None,
        horizontal_margin: int | None = None,
        vertical_margin: int | None = None,
        **kwargs,
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

        self.id_counter = 0
        self.active_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None = None) -> np.ndarray:
        self.check_inputs(dets=dets, img=img, embs=embs)

        if not self._margins_ready and hasattr(self, "w") and hasattr(self, "h"):
            self._maybe_set_margins(self.w, self.h)

        self.frame_count += 1

        boxes, scores, classes = self._split_detections(dets)
        det_inds = np.arange(len(dets)) if dets.size else np.empty((0,), dtype=int)

        hth, nth, mth = self._dynamic_thresholds(scores)

        next_active_tracks: list[Track] = []

        self._purge_stale_lost_tracks()

        track_pool = self.active_tracks + self.lost_tracks

        unmatched_tracks = np.array([], dtype=int)
        high_score = scores > hth
        if high_score.any():
            definite_boxes = boxes[high_score]
            definite_scores = scores[high_score]
            definite_classes = classes[high_score]
            definite_det_inds = det_inds[high_score]

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

        intermediate_score = np.logical_and(self.low_th < scores, scores < hth)
        if intermediate_score.any() and len(unmatched_tracks):
            possible_boxes = boxes[intermediate_score]
            possible_scores = scores[intermediate_score]
            possible_classes = classes[intermediate_score]
            possible_det_inds = det_inds[intermediate_score]

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

        if not (high_score.any() or intermediate_score.any()):
            next_lost_tracks = track_pool.copy()

        self._update_lost_tracks(next_lost_tracks)
        self.active_tracks = next_active_tracks.copy()

        outputs = [self._format_track(track) for track in next_active_tracks]
        return np.asarray(outputs, dtype=float) if outputs else self.empty_output(dtype=float)

    def _split_detections(self, dets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.is_obb:
            return self._split_obb_detections(dets)
        return self._split_aabb_detections(dets)

    def _split_aabb_detections(self, dets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if dets.size == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return dets[:, :4], dets[:, 4], dets[:, 5]

    def _split_obb_detections(self, dets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if dets.size == 0:
            return (
                np.empty((0, 5), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        return dets[:, :5], dets[:, 5], dets[:, 6]

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
            if track.state == TrackState.Lost_Central:
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
                if (self.l_margin < u < self.r_margin) and (self.t_margin < v < self.b_margin):
                    track.state = TrackState.Lost_Central
                else:
                    track.state = TrackState.Lost_Marginal

    def _maybe_set_margins(self, frame_width: int | None, frame_height: int | None) -> None:
        if frame_width is None or frame_height is None:
            return

        self.l_margin = 0.0
        self.r_margin = float(frame_width)
        if self.horizontal_margin is not None:
            self.l_margin = float(self.clamp(self.horizontal_margin, 0, frame_width))
            self.r_margin = float(
                self.clamp(frame_width - self.horizontal_margin, 0, frame_width)
            )

        self.t_margin = 0.0
        self.b_margin = float(frame_height)
        if self.vertical_margin is not None:
            self.t_margin = float(self.clamp(self.vertical_margin, 0, frame_height))
            self.b_margin = float(
                self.clamp(frame_height - self.vertical_margin, 0, frame_height)
            )

        self._margins_ready = True

    def _new_track(self, box: np.ndarray, frame_id: int, conf: float, cls: float, det_ind: int) -> Track:
        track = Track(
            bbox=box,
            last_frame=frame_id,
            track_id=self.id_counter,
            conf=float(conf),
            cls=int(cls),
            det_ind=int(det_ind),
        )
        self.id_counter += 1
        return track

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
    def _resolve_or_default(
        value: float | None, default: float, min_value: float, max_value: float
    ) -> float:
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
        sw = np.minimum(box1_width[:, None], box2_width) / (
            np.maximum(box1_width[:, None], box2_width) + eps
        )
        sh = np.minimum(box1_height[:, None], box2_height) / (
            np.maximum(box1_height[:, None], box2_height) + eps
        )

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

        h_intersection = (
            np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)
        ).clip(0)
        w_intersection = (
            np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
        ).clip(0)

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
