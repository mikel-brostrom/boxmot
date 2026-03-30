# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from pathlib import Path

import numpy as np
from torch import device

from boxmot.motion.cmc import get_cmc_method
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.strongsort.sort.detection import Detection
from boxmot.trackers.strongsort.sort.linear_assignment import \
    NearestNeighborDistanceMetric
from boxmot.trackers.strongsort.sort.tracker import Tracker
from boxmot.utils.ops import xyxy2tlwh


class StrongSort(BaseTracker):
    """
    Initialize the StrongSort tracker with various parameters.

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

    StrongSort-specific parameters:
    - min_conf (float): Minimum confidence threshold for detections.
    - max_cos_dist (float): Maximum cosine distance for ReID feature matching in Nearest Neighbor Distance Metric.
    - max_iou_dist (float): Maximum IoU distance for data association.
    - n_init (int): Number of consecutive frames required to confirm a track.
    - nn_budget (int): Maximum size of the feature library for Nearest Neighbor Distance Metric.
    - mc_lambda (float): Weight for motion consistency in the track state estimation.
    - ema_alpha (float): Alpha value for exponential moving average (EMA) update of appearance features.

    Attributes:
    - model: ReID model for appearance feature extraction.
    - tracker: StrongSort tracker instance.
    - cmc: Camera motion compensation object.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: device,
        half: bool,
        # StrongSort-specific parameters
        min_conf: float = 0.1,
        max_cos_dist: float = 0.2,
        max_iou_dist: float = 0.7,
        n_init: int = 3,
        nn_budget: int = 100,
        mc_lambda: float = 0.98,
        ema_alpha: float = 0.9,
        **kwargs,  # BaseTracker parameters
    ):
        init_args = {k: v for k, v in locals().items() if k not in ("self", "kwargs")}
        super().__init__(**init_args, _tracker_name="StrongSort", **kwargs)

        self.min_conf = min_conf
        self.max_cos_dist = max_cos_dist
        self.max_iou_dist = max_iou_dist
        self.n_init = n_init
        self.nn_budget = nn_budget
        self.mc_lambda = mc_lambda
        self.ema_alpha = ema_alpha

        self.model = ReidAutoBackend(
            weights=reid_weights,
            device=device,
            half=half,
        ).model

        self._next_track_id_value = 1
        self._trackers_by_active_tracks_id: dict[int, Tracker] = {}
        self.tracker = self._build_tracker(tracks=self.active_tracks)
        self._trackers_by_active_tracks_id[id(self.active_tracks)] = self.tracker
        self.cmc = self._build_cmc()
        self.active_tracks = self.tracker.tracks

    def _build_tracker(self, tracks: list | None = None) -> Tracker:
        return Tracker(
            metric=NearestNeighborDistanceMetric(
                "cosine", self.max_cos_dist, self.nn_budget
            ),
            max_iou_dist=self.max_iou_dist,
            max_age=self.max_age,
            max_obs=self.max_obs,
            n_init=self.n_init,
            mc_lambda=self.mc_lambda,
            ema_alpha=self.ema_alpha,
            tracks=tracks,
            next_track_id_fn=self._next_track_id,
        )

    @staticmethod
    def _build_cmc():
        return get_cmc_method("ecc")()

    def _next_track_id(self) -> int:
        track_id = self._next_track_id_value
        self._next_track_id_value += 1
        return track_id

    def _tracker_for_active_tracks(self) -> Tracker:
        track_list_id = id(self.active_tracks)
        tracker = self._trackers_by_active_tracks_id.get(track_list_id)
        if tracker is None:
            tracker = self._build_tracker(tracks=self.active_tracks)
            self._trackers_by_active_tracks_id[track_list_id] = tracker
        self.tracker = tracker
        self.active_tracks = tracker.tracks
        return tracker

    def _detection_boxes(self, dets: np.ndarray) -> np.ndarray:
        return self.detection_layout.boxes(dets)

    def _filter_detections(
        self, dets: np.ndarray, embs: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        dets = dets.astype(np.float32, copy=False)
        dets = self.detection_layout.with_detection_indices(dets)
        keep_mask = self.detection_layout.confidences(dets) >= self.min_conf
        filtered_dets = dets[keep_mask]
        filtered_embs = (
            embs.astype(np.float32, copy=False)[keep_mask] if embs is not None else None
        )
        return filtered_dets, filtered_embs

    def _appearance_features(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None
    ) -> np.ndarray:
        if embs is not None:
            embs = np.asarray(embs, dtype=np.float32)
            if embs.size:
                self.last_emb_size = embs.shape[1]
            return embs

        if len(dets) == 0:
            emb_size = 0 if self.last_emb_size is None else self.last_emb_size
            return np.empty((0, emb_size), dtype=np.float32)

        features = np.asarray(
            self.model.get_features(self._detection_boxes(dets), img),
            dtype=np.float32,
        )
        if features.size:
            self.last_emb_size = features.shape[1]
        return features

    def _create_detections(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray | None
    ) -> list[Detection]:
        if len(dets) == 0:
            return []

        boxes = self._detection_boxes(dets)
        features = self._appearance_features(dets, img, embs)
        tlwh = xyxy2tlwh(boxes)
        confs = self.detection_layout.confidences(dets)
        classes = self.detection_layout.classes(dets)
        det_indices = dets[:, -1]

        return [
            Detection(tlwh_box, conf, cls, det_ind, feat)
            for tlwh_box, conf, cls, det_ind, feat in zip(
                tlwh, confs, classes, det_indices, features
            )
        ]

    def _apply_camera_motion_compensation(
        self, dets: np.ndarray, img: np.ndarray
    ) -> None:
        if not self.tracker.tracks:
            return

        warp_matrix = np.asarray(
            self.cmc.apply(img, self._detection_boxes(dets)),
            dtype=np.float32,
        )
        if warp_matrix.shape == (2, 3):
            warp_matrix = np.vstack(
                (warp_matrix, np.array([0.0, 0.0, 1.0], dtype=np.float32))
            )

        track_boxes = np.asarray(
            [track.xyxy for track in self.tracker.tracks],
            dtype=np.float32,
        )
        if track_boxes.size == 0:
            return

        ones = np.ones((len(track_boxes), 1), dtype=np.float32)
        top_left = np.hstack((track_boxes[:, :2], ones))
        bottom_right = np.hstack((track_boxes[:, 2:], ones))
        warped_top_left = top_left @ warp_matrix.T
        warped_bottom_right = bottom_right @ warp_matrix.T

        widths_heights = warped_bottom_right[:, :2] - warped_top_left[:, :2]
        heights = np.maximum(widths_heights[:, 1], 1e-6)
        centers = warped_top_left[:, :2] + widths_heights / 2.0
        updated_means = np.column_stack(
            (
                centers[:, 0],
                centers[:, 1],
                widths_heights[:, 0] / heights,
                heights,
            )
        )

        for track, mean in zip(self.tracker.tracks, updated_means):
            track.mean[:4] = mean

    def _confirmed_tracks(self) -> list:
        return [
            track
            for track in self.tracker.tracks
            if track.is_confirmed() and track.time_since_update < 1
        ]

    def _prepare_output(self) -> np.ndarray:
        self.active_tracks = self.tracker.tracks
        tracks = self._confirmed_tracks()
        if not tracks:
            return self.empty_output(dtype=np.float32)

        boxes = np.asarray([track.to_tlbr() for track in tracks], dtype=np.float32)
        ids = np.asarray([track.id for track in tracks], dtype=np.float32)
        confs = np.asarray([track.conf for track in tracks], dtype=np.float32)
        classes = np.asarray([track.cls for track in tracks], dtype=np.float32)
        det_indices = np.asarray([track.det_ind for track in tracks], dtype=np.float32)

        return np.column_stack((boxes, ids, confs, classes, det_indices))

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        self.frame_count += 1
        self._tracker_for_active_tracks()

        dets, embs = self._filter_detections(dets, embs)
        self._apply_camera_motion_compensation(dets, img)
        detections = self._create_detections(dets, img, embs)

        self.tracker.predict()
        self.tracker.update(detections)

        return self._prepare_output()

    def reset(self):
        self.frame_count = 0
        self.last_emb_size = None
        self._first_frame_processed = False
        self._first_dets_processed = False
        self._plot_frame_idx = -1
        self._removed_first_seen.clear()
        self._removed_expired.clear()
        self._next_track_id_value = 1
        self._trackers_by_active_tracks_id = {}
        self.active_tracks = []
        if self.per_class_active_tracks is not None:
            self.per_class_active_tracks = {
                cls_id: [] for cls_id in range(self.nr_classes)
            }
        self.tracker = self._build_tracker(tracks=self.active_tracks)
        self._trackers_by_active_tracks_id[id(self.active_tracks)] = self.tracker
        self.cmc = self._build_cmc()
        self.active_tracks = self.tracker.tracks
