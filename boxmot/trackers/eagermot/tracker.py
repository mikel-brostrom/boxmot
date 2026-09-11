"""EagerMOT sensor fusion and two-stage association.

Adapted from Aleksandr Kim's EagerMOT implementation. The upstream MIT notice
is retained in this package's LICENSE.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from boxmot.structures import (
    Boxes3D,
    CameraModel,
    Detections,
    Detections3D,
    Frame,
    GeometryKind,
    MultimodalTracks,
    Tracks,
    Tracks3D,
)
from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.specs import TrackerCapabilities, TrackerFamily
from boxmot.trackers.common.tracking.per_class import ClassTrackState
from boxmot.trackers.eagermot.association import (
    greedy_association,
    similarity_3d,
)
from boxmot.trackers.eagermot.geometry import iou2d_matrix, project_box3d, transform_boxes3d
from boxmot.trackers.eagermot.motion import Kalman3D


@dataclass
class _Track:
    """One 3D motion model and its most recent supporting image observation."""

    id: int
    motion: Kalman3D
    cls: int
    confidence_3d: float
    history_observations: deque
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    time_since_2d_update: int = 10
    det_ind: int = -1
    det_ind_3d: int = -1
    bbox: np.ndarray | None = None
    conf: float = 0.0

    @property
    def xyxy(self) -> np.ndarray | None:
        """Expose the latest observed image box to the shared display API."""
        return self.bbox


class EagerMot(BaseTracker):
    """Fuse independent 2D/3D detections and preserve identities across dropouts.

    Call the inherited ``update`` with canonical ``Detections``, an independent
    ``detections_3d=Detections3D(...)`` batch, and ``camera=CameraModel(...)``.
    An empty 3D batch explicitly represents a depth-detector dropout. A 3D
    observation is needed to create a track; subsequent 2D observations can
    sustain it without updating its 3D Kalman measurement.

    The result contains current image observations in ``image_tracks`` and
    current 3D estimates in ``spatial_tracks``, sharing the same track IDs.
    Optional segmentation masks are preserved on image tracks. The two output
    collections have independent row counts and detection indices.

    Defaults reproduce the released KITTI car association/lifecycle profile.
    Classes are always associated separately, including when ``per_class`` is
    false. The scalar thresholds apply to every accepted detector class.
    """

    capabilities = TrackerCapabilities(
        family=TrackerFamily.MULTIMODAL,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        accepts_masks=True,
        accepts_frame=True,
        requires_detections_3d=True,
        accepts_detections_3d=True,
        requires_camera=True,
        accepts_camera=True,
    )
    _requires_detections_3d = True
    _requires_camera = True
    uses_frame_dimensions_for_association = False
    supports_kalman_noise = True

    def __init__(
        self,
        det_thresh: float = 0.0,
        det_thresh_3d: float = 0.0,
        max_age: int = 3,
        min_hits: int = 1,
        max_age_2d: int = 3,
        fusion_iou_threshold: float = 0.01,
        iou_threshold: float = 0.3,
        first_matching_method: str = "dist_2d_full",
        distance_threshold: float = 3.5,
        iou_3d_threshold: float = 0.01,
        is_angular: bool = False,
        per_class: bool = False,
        asso_func: str = "iou",
        kf_process_position_scale: float = 1.0,
        kf_process_velocity_scale: float = 1.0,
        kf_measurement_noise_scale: float = 1.0,
        kf_initial_position_scale: float = 1.0,
        kf_initial_velocity_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Configure fusion, 3D matching, and the image-only recovery stage.

        ``distance_threshold`` is a positive maximum distance for the source
        ``dist_2d``, ``dist_2d_dims``, and ``dist_2d_full`` methods. The last
        method includes center, dimension, and yaw disagreement.
        ``iou_3d_threshold`` applies only to ``first_matching_method='iou_3d'``.
        ``max_age_2d`` controls confidence decay after missing image support;
        ``max_age`` controls expiry after missing both sensor modalities.
        ``iou_threshold=1`` disables the second association stage, as upstream.
        The five ``kf_*_scale`` settings multiply the 3D filter's covariance
        priors. Position includes all seven box coordinates; velocity includes
        xyz derivatives and, with ``is_angular``, yaw velocity. Prediction
        remains one frame per update, independent of the supplied ego poses.
        """
        for name, value in (
            ("det_thresh", det_thresh),
            ("det_thresh_3d", det_thresh_3d),
            ("fusion_iou_threshold", fusion_iou_threshold),
            ("iou_threshold", iou_threshold),
            ("iou_3d_threshold", iou_3d_threshold),
        ):
            if isinstance(value, bool) or not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1].")
        for name, value in (("max_age", max_age), ("min_hits", min_hits), ("max_age_2d", max_age_2d)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1.")
        if isinstance(distance_threshold, bool) or not np.isfinite(distance_threshold) or distance_threshold <= 0:
            raise ValueError("distance_threshold must be finite and positive.")
        if first_matching_method not in {"dist_2d", "dist_2d_dims", "dist_2d_full", "iou_3d"}:
            raise ValueError("first_matching_method must be 'dist_2d', 'dist_2d_dims', 'dist_2d_full', or 'iou_3d'.")
        if not isinstance(is_angular, bool):
            raise TypeError("is_angular must be bool.")
        if asso_func != "iou":
            raise ValueError("EagerMot supports only asso_func='iou' for sensor fusion and image association.")
        super().__init__(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            per_class=per_class,
            asso_func=asso_func,
            kf_process_position_scale=kf_process_position_scale,
            kf_process_velocity_scale=kf_process_velocity_scale,
            kf_measurement_noise_scale=kf_measurement_noise_scale,
            kf_initial_position_scale=kf_initial_position_scale,
            kf_initial_velocity_scale=kf_initial_velocity_scale,
            **kwargs,
        )
        self.det_thresh_3d = det_thresh_3d
        self.max_age_2d = max_age_2d
        self.fusion_iou_threshold = fusion_iou_threshold
        self.first_matching_method = first_matching_method
        self.distance_threshold = distance_threshold
        self.iou_3d_threshold = iou_3d_threshold
        self.is_angular = is_angular
        self._tracks: list[_Track] = []
        self._uses_world_frame: bool | None = None

    def reset(self) -> None:
        """Clear identities, motion state, and pose mode for a new sequence."""
        self._reset_common_state()
        self._tracks = []
        self._uses_world_frame = None

    def get_active_tracks_for_display(self) -> list[_Track]:
        """Return observed image tracks; class gating is internal to fusion."""
        return list(self.active_tracks)

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray | None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reject calls that bypass the canonical multimodal input branch."""
        raise ValueError("EagerMot requires canonical detections, detections_3d, and camera on every update.")

    @staticmethod
    def _project(boxes: np.ndarray, camera: CameraModel) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-coordinate boxes while keeping invalid rows in place."""
        projections = np.zeros((len(boxes), 4), dtype=np.float64)
        valid = np.zeros(len(boxes), dtype=bool)
        projection = camera.projection.detach().numpy()
        for index, box in enumerate(boxes):
            projected = project_box3d(box, projection, camera.image_size)
            if projected is not None:
                projections[index] = projected
                valid[index] = True
        return projections, valid

    @staticmethod
    def _observe_image(track: _Track, index: int, detections: Detections) -> None:
        """Attach one current image observation without changing 3D motion."""
        track.det_ind = index
        track.bbox = detections.geometry.values[index].detach().numpy().copy()
        track.conf = float(detections.scores[index])
        track.time_since_2d_update = 0
        track.history_observations.append(track.bbox.copy())

    def _confirmed(self, track: _Track) -> bool:
        """Apply the source hit-count criterion and initial sequence warmup."""
        if self.frame_count < self.min_hits:
            return track.hits >= track.age
        return track.hits >= self.min_hits

    def _spatial_confidence(self, track: _Track) -> float:
        """Apply the source confidence decay when image evidence goes stale."""
        if track.time_since_2d_update < self.max_age_2d:
            return track.confidence_3d
        return track.confidence_3d / (2.0 * (track.time_since_2d_update + 1 - self.max_age_2d))

    def _results(self, detections: Detections, camera: CameraModel) -> MultimodalTracks:
        """Build independent canonical sensor outputs without fabricated masks."""
        current = sorted(
            (track for track in self._tracks if track.time_since_update == 0 and self._confirmed(track)),
            key=lambda track: track.id,
        )
        self.active_tracks = [track for track in current if track.det_ind >= 0]
        indices = torch.tensor([track.det_ind for track in self.active_tracks], dtype=torch.int64)
        selected = detections.select(indices)
        image_tracks = Tracks(
            geometry=selected.geometry,
            track_ids=torch.tensor([track.id for track in self.active_tracks], dtype=torch.int64),
            scores=selected.scores,
            class_ids=selected.class_ids,
            detection_indices=indices,
            sample_id=detections.sample_id,
            masks=selected.masks,
        )
        boxes = np.asarray([track.motion.box for track in current], dtype=np.float64).reshape(-1, 7)
        if camera.camera_to_world is not None:
            boxes = transform_boxes3d(boxes, camera.camera_to_world.detach().numpy(), inverse=True)
        spatial_tracks = Tracks3D(
            geometry=Boxes3D(torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32))),
            track_ids=torch.tensor([track.id for track in current], dtype=torch.int64),
            scores=torch.tensor([self._spatial_confidence(track) for track in current], dtype=torch.float32),
            class_ids=torch.tensor([track.cls for track in current], dtype=torch.int64),
            detection_indices=torch.tensor([track.det_ind_3d for track in current], dtype=torch.int64),
            sample_id=detections.sample_id,
        )
        # Populate the shared class-query views without dispatching fusion by class.
        if self.per_class:
            grouped_tracks: dict[int, list[_Track]] = {}
            for track in self._tracks:
                grouped_tracks.setdefault(track.cls, []).append(track)
            visible_ids = {track.id for track in self.active_tracks}
            self.class_track_states = {
                self._encode_kernel_class_id(class_id): ClassTrackState(
                    attrs={
                        "_tracks": tracks,
                        "active_tracks": [track for track in tracks if track.id in visible_ids],
                    }
                )
                for class_id, tracks in grouped_tracks.items()
            }
        return MultimodalTracks(image_tracks=image_tracks, spatial_tracks=spatial_tracks)

    def _track_multimodal(
        self,
        detections: Detections,
        detections_3d: Detections3D,
        camera: CameraModel,
        frame: Frame | np.ndarray | None,
    ) -> MultimodalTracks:
        """Fuse observations, associate in 3D, then recover using remaining 2D."""
        uses_world_frame = camera.camera_to_world is not None
        if self._uses_world_frame is not None and uses_world_frame != self._uses_world_frame:
            raise ValueError("camera_to_world must be supplied consistently throughout a sequence; reset to change it.")
        pose = camera.camera_to_world.detach().numpy() if uses_world_frame else np.eye(4)

        indices_2d = np.flatnonzero(detections.scores.detach().numpy() >= self.det_thresh)
        indices_3d = np.flatnonzero(detections_3d.scores.detach().numpy() >= self.det_thresh_3d)
        boxes_2d = detections.geometry.values.detach().numpy()[indices_2d]
        camera_boxes_3d = detections_3d.geometry.values.detach().numpy()[indices_3d]
        classes_2d = detections.class_ids.detach().numpy()[indices_2d]
        classes_3d = detections_3d.class_ids.detach().numpy()[indices_3d]
        boxes_3d = transform_boxes3d(camera_boxes_3d, pose) if uses_world_frame else camera_boxes_3d.copy()

        # Match independent sensor detections first, retaining original indices.
        projected, projectable = self._project(camera_boxes_3d, camera)
        fusion, _, unmatched_2d = greedy_association(
            iou2d_matrix(projected, boxes_2d),
            self.fusion_iou_threshold,
            allowed=projectable[:, None] & (classes_3d[:, None] == classes_2d[None, :]),
        )
        fused_image_indices = {int(i3): int(indices_2d[i2]) for i3, i2 in fusion}

        self._uses_world_frame = uses_world_frame
        self.frame_count += 1
        for track in self._tracks:
            track.age += 1
            track.time_since_update += 1
            track.time_since_2d_update += 1
            track.det_ind = -1
            track.det_ind_3d = -1

        # Prediction must also advance on frames with no 3D detections.
        predictions = Kalman3D.multi_predict([track.motion for track in self._tracks])
        track_classes = np.asarray([track.cls for track in self._tracks], dtype=np.int64)
        threshold = self.iou_3d_threshold if self.first_matching_method == "iou_3d" else -self.distance_threshold
        first_matches, unmatched_3d, unmatched_tracks = greedy_association(
            similarity_3d(boxes_3d, predictions, self.first_matching_method),
            threshold,
            allowed=classes_3d[:, None] == track_classes[None, :],
        )
        Kalman3D.multi_update(
            [self._tracks[track_index].motion for track_index in first_matches[:, 1]],
            boxes_3d[first_matches[:, 0]],
        )
        for detection_index, track_index in first_matches:
            track = self._tracks[track_index]
            track.det_ind_3d = int(indices_3d[detection_index])
            track.confidence_3d = float(detections_3d.scores[track.det_ind_3d])
            track.time_since_update = 0
            track.hits += 1
            if detection_index in fused_image_indices:
                self._observe_image(track, fused_image_indices[detection_index], detections)

        # The released source uses only 2D-only observations in its second stage.
        # Failed fused 3D observations become births, not image-based recoveries.
        leftover_tracks = [self._tracks[index] for index in unmatched_tracks]
        predicted_camera_boxes = predictions[unmatched_tracks]
        if uses_world_frame:
            predicted_camera_boxes = transform_boxes3d(predicted_camera_boxes, pose, inverse=True)
        projected_tracks, visible_tracks = self._project(predicted_camera_boxes, camera)
        second_matches = np.empty((0, 2), dtype=np.int64)
        if self.iou_threshold < 1.0:
            second_matches, _, _ = greedy_association(
                iou2d_matrix(boxes_2d[unmatched_2d], projected_tracks),
                self.iou_threshold,
                allowed=visible_tracks[None, :]
                & (classes_2d[unmatched_2d, None] == track_classes[None, unmatched_tracks]),
            )
        for detection_index, track_index in second_matches:
            track = leftover_tracks[track_index]
            self._observe_image(track, int(indices_2d[unmatched_2d[detection_index]]), detections)
            track.time_since_update = 0
            track.hits += 1

        for detection_index in unmatched_3d:
            original_index = int(indices_3d[detection_index])
            confidence = float(detections_3d.scores[original_index])
            track = _Track(
                id=self.id_allocator.alloc(),
                motion=Kalman3D(
                    boxes_3d[detection_index], is_angular=self.is_angular, noise_config=self.kalman_noise_config
                ),
                cls=int(classes_3d[detection_index]),
                confidence_3d=confidence,
                conf=confidence,
                det_ind_3d=original_index,
                history_observations=deque(maxlen=self.max_obs),
            )
            if detection_index in fused_image_indices:
                self._observe_image(track, fused_image_indices[detection_index], detections)
            self._tracks.append(track)

        self._tracks = [track for track in self._tracks if track.time_since_update < self.max_age]
        return self._results(detections, camera)
