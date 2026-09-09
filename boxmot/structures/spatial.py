"""Canonical camera-space observations and independent 2D/3D tracking results."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._validation import normalize_row_indices, validate_finite, validate_nonempty_string, validate_tensor
from .tracks import Tracks


@dataclass(frozen=True, slots=True, eq=False)
class Boxes3D:
    """Float32 ``[N,7]`` boxes: ``x,y,z,yaw,length,width,height`` in meters/radians.

    Coordinates follow the rectified-camera convention: x right, y down, z
    forward. The position is the bottom-face center; yaw rotates about +y.
    Construction preserves the caller's CPU-contiguous tensor without copying.
    """

    values: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def validate(self) -> None:
        """Validate finite camera-space boxes with strictly positive dimensions."""
        validate_tensor(self.values, name="Boxes3D.values", dtype=torch.float32, ndim=2)
        if self.values.shape[1] != 7:
            raise ValueError(f"Boxes3D.values must have shape [N, 7], got {tuple(self.values.shape)}.")
        validate_finite(self.values, name="Boxes3D.values")
        if len(self) and bool((self.values[:, 4:] <= 0).any()):
            raise ValueError("Boxes3D.values length, width, and height must be positive.")

    def select(self, indices: torch.Tensor) -> Boxes3D:
        """Select or reorder boxes while retaining the spatial convention."""
        return Boxes3D(self.values.index_select(0, normalize_row_indices(indices, count=len(self))))


def _validate_scores_and_classes(scores: torch.Tensor, class_ids: torch.Tensor, *, owner: str) -> None:
    """Validate common spatial row metadata without implicit conversion."""
    validate_tensor(scores, name=f"{owner}.scores", dtype=torch.float32, ndim=1)
    validate_tensor(class_ids, name=f"{owner}.class_ids", dtype=torch.int64, ndim=1)
    validate_finite(scores, name=f"{owner}.scores")
    if scores.numel() and bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError(f"{owner}.scores must be in the inclusive range [0, 1].")
    if class_ids.numel() and bool((class_ids < 0).any()):
        raise ValueError(f"{owner}.class_ids must be non-negative.")


@dataclass(frozen=True, slots=True, eq=False)
class Detections3D:
    """Independent 3D detector rows; their count need not match 2D detections."""

    geometry: Boxes3D
    scores: torch.Tensor
    class_ids: torch.Tensor
    sample_id: str

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.geometry)

    def validate(self) -> None:
        """Validate aligned spatial geometry, confidence, classes, and sample ID."""
        if not isinstance(self.geometry, Boxes3D):
            raise TypeError("Detections3D.geometry must be Boxes3D.")
        self.geometry.validate()
        _validate_scores_and_classes(self.scores, self.class_ids, owner="Detections3D")
        validate_nonempty_string(self.sample_id, name="Detections3D.sample_id")
        if len(self.scores) != len(self) or len(self.class_ids) != len(self):
            raise ValueError("Detections3D geometry, scores, and class IDs must be aligned.")

    def select(self, indices: torch.Tensor) -> Detections3D:
        """Select or reorder only this independent detector's rows."""
        selected = normalize_row_indices(indices, count=len(self))
        return Detections3D(
            geometry=self.geometry.select(selected),
            scores=self.scores.index_select(0, selected),
            class_ids=self.class_ids.index_select(0, selected),
            sample_id=self.sample_id,
        )


@dataclass(frozen=True, slots=True, eq=False)
class CameraModel:
    """One pinhole camera's projection, image size, and optional upright pose.

    ``projection`` is float32 ``[3,4]`` and maps homogeneous camera-space
    points to pixels after division by depth. ``image_size`` is (height,
    width). Optional float32 ``[4,4]`` ``camera_to_world`` maps these points
    to a fixed world frame using translation and rotation about +y only.
    None denotes a stationary camera with the identity transform.
    """

    projection: torch.Tensor
    image_size: tuple[int, int]
    camera_to_world: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject invalid intrinsics and poses outside the upright box model."""
        validate_tensor(self.projection, name="CameraModel.projection", dtype=torch.float32, ndim=2)
        if self.projection.shape != (3, 4):
            raise ValueError("CameraModel.projection must have shape [3, 4].")
        validate_finite(self.projection, name="CameraModel.projection")
        if not isinstance(self.image_size, tuple) or len(self.image_size) != 2:
            raise TypeError("CameraModel.image_size must be a (height, width) tuple.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.image_size):
            raise ValueError("CameraModel.image_size must contain positive integers.")
        projection = self.projection.detach()
        if int(torch.linalg.matrix_rank(projection[:, :3].to(torch.float64))) != 3:
            raise ValueError("CameraModel.projection must have nonsingular camera intrinsics.")
        if self.camera_to_world is None:
            return
        validate_tensor(self.camera_to_world, name="CameraModel.camera_to_world", dtype=torch.float32, ndim=2)
        if self.camera_to_world.shape != (4, 4):
            raise ValueError("CameraModel.camera_to_world must have shape [4, 4].")
        validate_finite(self.camera_to_world, name="CameraModel.camera_to_world")
        pose = self.camera_to_world.detach()
        if not torch.allclose(pose[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-6, rtol=0.0):
            raise ValueError("CameraModel.camera_to_world must be a homogeneous rigid transform.")
        rotation = pose[:3, :3].to(torch.float64)
        if not torch.allclose(rotation.T @ rotation, torch.eye(3, dtype=torch.float64), atol=1e-5, rtol=0.0):
            raise ValueError("CameraModel.camera_to_world rotation must be orthonormal.")
        if not torch.isclose(torch.linalg.det(rotation), torch.tensor(1.0, dtype=torch.float64), atol=1e-5, rtol=0.0):
            raise ValueError("CameraModel.camera_to_world must preserve orientation, without reflection.")
        if not torch.allclose(rotation[:, 1], torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), atol=1e-6, rtol=0.0):
            raise ValueError(
                "CameraModel.camera_to_world must preserve the upright +y axis; roll and pitch are unsupported."
            )


@dataclass(frozen=True, slots=True, eq=False)
class Tracks3D:
    """Camera-space 3D track rows, including objects outside the image view.

    ``detection_indices`` refers to the independent current Detections3D batch,
    with -1 for tracks without a current 3D observation.
    """

    geometry: Boxes3D
    track_ids: torch.Tensor
    scores: torch.Tensor
    class_ids: torch.Tensor
    detection_indices: torch.Tensor
    sample_id: str

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(self.geometry)

    def validate(self) -> None:
        """Validate spatial tracks independently of their visible 2D projections."""
        if not isinstance(self.geometry, Boxes3D):
            raise TypeError("Tracks3D.geometry must be Boxes3D.")
        self.geometry.validate()
        _validate_scores_and_classes(self.scores, self.class_ids, owner="Tracks3D")
        validate_tensor(self.track_ids, name="Tracks3D.track_ids", dtype=torch.int64, ndim=1)
        validate_tensor(self.detection_indices, name="Tracks3D.detection_indices", dtype=torch.int64, ndim=1)
        validate_nonempty_string(self.sample_id, name="Tracks3D.sample_id")
        if any(
            len(values) != len(self) for values in (self.track_ids, self.scores, self.class_ids, self.detection_indices)
        ):
            raise ValueError("Tracks3D geometry and all metadata must be aligned.")
        if self.track_ids.numel() and bool((self.track_ids < 0).any()):
            raise ValueError("Tracks3D.track_ids must be non-negative.")
        if self.track_ids.unique().numel() != self.track_ids.numel():
            raise ValueError("Tracks3D.track_ids must be unique.")
        if self.detection_indices.numel() and bool((self.detection_indices < -1).any()):
            raise ValueError("Tracks3D.detection_indices may only use -1 for an unmatched track.")

    def select(self, indices: torch.Tensor) -> Tracks3D:
        """Select or reorder aligned 3D tracking rows."""
        selected = normalize_row_indices(indices, count=len(self))
        return Tracks3D(
            geometry=self.geometry.select(selected),
            track_ids=self.track_ids.index_select(0, selected),
            scores=self.scores.index_select(0, selected),
            class_ids=self.class_ids.index_select(0, selected),
            detection_indices=self.detection_indices.index_select(0, selected),
            sample_id=self.sample_id,
        )


@dataclass(frozen=True, slots=True, eq=False)
class MultimodalTracks:
    """Independent image and spatial outputs sharing one track-ID namespace.

    The collections can differ in length: an off-camera 3D object needs no
    artificial image box. Matching IDs identify the same object in both views.
    """

    image_tracks: Tracks
    spatial_tracks: Tracks3D

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return len(set(self.image_tracks.track_ids.tolist()) | set(self.spatial_tracks.track_ids.tolist()))

    @property
    def sample_id(self) -> str:
        """Return the common frame identity."""
        return self.spatial_tracks.sample_id

    def validate(self) -> None:
        """Validate both collections and their shared identity metadata."""
        if not isinstance(self.image_tracks, Tracks) or not isinstance(self.spatial_tracks, Tracks3D):
            raise TypeError("MultimodalTracks requires image Tracks and spatial Tracks3D.")
        self.image_tracks.validate()
        self.spatial_tracks.validate()
        if self.image_tracks.sample_id != self.spatial_tracks.sample_id:
            raise ValueError("MultimodalTracks collections must identify the same sample.")
        classes = dict(zip(self.image_tracks.track_ids.tolist(), self.image_tracks.class_ids.tolist()))
        for track_id, class_id in zip(self.spatial_tracks.track_ids.tolist(), self.spatial_tracks.class_ids.tolist()):
            if track_id in classes and classes[track_id] != class_id:
                raise ValueError("MultimodalTracks must assign the same class to a shared track ID.")


__all__ = ("Boxes3D", "CameraModel", "Detections3D", "MultimodalTracks", "Tracks3D")
