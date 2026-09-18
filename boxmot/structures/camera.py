"""Camera calibration and ego poses for canonical sensor inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ._validation import validate_finite, validate_tensor


@dataclass(frozen=True, slots=True, eq=False)
class CameraModel:
    """One pinhole camera's projection, image size, and optional rigid pose.

    ``projection`` is float32 ``[3,4]`` and maps homogeneous camera-space
    points to pixels after division by depth. ``image_size`` is (height,
    width). Optional float32 ``[4,4]`` ``camera_to_world`` maps these points
    to a fixed world frame using a rigid rotation and translation. Roll and
    pitch are accepted; a tracker with yaw-only boxes approximates their
    orientation while transforming box centers with the complete pose.
    None denotes a stationary camera with the identity transform.
    """

    projection: torch.Tensor
    image_size: tuple[int, int]
    camera_to_world: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject invalid intrinsics and nonrigid or reflected camera poses."""
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


__all__ = ("CameraModel",)
