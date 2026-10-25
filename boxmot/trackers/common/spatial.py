"""NumPy calibration and output conversion for calibrated sensor tracking."""

from __future__ import annotations

import numpy as np
import torch

from boxmot.structures import CameraModel, Frame, MultimodalTracks
from boxmot.trackers.common.input import frame_image_size


def _camera_tensor(value: np.ndarray, *, name: str, shape: tuple[int, int]) -> torch.Tensor:
    """Copy a real calibration matrix into finite, contiguous float32 storage."""
    if type(value) is not np.ndarray:
        raise TypeError(f"{name} must be a plain numpy.ndarray.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}.")
    if not (np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.floating)):
        raise TypeError(f"{name} must use a real numeric dtype.")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be representable as finite float32 values.")
    return torch.from_numpy(values)


def prepare_camera(
    camera: CameraModel | np.ndarray,
    *,
    frame: Frame | np.ndarray | None,
    image_size: tuple[int, int] | None,
    camera_to_world: np.ndarray | None,
) -> CameraModel:
    """Validate calibration or construct it from NumPy matrices and image size."""
    if isinstance(camera, CameraModel):
        if image_size is not None or camera_to_world is not None:
            raise ValueError("Pass image_size and camera_to_world separately only with a NumPy camera projection.")
        camera.validate()
        return camera
    projection = _camera_tensor(camera, name="camera projection", shape=(3, 4))
    if image_size is None:
        if frame is None:
            raise ValueError("A NumPy camera projection requires image_size or a frame.")
        image_size = frame_image_size(frame)
    pose = (
        None
        if camera_to_world is None
        else _camera_tensor(camera_to_world, name="camera_to_world", shape=(4, 4))
    )
    return CameraModel(projection=projection, image_size=image_size, camera_to_world=pose)


def pack_numpy_multimodal_tracks(tracks: MultimodalTracks) -> tuple[np.ndarray, np.ndarray]:
    """Pack validated image/spatial tracks into independent float64 N×8/N×11 rows."""
    outputs = []
    for collection in (tracks.image_tracks, tracks.spatial_tracks):
        for name in ("track_ids", "class_ids", "detection_indices"):
            values = getattr(collection, name).numpy(force=True)
            if np.any(values > 2**53):
                raise ValueError(f"Packed {name} must be within the exact float64 integer range.")
        geometry = collection.geometry.values.numpy(force=True)
        columns = geometry.shape[1]
        rows = np.empty((len(collection), columns + 4), dtype=np.float64)
        rows[:, :columns] = geometry
        rows[:, columns] = collection.track_ids.numpy(force=True)
        rows[:, columns + 1] = collection.scores.numpy(force=True)
        rows[:, columns + 2] = collection.class_ids.numpy(force=True)
        rows[:, columns + 3] = collection.detection_indices.numpy(force=True)
        outputs.append(rows)
    return outputs[0], outputs[1]
