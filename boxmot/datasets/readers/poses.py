"""Absolute rigid ego poses in the canonical camera coordinate frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_camera_to_world_poses(path: Path, frame_count: int) -> np.ndarray:
    """Validate absolute camera-to-world poses without accumulating or inverting."""
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.shape != (frame_count, 4, 4):
        raise ValueError(f"Ego motion must have shape ({frame_count}, 4, 4): {path}")
    if (
        values.dtype.kind not in "fi"
        or not np.isfinite(values).all()
        or (np.abs(values) > np.finfo(np.float32).max).any()
    ):
        raise ValueError(f"Ego motion must contain finite float32 real numbers: {path}")
    values = np.asarray(values, dtype=np.float32)
    rotations = values[:, :3, :3].astype(np.float64)
    valid = np.isclose(values[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0).all(axis=1)
    valid &= np.isclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5, rtol=0).all(axis=(1, 2))
    valid &= np.isclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)
    valid &= np.isfinite(values).all(axis=(1, 2))
    if not valid.all():
        raise ValueError(f"Invalid rigid camera-to-world pose at frame {int(np.flatnonzero(~valid)[0])}: {path}")
    return np.ascontiguousarray(values)
