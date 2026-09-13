"""Camera calibration file readers independent of dataset layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from boxmot.structures import CameraModel


def read_kitti_projection(path: Path) -> torch.Tensor:
    """Read KITTI's rectified left-camera P2 projection, retaining all 12 values."""
    projection = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.partition(":")[0].strip() != "P2":
                continue
            try:
                if projection is not None:
                    raise ValueError("duplicate P2 calibration entry")
                values = [float(value) for value in line.partition(":")[2].split()]
                if (
                    len(values) != 12
                    or not np.isfinite(values).all()
                    or (np.abs(values) > np.finfo(np.float32).max).any()
                ):
                    raise ValueError("P2 must contain 12 finite float32 numbers")
                projection = torch.tensor(values, dtype=torch.float32).reshape(3, 4)
                CameraModel(projection, (1, 1))
            except ValueError as exc:
                raise ValueError(f"Invalid KITTI calibration at {path}:{line_number}: {exc}") from exc
    if projection is None:
        raise ValueError(f"KITTI calibration is missing P2: {path}")
    return projection
