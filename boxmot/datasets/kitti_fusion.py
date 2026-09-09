"""Lazy, aligned KITTI inputs for camera/LiDAR EagerMOT tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
import torch

from boxmot.datasets.trackrcnn import TrackRcnnSequence
from boxmot.structures import Boxes3D, CameraModel, Detections, Detections3D

_CAR_VARIANTS = {
    "t2-train": "results_tracking_car_auto_t2_train",
    "t3-trainval": "results_tracking_car_auto_t3_trainval",
}


@dataclass(frozen=True, slots=True)
class KittiFusionFrame:
    """Independent 2D/3D observations and camera metadata for one image frame."""

    frame_index: int
    image_size: tuple[int, int]
    detections: Detections
    detections_3d: Detections3D
    camera: CameraModel


def _projection(path: Path) -> torch.Tensor:
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


def _poses(path: Path, frame_count: int) -> np.ndarray:
    """Validate absolute camera-to-world poses without accumulating or inverting."""
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray) or values.shape != (frame_count, 4, 4):
        raise ValueError(f"KITTI ego motion must have shape ({frame_count}, 4, 4): {path}")
    if (
        values.dtype.kind not in "fi"
        or not np.isfinite(values).all()
        or (np.abs(values) > np.finfo(np.float32).max).any()
    ):
        raise ValueError(f"KITTI ego motion must contain finite float32 real numbers: {path}")
    values = np.asarray(values, dtype=np.float32)
    rotations = values[:, :3, :3].astype(np.float64)
    valid = np.isclose(values[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0).all(axis=1)
    valid &= np.isclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-5, rtol=0).all(axis=(1, 2))
    valid &= np.isclose(np.linalg.det(rotations), 1, atol=1e-5, rtol=0)
    valid &= np.isfinite(values).all(axis=(1, 2))
    if not valid.all():
        raise ValueError(f"Invalid rigid camera-to-world pose at frame {int(np.flatnonzero(~valid)[0])}: {path}")
    return np.ascontiguousarray(values)


class KittiFusionSequence(Sequence[KittiFusionFrame]):
    """Read a downloaded KITTI training sequence without decoding RGB images.

    ``data_root`` contains ``calib/training/calib``, ``ego_motion``,
    ``pointgnn/training`` and ``trackrcnn_detections``. ``image_root`` is the
    directory containing the image sequence folders, usually
    ``training/image_02``. Every image frame is retained, including frames
    without 2D predictions or with missing PointGNN files. Missing modality
    directories are errors. Image frames must be contiguous and zero-based.

    PointGNN car inputs are selected with ``car_variant='t3-trainval'`` or
    ``'t2-train'``. The latter download only contains the nine validation
    sequences. Car and pedestrian class IDs remain 1 and 2; cyclist rows
    are excluded because KITTI MOTS does not evaluate that class.

    PointGNN scores are nonnegative but can exceed one. They are mapped with
    ``s / (1 + s)`` to satisfy the canonical structure contract. This preserves
    ranking before float32 rounding and is not a calibrated probability.
    Keep the EagerMOT 3D score gate at zero for the source KITTI presets.
    Masks are decoded on indexing; unused TrackR-CNN embeddings are discarded.
    """

    def __init__(
        self,
        data_root: str | Path,
        image_root: str | Path,
        sequence_id: str,
        *,
        car_variant: str = "t3-trainval",
    ) -> None:
        if car_variant not in _CAR_VARIANTS:
            raise ValueError(f"KITTI car_variant must be one of {tuple(_CAR_VARIANTS)}.")
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.sequence_id = sequence_id
        self.car_variant = car_variant
        self._trackrcnn = TrackRcnnSequence(self.data_root / "trackrcnn_detections", self.image_root, sequence_id)
        self.frame_paths = self._trackrcnn.frame_paths
        self.image_size = self._trackrcnn.image_size
        self._projection = _projection(self.data_root / "calib/training/calib" / f"{sequence_id}.txt")
        self._poses = _poses(self.data_root / "ego_motion" / f"{sequence_id}.npy", len(self))
        CameraModel(self._projection, self.image_size, torch.from_numpy(self._poses[0]))
        pointgnn_root = self.data_root / "pointgnn/training"
        self._spatial_paths = tuple(
            pointgnn_root / folder / sequence_id / "data"
            for folder in (_CAR_VARIANTS[car_variant], "results_tracking_ped_cyl_auto_trainval")
        )
        self.missing_3d_frames: dict[str, tuple[int, ...]] = {}
        for name, directory in zip(("car", "pedestrian"), self._spatial_paths, strict=True):
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing KITTI PointGNN {name} sequence directory: {directory}")
            available: set[int] = set()
            for path in directory.glob("*.txt"):
                if path.name.startswith("._"):
                    continue
                if len(path.stem) != 6 or not path.stem.isascii() or not path.stem.isdecimal():
                    raise ValueError(f"PointGNN frame names must contain exactly six digits: {path}")
                frame_index = int(path.stem)
                if frame_index >= len(self):
                    raise ValueError(f"PointGNN frame {frame_index} has no corresponding image: {path}")
                available.add(frame_index)
            self.missing_3d_frames[name] = tuple(sorted(set(range(len(self))) - available))

    def _read_pointgnn(self, frame_index: int, sample_id: str) -> Detections3D:
        """Merge camera-space car/pedestrian boxes, retaining empty sensor frames."""
        boxes: list[list[float]] = []
        scores: list[float] = []
        classes: list[int] = []
        for directory in self._spatial_paths:
            path = directory / f"{frame_index:06d}.txt"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        fields = line.split()
                        if len(fields) != 16:
                            raise ValueError("PointGNN rows must have 16 KITTI detection fields")
                        if fields[0] not in {"Car", "Pedestrian", "Cyclist"}:
                            raise ValueError(f"unsupported PointGNN class {fields[0]!r}")
                        values = np.array(fields[1:], dtype=np.float64)
                        if not np.isfinite(values).all():
                            raise ValueError("all PointGNN numeric fields must be finite")
                        raw_score = float(values[-1])
                        if raw_score < 0:
                            raise ValueError("PointGNN score must be nonnegative")
                        geometry = values[7:14][[3, 4, 5, 6, 2, 1, 0]]
                        if (np.abs(geometry) > np.finfo(np.float32).max).any() or (geometry[4:] <= 0).any():
                            raise ValueError("3D box must have finite float32 coordinates and positive dimensions")
                        if (geometry[4:].astype(np.float32) <= 0).any():
                            raise ValueError("3D box dimensions must remain positive in float32")
                        if fields[0] == "Cyclist":
                            continue
                        boxes.append(geometry.tolist())
                        scores.append(raw_score / (1.0 + raw_score))
                        classes.append(1 if fields[0] == "Car" else 2)
                    except ValueError as exc:
                        raise ValueError(f"Invalid PointGNN input at {path}:{line_number}: {exc}") from exc
        return Detections3D(
            Boxes3D(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7)),
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(classes, dtype=torch.int64),
            sample_id,
        )

    def __len__(self) -> int:
        """Return the authoritative image frame count."""
        return len(self.frame_paths)

    @overload
    def __getitem__(self, index: int) -> KittiFusionFrame: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[KittiFusionFrame, ...]: ...

    def __getitem__(self, index: int | slice) -> KittiFusionFrame | tuple[KittiFusionFrame, ...]:
        """Decode one frame's masks and 3D inputs without loading RGB pixels."""
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("KITTI fusion indices must be integers or slices.")
        sample = self._trackrcnn[index]
        frame_index = sample.frame_index
        return KittiFusionFrame(
            frame_index=frame_index,
            image_size=self.image_size,
            detections=sample.detections,
            detections_3d=self._read_pointgnn(frame_index, sample.detections.sample_id),
            camera=CameraModel(self._projection, self.image_size, torch.from_numpy(self._poses[frame_index])),
        )


__all__ = ("KittiFusionFrame", "KittiFusionSequence")
