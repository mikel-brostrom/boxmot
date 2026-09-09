"""Lazy, aligned KITTI inputs for camera/LiDAR EagerMOT tracking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
import torch
from PIL import Image

from boxmot.datasets.kitti_mots import kitti_mots_frame_paths
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Detections3D, MaskBatch

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


@dataclass(frozen=True, slots=True)
class _ImageDetection:
    """Keep compressed masks until their frame is requested."""

    box: tuple[float, float, float, float]
    score: float
    class_id: int
    counts: bytes
    line_number: int


def _decode_mask(counts: bytes, image_size: tuple[int, int]) -> np.ndarray:
    """Decode validated compressed COCO runs without calling native code."""
    runs: list[int] = []
    pixels = image_size[0] * image_size[1]
    position = total = 0
    if not counts:
        raise ValueError("RLE counts must be non-empty ASCII bytes.")
    while position < len(counts):
        run = shift = 0
        while True:
            if position >= len(counts):
                raise ValueError("RLE contains a truncated run.")
            code = counts[position] - 48
            position += 1
            if code < 0 or code > 63 or shift >= 35:
                raise ValueError("RLE contains invalid compressed counts.")
            run |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    run |= -1 << shift
                break
        if len(runs) > 2:
            run += runs[-2]
        if run < 0 or total + run > pixels:
            raise ValueError("RLE run lengths exceed the mask dimensions or are negative.")
        runs.append(run)
        total += run
    if total != pixels:
        raise ValueError("RLE run lengths do not sum to the mask dimensions.")
    values = np.repeat(np.arange(len(runs)) % 2 == 1, runs)
    return np.ascontiguousarray(values.reshape(image_size, order="F"))


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
        if (
            not isinstance(sequence_id, str)
            or len(sequence_id) != 4
            or not sequence_id.isascii()
            or not sequence_id.isdecimal()
        ):
            raise ValueError("KITTI sequence_id must be an exact four-digit sequence name, such as '0000'.")
        if car_variant not in _CAR_VARIANTS:
            raise ValueError(f"KITTI car_variant must be one of {tuple(_CAR_VARIANTS)}.")
        self.data_root = Path(data_root).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.sequence_id = sequence_id
        self.car_variant = car_variant
        self.frame_paths = kitti_mots_frame_paths(self.image_root / sequence_id)
        if tuple(int(path.stem) for path in self.frame_paths) != tuple(range(len(self.frame_paths))):
            raise ValueError(
                f"KITTI fusion images must cover contiguous zero-based frames: {self.image_root / sequence_id}"
            )
        self.image_size: tuple[int, int] | None = None
        for path in self.frame_paths:
            with Image.open(path) as image:
                size = (image.height, image.width)
            if self.image_size is None:
                self.image_size = size
            elif size != self.image_size:
                raise ValueError(f"KITTI image dimensions {size} differ from {self.image_size}: {path}")
        assert self.image_size is not None
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
        self._trackrcnn_path = self.data_root / "trackrcnn_detections" / f"{sequence_id}.txt"
        self._image_detections = self._read_trackrcnn()

    def _read_trackrcnn(self) -> dict[int, tuple[_ImageDetection, ...]]:
        """Index 2D predictions by their native frame, keeping only compressed masks."""
        frames: dict[int, list[_ImageDetection]] = {}
        with self._trackrcnn_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    fields = line.split()
                    if len(fields) != 138:
                        raise ValueError("TrackR-CNN rows must have 138 fields (10 detection/mask + 128 embedding)")
                    integers = [fields[index] for index in (0, 6, 7, 8)]
                    if any(not value.isascii() or not value.isdecimal() for value in integers):
                        raise ValueError("frame, class, mask height and width must be nonnegative integers")
                    frame_index, class_id, height, width = map(int, integers)
                    if frame_index >= len(self):
                        raise ValueError(f"frame {frame_index} has no corresponding image")
                    if class_id not in {1, 2}:
                        raise ValueError("TrackR-CNN class must be 1 (car) or 2 (pedestrian)")
                    if (height, width) != self.image_size:
                        raise ValueError(f"mask dimensions {(height, width)} differ from images {self.image_size}")
                    values = np.array(fields[1:6], dtype=np.float64)
                    if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
                        raise ValueError("box and score must be finite float32 values")
                    x1, y1, x2, y2, score = values.astype(np.float32).tolist()
                    if x2 <= x1 or y2 <= y1 or not 0 <= values[-1] <= 1:
                        raise ValueError("box must have positive area and score must be between zero and one")
                    detection = _ImageDetection(
                        (x1, y1, x2, y2), score, class_id, fields[9].encode("ascii"), line_number
                    )
                    frames.setdefault(frame_index, []).append(detection)
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(
                        f"Invalid TrackR-CNN input at {self._trackrcnn_path}:{line_number}: {exc}"
                    ) from exc
        return {frame: tuple(rows) for frame, rows in frames.items()}

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
        frame_index = int(self.frame_paths[index].stem)
        sample_id = f"train:{self.sequence_id}:{frame_index}"
        rows = self._image_detections.get(frame_index, ())
        masks = np.empty((len(rows), *self.image_size), dtype=np.bool_)
        for mask, row in zip(masks, rows, strict=True):
            try:
                mask[:] = _decode_mask(row.counts, self.image_size)
            except ValueError as exc:
                raise ValueError(f"Invalid TrackR-CNN mask at {self._trackrcnn_path}:{row.line_number}: {exc}") from exc
        detections = Detections(
            geometry=Boxes(torch.tensor([row.box for row in rows], dtype=torch.float32).reshape(-1, 4)),
            scores=torch.tensor([row.score for row in rows], dtype=torch.float32),
            class_ids=torch.tensor([row.class_id for row in rows], dtype=torch.int64),
            sample_id=sample_id,
            masks=MaskBatch(torch.from_numpy(masks)),
        )
        return KittiFusionFrame(
            frame_index=frame_index,
            image_size=self.image_size,
            detections=detections,
            detections_3d=self._read_pointgnn(frame_index, sample_id),
            camera=CameraModel(self._projection, self.image_size, torch.from_numpy(self._poses[frame_index])),
        )


__all__ = ("KittiFusionFrame", "KittiFusionSequence")
