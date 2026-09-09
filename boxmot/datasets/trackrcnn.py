"""Lazy KITTI TrackR-CNN detections and masks with an authoritative image timeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
import torch
from PIL import Image

from boxmot.datasets.kitti_mots import kitti_mots_frame_paths
from boxmot.structures import Boxes, Detections, MaskBatch


@dataclass(frozen=True, slots=True)
class TrackRcnnFrame:
    """One image frame's path, dimensions and canonical 2D detector observations."""

    frame_index: int
    image_size: tuple[int, int]
    image_path: Path
    detections: Detections


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


class TrackRcnnSequence(Sequence[TrackRcnnFrame]):
    """Read 2D predictions using only TrackR-CNN text files and KITTI images.

    ``detections_root`` contains ``{sequence_id}.txt`` and ``image_root``
    contains the matching sequence's PNG directory, usually under
    ``training/image_02``. No camera calibration, ego poses or 3D detections
    are needed. Images define every time step, including frames with no
    predictions; their numeric frame indices must be contiguous and zero-based.

    Construction indexes image headers and compressed masks without decoding
    RGB pixels. Indexing decodes only that frame's masks. Boxes and masks
    remain aligned, including zero-area masks; filtering is the caller's
    responsibility. Classes retain KITTI MOTS IDs 1 (car) and 2 (pedestrian).
    The unused 128-dimensional TrackR-CNN embeddings are discarded.
    """

    def __init__(self, detections_root: str | Path, image_root: str | Path, sequence_id: str) -> None:
        if (
            not isinstance(sequence_id, str)
            or len(sequence_id) != 4
            or not sequence_id.isascii()
            or not sequence_id.isdecimal()
        ):
            raise ValueError("KITTI sequence_id must be an exact four-digit sequence name, such as '0000'.")
        self.detections_root = Path(detections_root).expanduser().resolve()
        self.image_root = Path(image_root).expanduser().resolve()
        self.sequence_id = sequence_id
        self.frame_paths = kitti_mots_frame_paths(self.image_root / sequence_id)
        if tuple(int(path.stem) for path in self.frame_paths) != tuple(range(len(self.frame_paths))):
            raise ValueError(f"KITTI images must cover contiguous zero-based frames: {self.image_root / sequence_id}")
        image_size = None
        for path in self.frame_paths:
            with Image.open(path) as image:
                size = (image.height, image.width)
            if image_size is None:
                image_size = size
            elif size != image_size:
                raise ValueError(f"KITTI image dimensions {size} differ from {image_size}: {path}")
        assert image_size is not None
        self.image_size = image_size
        self._detections_path = self.detections_root / f"{sequence_id}.txt"
        self._image_detections = self._read_detections()

    def _read_detections(self) -> dict[int, tuple[_ImageDetection, ...]]:
        """Index 2D predictions by their native frame, keeping only compressed masks."""
        frames: dict[int, list[_ImageDetection]] = {}
        with self._detections_path.open(encoding="utf-8") as handle:
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
                        f"Invalid TrackR-CNN input at {self._detections_path}:{line_number}: {exc}"
                    ) from exc
        return {frame: tuple(rows) for frame, rows in frames.items()}

    def __len__(self) -> int:
        """Return the authoritative image frame count."""
        return len(self.frame_paths)

    @overload
    def __getitem__(self, index: int) -> TrackRcnnFrame: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[TrackRcnnFrame, ...]: ...

    def __getitem__(self, index: int | slice) -> TrackRcnnFrame | tuple[TrackRcnnFrame, ...]:
        """Decode one frame's masks without loading RGB pixels."""
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("TrackR-CNN sequence indices must be integers or slices.")
        image_path = self.frame_paths[index]
        frame_index = int(image_path.stem)
        rows = self._image_detections.get(frame_index, ())
        masks = np.empty((len(rows), *self.image_size), dtype=np.bool_)
        for mask, row in zip(masks, rows, strict=True):
            try:
                mask[:] = _decode_mask(row.counts, self.image_size)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid TrackR-CNN mask at {self._detections_path}:{row.line_number}: {exc}"
                ) from exc
        detections = Detections(
            geometry=Boxes(torch.tensor([row.box for row in rows], dtype=torch.float32).reshape(-1, 4)),
            scores=torch.tensor([row.score for row in rows], dtype=torch.float32),
            class_ids=torch.tensor([row.class_id for row in rows], dtype=torch.int64),
            sample_id=f"train:{self.sequence_id}:{frame_index}",
            masks=MaskBatch(torch.from_numpy(masks)),
        )
        return TrackRcnnFrame(
            frame_index=frame_index,
            image_size=self.image_size,
            image_path=image_path,
            detections=detections,
        )


__all__ = ("TrackRcnnFrame", "TrackRcnnSequence")
