"""Official KITTI MOTS text records and canonical segmentation output.

MOTS uses zero-based frames and COCO compressed RLE in column-major order.
The optional codec is loaded only for segmentation evaluation and replay.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import numpy as np
import torch

from boxmot.pipelines import PipelineResult
from boxmot.structures import MaskBatch, Tracks


def _mask_api() -> ModuleType:
    """Load the optional native COCO codec with an actionable installation hint."""
    try:
        return importlib.import_module("pycocotools.mask")
    except ModuleNotFoundError as exc:
        # Box-only evaluation must remain usable without the optional MOTS extra.
        if exc.name not in {"pycocotools", "pycocotools.mask"}:
            raise
        raise ImportError(
            "MOTS segmentation evaluation requires pycocotools. Install it with `uv sync --extra cpu --extra mots`."
        ) from exc


def _nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"MOTS {name} must be a non-negative integer.")


def _validate_counts(counts: bytes, pixels: int) -> None:
    """Validate compressed COCO runs before passing untrusted data to C code."""
    if not isinstance(counts, bytes) or not counts:
        raise ValueError("MOTS RLE counts must be non-empty ASCII bytes.")
    runs: list[int] = []
    total = 0
    position = 0
    while position < len(counts):
        run = 0
        shift = 0
        while True:
            if position >= len(counts):
                raise ValueError("MOTS RLE contains a truncated run.")
            code = counts[position] - 48
            position += 1
            if code < 0 or code > 63 or shift >= 35:
                raise ValueError("MOTS RLE contains invalid compressed counts.")
            run |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    run |= -1 << shift
                break
        if len(runs) > 2:
            run += runs[-2]
        if run < 0 or total + run > pixels:
            raise ValueError("MOTS RLE run lengths exceed the mask dimensions or are negative.")
        runs.append(run)
        total += run
    if total != pixels:
        raise ValueError("MOTS RLE run lengths do not sum to the mask dimensions.")


@dataclass(frozen=True, slots=True)
class EncodedMask:
    """Immutable compressed COCO mask, with a fresh dictionary for codec calls."""

    height: int
    width: int
    counts: bytes

    def __post_init__(self) -> None:
        _nonnegative_integer(self.height, "mask height")
        _nonnegative_integer(self.width, "mask width")
        if not self.height or not self.width or self.height * self.width > 2**32 - 1:
            raise ValueError("MOTS mask dimensions must be positive and fit COCO's 32-bit run lengths.")
        _validate_counts(self.counts, self.height * self.width)

    @property
    def rle(self) -> dict[str, Any]:
        """Return the RLE representation expected by pycocotools."""
        return {"size": [self.height, self.width], "counts": self.counts}


@dataclass(frozen=True, slots=True)
class MOTSRow:
    """One official KITTI MOTS prediction, retaining exact integer identities."""

    frame_index: int
    track_id: int
    class_id: int
    encoded_mask: EncodedMask

    def __post_init__(self) -> None:
        _nonnegative_integer(self.frame_index, "frame index")
        _nonnegative_integer(self.track_id, "track ID")
        _nonnegative_integer(self.class_id, "class ID")
        if self.class_id not in {1, 2}:
            raise ValueError("MOTS prediction class IDs must be 1 (car) or 2 (pedestrian).")
        if not isinstance(self.encoded_mask, EncodedMask):
            raise TypeError("MOTS encoded_mask must be an EncodedMask.")

    @property
    def rle(self) -> dict[str, Any]:
        """Return this prediction's mask in the pycocotools representation."""
        return self.encoded_mask.rle


def encode_mots_mask(mask: np.ndarray | torch.Tensor) -> EncodedMask:
    """Encode a two-dimensional boolean mask in official compressed COCO RLE."""
    values = mask.numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
    if values.ndim != 2 or values.dtype != np.bool_:
        raise ValueError("MOTS masks must be two-dimensional boolean arrays.")
    if not all(values.shape):
        raise ValueError("MOTS mask dimensions must be positive.")
    rle = _mask_api().encode(np.asfortranarray(values, dtype=np.uint8))
    return EncodedMask(int(values.shape[0]), int(values.shape[1]), rle["counts"])


def write_mots_rows(handle: TextIO, rows: Iterable[MOTSRow]) -> None:
    """Write official space-separated MOTS rows without changing frame numbering."""
    for row in rows:
        mask = row.encoded_mask
        handle.write(
            f"{row.frame_index} {row.track_id} {row.class_id} "
            f"{mask.height} {mask.width} {mask.counts.decode('ascii')}\n"
        )


def read_mots_results(
    path: str | Path,
    *,
    frame_shapes: Mapping[int, tuple[int, int]] | None = None,
) -> dict[int, tuple[MOTSRow, ...]]:
    """Read and validate KITTI MOTS predictions, including cross-class overlap.

    If supplied, ``frame_shapes`` defines the valid zero-based frame indices
    and dimensions. Missing rows represent empty predictions for that frame.
    """
    api = _mask_api()
    frames: dict[int, list[MOTSRow]] = {}
    identities: dict[int, set[int]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError("MOTS rows must contain exactly six space-separated fields.")
                if any(not value.isascii() or not value.isdecimal() for value in fields[:5]):
                    raise ValueError("MOTS frame, ID, class, height and width must be non-negative integers.")
                frame_index, track_id, class_id, height, width = map(int, fields[:5])
                if frame_shapes is not None:
                    if frame_index not in frame_shapes:
                        raise ValueError(f"Unknown MOTS frame index {frame_index}.")
                    if (height, width) != frame_shapes[frame_index]:
                        raise ValueError(f"MOTS mask dimensions do not match frame {frame_index}.")
                encoded = EncodedMask(height, width, fields[5].encode("ascii"))
                row = MOTSRow(frame_index, track_id, class_id, encoded)
                frame_ids = identities.setdefault(frame_index, set())
                if track_id in frame_ids:
                    raise ValueError(f"Duplicate MOTS track ID {track_id} in frame {frame_index}.")
                if not api.area(row.rle):
                    raise ValueError("MOTS predictions must have non-empty masks.")
                frame_rows = frames.setdefault(frame_index, [])
                if frame_rows:
                    first_mask = frame_rows[0].encoded_mask
                    if (height, width) != (first_mask.height, first_mask.width):
                        raise ValueError(f"Inconsistent MOTS mask dimensions in frame {frame_index}.")
                    overlap = api.iou([row.rle], [other.rle for other in frame_rows], [0] * len(frame_rows))
                    if np.any(overlap > 0):
                        raise ValueError(f"Overlapping MOTS prediction masks in frame {frame_index}.")
                frame_ids.add(track_id)
                frame_rows.append(row)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid MOTS results at {path}:{line_number}: {exc}") from exc
    # TrackEval retains file order within each frame; Hungarian matching uses
    # that order to resolve exact ties, so sorting IDs can change the metrics.
    return {frame: tuple(rows) for frame, rows in sorted(frames.items())}


def prepare_mots_tracks(result: PipelineResult, image_size: tuple[int, int]) -> Tracks:
    """Attach real masks and resolve overlap by score, then the smaller track ID.

    Native track masks take precedence. Box trackers inherit their matched
    detection's mask; unmatched tracks and masks emptied by occlusion are
    omitted. Masks are never inferred from box geometry.
    """
    tracks = result.tracks
    if any(class_id not in {1, 2} for class_id in tracks.class_ids.tolist()):
        raise ValueError("MOTS prediction class IDs must be 1 (car) or 2 (pedestrian).")
    source_masks = tracks.masks if tracks.masks is not None else result.detections.masks
    if source_masks is not None and source_masks.image_size != image_size:
        raise ValueError(f"MOTS mask size {source_masks.image_size} does not match frame size {image_size}.")
    if source_masks is None and len(tracks):
        raise ValueError("MOTS replay requires published detection masks or native track masks.")
    track_ids = tracks.track_ids.tolist()
    scores = tracks.scores.tolist()
    detection_indices = tracks.detection_indices.tolist()
    occupied = torch.zeros(image_size, dtype=torch.bool)
    selected: dict[int, torch.Tensor] = {}
    for index in sorted(range(len(tracks)), key=lambda i: (-scores[i], track_ids[i])):
        source_index = index if tracks.masks is not None else detection_indices[index]
        if source_index < 0:
            continue
        if source_masks is None or source_index >= len(source_masks):
            raise ValueError("MOTS track detection index is outside the available mask batch.")
        mask = source_masks.values[source_index] & ~occupied
        if not mask.any():
            continue
        occupied |= mask
        selected[index] = mask
    indices = sorted(selected, key=lambda index: track_ids[index])
    values = (
        torch.stack([selected[index] for index in indices])
        if indices
        else torch.empty((0, *image_size), dtype=torch.bool)
    )
    return tracks.select(torch.tensor(indices, dtype=torch.int64)).with_masks(MaskBatch(values))


def tracks_to_mots_rows(tracks: Tracks, frame_index: int) -> list[MOTSRow]:
    """Serialize prepared, disjoint track masks to official MOTS rows."""
    _nonnegative_integer(frame_index, "frame index")
    if tracks.masks is None:
        raise ValueError("MOTS serialization requires track-aligned masks.")
    return [
        MOTSRow(frame_index, track_id, class_id, encode_mots_mask(mask))
        for track_id, class_id, mask in zip(tracks.track_ids.tolist(), tracks.class_ids.tolist(), tracks.masks.values)
    ]
