"""Decode identity-bearing image-box annotations without requiring 3D geometry."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

_INTEGER = re.compile(r"[+-]?[0-9]+")
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_CLASS = re.compile(r"[A-Za-z_]{1,254}")


@dataclass(frozen=True, slots=True)
class TrackingLabels2D:
    """Validated native KITTI tracking rows, including ignored classes and regions."""

    source_rows: tuple[str, ...]
    source_sha256: str


def read_kitti_tracking_labels_2d(
    path: Path, *, frame_count: int, cache_inputs: bool = False, cache_root: Path | None = None
) -> TrackingLabels2D:
    """Read 17-field tracking GT with valid identities, metadata, and image boxes.

    All rows remain available to the evaluator, including ``DontCare`` and
    neighboring classes. Missing 3D dimensions and locations may use KITTI's
    finite placeholders; a 2D annotation does not imply valid spatial data.
    """
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("KITTI tracking ground truth requires a positive integer frame count.")
    if cache_inputs:
        import json

        import numpy as np

        from boxmot.datasets.annotation_cache import load_cached_annotation

        def encode(source: Path) -> np.ndarray:
            """Store validated text as UTF-8 bytes without object arrays or pickle."""
            labels = _read_kitti_tracking_labels_2d(source, frame_count=frame_count)
            payload = json.dumps(
                {"source_rows": labels.source_rows, "source_sha256": labels.source_sha256}, separators=(",", ":")
            ).encode("utf-8")
            return np.frombuffer(payload, dtype=np.uint8)

        values = load_cached_annotation(
            path,
            reader=encode,
            format=f"boxmot.kitti-tracking-labels-2d/v1:frames={frame_count}",
            cache_root=cache_root,
        )
        payload = json.loads(values.tobytes().decode("utf-8"))
        return TrackingLabels2D(tuple(payload["source_rows"]), payload["source_sha256"])
    return _read_kitti_tracking_labels_2d(path, frame_count=frame_count)


def _read_kitti_tracking_labels_2d(path: Path, *, frame_count: int) -> TrackingLabels2D:
    """Validate native rows once before exposing or caching their exact text."""
    path = Path(path)
    payload = path.read_bytes()
    rows: list[str] = []
    identities: set[tuple[int, int]] = set()
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            if len(fields) != 17:
                raise ValueError("Rows must have 17 KITTI tracking label fields, including frame and track identity")
            if not line.isascii() or _CLASS.fullmatch(fields[2]) is None:
                raise ValueError("tracking annotations must use native ASCII text and class labels")
            if any(_INTEGER.fullmatch(fields[index]) is None for index in (0, 1, 3, 4)):
                raise ValueError("frame, track identity, truncation, and occlusion must be integers")
            frame_index, track_id = int(fields[0]), int(fields[1])
            if not 0 <= frame_index < frame_count:
                raise ValueError(f"frame index must be between 0 and {frame_count - 1}")
            dontcare = fields[2].casefold() == "dontcare"
            if not (0 <= track_id <= (1 << 63) - 1 or (dontcare and track_id == -1)):
                raise ValueError("track identities must be nonnegative int64 integers; only DontCare permits -1")
            truncation, occlusion = int(fields[3]), int(fields[4])
            if truncation not in ((-1, 0, 1, 2) if dontcare else (0, 1, 2)):
                raise ValueError("tracking truncation must be an integer in [0, 2]; only DontCare permits -1")
            if occlusion not in ((-1, 0, 1, 2, 3) if dontcare else (0, 1, 2, 3)):
                raise ValueError("tracking occlusion must be an integer in [0, 3]; only DontCare permits -1")
            values = [float(value) for value in fields[3:]]
            if any(_NUMBER.fullmatch(value) is None for value in fields[3:]) or not all(
                math.isfinite(value) for value in values
            ):
                raise ValueError("tracking annotation numbers must be finite ASCII decimals")
            left, top, right, bottom = values[3:7]
            if right <= left or bottom <= top:
                raise ValueError("tracking image bounds must have positive width and height")
            identity = frame_index, track_id
            if not dontcare:
                if identity in identities:
                    raise ValueError(f"repeated track identity in frame {frame_index}")
                identities.add(identity)
            rows.append(line.strip())
        except ValueError as error:
            raise ValueError(f"Invalid 2D tracking ground truth at {path}:{line_number}: {error}") from error
    return TrackingLabels2D(source_rows=tuple(rows), source_sha256=hashlib.sha256(payload).hexdigest())
