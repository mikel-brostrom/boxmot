"""Decode camera-space 3D detections and identity-bearing KITTI annotations."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot.structures import Boxes3D, Detections3D


def _class_mapping(
    classes: Mapping[str, Mapping[str, Any]], options: Mapping[str, Any], encoding: str
) -> tuple[dict[str, int], frozenset[int], frozenset[str]]:
    """Resolve source class names consistently for predictions and annotations."""
    class_ids = {str(name).casefold(): int(value["id"]) for name, value in classes.items()}
    ignored_ids = frozenset(int(value["id"]) for value in classes.values() if value.get("evaluation") == "ignore")
    class_map = dict(class_ids)
    mappings = options.get("class_map", {})
    if not isinstance(mappings, Mapping):
        raise ValueError(f"{encoding} class_map must be a mapping.")
    for label, target in mappings.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"{encoding} class_map source labels must be non-empty strings.")
        if isinstance(target, str):
            if target.casefold() not in class_ids:
                raise ValueError(f"Unknown configured 3D class name {target!r}.")
            target = class_ids[target.casefold()]
        if type(target) is not int or target not in class_ids.values():
            raise ValueError(f"3D class_map target {target!r} is not a configured class ID.")
        class_map[label.casefold()] = target
    ignored = options.get("ignore_classes", ())
    if (
        isinstance(ignored, (str, bytes))
        or not isinstance(ignored, Sequence)
        or any(not isinstance(name, str) or not name for name in ignored)
    ):
        raise ValueError(f"{encoding} ignore_classes must be a list of source class names.")
    return class_map, ignored_ids, frozenset(name.casefold() for name in ignored)


@dataclass(frozen=True)
class TrackingLabels3D:
    """Annotated boxes with zero-based frame indices, true identities and provenance."""

    frame_indices: np.ndarray
    track_ids: np.ndarray
    class_ids: np.ndarray
    boxes: np.ndarray
    row_count: int
    source_sha256: str


def read_kitti_tracking_labels(
    path: Path,
    *,
    frame_count: int,
    classes: Mapping[str, Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
) -> TrackingLabels3D:
    """Read 17-field KITTI tracking GT, preserving identities and annotation gaps.

    Source boxes use bottom-center camera coordinates, +y yaw, and h/w/l
    dimensions. Only explicitly ignored classes may omit valid 3D geometry,
    as KITTI's ``DontCare`` rows do. Detection scores are not GT annotations.
    """
    options = dict(options or {})
    conventions = {"coordinate_frame": "camera", "box_origin": "bottom-center", "dimensions": "hwl", "yaw_axis": "y"}
    unknown = set(options) - {"class_map", "ignore_classes", *conventions}
    if unknown:
        raise ValueError(f"Unsupported kitti-tracking-labels options: {', '.join(sorted(unknown))}.")
    for name, expected in conventions.items():
        if options.get(name, expected) != expected:
            raise ValueError(f"kitti-tracking-labels {name} must be {expected!r}.")
    class_map, ignored_ids, ignored_labels = _class_mapping(classes, options, "kitti-tracking-labels")
    payload = Path(path).read_bytes()
    records: list[tuple[int, int, int]] = []
    boxes: list[np.ndarray] = []
    identities: set[tuple[int, int, int]] = set()
    row_count = 0
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row_count += 1
        try:
            fields = line.split()
            if len(fields) != 17:
                raise ValueError("Rows must have 17 KITTI tracking label fields (including frame and track identity)")
            frame_index, track_id = int(fields[0]), int(fields[1])
            if not 0 <= frame_index < frame_count:
                raise ValueError(f"frame index must be between 0 and {frame_count - 1}")
            label = fields[2].casefold()
            if label in ignored_labels or class_map.get(label) in ignored_ids:
                continue
            if label not in class_map:
                raise ValueError(f"unsupported 3D annotation class {fields[2]!r}")
            if not 0 <= track_id <= np.iinfo(np.int64).max:
                raise ValueError("target track identities must be nonnegative int64 integers")
            values = np.asarray(fields[3:], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("all retained 3D annotation numeric fields must be finite")
            geometry = values[7:14][[3, 4, 5, 6, 2, 1, 0]]
            if (np.abs(geometry) > np.finfo(np.float32).max).any() or (geometry[4:] <= 0).any():
                raise ValueError("3D box must have finite float32 coordinates and positive dimensions")
            if (geometry[4:].astype(np.float32) <= 0).any():
                raise ValueError("3D box dimensions must remain positive in float32")
            identity = frame_index, class_map[label], track_id
            if identity in identities:
                raise ValueError(f"repeated class/identity in frame {frame_index}")
            identities.add(identity)
            records.append((frame_index, track_id, class_map[label]))
            boxes.append(geometry)
        except ValueError as exc:
            raise ValueError(f"Invalid 3D tracking ground truth at {path}:{line_number}: {exc}") from exc
    indices = np.asarray(records, dtype=np.int64).reshape(-1, 3)
    return TrackingLabels3D(
        frame_indices=indices[:, 0],
        track_ids=indices[:, 1],
        class_ids=indices[:, 2],
        boxes=np.asarray(boxes, dtype=np.float64).reshape(-1, 7),
        row_count=row_count,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


class KittiDetections3D:
    """Index per-frame detection files from one or more configured directories.

    KITTI rows describe bottom-center camera coordinates, y-axis rotation and
    height/width/length dimensions. They become canonical x/y/z/yaw/length/width/
    height rows without changing the coordinate frame. Scores must already be
    probabilities unless ``score_transform="odds"`` explicitly selects s/(1+s).
    """

    def __init__(
        self,
        directories: Sequence[Path],
        *,
        frame_count: int,
        classes: Mapping[str, Mapping[str, Any]],
        options: Mapping[str, Any] | None = None,
    ) -> None:
        options = dict(options or {})
        allowed_options = {
            "score_transform",
            "class_map",
            "ignore_classes",
            "coordinate_frame",
            "box_origin",
            "dimensions",
            "yaw_axis",
        }
        unknown = set(options) - allowed_options
        if unknown:
            raise ValueError(f"Unsupported kitti-detections options: {', '.join(sorted(unknown))}.")
        self.score_transform = options.get("score_transform", "identity")
        if not isinstance(self.score_transform, str) or self.score_transform not in {"identity", "odds"}:
            raise ValueError("kitti-detections score_transform must be 'identity' or 'odds'.")
        conventions = {
            "coordinate_frame": "camera",
            "box_origin": "bottom-center",
            "dimensions": "hwl",
            "yaw_axis": "y",
        }
        for name, expected in conventions.items():
            if options.get(name, expected) != expected:
                raise ValueError(f"kitti-detections {name} must be {expected!r}.")
        self.class_map, self.ignored_class_ids, self.ignore_classes = _class_mapping(
            classes, options, "kitti-detections"
        )
        self._paths: list[dict[int, Path]] = []
        self.missing_frames: dict[str, tuple[int, ...]] = {}
        for directory in directories:
            directory = Path(directory).expanduser().resolve()
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing 3D detection sequence directory: {directory}")
            indexed: dict[int, Path] = {}
            for path in directory.glob("*.txt"):
                if path.name.startswith("._"):
                    continue
                if not path.stem.isascii() or not path.stem.isdecimal():
                    raise ValueError(f"3D detection frame names must have nonnegative numeric stems: {path}")
                frame_index = int(path.stem)
                if frame_index >= frame_count:
                    raise ValueError(f"3D detection frame {frame_index} has no corresponding image: {path}")
                if frame_index in indexed:
                    raise ValueError(
                        f"Duplicate 3D detection frame index {frame_index}: {indexed[frame_index]} and {path}"
                    )
                indexed[frame_index] = path
            self._paths.append(indexed)
            self.missing_frames[str(directory)] = tuple(sorted(set(range(frame_count)) - indexed.keys()))

    def read(self, frame_index: int, sample_id: str) -> Detections3D:
        """Read and merge the requested frame while preserving absent predictions."""
        boxes: list[list[float]] = []
        scores: list[float] = []
        classes: list[int] = []
        for indexed in self._paths:
            path = indexed.get(frame_index)
            if path is None:
                continue
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        fields = line.split()
                        if len(fields) != 16:
                            raise ValueError("Rows must have 16 KITTI detection fields")
                        label = fields[0].casefold()
                        if label not in self.class_map and label not in self.ignore_classes:
                            raise ValueError(f"unsupported 3D detection class {fields[0]!r}")
                        values = np.array(fields[1:], dtype=np.float64)
                        if not np.isfinite(values).all():
                            raise ValueError("all 3D detection numeric fields must be finite")
                        score = float(values[-1])
                        if score < 0:
                            raise ValueError("3D detection score must be nonnegative")
                        if self.score_transform == "odds":
                            score /= 1.0 + score
                        elif score > 1:
                            raise ValueError(
                                "3D detection score must be between zero and one for identity score_transform"
                            )
                        geometry = values[7:14][[3, 4, 5, 6, 2, 1, 0]]
                        if (np.abs(geometry) > np.finfo(np.float32).max).any() or (geometry[4:] <= 0).any():
                            raise ValueError("3D box must have finite float32 coordinates and positive dimensions")
                        if (geometry[4:].astype(np.float32) <= 0).any():
                            raise ValueError("3D box dimensions must remain positive in float32")
                        if label in self.ignore_classes or self.class_map.get(label) in self.ignored_class_ids:
                            continue
                        boxes.append(geometry.tolist())
                        scores.append(score)
                        classes.append(self.class_map[label])
                    except ValueError as exc:
                        raise ValueError(f"Invalid 3D detection input at {path}:{line_number}: {exc}") from exc
        return Detections3D(
            Boxes3D(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7)),
            torch.tensor(scores, dtype=torch.float32),
            torch.tensor(classes, dtype=torch.int64),
            sample_id,
        )
