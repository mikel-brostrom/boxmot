"""Reusable KITTI 16-field camera-space 3D detection decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot.structures import Boxes3D, Detections3D


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
        class_ids = {str(name).casefold(): int(value["id"]) for name, value in classes.items()}
        self.ignored_class_ids = frozenset(
            int(value["id"]) for value in classes.values() if value.get("evaluation") == "ignore"
        )
        self.class_map = dict(class_ids)
        mappings = options.get("class_map", {})
        if not isinstance(mappings, Mapping):
            raise ValueError("kitti-detections class_map must be a mapping.")
        for label, target in mappings.items():
            if isinstance(target, str):
                if target.casefold() not in class_ids:
                    raise ValueError(f"Unknown configured 3D class name {target!r}.")
                target = class_ids[target.casefold()]
            if type(target) is not int or target not in class_ids.values():
                raise ValueError(f"3D class_map target {target!r} is not a configured class ID.")
            self.class_map[str(label).casefold()] = target
        ignored = options.get("ignore_classes", ())
        if isinstance(ignored, (str, bytes)) or not isinstance(ignored, Sequence):
            raise ValueError("kitti-detections ignore_classes must be a list of source class names.")
        self.ignore_classes = frozenset(str(name).casefold() for name in ignored)
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
