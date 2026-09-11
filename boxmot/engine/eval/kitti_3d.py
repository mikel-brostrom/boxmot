"""KITTI-format 3D results and volumetric tracking metrics.

This evaluates camera-space boxes with volumetric IoU. It does not implement
the official KITTI difficulty, visibility, or DontCare-region protocol.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from boxmot.datasets.readers.boxes3d import TrackingLabels3D
from boxmot.engine.eval.motmetrics import (
    SequenceData,
    _combine_bundles,
    _combine_bundles_class_averaged,
    _eval_bundle,
    _relabel_ids,
    _summary_from_bundle,
)
from boxmot.structures import CameraModel, Tracks3D
from boxmot.trackers.eagermot.geometry import iou3d_matrix, project_box3d
from boxmot.utils import logger as LOGGER

KITTI_3D_CLASSES = {1: "car", 2: "pedestrian"}
_RESULT_LABELS = {1: "Car", 2: "Pedestrian"}


def _nonnegative_integer(value: int, name: str) -> None:
    """Keep frame numbers and identities exact, including IDs above float precision."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"KITTI 3D {name} must be a nonnegative int64 integer.")


@dataclass(frozen=True, slots=True)
class Kitti3DRow:
    """One spatial prediction, retaining independent camera-space box7 geometry."""

    frame_index: int
    track_id: int
    class_id: int
    box: tuple[float, ...]
    score: float

    def __post_init__(self) -> None:
        _nonnegative_integer(self.frame_index, "frame index")
        _nonnegative_integer(self.track_id, "track ID")
        if self.class_id not in KITTI_3D_CLASSES:
            raise ValueError("KITTI 3D result classes must be car (1) or pedestrian (2).")
        values = np.asarray(self.box)
        if values.shape != (7,) or not np.isfinite(values).all() or np.any(values[4:] <= 0):
            raise ValueError("KITTI 3D results require finite box7 geometry with positive length, width, and height.")
        if np.any(np.abs(values) > np.finfo(np.float32).max) or np.any(values[4:].astype(np.float32) <= 0):
            raise ValueError("KITTI 3D result boxes must preserve finite geometry and positive dimensions in float32.")
        if not np.isfinite(self.score):
            raise ValueError("KITTI 3D result scores must be finite.")


def write_kitti_3d_rows(handle: TextIO, tracks: Tracks3D, frame_index: int, camera: CameraModel | None) -> int:
    """Write 18-field KITTI predictions without changing camera coordinates.

    Tracks already contain camera-space bottom-center boxes. The optional image
    bounds come from the camera projection; unavailable bounds, truncation, and
    occlusion use -1. Angles are wrapped to [-pi, pi). Invisible spatial tracks
    remain present in the result file.
    """
    _nonnegative_integer(frame_index, "frame index")
    tracks.validate()
    projection = None if camera is None else camera.projection.numpy()
    lines = []
    for index, box in enumerate(tracks.geometry.values.numpy()):
        row = Kitti3DRow(
            frame_index,
            int(tracks.track_ids[index]),
            int(tracks.class_ids[index]),
            tuple(map(float, box)),
            float(tracks.scores[index]),
        )
        x, y, z, yaw, length, width, height = row.box
        yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
        bounds = None if camera is None else project_box3d(box, projection, camera.image_size)
        bounds = (-1.0,) * 4 if bounds is None else bounds
        alpha = (yaw - np.arctan2(x, z) + np.pi) % (2 * np.pi) - np.pi
        geometry = (*bounds, height, width, length, x, y, z, yaw, row.score)
        values = " ".join(format(value, ".17g") for value in geometry)
        lines.append(f"{frame_index} {row.track_id} {_RESULT_LABELS[row.class_id]} -1 -1 {alpha:.17g} {values}\n")
    handle.writelines(lines)
    return len(lines)


def read_kitti_3d_results(path: str | Path, *, frame_count: int) -> dict[int, tuple[Kitti3DRow, ...]]:
    """Read strict 18-field KITTI tracking predictions, retaining within-frame order."""
    _nonnegative_integer(frame_count, "frame count")
    if not frame_count:
        raise ValueError("KITTI 3D frame count must be positive.")
    frames: dict[int, list[Kitti3DRow]] = {}
    identities: dict[int, set[int]] = {}
    class_ids = {name: class_id for class_id, name in KITTI_3D_CLASSES.items()}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                fields = line.split()
                if len(fields) != 18:
                    raise ValueError("Rows must have 18 KITTI tracking result fields, including score.")
                if any(not value.isascii() or not value.isdecimal() for value in fields[:2]):
                    raise ValueError("Frame and track identity must be nonnegative integers.")
                frame_index, track_id = map(int, fields[:2])
                if frame_index >= frame_count:
                    raise ValueError(f"Frame index must be between 0 and {frame_count - 1}.")
                label = fields[2].casefold()
                if label not in class_ids:
                    raise ValueError(f"Unsupported result class {fields[2]!r}; expected Car or Pedestrian.")
                values = np.asarray(fields[3:], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("All numeric KITTI 3D result fields must be finite.")
                row = Kitti3DRow(
                    frame_index,
                    track_id,
                    class_ids[label],
                    tuple(values[[10, 11, 12, 13, 9, 8, 7]]),
                    float(values[14]),
                )
                frame_ids = identities.setdefault(frame_index, set())
                if track_id in frame_ids:
                    raise ValueError(f"Duplicate track identity {track_id} in frame {frame_index}.")
                frame_ids.add(track_id)
                frames.setdefault(frame_index, []).append(row)
            except ValueError as error:
                raise ValueError(f"Invalid KITTI 3D results at {path}:{line_number}: {error}") from error
    return {frame: tuple(rows) for frame, rows in sorted(frames.items())}


def _sequence_data(
    sequence_id: str,
    annotations: TrackingLabels3D,
    predictions: Mapping[int, tuple[Kitti3DRow, ...]],
    frame_count: int,
    class_id: int,
) -> SequenceData:
    """Index true 3D observations and compact identities separately per sequence."""
    selected = np.flatnonzero(annotations.class_ids == class_id)
    frame_rows: list[list[int]] = [[] for _ in range(frame_count)]
    for index in selected:
        frame_rows[int(annotations.frame_indices[index])].append(int(index))
    gt_ids, tracker_ids, similarities = [], [], []
    for frame_index, indices in enumerate(frame_rows):
        rows = tuple(row for row in predictions.get(frame_index, ()) if row.class_id == class_id)
        boxes = np.asarray([row.box for row in rows], dtype=np.float64).reshape(-1, 7)
        gt_ids.append(annotations.track_ids[indices])
        tracker_ids.append(np.asarray([row.track_id for row in rows], dtype=np.int64))
        similarities.append(iou3d_matrix(annotations.boxes[indices], boxes))
    gt_ids, num_gt = _relabel_ids(gt_ids)
    tracker_ids, num_tracks = _relabel_ids(tracker_ids)
    return SequenceData(
        seq=sequence_id,
        gt_ids=gt_ids,
        tracker_ids=tracker_ids,
        similarity_scores=similarities,
        num_timesteps=frame_count,
        num_gt_dets=sum(map(len, gt_ids)),
        num_tracker_dets=sum(map(len, tracker_ids)),
        num_gt_ids=num_gt,
        num_tracker_ids=num_tracks,
    )


def evaluate_kitti_3d(
    prediction_dir: Path,
    output: Path,
    annotations: Mapping[str, TrackingLabels3D],
    frame_counts: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Evaluate saved 3D predictions with shared HOTA, CLEAR, and Identity metrics.

    HOTA uses volumetric IoU thresholds 0.05 through 0.95; CLEAR and Identity
    use 0.5. Ground truth comes from the dataset's filtered 3D annotation reader.
    Mask overlap and image visibility do not substitute for spatial overlap.
    """
    if not annotations or set(annotations) != set(frame_counts):
        raise ValueError("KITTI 3D evaluation requires matching nonempty annotation and frame-count sequence sets.")
    LOGGER.info("Evaluating camera-space 3D boxes with volumetric IoU (custom protocol)")
    bundles: dict[str, dict[str, Any]] = {name: {} for name in KITTI_3D_CLASSES.values()}
    for sequence_id, truth in annotations.items():
        frame_count = frame_counts[sequence_id]
        _nonnegative_integer(frame_count, "frame count")
        if not frame_count:
            raise ValueError("KITTI 3D frame count must be positive.")
        if np.any((truth.frame_indices < 0) | (truth.frame_indices >= frame_count)):
            raise ValueError(f"KITTI 3D annotations for {sequence_id!r} exceed its frame bounds.")
        if not set(truth.class_ids).issubset(KITTI_3D_CLASSES):
            raise ValueError("KITTI 3D annotations must contain only configured target car/pedestrian classes.")
        predictions = read_kitti_3d_results(Path(prediction_dir) / f"{sequence_id}.txt", frame_count=frame_count)
        for class_id, name in KITTI_3D_CLASSES.items():
            bundles[name][sequence_id] = _eval_bundle(
                _sequence_data(sequence_id, truth, predictions, frame_count, class_id)
            )
    combined = {name: _combine_bundles(values) for name, values in bundles.items()}
    results = {
        name: {
            **_summary_from_bundle(combined[name]),
            "per_sequence": {sequence_id: _summary_from_bundle(bundle) for sequence_id, bundle in values.items()},
        }
        for name, values in bundles.items()
    }
    results["cls_comb_cls_av"] = _summary_from_bundle(_combine_bundles_class_averaged(combined))
    results["cls_comb_det_av"] = _summary_from_bundle(_combine_bundles(combined))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = [name for name in results["car"] if name != "per_sequence"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", *fields])
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"class": name, **{key: values[key] for key in fields}})
    protocol = {
        "protocol": "boxmot-3d-iou-v1",
        "official_kitti_protocol": False,
        "geometry": "camera-space bottom-center x/y/z/yaw/length/width/height in meters and radians",
        "similarity": "volumetric 3D IoU",
        "corner_precision_m": 0.0001,
        "hota_iou_thresholds": [index / 20 for index in range(1, 20)],
        "clear_identity_iou_threshold": 0.5,
        "ground_truth": "All supplied target 3D annotations; configured non-target classes are excluded by the reader.",
        "limitations": "No official KITTI difficulty, visibility, truncation, or DontCare-region filtering.",
        "sequences": dict(frame_counts),
        "ground_truth_sha256": {name: value.source_sha256 for name, value in annotations.items()},
    }
    (output / "evaluation.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return results
