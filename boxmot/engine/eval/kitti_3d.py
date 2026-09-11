"""KITTI result export, official object AP, and TrackEval 2D tracking protocols."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from boxmot.datasets.readers.boxes3d import KittiObjectLabels, TrackingLabels3D
from boxmot.engine.eval.kitti_object_backend import evaluate_kitti_objects, resolve_kitti_object_backend
from boxmot.engine.eval.trackeval_reference import evaluate_trackeval_kitti, validate_trackeval_kitti_dependencies
from boxmot.structures import CameraModel, Tracks3D
from boxmot.trackers.eagermot.geometry import project_box3d
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


def validate_kitti_evaluation_dependencies() -> None:
    """Fail before replay when either official KITTI evaluator is unavailable."""
    validate_trackeval_kitti_dependencies()
    resolve_kitti_object_backend()


def _tracking_row(fields: list[str], identity_map: dict[int, int]) -> str:
    """Compact identities losslessly before TrackEval's floating-point parser."""
    fields = fields.copy()
    identity = int(fields[1])
    if identity >= 0:
        fields[1] = str(identity_map.setdefault(identity, len(identity_map)))
    if fields[2].casefold() == "person_sitting":
        fields[2] = "Person"
    return " ".join(fields)


def _validate_tracking_source(fields: list[str], *, sequence_id: str, frame_count: int) -> None:
    """Validate ignored raw rows too before passing them to official parsers."""
    try:
        if len(fields) != 17 or not 0 <= int(fields[0]) < frame_count:
            raise ValueError("expected 17 fields with a frame inside the sequence")
        int(fields[1])
        values = np.asarray(fields[3:], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("all numeric fields must be finite, including ignored classes")
        dontcare = fields[2].casefold() == "dontcare"
        if values[0] not in ((-1, 0, 1, 2) if dontcare else (0, 1, 2)):
            raise ValueError("tracking truncation must be an integer category in [0, 2]; DontCare also permits -1")
        if values[1] not in ((-1, 0, 1, 2, 3) if dontcare else (0, 1, 2, 3)):
            raise ValueError("tracking occlusion must be an integer in [0, 3]; DontCare also permits -1")
        if np.any(values[5:7] <= values[3:5]):
            raise ValueError("tracking image bounds must have positive width and height")
    except ValueError as error:
        raise ValueError(f"Invalid original KITTI tracking annotation row for {sequence_id!r}: {error}.") from error


def _export_official_inputs(
    prediction_dir: Path,
    output: Path,
    annotations: Mapping[str, TrackingLabels3D],
    frame_counts: Mapping[str, int],
    object_annotations: Mapping[str, KittiObjectLabels],
) -> tuple[Path, Path, Path, Path, list[str], dict[str, Any]]:
    """Export distinct, unfiltered official annotations with frame provenance."""
    if not annotations or set(annotations) != set(frame_counts) or set(annotations) != set(object_annotations):
        raise ValueError("KITTI evaluation requires matching nonempty tracking, object, and frame-count sequence sets.")
    root = output / "protocol_inputs"
    tracking_gt, tracking_predictions = root / "tracking" / "ground_truth", root / "tracking" / "predictions"
    object_gt, object_predictions = root / "objects" / "ground_truth", root / "objects" / "predictions"
    for directory in (tracking_gt / "label_02", tracking_predictions, object_gt, object_predictions):
        directory.mkdir(parents=True, exist_ok=True)
    frame_ids, frame_map, identity_maps, prediction_hashes = [], [], {}, {}
    for sequence_id, truth in annotations.items():
        frame_count = frame_counts[sequence_id]
        _nonnegative_integer(frame_count, "frame count")
        if not frame_count:
            raise ValueError("KITTI frame count must be positive.")
        objects = object_annotations[sequence_id]
        if len(objects.frame_rows) != frame_count:
            raise ValueError(f"KITTI object annotations for {sequence_id!r} must cover exactly {frame_count} frames.")
        if len(truth.source_rows) != truth.row_count:
            raise ValueError(f"KITTI tracking annotations for {sequence_id!r} must retain all original source rows.")
        path = Path(prediction_dir) / f"{sequence_id}.txt"
        read_kitti_3d_results(path, frame_count=frame_count)
        payload = path.read_bytes()
        prediction_hashes[sequence_id] = hashlib.sha256(payload).hexdigest()
        predictions_by_frame: dict[int, list[str]] = {}
        predicted_ids: dict[int, int] = {}
        prediction_rows = []
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            x1, y1, x2, y2 = map(float, fields[6:10])
            if x2 <= x1 or y2 <= y1 or (x1, y1, x2, y2) == (-1, -1, -1, -1):
                raise ValueError(
                    f"KITTI official scoring requires a valid projected 2D box for every spatial prediction; "
                    f"{path}:{line_number} has an unavailable or zero-area projection."
                )
            predictions_by_frame.setdefault(int(fields[0]), []).append(" ".join(fields[2:]))
            prediction_rows.append(_tracking_row(fields, predicted_ids))
        (tracking_predictions / f"{sequence_id}.txt").write_text(
            "".join(row + "\n" for row in prediction_rows), encoding="utf-8"
        )
        gt_ids: dict[int, int] = {}
        ground_truth_rows = []
        for line in truth.source_rows:
            fields = line.split()
            _validate_tracking_source(fields, sequence_id=sequence_id, frame_count=frame_count)
            ground_truth_rows.append(_tracking_row(fields, gt_ids))
        (tracking_gt / "label_02" / f"{sequence_id}.txt").write_text(
            "".join(row + "\n" for row in ground_truth_rows), encoding="utf-8"
        )
        identity_maps[sequence_id] = {"ground_truth": gt_ids, "predictions": predicted_ids}
        for frame_index, rows in enumerate(objects.frame_rows):
            if any(len(row.split()) != 15 for row in rows):
                raise ValueError(f"KITTI object annotations for {sequence_id!r} must retain 15-field source rows.")
            frame_id = f"{len(frame_ids):06d}"
            frame_ids.append(frame_id)
            frame_map.append({"id": frame_id, "sequence": sequence_id, "frame_index": frame_index})
            (object_gt / f"{frame_id}.txt").write_text("".join(row + "\n" for row in rows), encoding="utf-8")
            (object_predictions / f"{frame_id}.txt").write_text(
                "".join(row + "\n" for row in predictions_by_frame.get(frame_index, ())), encoding="utf-8"
            )
    provenance = {
        "frames": frame_map,
        "tracking_identity_maps": identity_maps,
        "prediction_sha256": prediction_hashes,
        "tracking_ground_truth_sha256": {name: truth.source_sha256 for name, truth in annotations.items()},
        "object_ground_truth_sha256": {name: truth.source_sha256 for name, truth in object_annotations.items()},
        "tracking_class_alias": {"Person_sitting": "Person"},
    }
    return tracking_gt, tracking_predictions, object_gt, object_predictions, frame_ids, provenance


def evaluate_kitti_3d(
    prediction_dir: Path,
    output: Path,
    annotations: Mapping[str, TrackingLabels3D],
    frame_counts: Mapping[str, int],
    object_annotations: Mapping[str, KittiObjectLabels],
) -> dict[str, dict[str, Any]]:
    """Score spatial outputs with official object AP and KITTI 2D tracking.

    KITTI's object devkit scores 2D/3D AP by difficulty against distinct object
    annotations. TrackEval scores projected 2D tracking against original tracking
    annotations, retaining its own distractor and visibility preprocessing.
    """
    validate_kitti_evaluation_dependencies()
    output = Path(output)
    gt, predictions, object_gt, object_predictions, frame_ids, provenance = _export_official_inputs(
        Path(prediction_dir), output, annotations, frame_counts, object_annotations
    )
    LOGGER.info("Evaluating official KITTI object AP and TrackEval KITTI 2D tracking")
    detection_results = evaluate_kitti_objects(object_gt, object_predictions, frame_ids, output)
    results = evaluate_trackeval_kitti(gt_folder=gt, tracker_folder=predictions, seq_info=frame_counts)
    (output / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = [name for name in results["car"] if name != "per_sequence"]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", *fields])
        writer.writeheader()
        for name, values in results.items():
            writer.writerow({"class": name, **{key: values[key] for key in fields}})
    (output / "detection_metrics.json").write_text(
        json.dumps(detection_results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "detection_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["geometry", "class", "difficulty", "AP40"])
        for geometry, classes in detection_results.items():
            for class_name, difficulties in classes.items():
                for difficulty, value in difficulties.items():
                    writer.writerow([geometry, class_name, difficulty, value])
    protocol = {
        "protocol": "kitti-official-object-and-trackeval-tracking",
        "tracking": {"evaluator": "trackeval==1.3.0", "dataset": "Kitti2DBox", "geometry": "2d"},
        "detection": {"evaluator": "KITTI object devkit", "geometry": ["2d", "3d"], "metric": "AP40"},
        "spatial_prediction_projection": "Existing camera projection of emitted 3D boxes, without tracker changes.",
        **provenance,
    }
    (output / "evaluation.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return results
