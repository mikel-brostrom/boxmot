"""Join saved 3D detections to annotated identities in the runtime coordinate frame."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.datasets.inputs import DatasetInputs, ModalityInput
from boxmot.datasets.readers.boxes3d import KittiDetections3D, read_kitti_tracking_labels
from boxmot.datasets.readers.calibration import read_kitti_projection
from boxmot.datasets.readers.frames import numeric_frame_paths
from boxmot.datasets.readers.poses import read_camera_to_world_poses
from boxmot.engine.calibration.kalman_data import CalibrationData, CalibrationTrack
from boxmot.trackers.eagermot.geometry import iou3d_matrix, transform_boxes3d

if TYPE_CHECKING:
    from boxmot.datasets.sensor_cache import SensorReplaySequence


def _single_path(modality: ModalityInput, encoding: str, role: str) -> Path:
    """Require the declared sensor encoding even for programmatic dataset inputs."""
    if modality.format != encoding or len(modality.paths) != 1:
        raise ValueError(f"3D KF calibration {role} requires one {encoding!r} input path.")
    return modality.paths[0]


def _digest_json(payload: Any) -> str:
    """Hash canonical metadata and ordered source-file records."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_digest(path: Path) -> str:
    """Hash file bytes without retaining complete prediction directories in memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _match_boxes(gt_boxes: np.ndarray, detected_boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match same-class camera-space boxes one-to-one using volumetric IoU >= 0.5."""
    if not len(gt_boxes) or not len(detected_boxes):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    similarities = iou3d_matrix(gt_boxes, detected_boxes)
    gated = np.where(similarities >= 0.5, similarities, 0.0)
    gt_indices, detection_indices = linear_sum_assignment(-gated)
    keep = gated[gt_indices, detection_indices] > 0.0
    return gt_indices[keep], detection_indices[keep]


def load_sensor_calibration_data(
    dataset: DatasetInputs,
    *,
    progress: Callable[[str], None] | None = None,
    cached_sequences: Mapping[str, SensorReplaySequence] | None = None,
) -> CalibrationData:
    """Match explicit 3D GT to predictions in the tracker's motion coordinates.

    Ground truth must contain annotated 3D boxes with persistent identities;
    neither detector boxes nor 2D instance masks substitute for annotations.
    Missing detector updates remain NaN rows; missing annotations remain gaps
    in each trajectory's zero-based frame indices. No image or mask pixels are
    decoded, and detector scores receive only their configured reader transform.
    Declared ego poses transform observations into world coordinates. Without
    poses, calibration retains camera coordinates, matching EagerMOT runtime.
    """
    if isinstance(dataset.fps, bool) or not np.isfinite(dataset.fps) or dataset.fps <= 0:
        raise ValueError("3D KF calibration requires a positive finite dataset fps.")
    target_classes = sorted(
        int(value["id"]) for value in dataset.classes.values() if value.get("evaluation") == "target"
    )
    if not target_classes:
        raise ValueError("3D KF calibration requires at least one target ground-truth class.")
    required = {"images", "ground_truth_3d", "detections_3d", "calibration"}
    for sequence in dataset.sequences:
        missing = required - sequence.modalities.keys()
        if missing:
            raise ValueError(
                f"3D KF calibration requires {', '.join(sorted(missing))} for sequence {sequence.sequence_id!r}. "
                "Declare ground_truth_3d with format kitti-tracking-labels and annotated 3D identities; "
                "2D masks and detector predictions cannot substitute for 3D ground truth."
            )
    statistics = dict.fromkeys(
        (
            "frames",
            "detections",
            "target_detections",
            "ground_truth_rows",
            "ground_truth",
            "filtered_ground_truth",
            "matched",
            "unmatched_ground_truth",
            "unmatched_detections",
            "trajectories",
        ),
        0,
    )
    tracks: list[CalibrationTrack] = []
    ground_truth_sources: list[dict[str, str]] = []
    input_sources: list[dict[str, str]] = []
    for sequence in dataset.sequences:
        sequence_id = sequence.sequence_id
        if progress is not None:
            progress(f"KF calibration: matching saved 3D detections to GT for {sequence_id}…")
        modalities = sequence.modalities
        cached = None
        if cached_sequences is not None:
            if sequence_id not in cached_sequences:
                raise ValueError(f"No cached sensor inputs for calibration sequence {sequence_id!r}.")
            cached = cached_sequences[sequence_id]
            cached.validate(dataset=dataset)
        images = _single_path(modalities["images"], "image-directory", "images")
        frame_paths = numeric_frame_paths(images, contiguous=True) if cached is None else cached.frame_paths
        frame_count = len(frame_paths)
        gt_path = _single_path(modalities["ground_truth_3d"], "kitti-tracking-labels", "ground_truth_3d")
        projection_path = _single_path(modalities["calibration"], "kitti-p2", "calibration")
        poses_path = (
            _single_path(modalities["poses"], "camera-to-world-npy", "poses") if "poses" in modalities else None
        )
        coordinate_frame = "camera" if poses_path is None else "world"
        for role in ("images", "calibration", "poses"):
            if role in modalities and modalities[role].options:
                raise ValueError(f"3D KF calibration {role} does not support reader options.")
        # Validate projection using the same reader as replay, although matching
        # uses 3D geometry and never projects boxes into the scoring image.
        predictions = modalities["detections_3d"]
        if predictions.format != "kitti-detections" or not predictions.paths:
            raise ValueError("3D KF calibration detections_3d requires kitti-detections input directories.")
        reader = poses = None
        if cached is None:
            read_kitti_projection(projection_path)
            poses = None if poses_path is None else read_camera_to_world_poses(poses_path, frame_count)
            ground_truth = read_kitti_tracking_labels(
                gt_path, frame_count=frame_count, classes=dataset.classes, options=modalities["ground_truth_3d"].options
            )
            reader = KittiDetections3D(
                predictions.paths, frame_count=frame_count, classes=dataset.classes, options=predictions.options
            )
        else:
            ground_truth = cached.ground_truth_3d()
            if ground_truth is None:
                raise ValueError(f"Cached sequence {sequence_id!r} has no 3D ground truth.")
        source_digest = _file_digest if cached is None else cached.source_sha256
        ground_truth_sources.append(
            {"sequence_id": sequence_id, "path": str(gt_path.resolve()), "sha256": ground_truth.source_sha256}
        )
        metadata = {
            "dataset_id": dataset.id,
            "split": dataset.split,
            "fps": dataset.fps,
            "coordinate_frame": coordinate_frame,
            "classes": dataset.classes,
            "modalities": {
                role: {"format": modalities[role].format, "options": modalities[role].options}
                for role in sorted(required | ({"poses"} if poses_path is not None else set()))
            },
        }
        input_sources.extend(
            (
                {"sequence_id": sequence_id, "role": "metadata", "sha256": _digest_json(metadata)},
                {
                    "sequence_id": sequence_id,
                    "role": "coordinate_frame",
                    "value": coordinate_frame,
                    "sha256": _digest_json(coordinate_frame),
                },
                {
                    "sequence_id": sequence_id,
                    "role": "images",
                    "path": str(images.resolve()),
                    "sha256": _digest_json([path.name for path in frame_paths]),
                },
            )
        )
        for role, path in (("calibration", projection_path), ("poses", poses_path)):
            if path is None:
                continue
            input_sources.append(
                {"sequence_id": sequence_id, "role": role, "path": str(path.resolve()), "sha256": source_digest(path)}
            )
        for directory in predictions.paths:
            if cached is None:
                contents = [
                    (path.name, source_digest(path))
                    for path in sorted(directory.glob("*.txt"))
                    if not path.name.startswith("._")
                ]
            else:
                contents = [
                    (name, digest)
                    for name, digest in cached.source_files(directory)
                    if name.endswith(".txt") and not name.startswith("._")
                ]
            input_sources.append(
                {
                    "sequence_id": sequence_id,
                    "role": "detections_3d",
                    "path": str(directory.resolve()),
                    "sha256": _digest_json(contents),
                }
            )
        statistics["ground_truth_rows"] += ground_truth.row_count
        statistics["ground_truth"] += len(ground_truth.boxes)
        statistics["filtered_ground_truth"] += ground_truth.row_count - len(ground_truth.boxes)
        frame_gt_indices: dict[int, list[int]] = defaultdict(list)
        for index, frame_index in enumerate(ground_truth.frame_indices):
            frame_gt_indices[int(frame_index)].append(index)
        observations: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
        for frame_index in range(frame_count):
            if cached is None:
                detected = reader.read(frame_index, f"{dataset.split}:{sequence_id}:{frame_index}")
                pose = None if poses is None else poses[frame_index]
            else:
                detected = cached.read_spatial(frame_index)
                camera = cached.read_camera(frame_index)
                if camera is None:
                    raise ValueError(
                        f"Cached sequence {sequence_id!r} has no camera calibration for frame {frame_index}."
                    )
                if (camera.camera_to_world is not None) != (poses_path is not None):
                    raise ValueError(f"Cached sequence {sequence_id!r} ego poses disagree with the declared inputs.")
                pose = None if camera.camera_to_world is None else camera.camera_to_world.numpy()
            detected_boxes = detected.geometry.values.numpy().astype(np.float64)
            detected_classes = detected.class_ids.numpy()
            scores = detected.scores.numpy().astype(np.float64)
            statistics["frames"] += 1
            statistics["detections"] += len(detected)
            statistics["target_detections"] += int(np.isin(detected_classes, target_classes).sum())
            gt_indices = np.asarray(frame_gt_indices[frame_index], dtype=np.int64)
            gt_boxes = ground_truth.boxes[gt_indices]
            gt_classes = ground_truth.class_ids[gt_indices]
            identities = ground_truth.track_ids[gt_indices]
            # Match before the runtime's upright yaw approximation discards
            # roll/pitch; transform both sides only after camera-space matching.
            state_gt = gt_boxes if pose is None else transform_boxes3d(gt_boxes, pose)
            state_detections = detected_boxes if pose is None else transform_boxes3d(detected_boxes, pose)
            for class_id in target_classes:
                class_gt = np.flatnonzero(gt_classes == class_id)
                class_detected = np.flatnonzero(detected_classes == class_id)
                gt_match, detection_match = _match_boxes(gt_boxes[class_gt], detected_boxes[class_detected])
                matched = dict(zip(class_gt[gt_match], class_detected[detection_match], strict=True))
                statistics["matched"] += len(matched)
                for gt_index in class_gt:
                    detection_index = matched.get(gt_index)
                    detected_box = np.full(7, np.nan) if detection_index is None else state_detections[detection_index]
                    score = np.nan if detection_index is None else scores[detection_index]
                    observations[(class_id, int(identities[gt_index]))].append(
                        (frame_index, state_gt[gt_index], detected_box, score)
                    )
        for (class_id, track_id), records in sorted(observations.items()):
            indices = np.asarray([record[0] for record in records], dtype=np.int64)
            tracks.append(
                CalibrationTrack(
                    sequence_id=sequence_id,
                    track_id=track_id,
                    class_id=class_id,
                    frame_indices=indices,
                    timestamps_s=indices.astype(np.float64) / dataset.fps,
                    gt_boxes=np.asarray([record[1] for record in records], dtype=np.float64),
                    detection_boxes=np.asarray([record[2] for record in records], dtype=np.float64),
                    scores=np.asarray([record[3] for record in records], dtype=np.float64),
                )
            )
    statistics["unmatched_ground_truth"] = statistics["ground_truth"] - statistics["matched"]
    statistics["unmatched_detections"] = statistics["target_detections"] - statistics["matched"]
    statistics["trajectories"] = len(tracks)
    return CalibrationData(
        tracks=tuple(tracks),
        statistics=statistics,
        ground_truth_sources=tuple(ground_truth_sources),
        input_sources=tuple(input_sources),
    )
