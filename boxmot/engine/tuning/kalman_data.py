"""Join cached detector predictions to eligible annotated trajectories for KF calibration."""

from __future__ import annotations

import argparse
import hashlib
import io
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.datasets.cached import CachedVisionDataset
from boxmot.engine.eval.motmetrics import (
    _aabb_gt_path,
    _aabb_iou_matrix,
    _index_rows_by_frame,
    _resolve_obb_gt_path,
    _rotated_iou_batch,
    build_dataset_eval_settings,
)
from boxmot.trackers.common.geometry.obb import xywha_to_corners


@dataclass(frozen=True)
class CalibrationTrack:
    """GT observations with matched detections; missing detections are NaN rows.

    Frame indices are zero-based delivered-frame indices. Geometry is ``xyxy``
    for AABB and ``cx, cy, w, h, angle`` (radians) for OBB. Unannotated frames
    are absent; their elapsed interval is preserved by indices and timestamps.
    """

    sequence_id: str
    track_id: int
    class_id: int
    frame_indices: np.ndarray
    timestamps_s: np.ndarray | None
    gt_boxes: np.ndarray
    detection_boxes: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class CalibrationData:
    """Matched trajectories plus matching coverage and exact annotation provenance."""

    tracks: tuple[CalibrationTrack, ...]
    statistics: dict[str, int]
    ground_truth_sources: tuple[dict[str, str], ...]
    match_iou: float = 0.5


def _read_ground_truth(path: Path, *, geometry: str, frame_count: int) -> tuple[np.ndarray, str]:
    """Parse annotation bytes strictly so malformed GT cannot become empty data."""

    payload = path.read_bytes()
    text = payload.decode("utf-8-sig")
    columns = 13 if geometry == "obb" else 9
    if not text.strip():
        return np.empty((0, columns), dtype=np.float64), hashlib.sha256(payload).hexdigest()
    try:
        rows = np.loadtxt(io.StringIO(text), delimiter="," if "," in text else None, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"Malformed calibration ground truth in {path}: {exc}") from exc
    if (geometry == "obb" and rows.shape[1] != 13) or (geometry == "aabb" and rows.shape[1] < 8):
        expected = "13 MMOT corner columns" if geometry == "obb" else "at least 8 MOT/VisDrone columns"
        raise ValueError(f"Calibration ground truth {path} requires {expected}; got {rows.shape[1]}.")
    if not np.isfinite(rows).all():
        raise ValueError(f"Calibration ground truth {path} contains non-finite values.")
    integer_columns = [0, 1, 11 if geometry == "obb" else 7]
    if np.any(rows[:, integer_columns] != np.floor(rows[:, integer_columns])):
        raise ValueError(f"Calibration ground truth {path} requires integer frame, identity, and class IDs.")
    if np.any((rows[:, 0] < 1) | (rows[:, 0] > frame_count)):
        raise ValueError(f"Calibration ground truth {path} has frame IDs outside 1..{frame_count}.")
    return rows, hashlib.sha256(payload).hexdigest()


def _gt_geometry(rows: np.ndarray, geometry: str) -> np.ndarray:
    """Retain true oriented geometry, never enclosing OBBs in axis-aligned boxes."""

    if geometry == "aabb":
        boxes = rows[:, 2:6].copy()
        boxes[:, 2:4] += boxes[:, :2]
        return boxes
    boxes = np.empty((len(rows), 5), dtype=np.float64)
    for index, row in enumerate(rows):
        center, size, angle = cv2.minAreaRect(row[2:10].reshape(4, 2).astype(np.float32))
        boxes[index] = (*center, *size, np.deg2rad(angle))
    return boxes


def _match_detections(
    gt_boxes: np.ndarray, detection_boxes: np.ndarray, geometry: str
) -> tuple[np.ndarray, np.ndarray]:
    """Find a deterministic one-to-one IoU assignment above the fixed 0.5 gate."""

    if not len(gt_boxes) or not len(detection_boxes):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if geometry == "obb":
        similarities = _rotated_iou_batch(
            xywha_to_corners(gt_boxes).reshape(-1, 8),
            xywha_to_corners(detection_boxes).reshape(-1, 8),
        )
    else:
        gt_xywh = gt_boxes.copy()
        gt_xywh[:, 2:] -= gt_xywh[:, :2]
        det_xywh = detection_boxes.copy()
        det_xywh[:, 2:] -= det_xywh[:, :2]
        similarities = _aabb_iou_matrix(gt_xywh, det_xywh)
    # Gate before assignment: subthreshold overlaps cannot displace a valid pair.
    gated = np.where(similarities >= 0.5, similarities, 0.0)
    gt_indices, detection_indices = linear_sum_assignment(-gated)
    matched = gated[gt_indices, detection_indices] > 0.0
    return gt_indices[matched], detection_indices[matched]


def load_calibration_data(
    args: argparse.Namespace,
    *,
    progress: Callable[[str], None] | None = None,
) -> CalibrationData:
    """Match immutable cached predictions to the GT view selected by eval setup.

    This reads only sample metadata and detection geometry, scores, and classes.
    Dataset sampling has already remapped GT frame IDs in ``eval_setup``; this
    adapter therefore uses the delivered frame indices without resampling.
    """

    geometry = str(args.geometry)
    if geometry not in {"aabb", "obb"}:
        raise ValueError("KF calibration requires AABB or OBB geometry.")
    gt_folder = Path(args.gt_folder)
    settings = build_dataset_eval_settings(args, gt_folder, args.seq_info)
    selected_classes = set(map(int, settings["class_ids"])) - set(map(int, settings["distractor_ids"]))
    requested_classes = getattr(args, "classes", None)
    if requested_classes is not None:
        selected_classes.intersection_update(map(int, requested_classes))
    if not selected_classes:
        raise ValueError("KF calibration requires at least one target ground-truth class.")
    selected_sequences = tuple(sorted(getattr(args, "sequence_names", None) or args.seq_info))
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
    tracks, sources = [], []
    for sequence_id in selected_sequences:
        if progress is not None:
            progress(f"KF calibration: matching cached detections to GT for {sequence_id}…")
        dataset = CachedVisionDataset._for_sequence(
            args.build_path,
            sequence_id=sequence_id,
            split=args.split,
            load_images=False,
            load_embeddings=False,
            load_masks=False,
        )
        if dataset.manifest.box_type != geometry:
            raise ValueError(f"Calibration geometry {geometry!r} does not match the cached build.")
        frame_count = int(args.seq_info[sequence_id])
        if len(dataset) != frame_count:
            raise ValueError(f"Calibration sequence {sequence_id!r} has a different cached and GT frame count.")
        if geometry == "obb":
            gt_path = _resolve_obb_gt_path(
                Path(args.source),
                gt_folder,
                sequence_id,
                flat_annotations=args.evaluation_config.get("annotation_layout") == "flat",
                # Select the evaluator's first existing candidate, then parse it
                # strictly below instead of silently skipping malformed files.
                load_gt=lambda _path: None,
            )
        else:
            gt_path = _aabb_gt_path(gt_folder, settings["gt_loc_format"], sequence_id)
        rows, digest = _read_ground_truth(gt_path, geometry=geometry, frame_count=frame_count)
        sources.append({"sequence_id": sequence_id, "path": str(gt_path.resolve()), "sha256": digest})
        statistics["ground_truth_rows"] += len(rows)
        frame_rows = _index_rows_by_frame(rows, frame_count)
        # Only GT-present observations enter a track. Missing detector updates
        # stay explicit; a GT annotation gap is represented by a frame/time gap.
        observations: dict[tuple[int, int], list[tuple[Any, ...]]] = defaultdict(list)
        timestamps = []
        previous_timestamp = None
        for expected_index, sample in enumerate(dataset):
            if sample.frame_index != expected_index:
                raise ValueError(f"Calibration sequence {sequence_id!r} requires contiguous delivered frame indices.")
            timestamp = sample.timestamp_s
            if timestamp is not None:
                if not np.isfinite(timestamp) or (previous_timestamp is not None and timestamp <= previous_timestamp):
                    raise ValueError(f"Calibration sequence {sequence_id!r} requires finite increasing timestamps.")
                previous_timestamp = timestamp
            elif bool(getattr(args, "variable_dt", False)):
                raise ValueError(f"Variable-dt KF calibration requires timestamps for every frame in {sequence_id!r}.")
            timestamps.append(timestamp)
            detections = sample.detections
            detection_boxes = detections.geometry.values.numpy().astype(np.float64)
            detection_classes = detections.class_ids.numpy()
            detection_scores = detections.scores.numpy().astype(np.float64)
            statistics["frames"] += 1
            statistics["detections"] += len(detections)
            statistics["target_detections"] += int(np.isin(detection_classes, tuple(selected_classes)).sum())
            frame_gt = frame_rows[expected_index]
            class_column, valid_column = (11, 10) if geometry == "obb" else (7, 6)
            keep = (
                np.isin(frame_gt[:, class_column], tuple(selected_classes))
                & (frame_gt[:, valid_column] > 0)
                & (frame_gt[:, 1] >= 0)
            )
            eligible = frame_gt[keep]
            boxes = _gt_geometry(eligible, geometry)
            sizes = boxes[:, 2:4] if geometry == "obb" else boxes[:, 2:4] - boxes[:, :2]
            positive_size = np.all(sizes > 0, axis=1)
            eligible, boxes = eligible[positive_size], boxes[positive_size]
            statistics["filtered_ground_truth"] += len(frame_gt) - len(eligible)
            statistics["ground_truth"] += len(eligible)
            identities = eligible[:, [class_column, 1]].astype(np.int64)
            if len(np.unique(identities, axis=0)) != len(identities):
                raise ValueError(
                    f"Calibration ground truth {gt_path} repeats a class/identity in frame {expected_index + 1}."
                )
            for class_id in sorted(selected_classes):
                class_gt = np.flatnonzero(eligible[:, class_column] == class_id)
                class_detections = np.flatnonzero(detection_classes == class_id)
                gt_matches, det_matches = _match_detections(
                    boxes[class_gt], detection_boxes[class_detections], geometry
                )
                matched = dict(zip(class_gt[gt_matches], class_detections[det_matches], strict=True))
                statistics["matched"] += len(matched)
                for gt_index in class_gt:
                    det_index = matched.get(gt_index)
                    detected_box = np.full(boxes.shape[1], np.nan) if det_index is None else detection_boxes[det_index]
                    score = np.nan if det_index is None else detection_scores[det_index]
                    observations[(class_id, int(eligible[gt_index, 1]))].append(
                        (expected_index, boxes[gt_index], detected_box, score)
                    )
        complete_timestamps = all(timestamp is not None for timestamp in timestamps)
        sequence_timestamps = np.asarray(timestamps, dtype=np.float64) if complete_timestamps else None
        for (class_id, track_id), records in sorted(observations.items()):
            indices = np.asarray([record[0] for record in records], dtype=np.int64)
            tracks.append(
                CalibrationTrack(
                    sequence_id=sequence_id,
                    track_id=track_id,
                    class_id=class_id,
                    frame_indices=indices,
                    timestamps_s=None if sequence_timestamps is None else sequence_timestamps[indices],
                    gt_boxes=np.asarray([record[1] for record in records]),
                    detection_boxes=np.asarray([record[2] for record in records]),
                    scores=np.asarray([record[3] for record in records]),
                )
            )
    statistics["unmatched_ground_truth"] = statistics["ground_truth"] - statistics["matched"]
    statistics["unmatched_detections"] = statistics["target_detections"] - statistics["matched"]
    statistics["trajectories"] = len(tracks)
    if not tracks:
        raise ValueError("KF calibration found no eligible ground-truth trajectories on the selected split.")
    return CalibrationData(tuple(tracks), statistics, tuple(sources))
