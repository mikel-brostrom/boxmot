"""Evaluate boxes against the bounding boxes of KITTI MOTS instance PNGs.

This evaluates visible instance bounds, not the separate KITTI 2D tracking
annotations. Predictions use BoxMOT's nine-column MOT format. No predicted
masks or COCO RLE dependency are needed. Ignore pixels are unit squares:
their exact intersection with a continuous prediction box is divided by the
full, unclipped box area. Only unmatched boxes with more than 50% ignore
coverage are removed, following the MOTS matched-before-ignore convention.
That convention is adapted from TrackEval (MIT, copyright (c) 2020 Jonathon
Luiten); see ``licenses/TrackEval.txt`` beside this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.engine.eval.motmetrics import (
    MetricBundle,
    SequenceData,
    _aabb_iou_matrix,
    _append_aggregate_results,
    _combine_bundles,
    _eval_bundle,
    _format_results,
    _load_eval_cfg,
    _relabel_ids,
    _sequence_names_from_paths,
)
from boxmot.engine.eval.mots import GroundTruthFrame, _load_gt_labels, _resolve_class_pairs, _validated_gt_frames

_FLOAT_EPS = np.finfo(float).eps


def _read_gt_boxes(
    path: Path, height: int, width: int, *, cache_inputs: bool = False, cache_root: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return native instance IDs, exclusive bounds in xywh, and ignore pixels."""
    labels = _load_gt_labels(path, cache_inputs=cache_inputs, cache_root=cache_root)
    if labels.shape != (height, width):
        raise ValueError(
            f"KITTI MOTS instance PNG dimensions {labels.shape} do not match image dimensions {(height, width)}: {path}"
        )
    object_ids = np.unique(labels)
    valid = (object_ids == 0) | (object_ids == 10000) | ((object_ids >= 1000) & (object_ids < 3000))
    if not valid.all():
        raise ValueError(f"KITTI MOTS instance PNG contains unsupported labels {object_ids[~valid].tolist()}: {path}")
    object_ids = object_ids[(object_ids != 0) & (object_ids != 10000)].astype(np.int64)
    boxes = np.empty((len(object_ids), 4), dtype=float)
    for index, object_id in enumerate(object_ids):
        ys, xs = np.nonzero(labels == object_id)
        boxes[index] = (xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min())
    return object_ids, boxes, labels == 10000


def _read_box_results(path: Path, selected_frames: set[int]) -> dict[int, np.ndarray]:
    """Read strict AABB9 rows, converting one-based result frames to catalog IDs."""
    contents = path.read_text().strip()
    frames: dict[int, list[list[float]]] = {}
    seen: set[tuple[int, int]] = set()
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = [float(value) for value in line.split(",")]
        except ValueError as exc:
            raise ValueError(f"Invalid KITTI box result at {path}:{line_number}") from exc
        if len(row) != 9 or not np.isfinite(row).all():
            raise ValueError(f"KITTI box results require nine finite MOT columns at {path}:{line_number}")
        frame, identity, x, y, width, height, _, class_id, _ = row
        if any(value != int(value) for value in (frame, identity, class_id)):
            raise ValueError(f"KITTI box result frame, ID, and class must be integers at {path}:{line_number}")
        frame_index = int(frame) - 1
        if frame_index not in selected_frames:
            raise ValueError(f"KITTI box result frame {int(frame)} is outside selected catalog frames: {path}")
        if identity < 0 or identity >= 2**63:
            raise ValueError(f"KITTI box result ID must be a non-negative int64 at {path}:{line_number}")
        if class_id not in (1, 2):
            raise ValueError(f"KITTI box result class must be car (1) or pedestrian (2) at {path}:{line_number}")
        if (
            width <= 0
            or height <= 0
            or width * height <= 0
            or not np.isfinite([x + width, y + height, width * height]).all()
        ):
            raise ValueError(f"KITTI box result must have finite, positive box area at {path}:{line_number}")
        key = (frame_index, int(identity))
        if key in seen:
            raise ValueError(f"Duplicate KITTI box result ID {int(identity)} on frame {int(frame)}: {path}")
        seen.add(key)
        frames.setdefault(frame_index, []).append(row)
    return {frame: np.asarray(rows, dtype=float) for frame, rows in frames.items()}


def _box_ignore_ioa(boxes: np.ndarray, ignore_mask: np.ndarray) -> np.ndarray:
    """Integrate unit-square ignore pixels over continuous xywh prediction boxes."""
    if not len(boxes) or not np.any(ignore_mask):
        return np.zeros(len(boxes), dtype=float)
    height, width = ignore_mask.shape
    integral = np.pad(ignore_mask.astype(float).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))

    def area_to(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the exact continuous summed-area function at each point."""
        x, y = np.clip(x, 0, width), np.clip(y, 0, height)
        x0, y0 = x.astype(int), y.astype(int)
        x1, y1 = np.minimum(x0 + 1, width), np.minimum(y0 + 1, height)
        dx, dy = x - x0, y - y0
        return (
            integral[y0, x0] * (1 - dx) * (1 - dy)
            + integral[y0, x1] * dx * (1 - dy)
            + integral[y1, x0] * (1 - dx) * dy
            + integral[y1, x1] * dx * dy
        )

    x0, y0, box_width, box_height = boxes.T
    x1, y1 = x0 + box_width, y0 + box_height
    intersection = area_to(x1, y1) - area_to(x0, y1) - area_to(x1, y0) + area_to(x0, y0)
    return np.clip(intersection / (box_width * box_height), 0, 1)


def _preprocess_box_frame(
    gt_boxes: np.ndarray, tracker_ids: np.ndarray, tracker_boxes: np.ndarray, ignore_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Match one class by box IoU before suppressing unmatched ignored boxes."""
    similarity = _aabb_iou_matrix(gt_boxes, tracker_boxes)
    unmatched = np.arange(len(tracker_ids))
    if len(gt_boxes) and len(tracker_ids):
        matching_scores = similarity.copy()
        matching_scores[matching_scores < 0.5 - _FLOAT_EPS] = -10000
        rows, cols = linear_sum_assignment(-matching_scores)
        unmatched = np.delete(unmatched, cols[matching_scores[rows, cols] > _FLOAT_EPS])
    if len(unmatched):
        ignore_ioa = _box_ignore_ioa(tracker_boxes[unmatched], ignore_mask)
        removed = unmatched[ignore_ioa > 0.5 + _FLOAT_EPS]
        tracker_ids = np.delete(tracker_ids, removed)
        similarity = np.delete(similarity, removed, axis=1)
    return tracker_ids, similarity


def _build_kitti_box_sequence_data(
    seq_name: str,
    gt_frames: Sequence[GroundTruthFrame],
    tracker_path: Path,
    class_pairs: Sequence[tuple[str, int]],
    num_timesteps: int,
    *,
    cache_inputs: bool = False,
    cache_root: Path | None = None,
) -> dict[str, SequenceData]:
    """Decode each PNG once and collect metric-ready box similarities by class."""
    tracker_frames = _read_box_results(tracker_path, {index for index, _, _, _ in gt_frames})
    gt_ids_by_class = {name: [np.empty(0, dtype=int) for _ in range(num_timesteps)] for name, _ in class_pairs}
    tracker_ids_by_class = {name: [np.empty(0, dtype=int) for _ in range(num_timesteps)] for name, _ in class_pairs}
    similarities = {name: [np.empty((0, 0), dtype=float) for _ in range(num_timesteps)] for name, _ in class_pairs}
    for frame_index, path, height, width in gt_frames:
        options = {"cache_inputs": True, "cache_root": cache_root} if cache_inputs else {}
        gt_ids, gt_boxes, ignore_mask = _read_gt_boxes(path, height, width, **options)
        rows = tracker_frames.pop(frame_index, np.empty((0, 9), dtype=float))
        for name, class_id in class_pairs:
            selected_gt = gt_ids // 1000 == class_id
            selected_rows = rows[rows[:, 7] == class_id]
            frame_tracker_ids, similarity = _preprocess_box_frame(
                gt_boxes[selected_gt], selected_rows[:, 1].astype(np.int64), selected_rows[:, 2:6], ignore_mask
            )
            gt_ids_by_class[name][frame_index] = gt_ids[selected_gt]
            tracker_ids_by_class[name][frame_index] = frame_tracker_ids
            similarities[name][frame_index] = similarity

    data: dict[str, SequenceData] = {}
    for name, _ in class_pairs:
        sequence_gt_ids, num_gt_ids = _relabel_ids(gt_ids_by_class[name])
        sequence_tracker_ids, num_tracker_ids = _relabel_ids(tracker_ids_by_class[name])
        data[name] = SequenceData(
            seq=seq_name,
            gt_ids=sequence_gt_ids,
            tracker_ids=sequence_tracker_ids,
            similarity_scores=similarities[name],
            num_timesteps=num_timesteps,
            num_gt_dets=sum(map(len, sequence_gt_ids)),
            num_tracker_dets=sum(map(len, sequence_tracker_ids)),
            num_gt_ids=num_gt_ids,
            num_tracker_ids=num_tracker_ids,
        )
    return data


def run_kitti_box_metrics(
    args: argparse.Namespace,
    seq_paths: Sequence[Path],
    save_dir: Path,
    gt_folder: Path,
    *,
    seq_info: Mapping[str, int | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate box predictions against selected KITTI MOTS instance bounds.

    ``evaluation_config['mots_gt_frames']`` supplies exact catalog PNG paths,
    dimensions, and zero-based frame indices, including FPS remapping. Result
    files use one-based frames and otherwise retain BoxMOT's AABB9 format.
    """
    del save_dir
    config = _load_eval_cfg(args)
    sequences = _sequence_names_from_paths(seq_paths, seq_info)
    if not sequences:
        raise ValueError("No KITTI sequences selected for box evaluation")
    annotations = config.get("mots_gt_frames")
    if not isinstance(annotations, Mapping):
        raise ValueError("KITTI box evaluation requires catalog-resolved mots_gt_frames")
    class_pairs = _resolve_class_pairs(args, config)
    per_class_sequence: dict[str, dict[str, MetricBundle]] = {name: {} for name, _ in class_pairs}
    for seq_name, num_timesteps in sorted(sequences.items()):
        if seq_name not in annotations:
            raise ValueError(f"No KITTI ground-truth frame metadata for {seq_name}")
        frames, num_timesteps = _validated_gt_frames(seq_name, annotations[seq_name], num_timesteps)
        options = {}
        if getattr(args, "cache_inputs", False):
            options.update(cache_inputs=True, cache_root=Path(gt_folder) / ".boxmot/replay_cache/annotations")
        sequence_data = _build_kitti_box_sequence_data(
            seq_name, frames, Path(args.exp_dir) / f"{seq_name}.txt", class_pairs, num_timesteps, **options
        )
        for name, data in sequence_data.items():
            per_class_sequence[name][seq_name] = _eval_bundle(data)
    combined = {name: _combine_bundles(per_class_sequence[name]) for name, _ in class_pairs}
    results = _format_results(combined, per_class_sequence)
    _append_aggregate_results(results, combined, include_obb_super_categories=False)
    return results
