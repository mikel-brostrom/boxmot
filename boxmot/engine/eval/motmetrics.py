"""In-repo MOT metrics used by the BoxMOT evaluator.

The implementation mirrors BoxMOT's MOTChallenge report contract: HOTA, CLEAR,
Identity, and Count summaries for AABB and OBB tracking result files. It is
intentionally self-contained so evaluation does not require an external metrics
package installation.
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.data.benchmark import (
    COCO_CLASSES,
    _ordered_benchmark_eval_class_names,
    resolve_eval_box_type,
    resolve_obb_eval_class_pairs,
)
from boxmot.engine.workflows.benchmark import find_dataset_cfg_for_source, load_evaluation_config_from_args
from boxmot.utils import logger as LOGGER

HOTA_ALPHA_VALUES: tuple[float, ...] = tuple(float(value) for value in np.arange(0.05, 0.99, 0.05))
_FLOAT_EPS = np.finfo(float).eps

DEFAULT_OBB_CLASS_NAME_TO_ID = {
    "car": 0,
    "bike": 1,
    "pedestrian": 2,
    "van": 3,
    "truck": 4,
    "bus": 5,
    "tricycle": 6,
    "awning-bike": 7,
}
DEFAULT_OBB_SUPER_CATEGORIES = {
    "HUMAN": ["pedestrian"],
    "VEHICLE": ["car", "van", "truck", "bus"],
    "BIKE": ["bike", "tricycle", "awning-bike"],
}

_HOTA_ARRAY_FIELDS = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA")
_HOTA_COUNT_ARRAY_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
_HOTA_FLOAT_FIELDS = ("HOTA(0)", "LocA(0)", "HOTALocA(0)")
_CLEAR_INTEGER_FIELDS = (
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "IDt",
    "IDa",
    "IDm",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_Frames",
)
_CLEAR_FLOAT_FIELDS = ("MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR", "sMOTA")
_CLEAR_EXTRA_FLOAT_FIELDS = ("CLR_F1", "FP_per_frame", "MOTAL", "MOTP_sum")
_CLEAR_SUMMED_FIELDS = (*_CLEAR_INTEGER_FIELDS, "MOTP_sum")
_IDENTITY_INTEGER_FIELDS = ("IDTP", "IDFN", "IDFP")
_IDENTITY_FLOAT_FIELDS = ("IDF1", "IDR", "IDP")
_COUNT_INTEGER_FIELDS = ("Dets", "GT_Dets", "IDs", "GT_IDs", "Frames")
_MAX_DENSE_COUNT_BINS = 4_000_000


@dataclass(frozen=True)
class SequenceData:
    """Metric-ready sequence data for one class."""

    seq: str
    gt_ids: list[np.ndarray]
    tracker_ids: list[np.ndarray]
    similarity_scores: list[np.ndarray]
    num_timesteps: int
    num_gt_dets: int
    num_tracker_dets: int
    num_gt_ids: int
    num_tracker_ids: int


@dataclass(frozen=True)
class IndexedSequenceRows:
    """GT and tracker rows partitioned into constant-time frame lookups."""

    gt: list[np.ndarray]
    tracker: list[np.ndarray]
    num_timesteps: int


@dataclass(frozen=True)
class AABBSequenceEvaluationTask:
    """Pickle-safe input for evaluating every class in one AABB sequence."""

    seq_name: str
    gt_path: Path
    tracker_path: Path
    class_pairs: tuple[tuple[str, int], ...]
    num_timesteps: int | None
    distractor_ids: frozenset[int]


@dataclass(frozen=True)
class OBBSequenceEvaluationTask:
    """Pickle-safe input for evaluating every class in one OBB sequence."""

    seq_name: str
    source: Path
    gt_folder: Path
    tracker_path: Path
    class_pairs: tuple[tuple[str, int], ...]
    num_timesteps: int | None


SequenceEvaluationTask = AABBSequenceEvaluationTask | OBBSequenceEvaluationTask
MetricBundle = dict[str, dict[str, Any]]


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _to_float(row: Mapping[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def _combine_alpha_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Combine one HOTA alpha threshold across sequences."""
    rows = list(rows)
    num_detections = sum(_to_float(row, "num_detections") for row in rows)
    num_objects = sum(_to_float(row, "num_objects") for row in rows)
    num_false_positives = sum(_to_float(row, "num_false_positives") for row in rows)

    deta = num_detections / max(1.0, num_objects + num_false_positives)
    assa_weighted_sum = sum(_to_float(row, "assa_alpha") * _to_float(row, "num_detections") for row in rows)
    assre_weighted_sum = sum(_to_float(row, "assre_alpha") * _to_float(row, "num_detections") for row in rows)
    assa = assa_weighted_sum / max(1.0, num_detections)
    assre = assre_weighted_sum / max(1.0, num_detections)
    hota = math.sqrt(max(0.0, deta * assa))

    return {
        "deta_alpha": deta,
        "assa_alpha": assa,
        "assre_alpha": assre,
        "hota_alpha": hota,
        "num_detections": num_detections,
        "num_objects": num_objects,
        "num_false_positives": num_false_positives,
    }


def _summarize_alpha_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"HOTA": 0.0, "DetA": 0.0, "AssA": 0.0, "AssRe": 0.0, "HOTA(0)": 0.0}

    return {
        "HOTA": _mean([_to_float(row, "hota_alpha") for row in rows]),
        "DetA": _mean([_to_float(row, "deta_alpha") for row in rows]),
        "AssA": _mean([_to_float(row, "assa_alpha") for row in rows]),
        "AssRe": _mean([_to_float(row, "assre_alpha") for row in rows]),
        "HOTA(0)": _to_float(rows[0], "hota_alpha"),
    }


def _read_csv_matrix(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 0), dtype=float)
    try:
        data = np.loadtxt(path, delimiter=",")
    except ValueError:
        return np.empty((0, 0), dtype=float)
    if data.size == 0:
        return np.empty((0, 0), dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data.astype(float, copy=False)


def _frame_count(seq_info: Mapping[str, int | None], seq_name: str, *arrays: np.ndarray) -> int:
    explicit = seq_info.get(seq_name)
    if explicit:
        return int(explicit)
    max_frame = 0
    for data in arrays:
        if data.size and data.shape[1] > 0:
            max_frame = max(max_frame, int(np.max(data[:, 0])))
    return max_frame


def _index_rows_by_frame(data: np.ndarray, num_timesteps: int) -> list[np.ndarray]:
    """Partition rows by integer frame ID with one scan of the input matrix."""
    if num_timesteps <= 0:
        return []

    num_columns = data.shape[1] if data.ndim == 2 else 0
    if data.size == 0:
        empty = np.empty((0, num_columns), dtype=data.dtype)
        return [empty] * num_timesteps

    frame_ids = data[:, 0].astype(int)
    if np.any(frame_ids[1:] < frame_ids[:-1]):
        order = np.argsort(frame_ids, kind="stable")
        frame_ids = frame_ids[order]
        data = data[order]

    boundaries = np.searchsorted(frame_ids, np.arange(1, num_timesteps + 2))
    return [data[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]


def _index_sequence_rows(
    *,
    seq_name: str,
    seq_info: Mapping[str, int | None],
    gt: np.ndarray,
    tracker: np.ndarray,
) -> IndexedSequenceRows:
    """Create reusable per-frame views for one raw GT/tracker sequence pair."""
    num_timesteps = _frame_count(seq_info, seq_name, gt, tracker)
    return IndexedSequenceRows(
        gt=_index_rows_by_frame(gt, num_timesteps),
        tracker=_index_rows_by_frame(tracker, num_timesteps),
        num_timesteps=num_timesteps,
    )


def _relabel_ids(frame_ids: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    lengths = np.fromiter((len(ids) for ids in frame_ids), dtype=int, count=len(frame_ids))
    if lengths.sum() == 0:
        return [ids.astype(int, copy=False) for ids in frame_ids], 0

    joined = np.concatenate(frame_ids).astype(int, copy=False)
    unique_ids, inverse = np.unique(joined, return_inverse=True)
    relabeled = np.split(inverse, np.cumsum(lengths)[:-1])
    return relabeled, len(unique_ids)


def _aabb_iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=np.float32)

    gt_width = np.maximum(gt_boxes[:, 2], 0.0)
    gt_height = np.maximum(gt_boxes[:, 3], 0.0)
    tracker_width = np.maximum(tracker_boxes[:, 2], 0.0)
    tracker_height = np.maximum(tracker_boxes[:, 3], 0.0)
    intersection_width = np.maximum(
        np.minimum(gt_boxes[:, None, 0] + gt_width[:, None], tracker_boxes[None, :, 0] + tracker_width)
        - np.maximum(gt_boxes[:, None, 0], tracker_boxes[None, :, 0]),
        0.0,
    )
    intersection_height = np.maximum(
        np.minimum(gt_boxes[:, None, 1] + gt_height[:, None], tracker_boxes[None, :, 1] + tracker_height)
        - np.maximum(gt_boxes[:, None, 1], tracker_boxes[None, :, 1]),
        0.0,
    )
    intersection = intersection_width * intersection_height
    gt_area = gt_width * gt_height
    tr_area = tracker_width * tracker_height
    union = gt_area[:, None] + tr_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=intersection != 0.0)


def _polygons_to_rotated_rects(polygons: np.ndarray) -> tuple[list, np.ndarray]:
    rects = []
    areas = np.empty(len(polygons), dtype=np.float64)
    for index, polygon in enumerate(polygons):
        points = polygon.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(points)
        rects.append(rect)
        areas[index] = rect[1][0] * rect[1][1]
    return rects, areas


def _rotated_iou_batch(gt_dets: np.ndarray, tracker_dets: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between GT and tracker OBB corner rows."""
    if len(gt_dets) == 0 or len(tracker_dets) == 0:
        return np.zeros((len(gt_dets), len(tracker_dets)), dtype=np.float32)

    gt_rects, gt_areas = _polygons_to_rotated_rects(gt_dets)
    tracker_rects, tracker_areas = _polygons_to_rotated_rects(tracker_dets)
    scores = np.zeros((len(gt_dets), len(tracker_dets)), dtype=np.float32)
    eps = _FLOAT_EPS

    gt_points = gt_dets.reshape(-1, 4, 2)
    tracker_points = tracker_dets.reshape(-1, 4, 2)
    gt_min = gt_points.min(axis=1)
    gt_max = gt_points.max(axis=1)
    tracker_min = tracker_points.min(axis=1)
    tracker_max = tracker_points.max(axis=1)
    candidates = (
        (gt_min[:, np.newaxis, 0] < tracker_max[np.newaxis, :, 0])
        & (gt_max[:, np.newaxis, 0] > tracker_min[np.newaxis, :, 0])
        & (gt_min[:, np.newaxis, 1] < tracker_max[np.newaxis, :, 1])
        & (gt_max[:, np.newaxis, 1] > tracker_min[np.newaxis, :, 1])
        & (gt_areas[:, np.newaxis] > eps)
        & (tracker_areas[np.newaxis, :] > eps)
    )

    for gt_index, tracker_index in zip(*np.nonzero(candidates)):
        ret, intersection = cv2.rotatedRectangleIntersection(
            gt_rects[gt_index],
            tracker_rects[tracker_index],
        )
        if ret == cv2.INTERSECT_NONE or intersection is None or len(intersection) == 0:
            continue
        inter_area = float(cv2.contourArea(intersection))
        union = gt_areas[gt_index] + tracker_areas[tracker_index] - inter_area
        if union > eps:
            scores[gt_index, tracker_index] = inter_area / union
    return scores


def _load_obb_gt_matrix(source: Path) -> np.ndarray:
    """Load OBB GT in the 13-column MMOT corner format."""
    data = _read_csv_matrix(source)
    if data.size == 0:
        return np.empty((0, 13), dtype=np.float32)
    if data.shape[1] == 13:
        return data.astype(np.float32, copy=False)
    raise ValueError(
        f"Unsupported OBB GT format in {source}: expected 13 columns in corner format, got {data.shape[1]}"
    )


def _resolve_obb_gt_path(
    source: Path,
    gt_folder: Path,
    seq_name: str,
    *,
    load_gt: Callable[[Path], np.ndarray] = _load_obb_gt_matrix,
) -> Path:
    seq_dir = source / seq_name
    candidates = [
        source.parent / "mot" / f"{seq_name}.txt",
        seq_dir / "gt" / "gt_temp.txt",
        gt_folder / seq_name / "gt" / "gt_temp.txt",
        seq_dir / "gt" / "gt.txt",
        gt_folder / seq_name / "gt" / "gt.txt",
        seq_dir / "gt" / "gt_obb_raw_temp.txt",
        gt_folder / seq_name / "gt" / "gt_obb_raw_temp.txt",
        seq_dir / "gt" / "gt_obb_temp.txt",
        gt_folder / seq_name / "gt" / "gt_obb_temp.txt",
        seq_dir / "gt" / "gt_obb_raw.txt",
        gt_folder / seq_name / "gt" / "gt_obb_raw.txt",
        seq_dir / "gt" / "gt_obb.txt",
        gt_folder / seq_name / "gt" / "gt_obb.txt",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            load_gt(candidate)
        except ValueError:
            continue
        return candidate
    raise FileNotFoundError(
        f"No OBB GT file found for sequence {seq_name}. "
        "Expected gt.txt/gt_temp.txt or gt_obb*.txt in 13-column corner format."
    )


def _build_sequence_data(
    *,
    seq_name: str,
    num_timesteps: int,
    frame_loader: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> SequenceData:
    gt_ids_by_frame: list[np.ndarray] = []
    tracker_ids_by_frame: list[np.ndarray] = []
    similarity_scores: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for frame_id in range(1, num_timesteps + 1):
        gt_ids, tracker_ids, similarity = frame_loader(frame_id)
        gt_ids = np.asarray(gt_ids, dtype=int)
        tracker_ids = np.asarray(tracker_ids, dtype=int)
        gt_ids_by_frame.append(gt_ids)
        tracker_ids_by_frame.append(tracker_ids)
        similarity_scores.append(np.asarray(similarity, dtype=float))
        num_gt_dets += len(gt_ids)
        num_tracker_dets += len(tracker_ids)

    gt_ids_by_frame, num_gt_ids = _relabel_ids(gt_ids_by_frame)
    tracker_ids_by_frame, num_tracker_ids = _relabel_ids(tracker_ids_by_frame)
    return SequenceData(
        seq=seq_name,
        gt_ids=gt_ids_by_frame,
        tracker_ids=tracker_ids_by_frame,
        similarity_scores=similarity_scores,
        num_timesteps=num_timesteps,
        num_gt_dets=num_gt_dets,
        num_tracker_dets=num_tracker_dets,
        num_gt_ids=num_gt_ids,
        num_tracker_ids=num_tracker_ids,
    )


def _build_aabb_sequence_data(
    *,
    seq_name: str,
    gt_path: Path,
    tracker_path: Path,
    class_id: int | None,
    distractor_ids: set[int],
    seq_info: Mapping[str, int | None],
    gt_min_confidence: float | None = None,
    indexed_rows: IndexedSequenceRows | None = None,
) -> SequenceData:
    if indexed_rows is None:
        indexed_rows = _index_sequence_rows(
            seq_name=seq_name,
            seq_info=seq_info,
            gt=_read_csv_matrix(gt_path),
            tracker=_read_csv_matrix(tracker_path),
        )

    def _load_frame(frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gt_frame = indexed_rows.gt[frame_id - 1]
        tracker_frame = indexed_rows.tracker[frame_id - 1]

        gt_ids = gt_frame[:, 1].astype(int) if gt_frame.size else np.empty(0, dtype=int)
        gt_boxes = gt_frame[:, 2:6] if gt_frame.size else np.empty((0, 4), dtype=np.float32)
        gt_zero = gt_frame[:, 6] if gt_frame.size and gt_frame.shape[1] > 6 else np.ones(len(gt_ids))
        gt_classes = (
            gt_frame[:, 7].astype(int) if gt_frame.size and gt_frame.shape[1] > 7 else np.ones(len(gt_ids), int)
        )

        tracker_ids = tracker_frame[:, 1].astype(int) if tracker_frame.size else np.empty(0, dtype=int)
        tracker_boxes = tracker_frame[:, 2:6] if tracker_frame.size else np.empty((0, 4), dtype=np.float32)
        tracker_classes = (
            tracker_frame[:, 7].astype(int)
            if tracker_frame.size and tracker_frame.shape[1] > 7
            else np.ones(len(tracker_ids), int)
        )

        tracker_keep = np.ones(len(tracker_ids), dtype=bool) if class_id is None else tracker_classes == class_id
        kept_tracker_ids = tracker_ids[tracker_keep]
        kept_tracker_boxes = tracker_boxes[tracker_keep] if tracker_boxes.size else tracker_boxes
        similarity = _aabb_iou_matrix(gt_boxes, kept_tracker_boxes)

        if distractor_ids and len(gt_ids) and len(kept_tracker_ids) and similarity.size:
            matching_scores = similarity.copy()
            matching_scores[matching_scores < 0.5 - _FLOAT_EPS] = 0
            match_rows, match_cols = linear_sum_assignment(-matching_scores)
            actually_matched = matching_scores[match_rows, match_cols] > _FLOAT_EPS
            match_rows = match_rows[actually_matched]
            match_cols = match_cols[actually_matched]
            remove_cols = match_cols[np.isin(gt_classes[match_rows], list(distractor_ids))]
            if remove_cols.size:
                kept_tracker_ids = np.delete(kept_tracker_ids, remove_cols, axis=0)
                kept_tracker_boxes = np.delete(kept_tracker_boxes, remove_cols, axis=0)
                similarity = np.delete(similarity, remove_cols, axis=1)

        class_keep = np.ones(len(gt_ids), dtype=bool) if class_id is None else gt_classes == class_id
        if gt_min_confidence is None:
            gt_keep = (gt_zero != 0) & class_keep
        else:
            gt_keep = (gt_zero >= gt_min_confidence) & class_keep

        kept_gt_ids = gt_ids[gt_keep]
        similarity = similarity[gt_keep, :] if similarity.size else np.empty((len(kept_gt_ids), len(kept_tracker_ids)))
        return kept_gt_ids, kept_tracker_ids, similarity

    return _build_sequence_data(
        seq_name=seq_name,
        num_timesteps=indexed_rows.num_timesteps,
        frame_loader=_load_frame,
    )


def _build_obb_sequence_data(
    *,
    seq_name: str,
    gt_path: Path,
    tracker_path: Path,
    class_id: int,
    seq_info: Mapping[str, int | None],
    indexed_rows: IndexedSequenceRows | None = None,
) -> SequenceData:
    if indexed_rows is None:
        gt = _load_obb_gt_matrix(gt_path)
        tracker = _read_csv_matrix(tracker_path)
        if tracker.size and tracker.shape[1] != 13:
            raise ValueError(
                f"Unsupported OBB tracker format in {tracker_path}: expected 13 columns, got {tracker.shape[1]}"
            )
        indexed_rows = _index_sequence_rows(
            seq_name=seq_name,
            seq_info=seq_info,
            gt=gt,
            tracker=tracker,
        )

    def _load_frame(frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gt_frame = indexed_rows.gt[frame_id - 1]
        tracker_frame = indexed_rows.tracker[frame_id - 1]

        gt_classes = gt_frame[:, 11].astype(int) if gt_frame.size else np.empty(0, dtype=int)
        tracker_classes = tracker_frame[:, 11].astype(int) if tracker_frame.size else np.empty(0, dtype=int)
        gt_keep = gt_classes == class_id
        tracker_keep = tracker_classes == class_id

        gt_ids = gt_frame[:, 1].astype(int)[gt_keep] if gt_frame.size else np.empty(0, dtype=int)
        tracker_ids = tracker_frame[:, 1].astype(int)[tracker_keep] if tracker_frame.size else np.empty(0, dtype=int)
        gt_polygons = gt_frame[:, 2:10][gt_keep] if gt_frame.size else np.empty((0, 8), dtype=np.float32)
        tracker_polygons = (
            tracker_frame[:, 2:10][tracker_keep] if tracker_frame.size else np.empty((0, 8), dtype=np.float32)
        )
        return gt_ids, tracker_ids, _rotated_iou_batch(gt_polygons, tracker_polygons)

    return _build_sequence_data(
        seq_name=seq_name,
        num_timesteps=indexed_rows.num_timesteps,
        frame_loader=_load_frame,
    )


def _compute_final_hota_fields(res: dict[str, Any]) -> dict[str, Any]:
    res["DetRe"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"])
    res["DetPr"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FP"])
    res["DetA"] = res["HOTA_TP"] / np.maximum(1, res["HOTA_TP"] + res["HOTA_FN"] + res["HOTA_FP"])
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    res["HOTA(0)"] = float(res["HOTA"][0])
    res["LocA(0)"] = float(res["LocA"][0])
    res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
    return res


def _occurrence_counts(frame_ids: Sequence[np.ndarray], num_ids: int) -> np.ndarray:
    """Count compact IDs across all frames with one vectorized pass."""
    nonempty = [ids for ids in frame_ids if len(ids)]
    if not nonempty:
        return np.zeros(num_ids, dtype=float)
    return np.bincount(np.concatenate(nonempty), minlength=num_ids).astype(float, copy=False)


def _encoded_value_counts(encoded: np.ndarray, num_values: int) -> tuple[np.ndarray, np.ndarray]:
    """Count encoded integer values, bounding dense scratch memory for long sequences."""
    if num_values <= _MAX_DENSE_COUNT_BINS:
        dense_counts = np.bincount(encoded, minlength=num_values)
        nonzero = np.flatnonzero(dense_counts)
        return nonzero, dense_counts[nonzero]
    return np.unique(encoded, return_counts=True)


def _hota_association_scores(
    alpha_indices: np.ndarray,
    gt_indices: np.ndarray,
    tracker_indices: np.ndarray,
    match_counts: np.ndarray,
    gt_id_count: np.ndarray,
    tracker_id_count: np.ndarray,
    true_positives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sparse matched-ID counts for every HOTA threshold."""
    if not len(match_counts):
        zeros = np.zeros_like(true_positives)
        return zeros, zeros.copy(), zeros.copy()

    tp_denominator = np.maximum(1, true_positives)
    squared_counts = match_counts * match_counts

    def _aggregate(denominator: np.ndarray) -> np.ndarray:
        return (
            np.bincount(
                alpha_indices,
                weights=squared_counts / np.maximum(1, denominator),
                minlength=len(true_positives),
            )
            / tp_denominator
        )

    ass_a = _aggregate(gt_id_count[gt_indices] + tracker_id_count[tracker_indices] - match_counts)
    ass_re = _aggregate(gt_id_count[gt_indices])
    ass_pr = _aggregate(tracker_id_count[tracker_indices])
    return ass_a, ass_re, ass_pr


def _eval_hota(
    data: SequenceData,
    alpha_values: Sequence[float] = HOTA_ALPHA_VALUES,
    *,
    id_counts: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    alpha_values = np.asarray(alpha_values, dtype=float)
    res: dict[str, Any] = {
        field: np.zeros(len(alpha_values), dtype=float) for field in (*_HOTA_ARRAY_FIELDS, *_HOTA_COUNT_ARRAY_FIELDS)
    }
    for field in _HOTA_FLOAT_FIELDS:
        res[field] = 0.0

    if data.num_tracker_dets == 0:
        res["HOTA_FN"] = data.num_gt_dets * np.ones(len(alpha_values), dtype=float)
        res["LocA"] = np.ones(len(alpha_values), dtype=float)
        return _compute_final_hota_fields(res)
    if data.num_gt_dets == 0:
        res["HOTA_FP"] = data.num_tracker_dets * np.ones(len(alpha_values), dtype=float)
        res["LocA"] = np.ones(len(alpha_values), dtype=float)
        return _compute_final_hota_fields(res)

    potential_matches_count = np.zeros((data.num_gt_ids, data.num_tracker_ids), dtype=float)
    if id_counts is None:
        gt_id_count = _occurrence_counts(data.gt_ids, data.num_gt_ids)
        tracker_id_count = _occurrence_counts(data.tracker_ids, data.num_tracker_ids)
    else:
        gt_id_count, tracker_id_count = id_counts

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if similarity.size:
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            np.divide(similarity, sim_iou_denom, out=sim_iou, where=sim_iou_denom > _FLOAT_EPS)
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

    denom = gt_id_count[:, np.newaxis] + tracker_id_count - potential_matches_count
    global_alignment_score = np.divide(
        potential_matches_count,
        denom,
        out=np.zeros_like(potential_matches_count),
        where=denom > _FLOAT_EPS,
    )
    matched_gt_ids: list[np.ndarray] = []
    matched_tracker_ids: list[np.ndarray] = []
    matched_similarities: list[np.ndarray] = []
    minimum_alpha = float(np.min(alpha_values))

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
            continue

        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
        match_rows, match_cols = linear_sum_assignment(-score_mat)
        frame_similarities = similarity[match_rows, match_cols]
        eligible = frame_similarities >= minimum_alpha - _FLOAT_EPS
        if np.any(eligible):
            matched_gt_ids.append(gt_ids_t[match_rows[eligible]])
            matched_tracker_ids.append(tracker_ids_t[match_cols[eligible]])
            matched_similarities.append(frame_similarities[eligible])

    if matched_similarities:
        joined_gt_ids = np.concatenate(matched_gt_ids)
        joined_tracker_ids = np.concatenate(matched_tracker_ids)
        joined_similarities = np.concatenate(matched_similarities)
        valid_matches = joined_similarities[:, np.newaxis] >= alpha_values - _FLOAT_EPS
        res["HOTA_TP"] = valid_matches.sum(axis=0, dtype=float)
        res["LocA"] = np.sum(joined_similarities[:, np.newaxis] * valid_matches, axis=0)

        alpha_indices, match_indices = np.nonzero(valid_matches.T)
        num_id_pairs = data.num_gt_ids * data.num_tracker_ids
        pair_indices = joined_gt_ids * data.num_tracker_ids + joined_tracker_ids
        encoded_pairs = alpha_indices * num_id_pairs + pair_indices[match_indices]
        unique_pairs, counts = _encoded_value_counts(encoded_pairs, len(alpha_values) * num_id_pairs)
        pair_indices = unique_pairs % num_id_pairs
        res["AssA"], res["AssRe"], res["AssPr"] = _hota_association_scores(
            unique_pairs // num_id_pairs,
            pair_indices // data.num_tracker_ids,
            pair_indices % data.num_tracker_ids,
            counts.astype(float, copy=False),
            gt_id_count,
            tracker_id_count,
            res["HOTA_TP"],
        )

    res["HOTA_FN"] = data.num_gt_dets - res["HOTA_TP"]
    res["HOTA_FP"] = data.num_tracker_dets - res["HOTA_TP"]

    res["LocA"] = np.maximum(1e-10, res["LocA"]) / np.maximum(1e-10, res["HOTA_TP"])
    return _compute_final_hota_fields(res)


def _compute_final_clear_fields(res: dict[str, Any]) -> dict[str, Any]:
    num_gt_ids = res["MT"] + res["ML"] + res["PT"]
    res["MTR"] = res["MT"] / np.maximum(1.0, num_gt_ids)
    res["MLR"] = res["ML"] / np.maximum(1.0, num_gt_ids)
    res["PTR"] = res["PT"] / np.maximum(1.0, num_gt_ids)
    res["CLR_Re"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["CLR_Pr"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + res["CLR_FP"])
    res["MODA"] = (res["CLR_TP"] - res["CLR_FP"]) / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["MOTA"] = (res["CLR_TP"] - res["CLR_FP"] - res["IDSW"]) / np.maximum(1.0, res["CLR_TP"] + res["CLR_FN"])
    res["MOTP"] = res["MOTP_sum"] / np.maximum(1.0, res["CLR_TP"])
    res["sMOTA"] = (res["MOTP_sum"] - res["CLR_FP"] - res["IDSW"]) / np.maximum(
        1.0,
        res["CLR_TP"] + res["CLR_FN"],
    )
    res["CLR_F1"] = res["CLR_TP"] / np.maximum(1.0, res["CLR_TP"] + 0.5 * res["CLR_FN"] + 0.5 * res["CLR_FP"])
    res["FP_per_frame"] = res["CLR_FP"] / np.maximum(1.0, res["CLR_Frames"])
    safe_log_idsw = np.log10(res["IDSW"]) if res["IDSW"] > 0 else res["IDSW"]
    res["MOTAL"] = (res["CLR_TP"] - res["CLR_FP"] - safe_log_idsw) / np.maximum(
        1.0,
        res["CLR_TP"] + res["CLR_FN"],
    )
    return res


def _eval_clear(
    data: SequenceData,
    threshold: float = 0.5,
    *,
    gt_id_count: np.ndarray | None = None,
) -> dict[str, Any]:
    res: dict[str, Any] = {field: 0 for field in (*_CLEAR_INTEGER_FIELDS, *_CLEAR_FLOAT_FIELDS)}
    for field in _CLEAR_EXTRA_FLOAT_FIELDS:
        res[field] = 0.0

    if data.num_tracker_dets == 0:
        res["CLR_FN"] = data.num_gt_dets
        res["ML"] = data.num_gt_ids
        res["MLR"] = 1.0
        return res
    if data.num_gt_dets == 0:
        res["CLR_FP"] = data.num_tracker_dets
        res["MLR"] = 1.0
        return res

    if gt_id_count is None:
        gt_id_count = _occurrence_counts(data.gt_ids, data.num_gt_ids)
    gt_matched_count = np.zeros(data.num_gt_ids)
    gt_frag_count = np.zeros(data.num_gt_ids)
    prev_tracker_id = np.nan * np.zeros(data.num_gt_ids)
    prev_timestep_tracker_id = np.nan * np.zeros(data.num_gt_ids)
    prev_gt_id = np.full(data.num_tracker_ids, -1, dtype=int)
    gt_ever_matched = np.zeros(data.num_gt_ids, dtype=bool)
    tracker_ever_matched = np.zeros(data.num_tracker_ids, dtype=bool)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if len(gt_ids_t) == 0:
            res["CLR_FP"] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            res["CLR_FN"] += len(gt_ids_t)
            continue

        score_mat = tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - _FLOAT_EPS] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        matched = score_mat[match_rows, match_cols] > _FLOAT_EPS
        match_rows = match_rows[matched]
        match_cols = match_cols[matched]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
        is_idsw = (~np.isnan(prev_matched_tracker_ids)) & (matched_tracker_ids != prev_matched_tracker_ids)
        res["IDSW"] += int(np.sum(is_idsw))

        # MOTMetrics identity-transition diagnostics complement ID switches:
        # a transfer reuses a tracker ID for another GT identity; an ascend is
        # a switch to a never-before-matched tracker ID; and a migrate is a
        # transfer to a never-before-matched GT identity. Compute all masks
        # before updating the persistent assignment state so simultaneous
        # assignments in one frame cannot influence each other.
        previous_gt_ids = prev_gt_id[matched_tracker_ids]
        is_transfer = (previous_gt_ids >= 0) & (previous_gt_ids != matched_gt_ids)
        is_ascend = is_idsw & ~tracker_ever_matched[matched_tracker_ids]
        is_migrate = is_transfer & ~gt_ever_matched[matched_gt_ids]
        res["IDt"] += int(np.sum(is_transfer))
        res["IDa"] += int(np.sum(is_ascend))
        res["IDm"] += int(np.sum(is_migrate))

        gt_matched_count[matched_gt_ids] += 1
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
        prev_gt_id[matched_tracker_ids] = matched_gt_ids
        gt_ever_matched[matched_gt_ids] = True
        tracker_ever_matched[matched_tracker_ids] = True
        prev_timestep_tracker_id[:] = np.nan
        prev_timestep_tracker_id[matched_gt_ids] = matched_tracker_ids
        currently_tracked = ~np.isnan(prev_timestep_tracker_id)
        gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

        num_matches = len(matched_gt_ids)
        res["CLR_TP"] += num_matches
        res["CLR_FN"] += len(gt_ids_t) - num_matches
        res["CLR_FP"] += len(tracker_ids_t) - num_matches
        if num_matches:
            res["MOTP_sum"] += float(np.sum(similarity[match_rows, match_cols]))

    tracked_ratio = gt_matched_count[gt_id_count > 0] / gt_id_count[gt_id_count > 0]
    res["MT"] = int(np.sum(np.greater(tracked_ratio, 0.8)))
    res["PT"] = int(np.sum(np.greater_equal(tracked_ratio, 0.2))) - res["MT"]
    res["ML"] = data.num_gt_ids - res["MT"] - res["PT"]
    res["Frag"] = int(np.sum(np.subtract(gt_frag_count[gt_frag_count > 0], 1)))
    res["CLR_Frames"] = data.num_timesteps
    return _compute_final_clear_fields(res)


def _compute_final_identity_fields(res: dict[str, Any]) -> dict[str, Any]:
    res["IDR"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFN"])
    res["IDP"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + res["IDFP"])
    res["IDF1"] = res["IDTP"] / np.maximum(1.0, res["IDTP"] + 0.5 * res["IDFP"] + 0.5 * res["IDFN"])
    return res


def _eval_identity(data: SequenceData, threshold: float = 0.5) -> dict[str, Any]:
    res: dict[str, Any] = {field: 0 for field in (*_IDENTITY_INTEGER_FIELDS, *_IDENTITY_FLOAT_FIELDS)}
    if data.num_tracker_dets == 0:
        res["IDFN"] = data.num_gt_dets
        return res
    if data.num_gt_dets == 0:
        res["IDFP"] = data.num_tracker_dets
        return res

    num_id_pairs = data.num_gt_ids * data.num_tracker_ids
    encoded_edges: list[np.ndarray] = []

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        matches_mask = np.greater_equal(similarity, threshold)
        match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
        if len(match_idx_gt):
            encoded_edges.append(gt_ids_t[match_idx_gt] * data.num_tracker_ids + tracker_ids_t[match_idx_tracker])

    if encoded_edges:
        potential_matches_count = np.bincount(
            np.concatenate(encoded_edges),
            minlength=num_id_pairs,
        ).astype(float, copy=False)
    else:
        potential_matches_count = np.zeros(num_id_pairs, dtype=float)
    potential_matches_count = potential_matches_count.reshape(data.num_gt_ids, data.num_tracker_ids)

    # Relative to leaving both IDs unmatched, pairing a GT/tracker ID saves twice
    # their potential match count. The full dummy-node assignment therefore has
    # the same optimum as this rectangular maximum-weight assignment.
    match_rows, match_cols = linear_sum_assignment(potential_matches_count, maximize=True)
    res["IDTP"] = int(potential_matches_count[match_rows, match_cols].sum())
    res["IDFN"] = data.num_gt_dets - res["IDTP"]
    res["IDFP"] = data.num_tracker_dets - res["IDTP"]
    return _compute_final_identity_fields(res)


def _eval_count(data: SequenceData) -> dict[str, Any]:
    return {
        "Dets": data.num_tracker_dets,
        "GT_Dets": data.num_gt_dets,
        "IDs": data.num_tracker_ids,
        "GT_IDs": data.num_gt_ids,
        "Frames": data.num_timesteps,
    }


def _eval_bundle(data: SequenceData) -> MetricBundle:
    id_counts = (
        _occurrence_counts(data.gt_ids, data.num_gt_ids),
        _occurrence_counts(data.tracker_ids, data.num_tracker_ids),
    )
    return {
        "HOTA": _eval_hota(data, id_counts=id_counts),
        "CLEAR": _eval_clear(data, gt_id_count=id_counts[0]),
        "Identity": _eval_identity(data),
        "Count": _eval_count(data),
    }


def _combine_sum(all_res: Mapping[str, dict[str, Any]], field: str) -> Any:
    values = [value[field] for value in all_res.values()]
    if values and isinstance(values[0], np.ndarray):
        return np.sum(values, axis=0)
    return sum(values)


def _combine_weighted_average(
    all_res: Mapping[str, dict[str, Any]],
    field: str,
    combined: dict[str, Any],
    *,
    weight_field: str,
) -> np.ndarray:
    weighted_sum = sum(value[field] * value[weight_field] for value in all_res.values())
    return np.divide(
        weighted_sum,
        np.maximum(1e-10, combined[weight_field]),
        out=np.zeros_like(weighted_sum, dtype=float),
        where=np.maximum(1e-10, combined[weight_field]) > 0,
    )


def _combine_hota_sequences(all_res: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    res: dict[str, Any] = {field: _combine_sum(all_res, field) for field in _HOTA_COUNT_ARRAY_FIELDS}
    for field in ("AssRe", "AssPr", "AssA"):
        res[field] = _combine_weighted_average(all_res, field, res, weight_field="HOTA_TP")
    loca_weighted_sum = sum(value["LocA"] * value["HOTA_TP"] for value in all_res.values())
    res["LocA"] = np.maximum(1e-10, loca_weighted_sum) / np.maximum(1e-10, res["HOTA_TP"])
    return _compute_final_hota_fields(res)


def _combine_clear_sequences(all_res: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    res = {field: _combine_sum(all_res, field) for field in _CLEAR_SUMMED_FIELDS}
    return _compute_final_clear_fields(res)


def _combine_identity_sequences(all_res: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    res = {field: _combine_sum(all_res, field) for field in _IDENTITY_INTEGER_FIELDS}
    return _compute_final_identity_fields(res)


def _combine_count(all_res: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    return {field: _combine_sum(all_res, field) for field in _COUNT_INTEGER_FIELDS}


def _combine_bundles(all_bundles: Mapping[str, MetricBundle]) -> MetricBundle:
    return {
        "HOTA": _combine_hota_sequences({key: value["HOTA"] for key, value in all_bundles.items()}),
        "CLEAR": _combine_clear_sequences({key: value["CLEAR"] for key, value in all_bundles.items()}),
        "Identity": _combine_identity_sequences({key: value["Identity"] for key, value in all_bundles.items()}),
        "Count": _combine_count({key: value["Count"] for key, value in all_bundles.items()}),
    }


def _class_average_hota(all_res: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    res = {field: _combine_sum(all_res, field) for field in _HOTA_COUNT_ARRAY_FIELDS}
    for field in (*_HOTA_ARRAY_FIELDS, *_HOTA_FLOAT_FIELDS):
        res[field] = np.mean([value[field] for value in all_res.values()], axis=0)
    return res


def _class_average_clear(all_res: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    res = {field: _combine_sum(all_res, field) for field in _CLEAR_INTEGER_FIELDS}
    for field in (*_CLEAR_FLOAT_FIELDS, *_CLEAR_EXTRA_FLOAT_FIELDS):
        res[field] = float(np.mean([value[field] for value in all_res.values()]))
    return res


def _class_average_identity(all_res: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    res = {field: _combine_sum(all_res, field) for field in _IDENTITY_INTEGER_FIELDS}
    for field in _IDENTITY_FLOAT_FIELDS:
        res[field] = float(np.mean([value[field] for value in all_res.values()]))
    return res


def _combine_bundles_class_averaged(all_bundles: Mapping[str, MetricBundle]) -> MetricBundle:
    return {
        "HOTA": _class_average_hota({key: value["HOTA"] for key, value in all_bundles.items()}),
        "CLEAR": _class_average_clear({key: value["CLEAR"] for key, value in all_bundles.items()}),
        "Identity": _class_average_identity({key: value["Identity"] for key, value in all_bundles.items()}),
        "Count": _combine_count({key: value["Count"] for key, value in all_bundles.items()}),
    }


def _percent(value: Any) -> float:
    return max(0.0, float(value) * 100.0)


def _count(value: Any) -> int:
    return max(0, int(value))


def _summary_from_bundle(bundle: MetricBundle) -> dict[str, Any]:
    hota = bundle["HOTA"]
    clear = bundle["CLEAR"]
    identity = bundle["Identity"]
    count = bundle["Count"]

    summary: dict[str, Any] = {}
    for field in _HOTA_ARRAY_FIELDS:
        summary[field] = _percent(np.mean(hota[field]))
    for field in _HOTA_FLOAT_FIELDS:
        summary[field] = _percent(hota[field])
    for field in _HOTA_COUNT_ARRAY_FIELDS:
        summary[field] = _count(np.sum(hota[field]))

    for field in _CLEAR_FLOAT_FIELDS:
        summary[field] = _percent(clear[field])
    for field in _CLEAR_INTEGER_FIELDS:
        if field in clear:
            summary[field] = _count(clear[field])
    summary["MOTP_sum"] = float(clear.get("MOTP_sum", 0.0))

    for field in _IDENTITY_FLOAT_FIELDS:
        summary[field] = _percent(identity[field])
    for field in _IDENTITY_INTEGER_FIELDS:
        summary[field] = _count(identity[field])

    for field in _COUNT_INTEGER_FIELDS:
        summary[field] = _count(count[field])
    return summary


def _sequence_names_from_paths(
    seq_paths: Sequence[Path],
    seq_info: Mapping[str, int | None] | None,
) -> dict[str, int | None]:
    if seq_info:
        return dict(seq_info)
    names = [seq_path.parent.name if seq_path.name == "img1" else seq_path.name for seq_path in seq_paths]
    return {name: None for name in names}


def build_dataset_eval_settings(
    args: argparse.Namespace,
    gt_folder: Path,
    seq_info: dict[str, int | None],
) -> dict[str, Any]:
    """Derive benchmark-specific AABB evaluation settings."""
    del gt_folder
    cfg: dict[str, Any] = {}
    try:
        cfg = load_evaluation_config_from_args(args)
    except FileNotFoundError:
        cfg = {}
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(f"Error loading benchmark config: {exc}")
        cfg = {}

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    eval_classes_cfg = bench_cfg.get("eval_classes") if isinstance(bench_cfg, dict) else None
    distractor_cfg = bench_cfg.get("distractor_classes") if isinstance(bench_cfg, dict) else None
    ignore_dataset_ids = bench_cfg.get("ignore_dataset_ids") if isinstance(bench_cfg, dict) else None

    layout_name = str(cfg.get("layout") or bench_cfg.get("layout") or "").lower() if isinstance(cfg, dict) else ""
    gt_loc_format = "{gt_folder}/{seq}/gt/gt_temp.txt"
    if (
        layout_name == "visdrone"
        or "visdrone" in getattr(args, "benchmark", "").lower()
        or "visdrone" in str(getattr(args, "source", "")).lower()
    ):
        gt_loc_format = "{gt_folder}/{seq}.txt"

    if ignore_dataset_ids is not None:
        distractor_ids = [int(class_id) for class_id in ignore_dataset_ids]
    elif isinstance(distractor_cfg, dict) and distractor_cfg:
        distractor_ids = [int(k) for k in distractor_cfg.keys()]
    else:
        distractor_ids = []

    if getattr(args, "remapped_class_ids", None):
        return {
            "classes_to_eval": args.remapped_class_names,
            "class_ids": args.remapped_class_ids,
            "distractor_ids": distractor_ids,
            "gt_loc_format": gt_loc_format,
            "seq_info": seq_info,
        }

    classes_to_eval: list[str] = []
    class_ids: list[int] = []

    if hasattr(args, "classes") and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        classes_to_eval = [COCO_CLASSES[int(index)] for index in class_indices]
        class_ids = [int(index) + 1 for index in class_indices]

    if isinstance(eval_classes_cfg, dict) and eval_classes_cfg:
        ordered = sorted(((int(k), v) for k, v in eval_classes_cfg.items()), key=lambda kv: kv[0])
        if class_ids:
            class_ids = [class_id for class_id, _ in ordered if class_id in class_ids]
            classes_to_eval = [name for class_id, name in ordered if class_id in class_ids]
        else:
            class_ids = [class_id for class_id, _ in ordered]
            classes_to_eval = [str(name) for _, name in ordered]

    if not classes_to_eval:
        classes_to_eval = ["person"]
    if not class_ids:
        class_ids = [1]

    pairs: list[tuple[str, int]] = []
    seen: set[str] = set()
    for name, class_id in zip(classes_to_eval, class_ids):
        normalized = str(name).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        pairs.append((normalized, int(class_id)))

    return {
        "classes_to_eval": [name for name, _ in pairs],
        "class_ids": [class_id for _, class_id in pairs],
        "distractor_ids": distractor_ids,
        "gt_loc_format": gt_loc_format,
        "seq_info": seq_info,
    }


def _load_eval_cfg(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_evaluation_config_from_args(args)
    if cfg:
        return cfg

    cfg = find_dataset_cfg_for_source(args.source) or {}
    if cfg:
        return cfg
    LOGGER.warning(f"Could not infer a dataset config for {args.source}. Class filtering might be incorrect.")
    return {}


def _aabb_gt_path(gt_folder: Path, gt_loc_format: str, seq_name: str) -> Path:
    return Path(gt_loc_format.format(gt_folder=gt_folder, seq=seq_name))


def _evaluate_aabb_sequence_task(task: AABBSequenceEvaluationTask) -> tuple[str, dict[str, MetricBundle]]:
    """Load and evaluate all requested classes for one AABB sequence."""
    seq_info = {task.seq_name: task.num_timesteps}
    distractor_ids = set(task.distractor_ids)
    indexed_rows = _index_sequence_rows(
        seq_name=task.seq_name,
        seq_info=seq_info,
        gt=_read_csv_matrix(task.gt_path),
        tracker=_read_csv_matrix(task.tracker_path),
    )
    class_results = {
        class_name: _eval_bundle(
            _build_aabb_sequence_data(
                seq_name=task.seq_name,
                gt_path=task.gt_path,
                tracker_path=task.tracker_path,
                class_id=class_id,
                distractor_ids=distractor_ids,
                seq_info=seq_info,
                indexed_rows=indexed_rows,
            )
        )
        for class_name, class_id in task.class_pairs
    }
    return task.seq_name, class_results


def _evaluate_obb_sequence_task(task: OBBSequenceEvaluationTask) -> tuple[str, dict[str, MetricBundle]]:
    """Load and evaluate all requested classes for one OBB sequence."""
    seq_info = {task.seq_name: task.num_timesteps}
    gt_matrices: dict[Path, np.ndarray] = {}

    def _load_gt(path: Path) -> np.ndarray:
        if path not in gt_matrices:
            gt_matrices[path] = _load_obb_gt_matrix(path)
        return gt_matrices[path]

    gt_path = _resolve_obb_gt_path(task.source, task.gt_folder, task.seq_name, load_gt=_load_gt)
    tracker = _read_csv_matrix(task.tracker_path)
    if tracker.size and tracker.shape[1] != 13:
        raise ValueError(
            f"Unsupported OBB tracker format in {task.tracker_path}: expected 13 columns, got {tracker.shape[1]}"
        )
    indexed_rows = _index_sequence_rows(
        seq_name=task.seq_name,
        seq_info=seq_info,
        gt=_load_gt(gt_path),
        tracker=tracker,
    )
    class_results = {
        class_name: _eval_bundle(
            _build_obb_sequence_data(
                seq_name=task.seq_name,
                gt_path=gt_path,
                tracker_path=task.tracker_path,
                class_id=class_id,
                seq_info=seq_info,
                indexed_rows=indexed_rows,
            )
        )
        for class_name, class_id in task.class_pairs
    }
    return task.seq_name, class_results


def _evaluate_sequence_task(task: SequenceEvaluationTask) -> tuple[str, dict[str, MetricBundle]]:
    """Evaluate one sequence in either the caller or a worker process."""
    if isinstance(task, AABBSequenceEvaluationTask):
        return _evaluate_aabb_sequence_task(task)
    return _evaluate_obb_sequence_task(task)


def _fast_process_context() -> BaseContext:
    """Prefer copy-on-write sequence workers on POSIX, matching motmetrics."""
    import multiprocessing

    if os.name == "posix" and "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _metric_worker_count(num_sequences: int, cpu_count: int | None = None) -> int:
    """Reserve two logical CPUs and never allocate more workers than sequences."""
    available_cpus = os.cpu_count() if cpu_count is None else cpu_count
    return min(num_sequences, max(1, (available_cpus or 1) - 2))


def _evaluate_sequence_tasks(
    tasks: Sequence[SequenceEvaluationTask],
) -> list[tuple[str, dict[str, MetricBundle]]]:
    """Evaluate sequence tasks serially or with an ordered process pool."""
    workers = _metric_worker_count(len(tasks))
    LOGGER.debug(f"Evaluating {len(tasks)} MOT sequence(s) with {max(1, workers)} metric worker process(es)")
    if workers <= 1:
        return [_evaluate_sequence_task(task) for task in tasks]

    import multiprocessing

    if multiprocessing.current_process().daemon:
        LOGGER.debug("Metric evaluation is already inside a daemon process; using the serial sequence path")
        return [_evaluate_sequence_task(task) for task in tasks]

    context = _fast_process_context()
    with context.Pool(processes=workers) as pool:
        return pool.map(_evaluate_sequence_task, tasks, chunksize=1)


def _evaluate_class_sequences(
    *,
    class_pairs: Sequence[tuple[str, int]],
    tasks: Sequence[SequenceEvaluationTask],
) -> tuple[dict[str, MetricBundle], dict[str, dict[str, MetricBundle]]]:
    per_class_sequence: dict[str, dict[str, MetricBundle]] = {class_name: {} for class_name, _ in class_pairs}

    for seq_name, class_results in _evaluate_sequence_tasks(tasks):
        for class_name, _ in class_pairs:
            per_class_sequence[class_name][seq_name] = class_results[class_name]

    class_combined = {class_name: _combine_bundles(per_class_sequence[class_name]) for class_name, _ in class_pairs}

    return class_combined, per_class_sequence


def _format_results(
    class_combined: Mapping[str, MetricBundle],
    per_class_sequence: Mapping[str, Mapping[str, MetricBundle]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for class_name, combined in class_combined.items():
        summary = _summary_from_bundle(combined)
        summary["per_sequence"] = {
            seq_name: _summary_from_bundle(bundle)
            for seq_name, bundle in per_class_sequence.get(class_name, {}).items()
        }
        results[class_name] = summary
    return results


def _append_aggregate_results(
    results: dict[str, dict[str, Any]],
    class_combined: Mapping[str, MetricBundle],
    *,
    include_obb_super_categories: bool,
) -> None:
    if len(class_combined) <= 1:
        return

    results["cls_comb_cls_av"] = {
        **_summary_from_bundle(_combine_bundles_class_averaged(class_combined)),
        "per_sequence": {},
    }
    results["cls_comb_det_av"] = {
        **_summary_from_bundle(_combine_bundles(class_combined)),
        "per_sequence": {},
    }
    if not include_obb_super_categories:
        return

    for super_name, members in DEFAULT_OBB_SUPER_CATEGORIES.items():
        selected = {name: class_combined[name] for name in members if name in class_combined}
        if selected:
            results[super_name] = {**_summary_from_bundle(_combine_bundles(selected)), "per_sequence": {}}


def run_motmetrics(
    args: argparse.Namespace,
    seq_paths: Sequence[Path],
    save_dir: Path,
    gt_folder: Path,
    *,
    seq_info: Mapping[str, int | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate MOT result text files with the in-repo motmetrics implementation."""
    del save_dir
    seq_info = _sequence_names_from_paths(seq_paths, seq_info)
    cfg = _load_eval_cfg(args)
    eval_box_type = resolve_eval_box_type(args, cfg)
    if eval_box_type == "obb":
        bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
        class_pairs = resolve_obb_eval_class_pairs(args, bench_cfg)
        if not class_pairs:
            class_pairs = list(DEFAULT_OBB_CLASS_NAME_TO_ID.items())
        tasks: list[SequenceEvaluationTask] = [
            OBBSequenceEvaluationTask(
                seq_name=seq_name,
                source=Path(args.source),
                gt_folder=Path(gt_folder),
                tracker_path=Path(args.exp_dir) / f"{seq_name}.txt",
                class_pairs=tuple(class_pairs),
                num_timesteps=seq_info[seq_name],
            )
            for seq_name in sorted(seq_info)
        ]

        class_combined, per_class_sequence = _evaluate_class_sequences(
            class_pairs=class_pairs,
            tasks=tasks,
        )
        results = _format_results(class_combined, per_class_sequence)
        _append_aggregate_results(results, class_combined, include_obb_super_categories=True)
        return results

    settings = build_dataset_eval_settings(args, gt_folder, dict(seq_info))
    class_pairs = list(zip(settings["classes_to_eval"], settings["class_ids"]))
    distractor_ids = set(settings.get("distractor_ids") or [])
    gt_loc_format = settings["gt_loc_format"]
    tasks = [
        AABBSequenceEvaluationTask(
            seq_name=seq_name,
            gt_path=_aabb_gt_path(gt_folder, gt_loc_format, seq_name),
            tracker_path=Path(args.exp_dir) / f"{seq_name}.txt",
            class_pairs=tuple(class_pairs),
            num_timesteps=seq_info[seq_name],
            distractor_ids=frozenset(distractor_ids),
        )
        for seq_name in sorted(seq_info)
    ]

    class_combined, per_class_sequence = _evaluate_class_sequences(
        class_pairs=class_pairs,
        tasks=tasks,
    )
    results = _format_results(class_combined, per_class_sequence)
    _append_aggregate_results(results, class_combined, include_obb_super_categories=False)
    return results


def evaluate_motchallenge_hota(
    sequence_files: Mapping[str, tuple[str | Path, str | Path]],
    *,
    alpha_values: Sequence[float] = HOTA_ALPHA_VALUES,
    gt_min_confidence: float = 1.0,
) -> dict[str, Any]:
    """Evaluate MOTChallenge text files with the BoxMOT HOTA implementation.

    This helper returns 0..1 ratios for parity tests. The CLI evaluator scales
    report metrics to percentages separately.
    """
    if not sequence_files:
        raise ValueError("sequence_files must contain at least one sequence")

    sequence_rows: dict[str, list[dict[str, float]]] = {}
    sequence_summaries: dict[str, dict[str, float]] = {}

    for name, (gt_path, tracker_path) in sequence_files.items():
        seq_data = _build_aabb_sequence_data(
            seq_name=name,
            gt_path=Path(gt_path),
            tracker_path=Path(tracker_path),
            class_id=None,
            distractor_ids=set(),
            seq_info={name: None},
            gt_min_confidence=gt_min_confidence,
        )
        hota = _eval_hota(seq_data, alpha_values=alpha_values)
        rows = [
            {
                "deta_alpha": float(hota["DetA"][index]),
                "assa_alpha": float(hota["AssA"][index]),
                "assre_alpha": float(hota["AssRe"][index]),
                "hota_alpha": float(hota["HOTA"][index]),
                "num_detections": float(hota["HOTA_TP"][index]),
                "num_objects": float(hota["HOTA_TP"][index] + hota["HOTA_FN"][index]),
                "num_false_positives": float(hota["HOTA_FP"][index]),
            }
            for index in range(len(alpha_values))
        ]
        sequence_rows[name] = rows
        sequence_summaries[name] = _summarize_alpha_rows(rows)

    combined_rows = [
        _combine_alpha_rows(rows[alpha_index] for rows in sequence_rows.values())
        for alpha_index in range(len(alpha_values))
    ]
    combined = _summarize_alpha_rows(combined_rows)
    combined["per_sequence"] = sequence_summaries
    return combined


def _known_motmetrics_class_names(args: argparse.Namespace, cfg: dict) -> list[str]:
    known: list[str] = []
    if getattr(args, "remapped_class_names", None):
        known.extend([str(name) for name in args.remapped_class_names])
    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    known.extend(_ordered_benchmark_eval_class_names(bench_cfg))
    known.extend(["cls_comb_cls_av", "cls_comb_det_av", "HUMAN", "VEHICLE", "BIKE", "all"])

    deduped: list[str] = []
    seen: set[str] = set()
    for name in known:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


__all__ = [
    "DEFAULT_OBB_CLASS_NAME_TO_ID",
    "DEFAULT_OBB_SUPER_CATEGORIES",
    "HOTA_ALPHA_VALUES",
    "_combine_alpha_rows",
    "_known_motmetrics_class_names",
    "_load_obb_gt_matrix",
    "build_dataset_eval_settings",
    "evaluate_motchallenge_hota",
    "run_motmetrics",
]
