"""In-repo MOT metrics used by the BoxMOT evaluator.

The implementation mirrors BoxMOT's MOTChallenge report contract: HOTA, CLEAR,
Identity, and Count summaries for AABB and OBB tracking result files. It is
intentionally self-contained so evaluation does not require an external metrics
package installation.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from boxmot.configs.benchmark import load_benchmark_cfg
from boxmot.data.benchmark import (
    COCO_CLASSES,
    _ordered_benchmark_eval_class_names,
    load_benchmark_cfg_from_args,
    resolve_eval_box_type,
    resolve_obb_eval_class_pairs,
)
from boxmot.utils import BENCHMARK_CONFIGS
from boxmot.utils import logger as LOGGER

HOTA_ALPHA_VALUES: tuple[float, ...] = tuple(float(value) for value in np.arange(0.05, 0.99, 0.05))

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
_CLEAR_INTEGER_FIELDS = ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "MT", "PT", "ML", "Frag", "CLR_Frames")
_CLEAR_FLOAT_FIELDS = ("MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR", "sMOTA")
_CLEAR_EXTRA_FLOAT_FIELDS = ("CLR_F1", "FP_per_frame", "MOTAL", "MOTP_sum")
_CLEAR_SUMMED_FIELDS = (*_CLEAR_INTEGER_FIELDS, "MOTP_sum")
_IDENTITY_INTEGER_FIELDS = ("IDTP", "IDFN", "IDFP")
_IDENTITY_FLOAT_FIELDS = ("IDF1", "IDR", "IDP")
_COUNT_INTEGER_FIELDS = ("Dets", "GT_Dets", "IDs", "GT_IDs", "Frames")


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


def _rows_for_frame(data: np.ndarray, frame_id: int) -> np.ndarray:
    if data.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    return data[data[:, 0].astype(int) == frame_id]


def _relabel_ids(frame_ids: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    unique_ids: list[int] = []
    for ids in frame_ids:
        unique_ids.extend(int(value) for value in np.unique(ids.astype(int)))

    if not unique_ids:
        return [ids.astype(int, copy=False) for ids in frame_ids], 0

    id_map = {raw_id: index for index, raw_id in enumerate(sorted(set(unique_ids)))}
    relabeled = [
        np.asarray([id_map[int(value)] for value in ids], dtype=int)
        if len(ids)
        else np.empty(0, dtype=int)
        for ids in frame_ids
    ]
    return relabeled, len(id_map)


def _aabb_iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=np.float32)

    gt_x1y1 = gt_boxes[:, :2]
    gt_x2y2 = gt_boxes[:, :2] + np.maximum(gt_boxes[:, 2:4], 0.0)
    tr_x1y1 = tracker_boxes[:, :2]
    tr_x2y2 = tracker_boxes[:, :2] + np.maximum(tracker_boxes[:, 2:4], 0.0)

    inter_x1y1 = np.maximum(gt_x1y1[:, None, :], tr_x1y1[None, :, :])
    inter_x2y2 = np.minimum(gt_x2y2[:, None, :], tr_x2y2[None, :, :])
    inter_wh = np.maximum(0.0, inter_x2y2 - inter_x1y1)
    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    gt_area = np.maximum(gt_boxes[:, 2], 0.0) * np.maximum(gt_boxes[:, 3], 0.0)
    tr_area = np.maximum(tracker_boxes[:, 2], 0.0) * np.maximum(tracker_boxes[:, 3], 0.0)
    union = gt_area[:, None] + tr_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


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
    eps = np.finfo(float).eps

    for gt_index, rect_a in enumerate(gt_rects):
        if gt_areas[gt_index] <= eps:
            continue
        for tracker_index, rect_b in enumerate(tracker_rects):
            if tracker_areas[tracker_index] <= eps:
                continue
            ret, intersection = cv2.rotatedRectangleIntersection(rect_a, rect_b)
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


def _resolve_obb_gt_path(args: argparse.Namespace, gt_folder: Path, seq_name: str) -> Path:
    seq_dir = Path(args.source) / seq_name
    candidates = [
        Path(args.source).parent / "mot" / f"{seq_name}.txt",
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
            _load_obb_gt_matrix(candidate)
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
) -> SequenceData:
    gt = _read_csv_matrix(gt_path)
    tracker = _read_csv_matrix(tracker_path)
    num_timesteps = _frame_count(seq_info, seq_name, gt, tracker)

    def _load_frame(frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gt_frame = _rows_for_frame(gt, frame_id)
        tracker_frame = _rows_for_frame(tracker, frame_id)

        gt_ids = gt_frame[:, 1].astype(int) if gt_frame.size else np.empty(0, dtype=int)
        gt_boxes = gt_frame[:, 2:6] if gt_frame.size else np.empty((0, 4), dtype=np.float32)
        gt_zero = gt_frame[:, 6] if gt_frame.size and gt_frame.shape[1] > 6 else np.ones(len(gt_ids))
        gt_classes = (
            gt_frame[:, 7].astype(int)
            if gt_frame.size and gt_frame.shape[1] > 7
            else np.ones(len(gt_ids), int)
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
            matching_scores[matching_scores < 0.5 - np.finfo(float).eps] = 0
            match_rows, match_cols = linear_sum_assignment(-matching_scores)
            actually_matched = matching_scores[match_rows, match_cols] > np.finfo(float).eps
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

    return _build_sequence_data(seq_name=seq_name, num_timesteps=num_timesteps, frame_loader=_load_frame)


def _build_obb_sequence_data(
    *,
    seq_name: str,
    gt_path: Path,
    tracker_path: Path,
    class_id: int,
    seq_info: Mapping[str, int | None],
) -> SequenceData:
    gt = _load_obb_gt_matrix(gt_path)
    tracker = _read_csv_matrix(tracker_path)
    if tracker.size and tracker.shape[1] != 13:
        raise ValueError(
            f"Unsupported OBB tracker format in {tracker_path}: expected 13 columns, got {tracker.shape[1]}"
        )
    num_timesteps = _frame_count(seq_info, seq_name, gt, tracker)

    def _load_frame(frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gt_frame = _rows_for_frame(gt, frame_id)
        tracker_frame = _rows_for_frame(tracker, frame_id)

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

    return _build_sequence_data(seq_name=seq_name, num_timesteps=num_timesteps, frame_loader=_load_frame)


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


def _eval_hota(data: SequenceData, alpha_values: Sequence[float] = HOTA_ALPHA_VALUES) -> dict[str, Any]:
    res: dict[str, Any] = {
        field: np.zeros(len(alpha_values), dtype=float)
        for field in (*_HOTA_ARRAY_FIELDS, *_HOTA_COUNT_ARRAY_FIELDS)
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
    gt_id_count = np.zeros((data.num_gt_ids, 1), dtype=float)
    tracker_id_count = np.zeros((1, data.num_tracker_ids), dtype=float)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if similarity.size:
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > np.finfo(float).eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1

    denom = gt_id_count + tracker_id_count - potential_matches_count
    global_alignment_score = np.divide(
        potential_matches_count,
        denom,
        out=np.zeros_like(potential_matches_count),
        where=denom > np.finfo(float).eps,
    )
    matches_counts = [np.zeros_like(potential_matches_count) for _ in alpha_values]

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if len(gt_ids_t) == 0:
            for alpha_index in range(len(alpha_values)):
                res["HOTA_FP"][alpha_index] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            for alpha_index in range(len(alpha_values)):
                res["HOTA_FN"][alpha_index] += len(gt_ids_t)
            continue

        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        for alpha_index, alpha in enumerate(alpha_values):
            matched = similarity[match_rows, match_cols] >= alpha - np.finfo(float).eps
            alpha_match_rows = match_rows[matched]
            alpha_match_cols = match_cols[matched]
            num_matches = len(alpha_match_rows)
            res["HOTA_TP"][alpha_index] += num_matches
            res["HOTA_FN"][alpha_index] += len(gt_ids_t) - num_matches
            res["HOTA_FP"][alpha_index] += len(tracker_ids_t) - num_matches
            if num_matches:
                res["LocA"][alpha_index] += float(np.sum(similarity[alpha_match_rows, alpha_match_cols]))
                matches_counts[alpha_index][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1

    for alpha_index in range(len(alpha_values)):
        matches_count = matches_counts[alpha_index]
        ass_a = matches_count / np.maximum(1, gt_id_count + tracker_id_count - matches_count)
        res["AssA"][alpha_index] = np.sum(matches_count * ass_a) / np.maximum(1, res["HOTA_TP"][alpha_index])
        ass_re = matches_count / np.maximum(1, gt_id_count)
        res["AssRe"][alpha_index] = np.sum(matches_count * ass_re) / np.maximum(1, res["HOTA_TP"][alpha_index])
        ass_pr = matches_count / np.maximum(1, tracker_id_count)
        res["AssPr"][alpha_index] = np.sum(matches_count * ass_pr) / np.maximum(1, res["HOTA_TP"][alpha_index])

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


def _eval_clear(data: SequenceData, threshold: float = 0.5) -> dict[str, Any]:
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

    gt_id_count = np.zeros(data.num_gt_ids)
    gt_matched_count = np.zeros(data.num_gt_ids)
    gt_frag_count = np.zeros(data.num_gt_ids)
    prev_tracker_id = np.nan * np.zeros(data.num_gt_ids)
    prev_timestep_tracker_id = np.nan * np.zeros(data.num_gt_ids)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        if len(gt_ids_t) == 0:
            res["CLR_FP"] += len(tracker_ids_t)
            continue
        if len(tracker_ids_t) == 0:
            res["CLR_FN"] += len(gt_ids_t)
            gt_id_count[gt_ids_t] += 1
            continue

        score_mat = tracker_ids_t[np.newaxis, :] == prev_timestep_tracker_id[gt_ids_t[:, np.newaxis]]
        score_mat = 1000 * score_mat + similarity
        score_mat[similarity < threshold - np.finfo(float).eps] = 0

        match_rows, match_cols = linear_sum_assignment(-score_mat)
        matched = score_mat[match_rows, match_cols] > np.finfo(float).eps
        match_rows = match_rows[matched]
        match_cols = match_cols[matched]

        matched_gt_ids = gt_ids_t[match_rows]
        matched_tracker_ids = tracker_ids_t[match_cols]
        prev_matched_tracker_ids = prev_tracker_id[matched_gt_ids]
        is_idsw = (~np.isnan(prev_matched_tracker_ids)) & (matched_tracker_ids != prev_matched_tracker_ids)
        res["IDSW"] += int(np.sum(is_idsw))

        gt_id_count[gt_ids_t] += 1
        gt_matched_count[matched_gt_ids] += 1
        not_previously_tracked = np.isnan(prev_timestep_tracker_id)
        prev_tracker_id[matched_gt_ids] = matched_tracker_ids
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

    potential_matches_count = np.zeros((data.num_gt_ids, data.num_tracker_ids))
    gt_id_count = np.zeros(data.num_gt_ids)
    tracker_id_count = np.zeros(data.num_tracker_ids)

    for gt_ids_t, tracker_ids_t, similarity in zip(data.gt_ids, data.tracker_ids, data.similarity_scores):
        matches_mask = np.greater_equal(similarity, threshold)
        match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
        potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[tracker_ids_t] += 1

    num_gt_ids = data.num_gt_ids
    num_tracker_ids = data.num_tracker_ids
    fp_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
    fn_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
    fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
    fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10
    for gt_id in range(num_gt_ids):
        fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
        fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
    for tracker_id in range(num_tracker_ids):
        fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
        fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
    fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
    fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

    match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)
    res["IDFN"] = int(fn_mat[match_rows, match_cols].sum())
    res["IDFP"] = int(fp_mat[match_rows, match_cols].sum())
    res["IDTP"] = int(gt_id_count.sum() - res["IDFN"])
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
    return {
        "HOTA": _eval_hota(data),
        "CLEAR": _eval_clear(data),
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
        benchmark_id = (
            getattr(args, "benchmark_id", None)
            or getattr(args, "dataset_id", None)
            or getattr(args, "benchmark", None)
        )
        if benchmark_id:
            cfg = load_benchmark_cfg(benchmark_id)
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
    cfg = load_benchmark_cfg_from_args(args)
    if cfg:
        return cfg

    cfg_name = (
        getattr(args, "benchmark_id", None)
        or getattr(args, "dataset_id", None)
        or getattr(args, "benchmark", str(Path(args.source).parent.name))
    )
    try:
        return load_benchmark_cfg(cfg_name)
    except FileNotFoundError:
        for config_file in BENCHMARK_CONFIGS.glob("*.yaml"):
            if config_file.stem in str(args.source):
                return load_benchmark_cfg(config_file.stem)
    LOGGER.warning(f"Could not find benchmark config for {cfg_name}. Class filtering might be incorrect.")
    return {}


def _aabb_gt_path(gt_folder: Path, gt_loc_format: str, seq_name: str) -> Path:
    return Path(gt_loc_format.format(gt_folder=gt_folder, seq=seq_name))


def _evaluate_class_sequences(
    *,
    class_pairs: Sequence[tuple[str, int]],
    seq_info: Mapping[str, int | None],
    load_sequence: Callable[[str, int], SequenceData],
) -> tuple[dict[str, MetricBundle], dict[str, dict[str, MetricBundle]]]:
    class_combined: dict[str, MetricBundle] = {}
    per_class_sequence: dict[str, dict[str, MetricBundle]] = {}

    for class_name, class_id in class_pairs:
        sequence_bundles = {
            seq_name: _eval_bundle(load_sequence(seq_name, class_id))
            for seq_name in sorted(seq_info.keys())
        }
        per_class_sequence[class_name] = sequence_bundles
        class_combined[class_name] = _combine_bundles(sequence_bundles)

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

        def _load_sequence(seq_name: str, class_id: int) -> SequenceData:
            return _build_obb_sequence_data(
                seq_name=seq_name,
                gt_path=_resolve_obb_gt_path(args, gt_folder, seq_name),
                tracker_path=Path(args.exp_dir) / f"{seq_name}.txt",
                class_id=class_id,
                seq_info=seq_info,
            )

        class_combined, per_class_sequence = _evaluate_class_sequences(
            class_pairs=class_pairs,
            seq_info=seq_info,
            load_sequence=_load_sequence,
        )
        results = _format_results(class_combined, per_class_sequence)
        _append_aggregate_results(results, class_combined, include_obb_super_categories=True)
        return results

    settings = build_dataset_eval_settings(args, gt_folder, dict(seq_info))
    class_pairs = list(zip(settings["classes_to_eval"], settings["class_ids"]))
    distractor_ids = set(settings.get("distractor_ids") or [])
    gt_loc_format = settings["gt_loc_format"]

    def _load_sequence(seq_name: str, class_id: int) -> SequenceData:
        return _build_aabb_sequence_data(
            seq_name=seq_name,
            gt_path=_aabb_gt_path(gt_folder, gt_loc_format, seq_name),
            tracker_path=Path(args.exp_dir) / f"{seq_name}.txt",
            class_id=class_id,
            distractor_ids=distractor_ids,
            seq_info=seq_info,
        )

    class_combined, per_class_sequence = _evaluate_class_sequences(
        class_pairs=class_pairs,
        seq_info=seq_info,
        load_sequence=_load_sequence,
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
