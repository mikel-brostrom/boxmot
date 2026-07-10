from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from boxmot.engine.eval.motmetrics import _combine_alpha_rows, evaluate_motchallenge_hota


def _load_motchallenge(path: Path, *, min_confidence: float | None = None) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if min_confidence is not None:
        data = data[data[:, 6] >= min_confidence]
    data[:, 2:4] -= 1
    return data


def _iou_matrix(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.empty((len(gt_boxes), len(tracker_boxes)), dtype=float)

    gt_x1y1 = gt_boxes[:, :2]
    gt_x2y2 = gt_boxes[:, :2] + gt_boxes[:, 2:4]
    tracker_x1y1 = tracker_boxes[:, :2]
    tracker_x2y2 = tracker_boxes[:, :2] + tracker_boxes[:, 2:4]

    inter_x1y1 = np.maximum(gt_x1y1[:, None, :], tracker_x1y1[None, :, :])
    inter_x2y2 = np.minimum(gt_x2y2[:, None, :], tracker_x2y2[None, :, :])
    inter_wh = np.maximum(0.0, inter_x2y2 - inter_x1y1)
    intersection = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    tracker_area = tracker_boxes[:, 2] * tracker_boxes[:, 3]
    union = gt_area[:, None] + tracker_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _trackeval_data_from_motchallenge(gt_path: Path, tracker_path: Path) -> dict:
    gt = _load_motchallenge(gt_path, min_confidence=1.0)
    tracker = _load_motchallenge(tracker_path)

    gt_id_map = {raw_id: idx for idx, raw_id in enumerate(sorted(set(gt[:, 1].astype(int))))}
    tracker_id_map = {raw_id: idx for idx, raw_id in enumerate(sorted(set(tracker[:, 1].astype(int))))}
    frames = sorted(set(gt[:, 0].astype(int)) | set(tracker[:, 0].astype(int)))

    gt_ids = []
    tracker_ids = []
    similarity_scores = []
    for frame in frames:
        gt_frame = gt[gt[:, 0].astype(int) == frame]
        tracker_frame = tracker[tracker[:, 0].astype(int) == frame]
        gt_ids.append(np.asarray([gt_id_map[int(raw_id)] for raw_id in gt_frame[:, 1]], dtype=int))
        tracker_ids.append(np.asarray([tracker_id_map[int(raw_id)] for raw_id in tracker_frame[:, 1]], dtype=int))
        similarity_scores.append(_iou_matrix(gt_frame[:, 2:6], tracker_frame[:, 2:6]))

    return {
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(tracker),
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tracker_id_map),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
    }


def _trackeval_hota_summary(trackeval_root: Path, sequence_files: dict[str, tuple[Path, Path]]) -> dict:
    sys.path.insert(0, str(trackeval_root))
    try:
        from trackeval.metrics.hota import HOTA
    finally:
        sys.path.pop(0)

    metric = HOTA()
    sequence_results = {
        name: metric.eval_sequence(_trackeval_data_from_motchallenge(gt_path, tracker_path))
        for name, (gt_path, tracker_path) in sequence_files.items()
    }
    combined = metric.combine_sequences(sequence_results)

    return {
        "HOTA": float(np.mean(combined["HOTA"])),
        "DetA": float(np.mean(combined["DetA"])),
        "AssA": float(np.mean(combined["AssA"])),
        "per_sequence": {
            name: {
                "HOTA": float(np.mean(rows["HOTA"])),
                "DetA": float(np.mean(rows["DetA"])),
                "AssA": float(np.mean(rows["AssA"])),
            }
            for name, rows in sequence_results.items()
        },
    }


def test_combine_alpha_rows_matches_trackeval_sequence_aggregation():
    combined = _combine_alpha_rows(
        [
            {
                "deta_alpha": 0.5,
                "assa_alpha": 0.25,
                "num_detections": 4,
                "num_objects": 6,
                "num_false_positives": 2,
            },
            {
                "deta_alpha": 0.2,
                "assa_alpha": 0.8,
                "num_detections": 1,
                "num_objects": 2,
                "num_false_positives": 2,
            },
        ]
    )

    assert combined["deta_alpha"] == pytest.approx(5 / 12)
    assert combined["assa_alpha"] == pytest.approx(((0.25 * 4) + (0.8 * 1)) / 5)
    assert combined["hota_alpha"] == pytest.approx((combined["deta_alpha"] * combined["assa_alpha"]) ** 0.5)


def test_in_repo_motmetrics_tud_hota_matches_trackeval():
    repo_root = Path(__file__).resolve().parents[2]
    py_motmetrics_root = repo_root / "py-motmetrics"
    trackeval_root = repo_root / "boxmot" / "engine" / "eval" / "trackeval" / "trackeval"
    data_root = py_motmetrics_root / "motmetrics" / "data"
    if not data_root.exists():
        pytest.skip("py-motmetrics checkout with TUD fixtures is not available")

    sequence_files = {
        "TUD-Campus": (
            data_root / "TUD-Campus" / "gt.txt",
            data_root / "TUD-Campus" / "test.txt",
        ),
        "TUD-Stadtmitte": (
            data_root / "TUD-Stadtmitte" / "gt.txt",
            data_root / "TUD-Stadtmitte" / "test.txt",
        ),
    }

    motmetrics_result = evaluate_motchallenge_hota(sequence_files)
    trackeval_result = _trackeval_hota_summary(trackeval_root, sequence_files)

    for metric in ("HOTA", "DetA", "AssA"):
        assert motmetrics_result[metric] == pytest.approx(trackeval_result[metric])
    for sequence_name in sequence_files:
        for metric in ("HOTA", "DetA", "AssA"):
            assert motmetrics_result["per_sequence"][sequence_name][metric] == pytest.approx(
                trackeval_result["per_sequence"][sequence_name][metric]
            )
