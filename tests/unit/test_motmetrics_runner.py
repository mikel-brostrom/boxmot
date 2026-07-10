from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from boxmot.engine.eval.motmetrics import run_motmetrics

PARITY_FIELDS = ("HOTA", "DetA", "AssA", "AssRe", "MOTA", "MOTP", "IDF1", "IDR", "IDP", "IDSW", "IDs")


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=float), delimiter=",", fmt="%g")


def _trackeval_root() -> Path:
    return Path(__file__).resolve().parents[2] / "boxmot" / "engine" / "eval" / "trackeval" / "trackeval"


def _load_trackeval_metric_classes():
    root = _trackeval_root()
    if not (root / "trackeval" / "metrics" / "hota.py").exists():
        pytest.skip("local TrackEval checkout is not available")

    sys.path.insert(0, str(root))
    try:
        from trackeval.metrics.clear import CLEAR
        from trackeval.metrics.count import Count
        from trackeval.metrics.hota import HOTA
        from trackeval.metrics.identity import Identity
    finally:
        sys.path.pop(0)

    return HOTA, CLEAR, Identity, Count


def _load_trackeval_obb_iou():
    root = _trackeval_root()
    if not (root / "trackeval" / "__init__.py").exists():
        pytest.skip("local TrackEval checkout is not available")

    sys.path.insert(0, str(root))
    try:
        from boxmot.engine.eval.trackeval.datasets.mot_challenge_obb import _rotated_iou_batch
    finally:
        sys.path.pop(0)
    return _rotated_iou_batch


def _relabel_frame_ids(frame_ids: list[np.ndarray]) -> tuple[list[np.ndarray], int]:
    raw_ids = sorted({int(value) for ids in frame_ids for value in ids})
    id_map = {raw_id: index for index, raw_id in enumerate(raw_ids)}
    return [
        np.asarray([id_map[int(value)] for value in ids], dtype=int)
        if len(ids)
        else np.empty(0, dtype=int)
        for ids in frame_ids
    ], len(id_map)


def _aabb_iou(gt_boxes: np.ndarray, tracker_boxes: np.ndarray) -> np.ndarray:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)

    gt_x2y2 = gt_boxes[:, :2] + gt_boxes[:, 2:4]
    tr_x2y2 = tracker_boxes[:, :2] + tracker_boxes[:, 2:4]
    inter_top_left = np.maximum(gt_boxes[:, None, :2], tracker_boxes[None, :, :2])
    inter_bottom_right = np.minimum(gt_x2y2[:, None, :], tr_x2y2[None, :, :])
    inter_wh = np.maximum(0.0, inter_bottom_right - inter_top_left)
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    tr_area = tracker_boxes[:, 2] * tracker_boxes[:, 3]
    union = gt_area[:, None] + tr_area[None, :] - inter_area
    return np.divide(inter_area, union, out=np.zeros_like(inter_area), where=union > 0)


def _trackeval_data_from_rows(
    gt_rows: np.ndarray,
    tracker_rows: np.ndarray,
    *,
    num_timesteps: int,
    similarity_fn,
    gt_box_slice: slice,
    tracker_box_slice: slice,
) -> dict[str, Any]:
    gt_ids_by_frame: list[np.ndarray] = []
    tracker_ids_by_frame: list[np.ndarray] = []
    similarity_scores: list[np.ndarray] = []
    num_gt_dets = 0
    num_tracker_dets = 0

    for frame_id in range(1, num_timesteps + 1):
        gt_frame = gt_rows[gt_rows[:, 0].astype(int) == frame_id] if gt_rows.size else np.empty((0, 0))
        tracker_frame = (
            tracker_rows[tracker_rows[:, 0].astype(int) == frame_id]
            if tracker_rows.size
            else np.empty((0, 0))
        )
        gt_ids = gt_frame[:, 1].astype(int) if gt_frame.size else np.empty(0, dtype=int)
        tracker_ids = tracker_frame[:, 1].astype(int) if tracker_frame.size else np.empty(0, dtype=int)
        gt_ids_by_frame.append(gt_ids)
        tracker_ids_by_frame.append(tracker_ids)

        gt_boxes = gt_frame[:, gt_box_slice] if gt_frame.size else np.empty((0, gt_box_slice.stop - gt_box_slice.start))
        tracker_boxes = (
            tracker_frame[:, tracker_box_slice]
            if tracker_frame.size
            else np.empty((0, tracker_box_slice.stop - tracker_box_slice.start))
        )
        similarity_scores.append(similarity_fn(gt_boxes, tracker_boxes))
        num_gt_dets += len(gt_ids)
        num_tracker_dets += len(tracker_ids)

    gt_ids_by_frame, num_gt_ids = _relabel_frame_ids(gt_ids_by_frame)
    tracker_ids_by_frame, num_tracker_ids = _relabel_frame_ids(tracker_ids_by_frame)
    return {
        "num_timesteps": num_timesteps,
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "num_gt_ids": num_gt_ids,
        "num_tracker_ids": num_tracker_ids,
        "gt_ids": gt_ids_by_frame,
        "tracker_ids": tracker_ids_by_frame,
        "similarity_scores": similarity_scores,
    }


def _trackeval_report_summary(data: dict[str, Any]) -> dict[str, float | int]:
    hota_cls, clear_cls, identity_cls, count_cls = _load_trackeval_metric_classes()
    hota = hota_cls().eval_sequence(data)
    clear = clear_cls({"PRINT_CONFIG": False}).eval_sequence(data)
    identity = identity_cls({"PRINT_CONFIG": False}).eval_sequence(data)
    count = count_cls().eval_sequence(data)

    summary: dict[str, float | int] = {}
    for field in ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA"):
        summary[field] = max(0.0, float(np.mean(hota[field])) * 100.0)
    for field in ("HOTA(0)", "LocA(0)", "HOTALocA(0)"):
        summary[field] = max(0.0, float(hota[field]) * 100.0)
    for field in ("MOTA", "MOTP", "MODA", "CLR_Re", "CLR_Pr", "MTR", "PTR", "MLR", "sMOTA"):
        summary[field] = max(0.0, float(clear[field]) * 100.0)
    for field in ("IDF1", "IDR", "IDP"):
        summary[field] = max(0.0, float(identity[field]) * 100.0)
    for field in ("IDSW", "MT", "PT", "ML", "Frag"):
        summary[field] = max(0, int(clear[field]))
    for field in ("IDTP", "IDFN", "IDFP"):
        summary[field] = max(0, int(identity[field]))
    for field in ("Dets", "GT_Dets", "IDs", "GT_IDs"):
        summary[field] = max(0, int(count[field]))
    return summary


def test_run_motmetrics_aabb_perfect_sequence(tmp_path):
    source = tmp_path / "source"
    exp_dir = tmp_path / "runs" / "exp"
    seq_name = "SEQ-01"

    _write_rows(
        source / seq_name / "gt" / "gt_temp.txt",
        [
            [1, 1, 0, 0, 10, 10, 1, 1, 1],
            [2, 1, 1, 0, 10, 10, 1, 1, 1],
        ],
    )
    _write_rows(
        exp_dir / f"{seq_name}.txt",
        [
            [1, 1, 0, 0, 10, 10, 0.9, 1, -1],
            [2, 1, 1, 0, 10, 10, 0.9, 1, -1],
        ],
    )

    args = Namespace(
        source=source,
        exp_dir=exp_dir,
        benchmark="",
        benchmark_id=None,
        dataset_id=None,
        remapped_class_ids=None,
        remapped_class_names=None,
        classes=None,
    )

    results = run_motmetrics(
        args,
        [source / seq_name / "img1"],
        tmp_path / "save",
        source,
        seq_info={seq_name: 2},
    )

    assert set(results) == {"person"}
    assert results["person"]["HOTA"] == pytest.approx(100.0)
    assert results["person"]["MOTA"] == pytest.approx(100.0)
    assert results["person"]["IDF1"] == pytest.approx(100.0)
    assert results["person"]["AssA"] == pytest.approx(100.0)
    assert results["person"]["AssRe"] == pytest.approx(100.0)
    assert results["person"]["IDSW"] == 0
    assert results["person"]["IDs"] == 1


def test_run_motmetrics_obb_perfect_sequence(tmp_path):
    source = tmp_path / "source"
    exp_dir = tmp_path / "runs" / "exp"
    seq_name = "data01-1"
    corners = [0, 0, 10, 0, 10, 10, 0, 10]

    _write_rows(
        source / seq_name / "gt" / "gt.txt",
        [[1, 7, *corners, 1, 0, 0]],
    )
    _write_rows(
        exp_dir / f"{seq_name}.txt",
        [[1, 3, *corners, 0.9, 0, -1]],
    )

    args = Namespace(
        source=source,
        exp_dir=exp_dir,
        benchmark="",
        benchmark_id=None,
        dataset_id=None,
        eval_box_type="obb",
        remapped_class_ids=[0],
        remapped_class_names=["car"],
        translated_benchmark_class_names=None,
        classes=None,
    )

    results = run_motmetrics(
        args,
        [source / seq_name],
        tmp_path / "save",
        source,
        seq_info={seq_name: 1},
    )

    assert set(results) == {"car"}
    assert results["car"]["HOTA"] == pytest.approx(100.0)
    assert results["car"]["MOTA"] == pytest.approx(100.0)
    assert results["car"]["IDF1"] == pytest.approx(100.0)
    assert results["car"]["AssA"] == pytest.approx(100.0)
    assert results["car"]["AssRe"] == pytest.approx(100.0)
    assert results["car"]["IDSW"] == 0
    assert results["car"]["IDs"] == 1


def test_run_motmetrics_aabb_matches_trackeval_report_metrics(tmp_path):
    source = tmp_path / "source"
    exp_dir = tmp_path / "runs" / "exp"
    seq_name = "SEQ-02"
    gt_rows = np.asarray(
        [
            [1, 1, 0.125, 0, 10.5, 10, 1, 1, 1],
            [1, 2, 30, 0.125, 10, 10.5, 1, 1, 1],
            [2, 1, 1.125, 0, 10.5, 10, 1, 1, 1],
            [2, 2, 30, 1.125, 10, 10.5, 1, 1, 1],
            [3, 1, 2.125, 0, 10.5, 10, 1, 1, 1],
        ],
        dtype=float,
    )
    tracker_rows = np.asarray(
        [
            [1, 10, 0.125, 0, 10.5, 10, 0.9, 1, -1],
            [1, 20, 30, 0.125, 10, 10.5, 0.9, 1, -1],
            [2, 20, 1.125, 0, 10.5, 10, 0.9, 1, -1],
            [2, 30, 30, 1.125, 10, 10.5, 0.9, 1, -1],
            [3, 20, 2.125, 0, 10.5, 10, 0.9, 1, -1],
            [3, 99, 80.125, 80, 10.5, 10, 0.2, 1, -1],
        ],
        dtype=float,
    )
    _write_rows(source / seq_name / "gt" / "gt_temp.txt", gt_rows.tolist())
    _write_rows(exp_dir / f"{seq_name}.txt", tracker_rows.tolist())

    args = Namespace(
        source=source,
        exp_dir=exp_dir,
        benchmark="",
        benchmark_id=None,
        dataset_id=None,
        remapped_class_ids=None,
        remapped_class_names=None,
        classes=None,
    )
    actual = run_motmetrics(args, [source / seq_name / "img1"], tmp_path / "save", source, seq_info={seq_name: 3})
    expected = _trackeval_report_summary(
        _trackeval_data_from_rows(
            gt_rows,
            tracker_rows,
            num_timesteps=3,
            similarity_fn=_aabb_iou,
            gt_box_slice=slice(2, 6),
            tracker_box_slice=slice(2, 6),
        )
    )

    for field in PARITY_FIELDS:
        assert actual["person"][field] == pytest.approx(expected[field], abs=1e-9)


def test_run_motmetrics_obb_matches_trackeval_report_metrics(tmp_path):
    source = tmp_path / "source"
    exp_dir = tmp_path / "runs" / "exp"
    seq_name = "data02-1"
    obb_iou = _load_trackeval_obb_iou()

    box_a = [0, 0, 10, 0, 10, 10, 0, 10]
    box_b = [30, 0, 40, 0, 40, 10, 30, 10]
    box_c = [1, 0, 11, 0, 11, 10, 1, 10]
    far_box = [80, 80, 90, 80, 90, 90, 80, 90]
    gt_rows = np.asarray(
        [
            [1, 1, *box_a, 1, 0, 0],
            [1, 2, *box_b, 1, 0, 0],
            [2, 1, *box_c, 1, 0, 0],
        ],
        dtype=np.float32,
    )
    tracker_rows = np.asarray(
        [
            [1, 10, *box_a, 0.9, 0, -1],
            [1, 20, *box_b, 0.9, 0, -1],
            [2, 20, *box_c, 0.9, 0, -1],
            [2, 99, *far_box, 0.2, 0, -1],
        ],
        dtype=np.float32,
    )
    _write_rows(source / seq_name / "gt" / "gt.txt", gt_rows.tolist())
    _write_rows(exp_dir / f"{seq_name}.txt", tracker_rows.tolist())

    args = Namespace(
        source=source,
        exp_dir=exp_dir,
        benchmark="",
        benchmark_id=None,
        dataset_id=None,
        eval_box_type="obb",
        remapped_class_ids=[0],
        remapped_class_names=["car"],
        translated_benchmark_class_names=None,
        classes=None,
    )
    actual = run_motmetrics(args, [source / seq_name], tmp_path / "save", source, seq_info={seq_name: 2})
    expected = _trackeval_report_summary(
        _trackeval_data_from_rows(
            gt_rows,
            tracker_rows,
            num_timesteps=2,
            similarity_fn=obb_iou,
            gt_box_slice=slice(2, 10),
            tracker_box_slice=slice(2, 10),
        )
    )

    for field in PARITY_FIELDS:
        assert actual["car"][field] == pytest.approx(expected[field], abs=1e-9)
