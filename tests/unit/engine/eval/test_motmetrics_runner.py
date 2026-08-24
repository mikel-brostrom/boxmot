from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from boxmot.engine.eval import motmetrics as motmetrics_module
from boxmot.engine.eval.motmetrics import SequenceData, run_motmetrics

PARITY_FIELDS = ("HOTA", "DetA", "AssA", "AssRe", "MOTA", "MOTP", "IDF1", "IDR", "IDP", "IDSW", "IDs")
AABB_LEGACY_GOLDEN = {
    "HOTA": 58.92556509887896,
    "DetA": 83.33333333333334,
    "AssA": 41.66666666666667,
    "AssRe": 53.333333333333336,
    "MOTA": 40.0,
    "MOTP": 100.0,
    "IDF1": 54.54545454545454,
    "IDR": 60.0,
    "IDP": 50.0,
    "IDSW": 2,
    "IDs": 4,
}
OBB_LEGACY_GOLDEN = {
    "HOTA": 57.73502691896258,
    "DetA": 75.0,
    "AssA": 44.44444444444445,
    "AssRe": 66.66666666666666,
    "MOTA": 33.33333333333333,
    "MOTP": 100.0,
    "IDF1": 57.14285714285714,
    "IDR": 66.66666666666666,
    "IDP": 50.0,
    "IDSW": 1,
    "IDs": 3,
}


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=float), delimiter=",", fmt="%g")


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
        experiment_id=None,
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
    assert results["person"]["IDt"] == 0
    assert results["person"]["IDa"] == 0
    assert results["person"]["IDm"] == 0
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
        experiment_id=None,
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


def test_run_motmetrics_aabb_matches_legacy_golden_report_metrics(tmp_path):
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
        experiment_id=None,
        dataset_id=None,
        remapped_class_ids=None,
        remapped_class_names=None,
        classes=None,
    )
    actual = run_motmetrics(args, [source / seq_name / "img1"], tmp_path / "save", source, seq_info={seq_name: 3})

    for field in PARITY_FIELDS:
        assert actual["person"][field] == pytest.approx(AABB_LEGACY_GOLDEN[field], abs=1e-9)


def test_run_motmetrics_obb_matches_legacy_golden_report_metrics(tmp_path):
    source = tmp_path / "source"
    exp_dir = tmp_path / "runs" / "exp"
    seq_name = "data02-1"

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
        experiment_id=None,
        dataset_id=None,
        eval_box_type="obb",
        remapped_class_ids=[0],
        remapped_class_names=["car"],
        translated_benchmark_class_names=None,
        classes=None,
    )
    actual = run_motmetrics(args, [source / seq_name], tmp_path / "save", source, seq_info={seq_name: 2})

    for field in PARITY_FIELDS:
        assert actual["car"][field] == pytest.approx(OBB_LEGACY_GOLDEN[field], abs=1e-9)


def test_index_rows_by_frame_handles_unsorted_and_missing_frames():
    data = np.asarray(
        [
            [2, 20],
            [1, 10],
            [2, 21],
            [4, 40],
        ],
        dtype=float,
    )

    indexed = motmetrics_module._index_rows_by_frame(data, num_timesteps=3)

    assert [rows[:, 1].tolist() for rows in indexed] == [[10.0], [20.0, 21.0], []]


def test_identity_uses_rectangular_maximum_weight_assignment(monkeypatch):
    data = SequenceData(
        seq="identity",
        gt_ids=[np.asarray([0, 1]), np.asarray([0, 1])],
        tracker_ids=[np.asarray([0, 1]), np.asarray([1, 2])],
        similarity_scores=[
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ],
        num_timesteps=2,
        num_gt_dets=4,
        num_tracker_dets=4,
        num_gt_ids=2,
        num_tracker_ids=3,
    )
    original_assignment = motmetrics_module.linear_sum_assignment
    calls: list[tuple[tuple[int, ...], bool]] = []

    def _record_assignment(cost_matrix, *, maximize=False):
        calls.append((cost_matrix.shape, maximize))
        return original_assignment(cost_matrix, maximize=maximize)

    monkeypatch.setattr(motmetrics_module, "linear_sum_assignment", _record_assignment)

    result = motmetrics_module._eval_identity(data)

    assert calls == [((2, 3), True)]
    assert result["IDTP"] == 2
    assert result["IDFN"] == 2
    assert result["IDFP"] == 2
    assert result["IDF1"] == pytest.approx(0.5)


def test_clear_reports_motmetrics_identity_transition_counts():
    data = SequenceData(
        seq="identity-transitions",
        gt_ids=[
            np.asarray([0, 1]),
            np.asarray([0, 1]),
            np.asarray([0]),
            np.asarray([2]),
        ],
        tracker_ids=[
            np.asarray([0, 1]),
            np.asarray([0, 1]),
            np.asarray([2]),
            np.asarray([2]),
        ],
        similarity_scores=[
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            np.asarray([[1.0]]),
            np.asarray([[1.0]]),
        ],
        num_timesteps=4,
        num_gt_dets=6,
        num_tracker_dets=6,
        num_gt_ids=3,
        num_tracker_ids=3,
    )

    result = motmetrics_module._eval_clear(data)

    assert result["IDSW"] == 3
    assert result["IDt"] == 3
    assert result["IDa"] == 1
    assert result["IDm"] == 1


def test_identity_transition_counts_are_summed_across_sequences():
    sequence = SequenceData(
        seq="transfer",
        gt_ids=[np.asarray([0]), np.asarray([1])],
        tracker_ids=[np.asarray([0]), np.asarray([0])],
        similarity_scores=[np.asarray([[1.0]]), np.asarray([[1.0]])],
        num_timesteps=2,
        num_gt_dets=2,
        num_tracker_dets=2,
        num_gt_ids=2,
        num_tracker_ids=1,
    )
    bundle = motmetrics_module._eval_bundle(sequence)

    combined = motmetrics_module._combine_bundles({"one": bundle, "two": bundle})
    summary = motmetrics_module._summary_from_bundle(combined)

    assert summary["IDt"] == 2
    assert summary["IDa"] == 0
    assert summary["IDm"] == 2


def test_rotated_iou_rejects_disjoint_aabbs_before_exact_intersection(monkeypatch):
    box = [0, 0, 10, 0, 10, 10, 0, 10]
    far_box = [100, 100, 110, 100, 110, 110, 100, 110]
    original_intersection = motmetrics_module.cv2.rotatedRectangleIntersection
    calls = 0

    def _record_intersection(rect_a, rect_b):
        nonlocal calls
        calls += 1
        return original_intersection(rect_a, rect_b)

    monkeypatch.setattr(motmetrics_module.cv2, "rotatedRectangleIntersection", _record_intersection)

    result = motmetrics_module._rotated_iou_batch(
        np.asarray([box], dtype=float),
        np.asarray([box, far_box], dtype=float),
    )

    assert calls == 1
    np.testing.assert_allclose(result, [[1.0, 0.0]])


@pytest.mark.parametrize("eval_box_type", ["aabb", "obb"])
def test_run_motmetrics_loads_each_multiclass_sequence_once(tmp_path, monkeypatch, eval_box_type):
    source = tmp_path / "source"
    exp_dir = tmp_path / "runs" / "exp"
    seq_names = ("SEQ-01", "SEQ-02")
    if eval_box_type == "obb":
        box_a = [0, 0, 10, 0, 10, 10, 0, 10]
        box_b = [20, 0, 30, 0, 30, 10, 20, 10]
        gt_rows = [[1, 1, *box_a, 1, 0, 0], [1, 2, *box_b, 1, 1, 0]]
        tracker_rows = [[1, 10, *box_a, 0.9, 0, -1], [1, 20, *box_b, 0.9, 1, -1]]
    else:
        gt_rows = [[1, 1, 0, 0, 10, 10, 1, 1, 1], [1, 2, 20, 0, 10, 10, 1, 2, 1]]
        tracker_rows = [[1, 10, 0, 0, 10, 10, 0.9, 1, -1], [1, 20, 20, 0, 10, 10, 0.9, 2, -1]]
    gt_paths = {
        seq_name: source / seq_name / "gt" / ("gt.txt" if eval_box_type == "obb" else "gt_temp.txt")
        for seq_name in seq_names
    }
    tracker_paths = {seq_name: exp_dir / f"{seq_name}.txt" for seq_name in seq_names}
    for seq_name in seq_names:
        _write_rows(gt_paths[seq_name], gt_rows)
        _write_rows(tracker_paths[seq_name], tracker_rows)

    original_read = motmetrics_module._read_csv_matrix
    reads: list[Path] = []

    def _record_read(path):
        reads.append(Path(path))
        return original_read(path)

    monkeypatch.setattr(motmetrics_module, "_read_csv_matrix", _record_read)
    monkeypatch.setattr(motmetrics_module.os, "cpu_count", lambda: 1)
    args = Namespace(
        source=source,
        exp_dir=exp_dir,
        benchmark="",
        experiment_id=None,
        dataset_id=None,
        eval_box_type=eval_box_type,
        remapped_class_ids=[0, 1] if eval_box_type == "obb" else [1, 2],
        remapped_class_names=["first", "second"],
        translated_benchmark_class_names=None,
        classes=None,
    )

    run_motmetrics(
        args,
        [source / seq_name for seq_name in seq_names]
        if eval_box_type == "obb"
        else [source / seq_name / "img1" for seq_name in seq_names],
        tmp_path / "save",
        source,
        seq_info={seq_name: 1 for seq_name in seq_names},
    )

    for seq_name in seq_names:
        assert reads.count(gt_paths[seq_name]) == 1
        assert reads.count(tracker_paths[seq_name]) == 1


@pytest.mark.parametrize("eval_box_type", ["aabb", "obb"])
def test_parallel_sequence_evaluation_matches_serial_results(tmp_path, monkeypatch, eval_box_type):
    source = tmp_path / "source"
    exp_dir = tmp_path / "runs" / "exp"
    seq_names = ("SEQ-01", "SEQ-02")
    if eval_box_type == "obb":
        box = [0, 0, 10, 0, 10, 10, 0, 10]
        gt_rows = [[1, 1, *box, 1, 0, 0], [2, 1, *box, 1, 0, 0]]
        tracker_rows = [[1, 10, *box, 0.9, 0, -1], [2, 10, *box, 0.9, 0, -1]]
        gt_name = "gt.txt"
    else:
        gt_rows = [[1, 1, 0, 0, 10, 10, 1, 1, 1], [2, 1, 0, 0, 10, 10, 1, 1, 1]]
        tracker_rows = [[1, 10, 0, 0, 10, 10, 0.9, 1, -1], [2, 10, 0, 0, 10, 10, 0.9, 1, -1]]
        gt_name = "gt_temp.txt"

    for seq_name in seq_names:
        _write_rows(source / seq_name / "gt" / gt_name, gt_rows)
        _write_rows(exp_dir / f"{seq_name}.txt", tracker_rows)

    common_args = {
        "source": source,
        "exp_dir": exp_dir,
        "benchmark": "",
        "experiment_id": None,
        "dataset_id": None,
        "eval_box_type": eval_box_type,
        "remapped_class_ids": [0] if eval_box_type == "obb" else None,
        "remapped_class_names": ["car"] if eval_box_type == "obb" else None,
        "translated_benchmark_class_names": None,
        "classes": None,
    }
    seq_paths = [source / seq_name for seq_name in seq_names]
    if eval_box_type == "aabb":
        seq_paths = [path / "img1" for path in seq_paths]

    monkeypatch.setattr(motmetrics_module.os, "cpu_count", lambda: 1)
    serial = run_motmetrics(
        Namespace(**common_args),
        seq_paths,
        tmp_path / "serial",
        source,
        seq_info={seq_name: 2 for seq_name in seq_names},
    )
    monkeypatch.setattr(motmetrics_module.os, "cpu_count", lambda: 8)
    parallel = run_motmetrics(
        Namespace(**common_args),
        seq_paths,
        tmp_path / "parallel",
        source,
        seq_info={seq_name: 2 for seq_name in seq_names},
    )

    assert parallel == serial


@pytest.mark.parametrize(
    ("num_sequences", "cpu_count", "expected"),
    [(1, 8, 1), (3, 8, 3), (12, 8, 6), (3, 2, 1), (3, 1, 1), (3, 0, 1)],
)
def test_metric_worker_count_reserves_two_computer_cpus(num_sequences, cpu_count, expected):
    assert motmetrics_module._metric_worker_count(num_sequences, cpu_count) == expected
