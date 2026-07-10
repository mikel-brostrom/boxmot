from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from boxmot.engine.eval.motmetrics import run_motmetrics

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
        benchmark_id=None,
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
        benchmark_id=None,
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
