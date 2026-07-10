from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from boxmot.engine.eval.motmetrics import run_motmetrics


def _write_rows(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(rows, dtype=np.float32), delimiter=",", fmt="%g")


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
