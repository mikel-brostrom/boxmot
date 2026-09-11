"""Official KITTI preprocessing with independent object and tracking annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from boxmot.datasets.readers.boxes3d import KittiObjectLabels, TrackingLabels3D, read_kitti_tracking_labels
from boxmot.engine.eval import kitti_3d
from boxmot.engine.eval.trackeval_reference import evaluate_trackeval_kitti, validate_trackeval_kitti_dependencies


def _gt(
    frame: int,
    identity: int,
    label: str = "Car",
    box: tuple[int, int, int, int] = (0, 0, 50, 50),
    truncation: int = 0,
    occlusion: int = 0,
) -> str:
    """Represent native tracking visibility metadata, including distractors."""
    bounds = " ".join(map(str, box))
    return f"{frame} {identity} {label} {truncation} {occlusion} 0 {bounds} 2 2 4 0 2 10 0"


def _prediction(row: str, identity: int, label: str | None = None) -> str:
    """Preserve projected bounds and 3D geometry in canonical 18-field output."""
    fields = row.split()
    fields[1] = str(identity)
    if label is not None:
        fields[2] = label
    fields[3:5] = ["-1", "-1"]
    return " ".join(fields) + " 0.9"


def _annotations(path: Path, rows: list[str], frame_count: int) -> TrackingLabels3D:
    path.write_text("".join(row + "\n" for row in rows))
    return read_kitti_tracking_labels(
        path,
        frame_count=frame_count,
        classes={"car": {"id": 1}, "pedestrian": {"id": 2}},
        options={"ignore_classes": ["Van", "Person", "Person_sitting", "DontCare"]},
    )


@pytest.fixture
def installed_trackeval() -> None:
    """Run against the installed pinned package, without local reference copies."""
    pytest.importorskip("trackeval")
    validate_trackeval_kitti_dependencies()


@pytest.mark.usefixtures("installed_trackeval")
def test_official_tracking_preprocessing_keeps_distractors_until_trackeval(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    (gt / "label_02").mkdir(parents=True)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    rows = [
        _gt(0, 1),
        _gt(0, 2, "Van", (60, 0, 110, 50)),
        _gt(0, 3, "Car", (120, 0, 170, 50), truncation=1),
        _gt(0, 4, "Car", (180, 0, 230, 50), occlusion=3),
        _gt(0, -1, "DontCare", (240, 0, 290, 50)),
        _gt(0, 5, "Person", (0, 60, 50, 110)),
        _gt(0, 6, "Pedestrian", (60, 60, 110, 110)),
    ]
    (gt / "label_02" / "0000.txt").write_text("\n".join(rows) + "\n")
    prediction_rows = [
        _prediction(row, index + 10, "Car" if index <= 4 else "Pedestrian") for index, row in enumerate(rows)
    ]
    prediction_rows.append(_prediction(_gt(0, 7, box=(300, 0, 325, 25)), 17))
    (predictions / "0000.txt").write_text("\n".join(prediction_rows) + "\n")

    result = evaluate_trackeval_kitti(gt_folder=gt, tracker_folder=predictions, seq_info={"0000": 1})

    for class_name in ("car", "pedestrian"):
        assert result[class_name]["HOTA"] == result[class_name]["IDF1"] == 100
        assert result[class_name]["GT_Dets"] == result[class_name]["Dets"] == 1
    assert result["cls_comb_det_av"]["Dets"] == 2
    assert result["cls_comb_det_av"]["Frames"] == 1


@pytest.mark.usefixtures("installed_trackeval")
def test_tracking_metrics_retain_empty_frames_and_sequence_identity_boundaries(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    (gt / "label_02").mkdir(parents=True)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for sequence in ("0000", "0001"):
        rows = [_gt(frame, 1) for frame in (0, 2)]
        (gt / "label_02" / f"{sequence}.txt").write_text("\n".join(rows) + "\n")
        (predictions / f"{sequence}.txt").write_text("\n".join(_prediction(row, 8) for row in rows) + "\n")

    results = evaluate_trackeval_kitti(gt_folder=gt, tracker_folder=predictions, seq_info={"0000": 3, "0001": 3})

    car = results["car"]
    assert car["HOTA"] == car["IDF1"] == car["MOTA"] == 100
    assert car["GT_Dets"] == car["Dets"] == 4
    assert car["GT_IDs"] == car["IDs"] == 2
    assert car["Frames"] == 6
    assert car["per_sequence"]["0000"]["Frames"] == 3


@pytest.mark.usefixtures("installed_trackeval")
def test_tracking_identity_switch_is_scored_by_installed_pipeline(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    (gt / "label_02").mkdir(parents=True)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    rows = [_gt(frame, 1) for frame in (0, 1)]
    (gt / "label_02" / "0000.txt").write_text("\n".join(rows))
    (predictions / "0000.txt").write_text("\n".join(_prediction(row, 8 + index) for index, row in enumerate(rows)))

    result = evaluate_trackeval_kitti(gt_folder=gt, tracker_folder=predictions, seq_info={"0000": 2})["car"]

    assert result["IDSW"] == 1
    assert result["IDF1"] == result["MOTA"] == 50


@pytest.mark.usefixtures("installed_trackeval")
@pytest.mark.parametrize(("has_gt", "has_prediction"), [(False, False), (False, True), (True, False)])
def test_official_tracking_accepts_empty_annotations_and_predictions(
    tmp_path: Path, has_gt: bool, has_prediction: bool
) -> None:
    gt = tmp_path / "gt"
    (gt / "label_02").mkdir(parents=True)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    row = _gt(0, 1)
    (gt / "label_02" / "0000.txt").write_text(row if has_gt else "")
    (predictions / "0000.txt").write_text(_prediction(row, 8) if has_prediction else "")
    result = evaluate_trackeval_kitti(gt_folder=gt, tracker_folder=predictions, seq_info={"0000": 3})["car"]

    assert result["GT_Dets"] == result["CLR_FN"] == int(has_gt)
    assert result["Dets"] == result["CLR_FP"] == int(has_prediction)
    assert result["Frames"] == 3


@pytest.mark.usefixtures("installed_trackeval")
def test_official_tracking_metric_geometry_is_projected_2d(tmp_path: Path) -> None:
    gt = tmp_path / "gt"
    (gt / "label_02").mkdir(parents=True)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    row = _gt(0, 1)
    (gt / "label_02" / "0000.txt").write_text(row)
    fields = _prediction(row, 8).split()
    fields[15] = "1000"
    (predictions / "0000.txt").write_text(" ".join(fields))
    result = evaluate_trackeval_kitti(gt_folder=gt, tracker_folder=predictions, seq_info={"0000": 1})["car"]

    assert result["HOTA"] == result["IDF1"] == result["MOTA"] == 100


@pytest.mark.usefixtures("installed_trackeval")
def test_official_exports_keep_distinct_object_labels_scores_and_frame_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    large_id = 2**53 + 1
    tracking_rows = [_gt(0, large_id), _gt(2, large_id), _gt(0, 4, "Person_sitting", (70, 0, 120, 50))]
    truth = _annotations(tmp_path / "tracking.txt", tracking_rows, 3)
    prediction_rows = [_prediction(tracking_rows[0], large_id + 10), _prediction(tracking_rows[1], large_id + 10)]
    (predictions / "0000.txt").write_text("\n".join(prediction_rows) + "\n")
    object_row = "Car 0.37 1 0 2 3 44 49 2 2 4 0 2 10 0"
    objects = KittiObjectLabels(frame_rows=((object_row,), (), (object_row,)), source_sha256="b" * 64)
    expected_ap = {
        geometry: {name: {"easy": None, "moderate": 63.5, "hard": 81.25} for name in ("car", "pedestrian")}
        for geometry in ("2d", "3d")
    }

    def object_backend(gt_dir: Path, prediction_dir: Path, frame_ids: list[str], output: Path) -> dict[str, Any]:
        assert frame_ids == ["000000", "000001", "000002"]
        assert (gt_dir / "000000.txt").read_text() == object_row + "\n"
        assert (gt_dir / "000001.txt").read_text() == (prediction_dir / "000001.txt").read_text() == ""
        assert (prediction_dir / "000000.txt").read_text() == " ".join(prediction_rows[0].split()[2:]) + "\n"
        return expected_ap

    monkeypatch.setattr(kitti_3d, "evaluate_kitti_objects", object_backend)
    monkeypatch.setattr(kitti_3d, "resolve_kitti_object_backend", lambda: tmp_path / "backend")
    output = tmp_path / "metrics"
    results = kitti_3d.evaluate_kitti_3d(predictions, output, {"0000": truth}, {"0000": 3}, {"0000": objects})

    assert results["car"]["HOTA"] == results["car"]["IDF1"] == 100
    assert results["car"]["GT_Dets"] == 2
    assert json.loads((output / "metrics.json").read_text()) == results
    assert json.loads((output / "detection_metrics.json").read_text()) == expected_ap
    assert len((output / "detection_metrics.csv").read_text().splitlines()) == 13
    protocol = json.loads((output / "evaluation.json").read_text())
    assert protocol["tracking"]["geometry"] == "2d"
    assert protocol["tracking"]["evaluator"] == "trackeval==1.3.0"
    assert protocol["detection"]["geometry"] == ["2d", "3d"]
    assert protocol["detection"]["metric"] == "AP40"
    assert protocol["frames"][-1] == {"id": "000002", "sequence": "0000", "frame_index": 2}
    assert protocol["tracking_identity_maps"]["0000"]["ground_truth"][str(large_id)] == 0
    exported = (output / "protocol_inputs/tracking/ground_truth/label_02/0000.txt").read_text()
    assert " Person " in exported
    assert len(exported.splitlines()) == len(tracking_rows)
    assert (predictions / "0000.txt").read_text() == "\n".join(prediction_rows) + "\n"


@pytest.mark.parametrize("bounds", ["-1 -1 -1 -1", "1 2 1 4", "1 4 3 4"])
def test_unavailable_projection_is_rejected_before_official_scoring(tmp_path: Path, bounds: str) -> None:
    truth = _annotations(tmp_path / "truth.txt", [_gt(0, 1)], 1)
    objects = KittiObjectLabels(frame_rows=((),), source_sha256="a" * 64)
    fields = _prediction(_gt(0, 1), 8).split()
    fields[6:10] = bounds.split()
    (tmp_path / "0000.txt").write_text(" ".join(fields))

    with pytest.raises(ValueError, match="valid projected 2D box"):
        kitti_3d._export_official_inputs(tmp_path, tmp_path / "output", {"0000": truth}, {"0000": 1}, {"0000": objects})


def test_export_requires_original_tracking_rows_and_complete_object_frames(tmp_path: Path) -> None:
    truth = _annotations(tmp_path / "truth.txt", [_gt(0, 1)], 1)
    objects = KittiObjectLabels(frame_rows=(), source_sha256="a" * 64)
    with pytest.raises(ValueError, match="cover exactly 1 frames"):
        kitti_3d._export_official_inputs(tmp_path, tmp_path / "output", {"0000": truth}, {"0000": 1}, {"0000": objects})


def test_dependency_preflight_runs_both_official_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    called = []
    monkeypatch.setattr(kitti_3d, "validate_trackeval_kitti_dependencies", lambda: called.append("trackeval"))
    monkeypatch.setattr(kitti_3d, "resolve_kitti_object_backend", lambda: called.append("object"))
    kitti_3d.validate_kitti_evaluation_dependencies()
    assert called == ["trackeval", "object"]


@pytest.mark.parametrize(("index", "value"), [(3, "0.37"), (4, "0.5"), (6, "nan"), (8, "0"), (13, "inf")])
def test_unfiltered_ignored_tracking_rows_are_validated(tmp_path: Path, index: int, value: str) -> None:
    fields = _gt(0, 1, "Van").split()
    fields[index] = value
    truth = _annotations(tmp_path / "truth.txt", [" ".join(fields)], 1)
    assert len(truth.boxes) == 0
    objects = KittiObjectLabels(frame_rows=((),), source_sha256="a" * 64)
    (tmp_path / "0000.txt").write_text("")
    with pytest.raises(ValueError, match="Invalid original KITTI tracking annotation"):
        kitti_3d._export_official_inputs(tmp_path, tmp_path / "output", {"0000": truth}, {"0000": 1}, {"0000": objects})
