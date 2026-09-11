"""Independent spatial result serialization and volumetric tracking evaluation."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from boxmot.datasets.readers.boxes3d import TrackingLabels3D, read_kitti_tracking_labels
from boxmot.engine.eval.kitti_3d import evaluate_kitti_3d, read_kitti_3d_results, write_kitti_3d_rows
from boxmot.structures import Boxes3D, CameraModel, Tracks3D

_BOX = (0.0, 2.0, 10.0, 0.0, 4.0, 2.0, 2.0)
_VALID_ROW = "0 8 Car -1 -1 0 10 20 30 40 2 2 4 0 2 10 0 0.9"


def _tracks(boxes: list[tuple[float, ...]], ids: list[int], classes: list[int]) -> Tracks3D:
    """Construct CPU spatial output, including unobserved tracks without masks."""
    return Tracks3D(
        geometry=Boxes3D(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 7)),
        track_ids=torch.tensor(ids, dtype=torch.int64),
        scores=torch.full((len(ids),), 0.9, dtype=torch.float32),
        class_ids=torch.tensor(classes, dtype=torch.int64),
        detection_indices=torch.full((len(ids),), -1, dtype=torch.int64),
        sample_id="0000:0",
    )


def _truth(rows: list[tuple[int, int, int, tuple[float, ...]]]) -> TrackingLabels3D:
    """Retain native zero-based frame indices and sparse ground-truth identities."""
    return TrackingLabels3D(
        frame_indices=np.array([row[0] for row in rows], dtype=np.int64),
        track_ids=np.array([row[1] for row in rows], dtype=np.int64),
        class_ids=np.array([row[2] for row in rows], dtype=np.int64),
        boxes=np.array([row[3] for row in rows], dtype=np.float64).reshape(-1, 7),
        row_count=len(rows),
        source_sha256="a" * 64,
    )


def _save(tmp_path: Path, sequence: str, frames: list[tuple[int, Tracks3D]]) -> Path:
    """Write persisted predictions independently from annotation construction."""
    directory = tmp_path / "predictions"
    directory.mkdir(exist_ok=True)
    with (directory / f"{sequence}.txt").open("w") as handle:
        for frame, tracks in frames:
            write_kitti_3d_rows(handle, tracks, frame, None)
    return directory


def test_writer_round_trip_retains_camera_geometry_and_exact_large_ids(tmp_path: Path) -> None:
    pose = torch.eye(4)
    pose[0, 3] = 100
    camera = CameraModel(
        projection=torch.tensor([[100.0, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]]),
        image_size=(100, 100),
        camera_to_world=pose,
    )
    boxes = [_BOX, (0.0, 2.0, -10.0, 0.0, 4.0, 2.0, 2.0)]
    tracks = _tracks(boxes, [2**53 + 1, 42], [1, 2])
    before = tracks.geometry.values.clone()
    output = tmp_path / "0000.txt"
    with output.open("w") as handle:
        assert write_kitti_3d_rows(handle, tracks, 2, camera) == 2

    lines = [line.split() for line in output.read_text().splitlines()]
    assert all(len(row) == 18 for row in lines)
    assert lines[0][:6] == ["2", str(2**53 + 1), "Car", "-1", "-1", "0"]
    np.testing.assert_array_equal(np.array(lines[0][6:10], dtype=float), [28, 50, 72, 72])
    np.testing.assert_array_equal(np.array(lines[0][10:17], dtype=float), [2, 2, 4, 0, 2, 10, 0])
    assert lines[1][6:10] == ["-1"] * 4
    rows = read_kitti_3d_results(output, frame_count=3)[2]
    assert [row.track_id for row in rows] == [2**53 + 1, 42]
    assert [row.class_id for row in rows] == [1, 2]
    np.testing.assert_array_equal([row.box for row in rows], boxes)
    assert rows[0].score == pytest.approx(0.9)
    torch.testing.assert_close(tracks.geometry.values, before, rtol=0, atol=0)


def test_writer_rejects_unknown_class_before_emitting_partial_frame() -> None:
    handle = io.StringIO()
    with pytest.raises(ValueError, match="classes"):
        write_kitti_3d_rows(handle, _tracks([_BOX, _BOX], [1, 2], [1, 3]), 0, None)
    assert handle.getvalue() == ""


def test_writer_wraps_accumulated_yaw_without_changing_the_input(tmp_path: Path) -> None:
    box = (*_BOX[:3], 4 * np.pi + 0.5, *_BOX[4:])
    tracks = _tracks([box], [8], [1])
    original_yaw = float(tracks.geometry.values[0, 3])
    prediction_dir = _save(tmp_path, "0000", [(0, tracks)])
    row = read_kitti_3d_results(prediction_dir / "0000.txt", frame_count=1)[0][0]

    assert -np.pi <= row.box[3] < np.pi
    assert row.box[3] == pytest.approx(0.5, abs=1e-6)
    assert np.sin(row.box[3]) == pytest.approx(np.sin(original_yaw))
    assert np.cos(row.box[3]) == pytest.approx(np.cos(original_yaw))
    assert float(tracks.geometry.values[0, 3]) == original_yaw


@pytest.mark.parametrize(
    ("index", "value", "message"),
    [
        (0, "-1", "nonnegative integers"),
        (0, "1", "Frame index"),
        (0, "0.0", "nonnegative integers"),
        (1, "8.5", "nonnegative integers"),
        (1, str(2**63), "int64"),
        (1, "-1", "nonnegative integers"),
        (2, "Cyclist", "Unsupported result class"),
        (6, "nan", "finite"),
        (10, "0", "positive"),
        (11, "-2", "positive"),
        (12, "0", "positive"),
        (12, "1e-300", "float32"),
        (13, "inf", "finite"),
        (13, "1e300", "float32"),
        (17, "nan", "finite"),
    ],
)
def test_reader_rejects_malformed_fields(tmp_path: Path, index: int, value: str, message: str) -> None:
    fields = _VALID_ROW.split()
    fields[index] = value
    path = tmp_path / "bad.txt"
    path.write_text(" ".join(fields) + "\n")

    with pytest.raises(ValueError, match=message) as error:
        read_kitti_3d_results(path, frame_count=1)
    assert f"{path}:1:" in str(error.value)


@pytest.mark.parametrize("row", ["broken", " ".join(_VALID_ROW.split()[:-1]), _VALID_ROW + " extra"])
def test_reader_requires_result_score_and_exact_schema(tmp_path: Path, row: str) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(row)
    with pytest.raises(ValueError, match="18 KITTI"):
        read_kitti_3d_results(path, frame_count=1)


def test_reader_rejects_duplicate_id_even_across_classes(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text(_VALID_ROW + "\n" + _VALID_ROW.replace("Car", "Pedestrian") + "\n")
    with pytest.raises(ValueError, match="Duplicate track identity 8 in frame 0"):
        read_kitti_3d_results(path, frame_count=1)


def test_reader_preserves_within_frame_order_and_missing_empty_frames(tmp_path: Path) -> None:
    path = tmp_path / "ordered.txt"
    path.write_text("2 9" + _VALID_ROW[3:] + "\n" + _VALID_ROW + "\n" + "2 7" + _VALID_ROW[3:])
    rows = read_kitti_3d_results(path, frame_count=4)
    assert list(rows) == [0, 2]
    assert [row.track_id for row in rows[2]] == [9, 7]


def test_perfect_3d_evaluation_persists_metric_and_protocol_reports(tmp_path: Path) -> None:
    tracks = _tracks([_BOX, _BOX], [8, 9], [1, 2])
    prediction_dir = _save(tmp_path, "0000", [(0, tracks), (2, tracks)])
    truth = _truth([(frame, identity, class_id, _BOX) for frame in [0, 2] for identity, class_id in [(18, 1), (19, 2)]])
    output = tmp_path / "metrics"

    results = evaluate_kitti_3d(prediction_dir, output, {"0000": truth}, {"0000": 3})

    for class_name in ("car", "pedestrian", "cls_comb_cls_av", "cls_comb_det_av"):
        for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "MOTP"):
            assert results[class_name][metric] == 100
    assert results["car"]["Frames"] == 3
    assert results["car"]["GT_Dets"] == results["car"]["Dets"] == 2
    assert results["car"]["per_sequence"]["0000"]["Frames"] == 3
    assert json.loads((output / "metrics.json").read_text()) == results
    assert len((output / "metrics.csv").read_text().splitlines()) == 5
    protocol = json.loads((output / "evaluation.json").read_text())
    assert protocol["official_kitti_protocol"] is False
    assert protocol["similarity"] == "volumetric 3D IoU"
    assert protocol["clear_identity_iou_threshold"] == 0.5
    assert protocol["hota_iou_thresholds"] == [index / 20 for index in range(1, 20)]
    assert protocol["ground_truth_sha256"] == {"0000": "a" * 64}


def test_identity_switch_affects_3d_association_metrics(tmp_path: Path) -> None:
    prediction_dir = _save(
        tmp_path, "0000", [(frame, _tracks([_BOX], [identity], [1])) for frame, identity in [(0, 8), (1, 9)]]
    )
    truth = _truth([(0, 18, 1, _BOX), (1, 18, 1, _BOX)])
    result = evaluate_kitti_3d(prediction_dir, tmp_path / "metrics", {"0000": truth}, {"0000": 2})["car"]

    assert result["IDSW"] == 1
    assert result["IDF1"] == result["MOTA"] == 50
    assert result["HOTA"] == pytest.approx(np.sqrt(0.5) * 100)


@pytest.mark.parametrize(
    ("box", "expected_hota"),
    [
        ((0.0, 2.0, 20.0, 0.0, 4.0, 2.0, 2.0), 0),
        ((0.0, 2.0, 10.0, np.pi / 2, 4.0, 2.0, 2.0), 600 / 19),
        ((0.0, 5.0, 10.0, 0.0, 4.0, 2.0, 2.0), 0),
    ],
)
def test_matching_uses_volume_depth_height_and_yaw(
    tmp_path: Path, box: tuple[float, ...], expected_hota: float
) -> None:
    prediction_dir = _save(tmp_path, "0000", [(0, _tracks([box], [8], [1]))])
    result = evaluate_kitti_3d(prediction_dir, tmp_path / "metrics", {"0000": _truth([(0, 18, 1, _BOX)])}, {"0000": 1})[
        "car"
    ]

    assert result["HOTA"] == pytest.approx(expected_hota)
    assert result["CLR_TP"] == result["IDTP"] == 0
    assert result["CLR_FN"] == result["CLR_FP"] == 1


def test_sequences_do_not_share_identity_namespace_and_classes_do_not_match(tmp_path: Path) -> None:
    prediction_dir = _save(tmp_path, "0000", [(0, _tracks([_BOX, _BOX], [8, 9], [1, 2]))])
    _save(tmp_path, "0001", [(0, _tracks([_BOX], [8], [1]))])
    truth = _truth([(0, 18, 1, _BOX), (0, 19, 2, _BOX)])
    results = evaluate_kitti_3d(
        prediction_dir, tmp_path / "metrics", {"0000": truth, "0001": truth}, {"0000": 1, "0001": 1}
    )

    assert results["car"]["HOTA"] == 100
    assert results["car"]["GT_IDs"] == results["car"]["IDs"] == 2
    assert results["car"]["Frames"] == 2
    assert results["pedestrian"]["CLR_TP"] == results["pedestrian"]["CLR_FN"] == 1
    assert results["cls_comb_det_av"]["Dets"] == 3
    assert results["cls_comb_det_av"]["GT_Dets"] == 4


@pytest.mark.parametrize(("has_gt", "has_prediction"), [(False, False), (False, True), (True, False)])
def test_empty_ground_truth_and_predictions(tmp_path: Path, has_gt: bool, has_prediction: bool) -> None:
    frames = [(0, _tracks([_BOX], [8], [1]))] if has_prediction else []
    prediction_dir = _save(tmp_path, "0000", frames)
    truth = _truth([(0, 18, 1, _BOX)] if has_gt else [])
    result = evaluate_kitti_3d(prediction_dir, tmp_path / "metrics", {"0000": truth}, {"0000": 2})["car"]

    assert result["GT_Dets"] == result["CLR_FN"] == int(has_gt)
    assert result["Dets"] == result["CLR_FP"] == int(has_prediction)
    assert result["Frames"] == 2


def test_explicitly_ignored_gt_classes_are_excluded_by_reader(tmp_path: Path) -> None:
    gt_path = tmp_path / "ground-truth.txt"
    gt_path.write_text(
        "0 18 Car 0 0 0 10 20 30 40 2 2 4 0 2 10 0\n"
        "0 -1 DontCare -1 -1 -10 0 0 10 10 -1 -1 -1 -1000 -1000 -1000 -10\n"
        "0 19 Person 0 0 0 10 20 30 40 2 2 4 0 2 10 0\n"
    )
    truth = read_kitti_tracking_labels(
        gt_path,
        frame_count=1,
        classes={"car": {"id": 1}, "pedestrian": {"id": 2}},
        options={"ignore_classes": ["DontCare", "Person"]},
    )
    prediction_dir = _save(tmp_path, "0000", [(0, _tracks([_BOX], [8], [1]))])
    result = evaluate_kitti_3d(prediction_dir, tmp_path / "metrics", {"0000": truth}, {"0000": 1})["car"]
    assert truth.row_count == 3
    assert result["GT_Dets"] == 1
    assert result["HOTA"] == 100


def test_missing_predictions_cannot_silently_be_scored_as_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        evaluate_kitti_3d(tmp_path, tmp_path / "metrics", {"0000": _truth([])}, {"0000": 1})


@pytest.mark.parametrize("frame_counts", [{}, {"0001": 1}, {"0000": 0}])
def test_invalid_evaluation_sequence_counts_are_rejected(tmp_path: Path, frame_counts: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        evaluate_kitti_3d(tmp_path, tmp_path / "metrics", {"0000": _truth([])}, frame_counts)
