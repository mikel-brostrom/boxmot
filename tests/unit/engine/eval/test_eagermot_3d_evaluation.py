"""Exercise spatial scoring through the public eval command and cached replay."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from PIL import Image

from boxmot.engine.cli import boxmot
from boxmot.engine.eval import eagermot_kitti as replay
from boxmot.engine.eval import kitti_3d
from boxmot.engine.eval.evaluator import run_eval
from boxmot.engine.eval.kitti_3d import read_kitti_3d_results
from boxmot.trackers.eagermot.geometry import project_box3d
from tests.unit.engine.eval.test_eagermot_kitti import _arguments, _fixture


def _spatial_fixture(root: Path, *, with_objects: bool = False) -> SimpleNamespace:
    """Supply identity-bearing 3D GT; exact per-image object labels are optional."""
    data = _fixture(root)
    paths = data.reader_paths
    config = yaml.safe_load(data.dataset.read_text())
    config["modalities"]["ground_truth_3d"] = {"format": "kitti-tracking-labels", "path": "labels/{sequence}.txt"}
    if with_objects:
        config["modalities"]["ground_truth_objects"] = {"format": "kitti-object-labels", "path": "objects/{sequence}"}
    labels = root / "labels/0002.txt"
    labels.parent.mkdir()
    object_dir = root / "objects/0002"
    if with_objects:
        object_dir.mkdir(parents=True)
    projection = np.array([[400, 0, 240, 0], [0, 400, 120, 0], [0, 0, 1, 0]])
    paths["calibration"].write_text("P2: " + " ".join(map(str, projection.flat)) + "\n")
    np.save(paths["poses"], np.repeat(np.eye(4)[None], 45, axis=0))
    paths["detections_2d"].write_text("")
    rows = []
    for frame in range(45):
        Image.fromarray(np.zeros((240, 480, 3), dtype=np.uint8)).save(paths["images"] / f"{frame:06d}.png")
        object_rows = []
        for identity, (kind, label, x, length) in enumerate(
            (
                ("car_detections_3d", "Car", -5, 4),
                ("pedestrian_detections_3d", "Pedestrian", 5, 1),
            )
        ):
            bounds = project_box3d(np.array([x, 0, 20, 0, length, 1, 3]), projection, (240, 480))
            fields = [label, 0, 0, 0, *bounds, 3, 1, length, x, 0, 20, 0]
            row = " ".join(map(str, fields))
            (paths[kind] / f"{frame:06d}.txt").write_text(row + " 120\n")
            rows.append(f"{frame} {identity} {row}")
            object_rows.append(row)
        if with_objects:
            (object_dir / f"{frame:06d}.txt").write_text("\n".join(object_rows) + "\n")
    labels.write_text("\n".join(rows) + "\n")
    data.dataset.write_text(yaml.safe_dump(config))
    shutil.rmtree(data.ground_truth)
    return data


@pytest.mark.parametrize("cache_inputs", (False, True))
def test_cli_scores_spatial_tracks_without_reading_or_preparing_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_inputs: bool
) -> None:
    """3D tracking ignores absent optional labels and never requires official AP tools."""
    data = _spatial_fixture(tmp_path, with_objects=True)
    shutil.rmtree(tmp_path / "objects")
    data.reader_paths["detections_2d"].write_text("")

    def forbid_masks(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("3D scoring must not load GT masks or prepare MOTS output.")

    monkeypatch.setattr(replay, "kitti_mots_annotations", forbid_masks)
    monkeypatch.setattr(replay, "prepare_mots_tracks", forbid_masks)
    monkeypatch.setattr(replay, "evaluate_kitti_mots", forbid_masks)
    monkeypatch.setattr(kitti_3d, "validate_trackeval_kitti_dependencies", forbid_masks)
    monkeypatch.setattr(kitti_3d, "resolve_kitti_object_backend", forbid_masks)
    monkeypatch.setattr(kitti_3d, "evaluate_kitti_objects", forbid_masks)
    monkeypatch.setattr(kitti_3d, "evaluate_trackeval_kitti", forbid_masks)
    arguments = [*_arguments(data), "--eval-3d", "--cache-inputs" if cache_inputs else "--no-cache-inputs"]
    invocation = CliRunner().invoke(boxmot, arguments)

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    output = data.project / "val"
    metrics = json.loads((output / "metrics.json").read_text())
    for class_name in ("car", "pedestrian"):
        assert metrics[class_name]["HOTA"] == pytest.approx(100)
        assert metrics[class_name]["MOTA"] == metrics[class_name]["IDF1"] == pytest.approx(100)
        assert metrics[class_name]["GT_Dets"] == metrics[class_name]["Dets"] == 45
        assert metrics[class_name]["Frames"] == 45
    assert not (output / "mots").exists()
    predictions = read_kitti_3d_results(output / "kitti_3d/0002.txt", frame_count=45)
    assert set(predictions) == set(range(45))
    assert len(predictions[0]) == len(predictions[2]) == 2
    assert [row.track_id for row in predictions[0]] == [row.track_id for row in predictions[2]]
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["eval_3d"] is True
    assert "ground_truth" not in manifest["sequence_inputs"]["0002"]
    assert "ground_truth_objects" not in manifest["sequence_inputs"]["0002"]
    assert manifest["sequence_inputs"]["0002"]["ground_truth_3d"]["format"] == "kitti-tracking-labels"
    protocol = json.loads((output / "evaluation.json").read_text())
    assert protocol["tracking"]["geometry"] == "3d"
    assert "volumetric" in json.dumps(protocol).lower()
    assert not (output / "detection_metrics.json").exists()
    assert not (output / "tracking_2d_metrics.json").exists()
    assert "AP40" not in invocation.output and "Easy" not in invocation.output
    assert "3D tracking — volumetric IoU" in invocation.output


def test_python_eval_preserves_selected_3d_mode_and_previous_results(tmp_path: Path) -> None:
    data = _spatial_fixture(tmp_path)
    args = SimpleNamespace(dataset=data.dataset, tracker="eagermot", project=data.project, eval_3d=True)
    result = run_eval(args, verbose=False, show_progress=False)
    assert result.summary["HOTA"] == pytest.approx(100)
    assert result.detection_metrics is None
    assert result.tracking_2d_metrics is None
    assert "detection_metrics" not in result.to_dict()
    assert result.args.eval_3d is True
    assert result.args.eval_masks is False
    original = (result.exp_dir / "kitti_3d/0002.txt").read_bytes()
    second = run_eval(args, verbose=False, show_progress=False)
    assert second.exp_dir == data.project / "val2"
    assert second.raw == result.raw
    assert (result.exp_dir / "kitti_3d/0002.txt").read_bytes() == original


@pytest.mark.parametrize(("cache_inputs", "calibrate_kf"), ((False, False), (True, False), (True, True)))
def test_optional_ap_preserves_main_3d_scores_and_adds_separate_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_inputs: bool, calibrate_kf: bool
) -> None:
    data = _spatial_fixture(tmp_path, with_objects=True)
    args = SimpleNamespace(
        dataset=data.dataset,
        tracker="eagermot",
        project=data.project,
        eval_3d=True,
        cache_inputs=cache_inputs,
        calibrate_kf=calibrate_kf,
    )
    baseline = run_eval(args, verbose=False, show_progress=False)
    detection_metrics = {
        geometry: {label: {"easy": 71.0, "moderate": 65.0, "hard": 55.0} for label in ("car", "pedestrian")}
        for geometry in ("2d", "3d")
    }
    projected_metrics = {
        label: {"HOTA": 34.0, "MOTA": 35.0, "IDF1": 36.0, "AssA": 37.0, "AssRe": 38.0, "IDSW": 3, "IDs": 9}
        for label in ("car", "pedestrian", "cls_comb_cls_av", "cls_comb_det_av")
    }
    called: list[str] = []

    def object_backend(gt_dir: Path, prediction_dir: Path, frame_ids: list[str], output: Path) -> dict:
        assert len(frame_ids) == 45
        assert (gt_dir / "000000.txt").is_file()
        assert (prediction_dir / "000000.txt").is_file()
        called.append("objects")
        return detection_metrics

    def tracking_backend(*, gt_folder: Path, tracker_folder: Path, seq_info: dict[str, int]) -> dict:
        assert seq_info == {"0002": 45}
        assert (gt_folder / "label_02/0002.txt").is_file()
        assert (tracker_folder / "0002.txt").is_file()
        called.append("projected tracking")
        return projected_metrics

    monkeypatch.setattr(replay, "validate_kitti_evaluation_dependencies", lambda **_kwargs: None)
    monkeypatch.setattr(kitti_3d, "validate_kitti_evaluation_dependencies", lambda **_kwargs: None)
    monkeypatch.setattr(kitti_3d, "evaluate_kitti_objects", object_backend)
    monkeypatch.setattr(kitti_3d, "evaluate_trackeval_kitti", tracking_backend)
    result = run_eval(SimpleNamespace(**{**vars(args), "eval_ap": True}), verbose=False, show_progress=False)

    assert called == ["objects", "projected tracking"]
    assert result.raw == baseline.raw
    assert result.summary == baseline.summary
    assert result.detection_metrics == detection_metrics
    assert result.tracking_2d_metrics == projected_metrics
    assert result.to_dict()["detection_metrics"] == detection_metrics
    assert result.to_dict()["tracking_2d_metrics"] == projected_metrics
    assert json.loads((result.exp_dir / "metrics.json").read_text()) == baseline.raw
    assert json.loads((result.exp_dir / "detection_metrics.json").read_text()) == detection_metrics
    assert json.loads((result.exp_dir / "tracking_2d_metrics.json").read_text()) == projected_metrics
    assert (result.exp_dir / "tracking_2d_metrics.csv").is_file()
    if calibrate_kf:
        calibration = json.loads((result.exp_dir / "kf-tuning/calibration.json").read_text())
        assert calibration["statistics"]["matched"] == 90
        assert (result.exp_dir / "kf-tuning/calibrated.yaml").is_file()
    assert "AP40" in result.render()
    assert "3D tracking — volumetric IoU" in result.render()
    assert "2D tracking" in result.render()
    assert "AP40" not in baseline.render()
    invocation = CliRunner().invoke(
        boxmot,
        [
            *_arguments(data),
            "--eval-3d",
            "--eval-ap",
            "--cache-inputs" if cache_inputs else "--no-cache-inputs",
            *(["--calibrate-kf"] if calibrate_kf else []),
        ],
    )
    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    assert "AP40" in invocation.output
    assert "Easy" in invocation.output and "Moderate" in invocation.output and "Hard" in invocation.output
    assert "3D tracking — volumetric IoU" in invocation.output
    assert "2D tracking" in invocation.output


def test_conflicting_scoring_flags_fail_before_dataset_payloads(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--eval-3d", "--eval-masks"])
    assert invocation.exit_code == 2
    assert "Choose either --eval-3d or --eval-masks" in invocation.output
    assert not data.project.exists()


def test_optional_ap_requires_object_labels_before_tracking_or_dependency_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _spatial_fixture(tmp_path)

    def forbid_replay(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Missing annotations must fail before preparing replay.")

    monkeypatch.setattr(replay, "validate_kitti_evaluation_dependencies", forbid_replay)
    monkeypatch.setattr(replay, "_track_frame", forbid_replay)
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--eval-3d", "--eval-ap"])
    assert invocation.exit_code == 2
    assert "ground_truth_objects" in invocation.output
    assert not data.project.exists()


def test_optional_ap_requires_3d_mode_before_dataset_payloads(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--eval-ap"])
    assert invocation.exit_code == 2
    assert "--eval-ap requires --eval-3d" in invocation.output
    assert not data.project.exists()


def test_missing_optional_ap_evaluators_fail_before_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _spatial_fixture(tmp_path, with_objects=True)

    def missing(**_kwargs: Any) -> None:
        raise RuntimeError(
            "Install official KITTI evaluators with boxmot install --extra trackeval --kitti-devkit PATH."
        )

    monkeypatch.setattr(replay, "validate_kitti_evaluation_dependencies", missing)
    with pytest.raises(RuntimeError, match="boxmot install"):
        run_eval(
            SimpleNamespace(dataset=data.dataset, tracker="eagermot", project=data.project, eval_3d=True, eval_ap=True),
            verbose=False,
            show_progress=False,
        )
    assert not data.project.exists()
