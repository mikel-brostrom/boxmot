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
from boxmot.engine.eval.evaluator import run_eval
from boxmot.engine.eval.kitti_3d import read_kitti_3d_results
from boxmot.trackers.eagermot.geometry import project_box3d
from tests.unit.engine.eval.test_eagermot_kitti import _arguments, _fixture


@pytest.fixture
def official_evaluators() -> None:
    """Run integration checks only when explicitly installed optional evaluators exist."""
    pytest.importorskip("trackeval")
    from boxmot.engine.eval.kitti_object_backend import resolve_kitti_object_backend

    try:
        resolve_kitti_object_backend()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        pytest.skip(str(error))


def _spatial_fixture(root: Path) -> SimpleNamespace:
    """Supply exact easy object GT on 45 frames, independently of identity GT."""
    data = _fixture(root)
    paths = data.reader_paths
    config = yaml.safe_load(data.dataset.read_text())
    config["modalities"]["ground_truth_3d"] = {"format": "kitti-tracking-labels", "path": "labels/{sequence}.txt"}
    config["modalities"]["ground_truth_objects"] = {"format": "kitti-object-labels", "path": "objects/{sequence}"}
    labels = root / "labels/0002.txt"
    labels.parent.mkdir()
    object_dir = root / "objects/0002"
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
        (object_dir / f"{frame:06d}.txt").write_text("\n".join(object_rows) + "\n")
    labels.write_text("\n".join(rows) + "\n")
    data.dataset.write_text(yaml.safe_dump(config))
    shutil.rmtree(data.ground_truth)
    return data


@pytest.mark.parametrize("cache_inputs", (False, True))
def test_cli_scores_spatial_tracks_without_reading_or_preparing_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_inputs: bool, official_evaluators: None
) -> None:
    """3D scoring survives absent mask GT and retains spatial-only output rows."""
    data = _spatial_fixture(tmp_path)
    data.reader_paths["detections_2d"].write_text("")

    def forbid_masks(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("3D scoring must not load GT masks or prepare MOTS output.")

    monkeypatch.setattr(replay, "kitti_mots_annotations", forbid_masks)
    monkeypatch.setattr(replay, "prepare_mots_tracks", forbid_masks)
    monkeypatch.setattr(replay, "evaluate_kitti_mots", forbid_masks)
    arguments = [*_arguments(data), "--eval-3d", "--cache-inputs" if cache_inputs else "--no-cache-inputs"]
    invocation = CliRunner().invoke(boxmot, arguments)

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    output = data.project / "val"
    metrics = json.loads((output / "metrics.json").read_text())
    for class_name in ("car", "pedestrian"):
        assert metrics[class_name]["HOTA"] == pytest.approx(100)
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
    assert manifest["sequence_inputs"]["0002"]["ground_truth_3d"]["format"] == "kitti-tracking-labels"
    protocol = json.loads((output / "evaluation.json").read_text())
    assert protocol["protocol"] == "kitti-official-object-and-trackeval-tracking"
    assert protocol["tracking"]["geometry"] == "2d"
    detection = json.loads((output / "detection_metrics.json").read_text())
    for geometry in ("2d", "3d"):
        for class_name in ("car", "pedestrian"):
            assert detection[geometry][class_name] == pytest.approx(dict.fromkeys(("easy", "moderate", "hard"), 100))
    assert "Easy" in invocation.output and "Moderate" in invocation.output and "Hard" in invocation.output
    assert "AP40" in invocation.output and "2D tracking" in invocation.output


def test_python_eval_preserves_selected_3d_mode_and_previous_results(tmp_path: Path, official_evaluators: None) -> None:
    data = _spatial_fixture(tmp_path)
    args = SimpleNamespace(dataset=data.dataset, tracker="eagermot", project=data.project, eval_3d=True)
    result = run_eval(args, verbose=False, show_progress=False)
    assert result.summary["HOTA"] == pytest.approx(100)
    assert result.detection_metrics["3d"]["car"]["easy"] == pytest.approx(100)
    assert result.to_dict()["detection_metrics"] == result.detection_metrics
    assert result.args.eval_3d is True
    assert result.args.eval_masks is False
    original = (result.exp_dir / "kitti_3d/0002.txt").read_bytes()
    second = run_eval(args, verbose=False, show_progress=False)
    assert second.exp_dir == data.project / "val2"
    assert second.raw == result.raw
    assert (result.exp_dir / "kitti_3d/0002.txt").read_bytes() == original


def test_conflicting_scoring_flags_fail_before_dataset_payloads(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--eval-3d", "--eval-masks"])
    assert invocation.exit_code == 2
    assert "Choose either --eval-3d or --eval-masks" in invocation.output
    assert not data.project.exists()


def test_missing_object_labels_fail_before_tracking_or_dependency_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _spatial_fixture(tmp_path)
    config = yaml.safe_load(data.dataset.read_text())
    del config["modalities"]["ground_truth_objects"]
    data.dataset.write_text(yaml.safe_dump(config))

    def forbid_replay(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Missing annotations must fail before preparing replay.")

    monkeypatch.setattr(replay, "validate_kitti_evaluation_dependencies", forbid_replay)
    monkeypatch.setattr(replay, "_track_frame", forbid_replay)
    invocation = CliRunner().invoke(boxmot, [*_arguments(data), "--eval-3d"])
    assert invocation.exit_code == 2
    assert "ground_truth_objects" in invocation.output
    assert not data.project.exists()


def test_missing_evaluators_fail_before_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = _spatial_fixture(tmp_path)

    def missing() -> None:
        raise RuntimeError(
            "Install official KITTI evaluators with boxmot install --extra trackeval --kitti-devkit PATH."
        )

    monkeypatch.setattr(replay, "validate_kitti_evaluation_dependencies", missing)
    with pytest.raises(RuntimeError, match="boxmot install"):
        run_eval(
            SimpleNamespace(dataset=data.dataset, tracker="eagermot", project=data.project, eval_3d=True),
            verbose=False,
            show_progress=False,
        )
    assert not data.project.exists()
