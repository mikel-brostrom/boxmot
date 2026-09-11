"""Exercise spatial scoring through the public eval command and cached replay."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.eval import eagermot_kitti as replay
from boxmot.engine.eval.evaluator import run_eval
from boxmot.engine.eval.kitti_3d import read_kitti_3d_results
from tests.unit.engine.eval.test_eagermot_kitti import _arguments, _fixture


def _spatial_fixture(root: Path) -> SimpleNamespace:
    """Keep 3D identity GT while removing the mask GT payload entirely."""
    data = _fixture(root)
    config = yaml.safe_load(data.dataset.read_text())
    config["modalities"]["ground_truth_3d"] = {"format": "kitti-tracking-labels", "path": "labels/{sequence}.txt"}
    labels = root / "labels/0002.txt"
    labels.parent.mkdir()
    rows = []
    for frame in (0, 2):
        for identity, kind in enumerate(("car_detections_3d", "pedestrian_detections_3d")):
            fields = (data.reader_paths[kind] / f"{frame:06d}.txt").read_text().split()
            rows.append(f"{frame} {identity} " + " ".join(fields[:-1]))
    labels.write_text("\n".join(rows) + "\n")
    data.dataset.write_text(yaml.safe_dump(config))
    shutil.rmtree(data.ground_truth)
    return data


@pytest.mark.parametrize("cache_inputs", (False, True))
def test_cli_scores_spatial_tracks_without_reading_or_preparing_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_inputs: bool
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
        assert metrics[class_name]["GT_Dets"] == metrics[class_name]["Dets"] == 2
        assert metrics[class_name]["Frames"] == 3
    assert not (output / "mots").exists()
    predictions = read_kitti_3d_results(output / "kitti_3d/0002.txt", frame_count=3)
    assert set(predictions) == {0, 2}
    assert len(predictions[0]) == len(predictions[2]) == 2
    assert [row.track_id for row in predictions[0]] == [row.track_id for row in predictions[2]]
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["eval_3d"] is True
    assert "ground_truth" not in manifest["sequence_inputs"]["0002"]
    assert manifest["sequence_inputs"]["0002"]["ground_truth_3d"]["format"] == "kitti-tracking-labels"
    protocol = json.loads((output / "evaluation.json").read_text())
    assert protocol["similarity"] == "volumetric 3D IoU"
    assert protocol["official_kitti_protocol"] is False


def test_python_eval_preserves_selected_3d_mode_and_previous_results(tmp_path: Path) -> None:
    data = _spatial_fixture(tmp_path)
    args = SimpleNamespace(dataset=data.dataset, tracker="eagermot", project=data.project, eval_3d=True)
    result = run_eval(args, verbose=False, show_progress=False)
    assert result.summary["HOTA"] == pytest.approx(100)
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
