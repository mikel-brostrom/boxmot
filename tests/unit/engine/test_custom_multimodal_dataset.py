"""Replay and tune a user's portable sensor bundle through the public commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from tests.unit.engine.eval.test_eagermot_kitti import _fixture


def _custom_dataset(tmp_path: Path) -> Path:
    """Author custom names and splits, then relocate the complete relative-path bundle."""
    data = _fixture(tmp_path / "source-bundle")
    original = data.root / "sequences/training"
    recordings = original.with_name("recordings")
    original.rename(recordings)
    (recordings / "0002").rename(recordings / "drive-001")
    shutil.copytree(recordings / "drive-001", recordings / "drive-002")

    dataset = yaml.safe_load(data.dataset.read_text(encoding="utf-8"))
    spatial_paths = []
    for role, reader_key in (
        ("image", "detections_2d"),
        ("car", "car_detections_3d"),
        ("pedestrian", "pedestrian_detections_3d"),
    ):
        directory = data.reader_paths[reader_key].parent.parent
        (directory / "training").rename(directory / "recordings")
        extension = ".txt" if role == "image" else ""
        original = directory / "recordings" / f"0002{extension}"
        first = original.with_name(f"drive-001{extension}")
        second = original.with_name(f"drive-002{extension}")
        original.rename(first)
        if first.is_dir():
            shutil.copytree(first, second)
        else:
            shutil.copyfile(first, second)
        renamed = directory.with_name(f"custom-{role}-detector")
        directory.rename(renamed)
        template = f"{renamed.relative_to(data.root)}/{{partition}}/{{sequence}}{extension}"
        if role == "image":
            dataset["modalities"]["detections_2d"]["path"] = template
        else:
            spatial_paths.append(template)
    dataset["modalities"]["detections_3d"]["paths"] = spatial_paths

    dataset.update(
        id="custom-sensor-rig",
        fps=12.5,
        default_split="validation",
        splits={
            "development": {"partition": "recordings", "sequences": ["drive-002"], "has_ground_truth": True},
            "validation": {"partition": "recordings", "sequences": ["drive-001"], "has_ground_truth": True},
        },
    )
    data.dataset.write_text(yaml.safe_dump(dataset), encoding="utf-8")
    relocated = data.root.with_name("my-dataset")
    data.root.rename(relocated)
    return relocated


def test_eval_accepts_relocated_custom_bundle_and_uses_authored_default_split(tmp_path: Path) -> None:
    """An own-named folder must evaluate without KITTI sequence or split conventions."""
    dataset = _custom_dataset(tmp_path)
    project = tmp_path / "evaluation"
    invocation = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            str(dataset),
            "--tracker",
            "eagermot",
            "--sequence",
            "drive-001",
            "--project",
            str(project),
        ],
    )

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    output = project / "validation"
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["dataset_id"] == "custom-sensor-rig"
    assert manifest["dataset_config"] == str((dataset / "dataset.yaml").resolve())
    assert manifest["split"] == "validation"
    assert manifest["sequences"] == {"drive-001": 3}
    assert manifest["fps"] == 12.5
    assert manifest["visualization"]["video_fps"] == 12.5
    assert manifest["sequence_inputs"]["drive-001"]["ground_truth"]["paths"] == [
        str((dataset / "sequences/recordings/drive-001/ground_truth").resolve())
    ]
    assert {path.name for path in (output / "mots").iterdir()} == {"drive-001.txt"}
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    for class_name in ("car", "pedestrian"):
        assert metrics[class_name]["HOTA"] == pytest.approx(100)
        assert set(metrics[class_name]["per_sequence"]) == {"drive-001"}


def test_tune_uses_requested_custom_split_in_actual_saved_sensor_trial(tmp_path: Path) -> None:
    """Tuning must preserve user sequence names throughout the trial and best profile."""
    optuna = pytest.importorskip("optuna")
    dataset = _custom_dataset(tmp_path)
    project = tmp_path / "tuning"
    invocation = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--dataset",
            str(dataset),
            "--tracker",
            "eagermot",
            "--split",
            "development",
            "--sequence",
            "drive-002",
            "--n-trials",
            "1",
            "--seed",
            "0",
            "--project",
            str(project),
        ],
    )

    assert invocation.exit_code == 0, (invocation.output, invocation.exception)
    output = project / "development"
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["dataset_id"] == "custom-sensor-rig"
    assert manifest["split"] == "development"
    assert manifest["sequences"] == {"drive-002": 3}
    assert manifest["fps"] == 12.5
    assert manifest["completed_trials"] == 1
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{(output / 'study.sqlite3').as_uri()}?uri=true")
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.best_value == pytest.approx(100)
    assert set(yaml.safe_load((output / "best.yaml").read_text(encoding="utf-8"))) == {"car", "pedestrian"}
    trial = output / "trials/0000"
    assert {path.name for path in (trial / "mots").iterdir()} == {"drive-002.txt"}
    metrics = json.loads((trial / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["cls_comb_cls_av"]["HOTA"] == pytest.approx(study.best_value)
    assert set(metrics["car"]["per_sequence"]) == {"drive-002"}
    assert not (project / "validation").exists()
