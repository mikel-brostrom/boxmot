"""Saved box datasets bypass detectors and retain explicit appearance selection."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


def _saved_dataset(root: Path) -> Path:
    """Declare existing predictions without requiring any decoded fixture pixels."""
    fixture = sensor_dataset_fixture(root)
    config = yaml.safe_load(fixture.dataset.read_text())
    config["id"] = "saved-kitti-boxes"
    config["classes"] = {"target": {"car": 1, "pedestrian": 2}}
    config["modalities"] = {
        key: value for key, value in config["modalities"].items() if key in {"images", "detections_2d"}
    }
    config["modalities"]["detections_2d"]["options"] = {"load_masks": False}
    config["modalities"]["ground_truth"] = {"format": "kitti-tracking-labels", "path": "labels/{sequence}.txt"}
    (root / "labels").mkdir()
    (root / "labels/0002.txt").touch()
    fixture.dataset.write_text(yaml.safe_dump(config))
    return fixture.dataset


@pytest.mark.parametrize("tracker,reid", [("occluboost", "osnet-x0-25-msmt17"), ("bytetrack", None)])
@pytest.mark.parametrize("backend", ("python", "cpp"))
def test_saved_dataset_dispatches_without_detector_and_preserves_reid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracker: str, reid: str | None, backend: str
) -> None:
    dataset = _saved_dataset(tmp_path)
    captured = {}
    command = importlib.import_module("boxmot.engine.commands.eval")
    monkeypatch.setattr(
        command, "_prepare_replay_build", lambda *_args, **_kwargs: pytest.fail("Detector build requested")
    )
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.saved_detections",
        SimpleNamespace(main=lambda args: captured.update(args=vars(args))),
    )
    args = [
        "eval",
        "--dataset",
        str(dataset),
        "--tracker",
        tracker,
        "--tracker-backend",
        backend,
        "--cache-inputs",
        "--sequence",
        "0002",
    ]
    if reid:
        args += ["--reid", reid]
    result = CliRunner().invoke(boxmot, args)

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"]["dataset"] == dataset.resolve()
    assert captured["args"]["reid"] == reid
    assert captured["args"]["saved_detections"] is True
    assert captured["args"]["sequence_names"] == ("0002",)
    assert captured["args"]["sequence_workers"] == 1
    assert captured["args"]["cache_inputs"] is True
    assert captured["args"]["tracker_backend"] == backend
    assert "detector" not in captured["args"]


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--tracker", "occluboost"], "Add --reid"),
        (["--tracker", "occluboost", "--tracker-backend", "cpp"], "Add --reid"),
        (["--tracker", "botsort", "--tracker-backend", "cpp"], "Add --reid"),
        (["--tracker", "bytetrack", "--reid", "osnet-x0-25-msmt17"], "does not use embeddings"),
        (["--tracker", "maf_hda"], "requires instance masks"),
        (["--tracker", "eagermot"], "requires 3D detections"),
        (["--tracker", "bytetrack", "--detector", "yolo26n"], "does not support --detector"),
        (["--tracker", "bytetrack", "--build", "a-build"], "does not support --build"),
        (["--tracker", "bytetrack", "--eval-3d"], "does not support --eval-3d"),
        (["--tracker", "bytetrack", "--calibrate-kf"], "does not support --calibrate-kf"),
        (["--tracker", "bytetrack", "--sequence-workers", "2"], "requires --sequence-workers 1"),
    ],
)
def test_saved_dataset_rejects_unconsumed_or_missing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str], message: str
) -> None:
    dataset = _saved_dataset(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.saved_detections",
        SimpleNamespace(main=lambda _args: pytest.fail("Unexpected replay")),
    )
    result = CliRunner().invoke(boxmot, ["eval", "--dataset", str(dataset), *flags])
    assert result.exit_code == 2, result.output
    assert message in result.output


@pytest.mark.parametrize("backend", ("python", "cpp"))
def test_saved_dataset_accepts_appearance_disabled_tracker_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    dataset = _saved_dataset(tmp_path)
    profile = tmp_path / "motion.yaml"
    profile.write_text("use_embeddings: false\n")
    captured = {}
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.saved_detections",
        SimpleNamespace(main=lambda args: captured.update(args=vars(args))),
    )
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            str(dataset),
            "--tracker",
            "occluboost",
            "--tracker-backend",
            backend,
            "--tracker-config",
            str(profile),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"]["reid"] is None
