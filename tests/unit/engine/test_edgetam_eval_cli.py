"""Ground-truth prompted evaluation bypasses detector workflows and rejects unused flags."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def test_edgetam_eval_dispatches_without_materialization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The normal namespace and workflow loader preserve prompt settings without loading models."""
    command = importlib.import_module("boxmot.engine.commands.eval")
    captured = {}

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Prompted evaluation must bypass detector and saved-input preparation")

    for name in ("_prepare_replay_build", "_prepare_saved_2d_evaluation", "_prepare_sensor_evaluation"):
        monkeypatch.setattr(command, name, unexpected)
    monkeypatch.setitem(
        sys.modules,
        "boxmot.engine.eval.prompted",
        SimpleNamespace(main=lambda args: captured.update(vars(args))),
    )
    config = tmp_path / "edgetam.yaml"
    config.write_text("max_objects: 128\nbatch_size: 4\n", encoding="utf-8")
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            "mot17-mini",
            "--tracker",
            "edgetam",
            "--prompt-source",
            "first-visible",
            "--prompt-min-visibility",
            "0.95",
            "--prompt-weights",
            "models/my edgetam.pt",
            "--tracker-config",
            str(config),
            "--split",
            "train",
            "--sequence",
            "MOT17-02-FRCNN",
            "--data-root",
            str(tmp_path),
            "--project",
            str(tmp_path / "results"),
            "--device",
            "cpu",
            "--show",
            "--save",
            "--cache-inputs",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["tracker"] == "edgetam"
    assert captured["tracker_backend"] == "python"
    assert captured["prompt_source"] == "first-visible"
    assert captured["prompt_min_visibility"] == 0.95
    assert captured["prompt_weights"] == Path("models/my edgetam.pt")
    assert captured["tracker_config"] == str(config)
    assert captured["sequence_workers"] == 1
    assert captured["sequence_names"] == ("MOT17-02-FRCNN",)
    assert captured["split"] == "train"
    assert captured["show"] and captured["save"] and captured["cache_inputs"]
    assert captured["per_class"] is False
    assert "detector" not in captured and "reid" not in captured


def test_edgetam_eval_uses_strict_prompt_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unspecified worker count becomes one and visibility requires a fully visible person."""
    command = importlib.import_module("boxmot.engine.commands.eval")
    captured = {}
    monkeypatch.setattr(
        command, "_dispatch_cli_workflow", lambda _ctx, _mode, _target, payload: captured.update(payload)
    )
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--dataset", "mot17-mini", "--tracker", "edgetam", "--prompt-source", "first-visible"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["prompt_min_visibility"] == 1.0
    assert captured["prompt_weights"] == Path("edgetam.pt")
    assert captured["sequence_workers"] == 1


@pytest.mark.parametrize(
    "options, message",
    [
        (["--detector", "yolo26n"], "--detector"),
        (["--reid", "osnet-x0-25-msmt17"], "--reid"),
        (["--build", "some-build"], "--build"),
        (["--build-root", "builds"], "--build-root"),
        (["--fps", "15"], "--fps"),
        (["--eval-masks"], "--eval-masks"),
        (["--eval-3d"], "--eval-3d"),
        (["--eval-ap"], "--eval-ap"),
        (["--postprocessing", "gsi"], "--postprocessing"),
        (["--calibrate-kf"], "--calibrate-kf"),
        (["--asso-func", "iou"], "--asso-func"),
        (["--cmc-policy", "auto"], "--cmc-policy"),
        (["--variable-dt"], "--variable-dt"),
        (["--fixed-dt"], "--variable-dt"),
        (["--compare-trackeval"], "--compare-trackeval"),
        (["--edgetam"], "--edgetam"),
        (["--no-edgetam"], "--edgetam"),
        (["--mask-guidance-weights", "edgetam.pt"], "--mask-guidance-weights"),
        (["--mask-guidance-max-objects", "96"], "--mask-guidance-max-objects"),
        (["--allow-noncanonical-build"], "--allow-noncanonical-build"),
        (["--tracker-backend", "cpp"], "--tracker-backend python"),
        (["--per-class"], "--per-class"),
        (["--sequence-workers", "2"], "--sequence-workers 1"),
    ],
)
def test_edgetam_eval_rejects_unused_options_before_dispatch(
    monkeypatch: pytest.MonkeyPatch, options: list[str], message: str
) -> None:
    """Even explicit default-valued irrelevant flags are rejected before model setup."""
    command = importlib.import_module("boxmot.engine.commands.eval")
    monkeypatch.setattr(command, "_dispatch_cli_workflow", lambda *_args: pytest.fail("Unexpected workflow dispatch"))
    result = CliRunner().invoke(
        boxmot,
        ["eval", "--dataset", "mot17-mini", "--tracker", "edgetam", "--prompt-source", "first-visible", *options],
    )
    assert result.exit_code == 2, (result.output, result.exception)
    assert "EdgeTAM" in result.output
    assert message in result.output


@pytest.mark.parametrize(
    "options",
    [["--prompt-source", "first-visible"], ["--prompt-min-visibility", "1"], ["--prompt-weights", "edgetam.pt"]],
)
def test_prompt_eval_options_require_edgetam(options: list[str]) -> None:
    """Other trackers must not silently consume EdgeTAM-only options."""
    result = CliRunner().invoke(boxmot, ["eval", "--dataset", "mot17-mini", "--tracker", "bytetrack", *options])
    assert result.exit_code == 2, result.output
    assert "requires --tracker edgetam" in result.output


def test_edgetam_eval_requires_explicit_prompt_source() -> None:
    """GT initialization requires an explicit opt-in for EdgeTAM evaluation."""
    result = CliRunner().invoke(boxmot, ["eval", "--dataset", "mot17-mini", "--tracker", "edgetam"])
    assert result.exit_code == 2, result.output
    assert "requires --prompt-source first-visible" in result.output


def test_edgetam_eval_requires_dataset_instead_of_experiment() -> None:
    """Perception experiments do not describe the prompted protocol."""
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--tracker",
            "edgetam",
            "--prompt-source",
            "first-visible",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "requires --dataset, without --experiment" in result.output


@pytest.mark.parametrize("visibility", ["-0.1", "1.1", "nan"])
def test_edgetam_eval_rejects_invalid_visibility(visibility: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Visibility is a fraction bounded by zero and one."""
    command = importlib.import_module("boxmot.engine.commands.eval")
    monkeypatch.setattr(command, "_dispatch_cli_workflow", lambda *_args: pytest.fail("Unexpected workflow dispatch"))
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--dataset",
            "mot17-mini",
            "--tracker",
            "edgetam",
            "--prompt-source",
            "first-visible",
            "--prompt-min-visibility",
            visibility,
        ],
    )
    assert result.exit_code == 2, result.output
    assert "--prompt-min-visibility" in result.output
