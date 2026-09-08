"""The frame-loss workflow has a small, model-free CLI boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.engine.config import build_mode_namespace


def test_time_variant_dispatch_uses_only_dataset_derivation_options(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        _support, "_run_engine_workflow", lambda module, args: captured.update(module=module, args=args)
    )

    result = CliRunner().invoke(boxmot, ["time-variant", "--sequence", "MOT17-10-FRCNN", "--build", "parent-build"])

    assert result.exit_code == 0, result.output
    assert captured["module"] == "boxmot.engine.dataset_variants.workflow"
    assert vars(captured["args"]) == {
        "dataset": "mot17",
        "split": "ablation",
        "sequence": "MOT17-10-FRCNN",
        "build": "parent-build",
        "build_root": None,
        "data_root": None,
        "name": None,
        "seed": 0,
    }


def test_time_variant_accepts_source_paths_and_reproducible_selection(monkeypatch, tmp_path) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.update(args=args))
    dataset = tmp_path / "source.yaml"
    parent = tmp_path / "builds" / "parent"
    result = CliRunner().invoke(
        boxmot,
        [
            "time-variant",
            "--dataset",
            str(dataset),
            "--split",
            "train",
            "--sequence",
            "camera-1",
            "--build",
            str(parent),
            "--build-root",
            str(tmp_path / "builds"),
            "--data-root",
            str(tmp_path / "datasets"),
            "--name",
            "camera-1-drops",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.dataset == str(dataset)
    assert args.build == str(parent)
    assert args.split == "train"
    assert args.sequence == "camera-1"
    assert args.build_root == tmp_path / "builds"
    assert args.data_root == tmp_path / "datasets"
    assert args.name == "camera-1-drops"
    assert args.seed == 42


@pytest.mark.parametrize("arguments,missing", [([], "--sequence"), (["--sequence", "camera-1"], "--build")])
def test_time_variant_requires_a_sequence_and_parent_build(arguments, missing) -> None:
    result = CliRunner().invoke(boxmot, ["time-variant", *arguments])
    assert result.exit_code == 2
    assert f"Missing option '{missing}'" in result.output


@pytest.mark.parametrize("seed", ["-1", "1.5"])
def test_time_variant_rejects_invalid_seeds_before_workflow(monkeypatch, seed) -> None:
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda *args: pytest.fail("invalid CLI reached workflow"))
    result = CliRunner().invoke(boxmot, ["time-variant", "--sequence", "camera-1", "--build", "parent", "--seed", seed])
    assert result.exit_code == 2
    assert "--seed" in result.output


@pytest.mark.parametrize("option", ["--detector", "--tracker", "--fps"])
def test_time_variant_does_not_accept_inference_or_synthetic_timestamp_controls(option) -> None:
    result = CliRunner().invoke(
        boxmot, ["time-variant", "--sequence", "camera-1", "--build", "parent", option, "value"]
    )
    assert result.exit_code == 2
    assert f"No such option '{option}'" in result.output


def test_time_variant_namespace_filters_shared_tracking_defaults() -> None:
    args = build_mode_namespace(
        "time-variant",
        {
            "sequence": "camera-1",
            "build": "parent",
            "data_root": "datasets",
            "build_root": "builds",
            "tracker": "botsort",
            "fps": 200,
            "detector": "unused",
        },
    )
    assert vars(args) == {
        "dataset": "mot17",
        "split": "ablation",
        "sequence": "camera-1",
        "build": "parent",
        "data_root": Path("datasets"),
        "build_root": Path("builds"),
        "name": None,
        "seed": 0,
    }
