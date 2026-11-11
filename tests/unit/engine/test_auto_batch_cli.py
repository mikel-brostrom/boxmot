"""Materialization CLI contracts for optional CUDA batch sizing."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.runtime import build_mode_namespace


def test_materialize_auto_batch_options_reach_workflow(monkeypatch) -> None:
    captured = {}

    def main(args):
        captured.update(vars(args))

    monkeypatch.setitem(sys.modules, "boxmot.engine.materialization.workflow", SimpleNamespace(main=main))
    result = CliRunner().invoke(
        boxmot,
        [
            "materialize",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--device",
            "1",
            "--auto-batch",
            "--auto-batch-max-size",
            "512",
            "--auto-batch-memory-fraction",
            "0.7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["auto_batch"] is True
    assert captured["auto_batch_max_size"] == 512
    assert captured["auto_batch_memory_fraction"] == 0.7
    assert captured["device"] == "cuda:1"


def test_auto_batch_is_opt_in_for_materialization_python_api() -> None:
    defaults = build_mode_namespace("materialize", {})
    assert defaults.auto_batch is False
    assert defaults.auto_batch_max_size == 256
    assert defaults.auto_batch_memory_fraction == 0.8
    selected = build_mode_namespace("materialize", {"auto_batch": True, "auto_batch_max_size": 128})
    assert selected.auto_batch is True
    assert selected.auto_batch_max_size == 128


@pytest.mark.parametrize(
    "option",
    [
        ["--auto-batch-max-size", "0"],
        ["--auto-batch-memory-fraction", "0"],
        ["--auto-batch-memory-fraction", "1"],
    ],
)
def test_materialize_rejects_invalid_profiling_limits(option) -> None:
    result = CliRunner().invoke(
        boxmot,
        [
            "materialize",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            *option,
        ],
    )
    assert result.exit_code == 2


def test_time_variant_rejects_auto_batch() -> None:
    result = CliRunner().invoke(boxmot, ["materialize", "--time-variant", "--auto-batch"])
    assert result.exit_code == 2
    assert "--auto-batch cannot be combined with --time-variant" in result.output
