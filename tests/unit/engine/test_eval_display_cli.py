"""Evaluation display flags reach replay independently of Kalman calibration."""

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support


@pytest.mark.parametrize("flags", [[], ["--show"], ["--save"], ["--show", "--save"]])
@pytest.mark.parametrize("calibration", [False, True])
def test_eval_dispatch_preserves_display_flags(monkeypatch, flags, calibration) -> None:
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.update(args=args))
    argv = ["eval", "--dataset", "fixture", "--build", "fixture-build", *flags]
    if calibration:
        argv.append("--calibrate-kf")

    result = CliRunner().invoke(boxmot, argv)

    assert result.exit_code == 0, result.output
    assert captured["args"].show is ("--show" in flags)
    assert captured["args"].save is ("--save" in flags)
    assert captured["args"].calibrate_kf is calibration


def test_eval_help_describes_visualization_after_calibration() -> None:
    result = CliRunner().invoke(boxmot, ["eval", "--help"])

    assert result.exit_code == 0, result.output
    assert "--show" in result.output
    assert "--save" in result.output
    assert "after calibration when --calibrate-kf is enabled" in " ".join(result.output.split())
