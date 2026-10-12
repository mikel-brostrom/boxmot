"""Joint selection requires a supported runtime before preparing input data."""

from __future__ import annotations

import importlib

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES


@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_joint_selection_forwards_both_flags(monkeypatch: pytest.MonkeyPatch, tracker: str) -> None:
    """Joint selection preserves both flags when dispatching an eligible tracker."""
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.update(args=args))
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture",
            "--tracker",
            tracker,
            "--select-kf",
            "--select-cmc",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"].select_cmc is captured["args"].select_kf is True
    assert captured["args"].tracker == tracker


@pytest.mark.parametrize(
    "flags,message",
    [
        ([], "requires --select-kf"),
        (["--select-kf", "--tracker-backend", "cpp"], "requires a Python 2D Kalman tracker"),
        (["--select-kf", "--tracker", "sfsort"], "requires a Python 2D Kalman tracker"),
        (["--select-kf", "--tracker", "eagermot"], "requires a Python 2D Kalman tracker"),
        (["--select-kf", "--resume-tune", "old"], "cannot be combined with --resume-tune"),
    ],
)
def test_invalid_joint_selection_fails_before_input_preparation(
    monkeypatch: pytest.MonkeyPatch, flags: list[str], message: str
) -> None:
    """Invalid camera/filter requests fail before opening dataset inputs."""
    command = importlib.import_module("boxmot.engine.commands.tune")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid joint selection must not prepare input data.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)
    monkeypatch.setattr(command, "_prepare_sensor_tuning", unexpected)
    result = CliRunner().invoke(
        boxmot, ["tune", "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--select-cmc", *flags]
    )
    assert result.exit_code == 2, (result.output, result.exception)
    assert message in result.output
