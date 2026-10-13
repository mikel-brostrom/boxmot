"""Select the filter before tuning without materializing unsupported requests."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES


@pytest.mark.parametrize("calibrate", [False, True])
@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_selection_dispatches_without_requiring_calibration_flag(
    monkeypatch: pytest.MonkeyPatch, calibrate: bool, tracker: str
) -> None:
    """The selection workflow itself owns calibration and preserves explicit refinements."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.update(args=args))
    flags = ["--calibrate-kf"] if calibrate else []
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture-build",
            "--tracker",
            tracker,
            "--select-kf",
            "--tune-kf",
            "measurement_noise_scale",
            *flags,
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"].select_kf is True
    assert captured["args"].calibrate_kf is calibrate
    assert captured["args"].tune_kf == ("measurement_noise_scale",)


@pytest.mark.parametrize(
    "tracker,backend", [("bytetrack", "cpp"), ("sfsort", "python"), ("maf_hda", "python"), ("eagermot", "python")]
)
def test_unsupported_selection_rejected_before_inputs(
    monkeypatch: pytest.MonkeyPatch, tracker: str, backend: str
) -> None:
    """Selection is restricted to searchable Python 2D models before any I/O."""
    command = importlib.import_module("boxmot.engine.commands.tune")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Unsupported selection must not prepare sensor or perception inputs.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)
    monkeypatch.setattr(command, "_prepare_sensor_tuning", unexpected)
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--select-kf",
            "--tracker",
            tracker,
            "--tracker-backend",
            backend,
        ],
    )

    assert result.exit_code == 2, result.output
    assert "--select-kf" in result.output
    assert "Python" in result.output


def test_selection_cannot_replace_a_resumed_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved choice can be resumed without running another selection."""
    command = importlib.import_module("boxmot.engine.commands.tune")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Fresh selection cannot replace a resumed search before input preparation.")

    monkeypatch.setattr(command, "_prepare_sensor_tuning", unexpected)
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--select-kf",
            "--resume-tune",
            "previous-run",
        ],
    )
    assert result.exit_code == 2
    assert "--select-kf cannot be combined with --resume-tune" in result.output


def test_selection_rejects_conflicting_profile_timing_before_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Selection honors input profile validation before fitting new priors."""
    command = importlib.import_module("boxmot.engine.commands.tune")
    profile = tmp_path / "seconds.yaml"
    profile.write_text("tracker: bytetrack\nkalman.variable_dt: true\nkalman.noise.time_unit: seconds\n")

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Timing conflicts must be rejected before materialization.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)
    result = CliRunner().invoke(
        boxmot,
        [
            "tune",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--select-kf",
            "--tracker",
            "bytetrack",
            "--tracker-config",
            str(profile),
            "--fixed-dt",
        ],
    )
    assert result.exit_code == 2
    assert "kalman.noise.time_unit" in result.output


def test_selection_help_is_tune_only() -> None:
    """Evaluation and tracking do not silently select a different motion model."""
    result = CliRunner().invoke(boxmot, ["tune", "--help"], terminal_width=120)
    assert result.exit_code == 0, result.output
    assert "--select-kf" in result.output
    assert "Compare default, partial and full KF calibration" in " ".join(result.output.split())
    assert "held-out detection/GT predictions" in " ".join(result.output.split())
    for mode in ("eval", "track"):
        result = CliRunner().invoke(boxmot, [mode, "--help"], terminal_width=120)
        assert "--select-kf" not in result.output
