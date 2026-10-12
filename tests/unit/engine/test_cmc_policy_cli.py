"""Sequence CMC policies are explicit fixed-profile image evaluation inputs."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.commands import _support
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES


def test_eval_forwards_policy_and_saved_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Evaluating a saved best config must preserve it alongside the CMC policy."""
    captured = {}
    policy = tmp_path / "cmc.yaml"
    policy.write_text("method: sof\ndefault: false\nsequences:\n  moving: true\n")
    config = tmp_path / "best.yaml"
    config.write_text("tracker: bytetrack\ntrack_thresh: 0.623\nkalman:\n  type: xyah\n")
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.update(args=args))
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture-build",
            "--tracker",
            "bytetrack",
            "--tracker-config",
            str(config),
            "--cmc-policy",
            str(policy),
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"].cmc_policy == str(policy)
    assert captured["args"].tracker_config == str(config)
    assert captured["args"].calibrate_kf is False


@pytest.mark.parametrize(
    "flags,contents,message",
    [
        (["--calibrate-kf"], "default: false", "cannot be combined with --calibrate-kf"),
        (["--tracker-backend", "cpp"], "default: false", "Python"),
        ([], "sequences:\n  moving: 'false'", "bool"),
        ([], "method: invalid", "method"),
    ],
)
def test_invalid_policy_fails_before_materialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str], contents: str, message: str
) -> None:
    """Reject incompatible or malformed policy before preparing expensive inputs."""
    command = importlib.import_module("boxmot.engine.commands.eval")
    policy = tmp_path / "cmc.yaml"
    policy.write_text(contents)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid policy must fail before materialization.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--tracker",
            "bytetrack",
            "--cmc-policy",
            str(policy),
            *flags,
        ],
    )
    assert result.exit_code == 2, (result.output, result.exception)
    assert message in result.output


def test_evaluator_forwards_loaded_policy_without_altering_kf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The policy reaches replay while the selected KF and thresholds stay fixed."""
    from boxmot.engine.eval import evaluator
    from boxmot.engine.eval.replay import ReplayResult
    from tests.unit.engine.eval.test_eval_display_workflow import _args

    policy = tmp_path / "cmc.yaml"
    policy.write_text("method: sof\ndefault: false\nsequences:\n  sequence: true\n")
    config = tmp_path / "best.yaml"
    config.write_text("tracker: bytetrack\ntrack_thresh: 0.623\nkalman:\n  type: xyah\n")

    def replay(build, spec, **kwargs):
        resolved = kwargs["cmc_policy"].apply(spec, "sequence")
        assert resolved.option_dict["use_cmc"] is True
        assert resolved.option_dict["track_thresh"] == 0.623
        assert resolved.option_dict["kalman.type"] == "xyah"
        assert spec.option_dict["use_cmc"] is False
        return ReplayResult(build, kwargs["output_dir"], (), 2, 0)

    monkeypatch.setattr(evaluator, "replay_build", replay)
    monkeypatch.setattr(evaluator, "run_motmetrics", lambda *args, **kwargs: {"HOTA": 60.0})
    args = _args(tmp_path, cmc_policy=policy, tracker_config=config)
    result = evaluator.run_eval(args, setup=False, output_dir=tmp_path / "output")
    assert result.summary == {"HOTA": 60.0}


def test_cmc_policy_help_is_eval_only() -> None:
    """This policy evaluates saved settings and does not claim joint KF calibration."""
    for mode in ("eval", "tune", "track"):
        result = CliRunner().invoke(boxmot, [mode, "--help"])
        assert result.exit_code == 0
        assert ("--cmc-policy" in result.output) is (mode == "eval")


@pytest.mark.parametrize("policy", ["auto", "off"])
@pytest.mark.parametrize("tracker", sorted(KALMAN_TRACKER_NAMES))
def test_eval_accepts_camera_analysis_modes(monkeypatch, policy, tracker):
    """Automatic and disabled policies are values rather than YAML paths."""
    captured = {}
    monkeypatch.setattr(_support, "_run_engine_workflow", lambda module, args: captured.update(args=args))
    result = CliRunner().invoke(
        boxmot,
        [
            "eval",
            "--experiment",
            "mot17/ablation-yolox-lmbn.yaml",
            "--build",
            "fixture-build",
            "--tracker",
            tracker,
            "--cmc-policy",
            policy,
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"].cmc_policy == policy


@pytest.mark.parametrize(
    "flags",
    [
        ["--tracker", "sfsort"],
        ["--tracker", "eagermot"],
        ["--tracker", "bytetrack", "--tracker-backend", "cpp"],
    ],
)
def test_auto_policy_rejects_unsupported_tracker_before_materialization(monkeypatch, flags):
    command = importlib.import_module("boxmot.engine.commands.eval")

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported automatic policy must fail before materialization.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)
    result = CliRunner().invoke(
        boxmot, ["eval", "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--cmc-policy", "auto", *flags]
    )
    assert result.exit_code == 2
    assert "Python 2D Kalman tracker" in result.output


def test_saved_detection_api_rejects_policy_before_loading_inputs() -> None:
    """Python callers also receive an error instead of silently ignoring the policy."""
    from boxmot.engine.eval.saved_detections import run_saved_detections

    with pytest.raises(ValueError, match="cmc-policy requires a materialized image build"):
        run_saved_detections(SimpleNamespace(cmc_policy="policy.yaml"))
