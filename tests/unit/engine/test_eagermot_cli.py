"""CLI contracts for saved KITTI sensor evaluation and class-profile tuning."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot


def _sensor_arguments(command: str, root: Path) -> list[str]:
    """Supply existing input roots without coupling CLI checks to sensor payloads."""
    return [command, "--data-root", str(root), "--images", str(root), "--instances", str(root)]


def test_tune_eagermot_dispatches_default_profiles_trial_count_and_cpu_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    def run(args: SimpleNamespace) -> Path:
        captured["args"] = args
        return tmp_path / "results"

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.tuning.eagermot_kitti", SimpleNamespace(run_eagermot_kitti_tuning=run)
    )
    result = CliRunner().invoke(boxmot, _sensor_arguments("tune-eagermot", tmp_path))

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.tracker == "eagermot"
    assert args.tracker_backend == "python"
    assert args.data_root == args.images == args.instances == tmp_path
    assert args.sequence_names == ()
    assert args.split == "val"
    assert args.pointgnn_car == "t2-train"
    assert args.n_trials == 50
    assert args.seed == 0
    assert args.project == Path("runs/eagermot-tune")
    assert f"Results: {tmp_path / 'results'}" in result.output
    assert f"Best profiles: {tmp_path / 'results' / 'best.yaml'}" in result.output


def test_tune_eagermot_preserves_explicit_sensor_selection_and_sampling_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    def run(args: SimpleNamespace) -> Path:
        captured["args"] = args
        return args.project / args.split

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.tuning.eagermot_kitti", SimpleNamespace(run_eagermot_kitti_tuning=run)
    )
    result = CliRunner().invoke(
        boxmot,
        [
            *_sensor_arguments("tune-eagermot", tmp_path),
            "--split",
            "fulltrain",
            "--pointgnn-car",
            "t3-trainval",
            "--sequence",
            "0006",
            "--sequence",
            "0002",
            "--n-trials",
            "1",
            "--seed",
            "19",
            "--project",
            str(tmp_path / "tuning"),
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.split == "fulltrain"
    assert args.pointgnn_car == "t3-trainval"
    assert args.sequence_names == ("0006", "0002")
    assert args.n_trials == 1
    assert args.seed == 19
    assert args.project == tmp_path / "tuning"


@pytest.mark.parametrize("n_trials", ("0", "-1", "1.5"))
def test_tune_eagermot_rejects_invalid_trial_counts_before_dispatch(tmp_path: Path, n_trials: str) -> None:
    result = CliRunner().invoke(boxmot, [*_sensor_arguments("tune-eagermot", tmp_path), "--n-trials", n_trials])

    assert result.exit_code == 2
    assert "Invalid value for '--n-trials'" in result.output


@pytest.mark.parametrize("seed", ("-1", str(2**32)))
def test_tune_eagermot_rejects_out_of_range_seeds_before_dispatch(tmp_path: Path, seed: str) -> None:
    result = CliRunner().invoke(boxmot, [*_sensor_arguments("tune-eagermot", tmp_path), "--seed", seed])

    assert result.exit_code == 2
    assert "Invalid value for '--seed'" in result.output


def test_eval_eagermot_passes_class_profile_yaml_to_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    def run(args: SimpleNamespace) -> Path:
        captured["args"] = args
        return tmp_path / "evaluation"

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    profiles = tmp_path / "best.yaml"
    profiles.write_text("car: {}\npedestrian: {}\n", encoding="utf-8")
    result = CliRunner().invoke(
        boxmot, [*_sensor_arguments("eval-eagermot", tmp_path), "--class-config", str(profiles)]
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    assert args.class_config == profiles
    assert args.tracker == "eagermot"
    assert args.tracker_backend == "python"
    assert args.project == Path("runs/eagermot")
    assert args.show is False
    assert args.save is False
    assert args.show_3d is False


@pytest.mark.parametrize(("show", "save"), ((False, True), (True, False), (True, True)))
@pytest.mark.parametrize("show_3d", (False, True))
def test_eval_eagermot_dispatches_preview_and_video_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, show: bool, save: bool, show_3d: bool
) -> None:
    captured = {}

    def run(args: SimpleNamespace) -> Path:
        captured["args"] = args
        return tmp_path / "evaluation"

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    arguments = _sensor_arguments("eval-eagermot", tmp_path)
    if show:
        arguments.append("--show")
    if save:
        arguments.append("--save")
    if show_3d:
        arguments.append("--show-3d")
    result = CliRunner().invoke(boxmot, arguments)

    assert result.exit_code == 0, result.output
    assert captured["args"].show is show
    assert captured["args"].save is save
    assert captured["args"].show_3d is show_3d


def test_eval_eagermot_rejects_3d_overlay_without_preview_or_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_run(_args: SimpleNamespace) -> Path:
        pytest.fail("Invalid visualization flags must be rejected before loading sensor data.")

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=unexpected_run)
    )
    result = CliRunner().invoke(boxmot, [*_sensor_arguments("eval-eagermot", tmp_path), "--show-3d"])

    assert result.exit_code == 2
    assert "--show-3d requires --show or --save" in result.output


def test_eval_eagermot_reports_video_writer_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    message = "Could not open MP4 video writer."

    def fail(_args: SimpleNamespace) -> Path:
        raise OSError(message)

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=fail))
    result = CliRunner().invoke(boxmot, [*_sensor_arguments("eval-eagermot", tmp_path), "--save"])

    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert "Traceback" not in result.output


def test_eval_eagermot_help_describes_preview_and_saved_videos() -> None:
    result = CliRunner().invoke(boxmot, ["eval-eagermot", "--help"], terminal_width=120)

    assert result.exit_code == 0, result.output
    assert "--show" in result.output
    assert "tracked masks, IDs, and classes" in result.output
    assert "--save" in result.output
    assert "results/videos" in result.output
    assert "--show-3d" in result.output
    assert "estimated tracked 3D cuboids" in result.output


@pytest.mark.parametrize("use_directory", (False, True))
def test_eval_eagermot_requires_an_existing_class_profile_file(tmp_path: Path, use_directory: bool) -> None:
    config = tmp_path if use_directory else tmp_path / "missing.yaml"
    result = CliRunner().invoke(boxmot, [*_sensor_arguments("eval-eagermot", tmp_path), "--class-config", str(config)])

    assert result.exit_code == 2
    assert "Invalid value for '--class-config'" in result.output


@pytest.mark.parametrize("error_type", (ValueError, FileNotFoundError, ImportError))
def test_tune_eagermot_reports_actionable_runner_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
) -> None:
    message = "Selected sequence lacks KITTI ground truth."

    def fail(_args: SimpleNamespace) -> Path:
        raise error_type(message)

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.tuning.eagermot_kitti", SimpleNamespace(run_eagermot_kitti_tuning=fail)
    )
    result = CliRunner().invoke(boxmot, _sensor_arguments("tune-eagermot", tmp_path))

    assert result.exit_code == 1
    assert f"Error: {message}" in result.output
    assert "Traceback" not in result.output


def test_tune_eagermot_help_explains_joint_profiles_without_a_per_class_flag() -> None:
    result = CliRunner().invoke(boxmot, ["tune-eagermot", "--help"], terminal_width=120)

    assert result.exit_code == 0, result.output
    assert "car and pedestrian" in result.output
    assert "class-average KITTI mask HOTA" in result.output
    assert "--per-class" not in result.output
    assert "--show" not in result.output
    assert "--save" not in result.output
    assert "--show-3d" not in result.output
