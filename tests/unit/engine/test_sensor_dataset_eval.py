"""Saved KITTI sensor evaluation through the shared eval command."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.eval.results import ValidationResult
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


def _arguments(root: Path) -> list[str]:
    """Declare portable sensor inputs without loading their optional runtime."""
    dataset = sensor_dataset_fixture(root).dataset
    return ["eval", "--dataset", str(dataset), "--tracker", "eagermot"]


def _forbid_perception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sensor evaluation must read its saved inputs without preparing an image build."""
    command = importlib.import_module("boxmot.engine.commands.eval")

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Sensor datasets must skip perception preparation and cached replay.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)


@pytest.mark.parametrize("cpu_count", (2, 8, 32))
@pytest.mark.parametrize("configured_workers", (None, 4))
@pytest.mark.parametrize(
    "options",
    (
        [],
        ["--sequence-workers", "3"],
        ["--sequence-workers", "20"],
        ["--show"],
        ["--sequence", "0002", "--sequence", "0006"],
    ),
)
def test_sensor_eval_resolves_worker_count_for_all_selected_sequences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str], cpu_count: int, configured_workers: int | None
) -> None:
    """The public command applies automatic or explicit limits to the actual split."""
    data = sensor_dataset_fixture(tmp_path)
    names = ["0002", "0006", "0007", "0008", "0010", "0013", "0014", "0016", "0018"]
    monkeypatch.setattr("boxmot.engine.config.runtime.os.cpu_count", lambda: cpu_count)
    if configured_workers is not None:
        from boxmot.engine.config import runtime

        defaults = yaml.safe_load(runtime.RUNTIME_DEFAULTS_PATH.read_text())
        defaults["runtime"]["sequence_workers"] = configured_workers
        defaults_path = tmp_path / "runtime.yaml"
        defaults_path.write_text(yaml.safe_dump(defaults))
        monkeypatch.setattr(runtime, "RUNTIME_DEFAULTS_PATH", defaults_path)
    for name in names[1:]:
        shutil.copytree(tmp_path / "sequences/training/0002", tmp_path / "sequences/training" / name)
        for model in ("pointgnn-car", "pointgnn-pedestrian"):
            shutil.copytree(
                tmp_path / f"predictions/{model}/training/0002", tmp_path / f"predictions/{model}/training" / name
            )
        shutil.copyfile(
            tmp_path / "predictions/trackrcnn/training/0002.txt",
            tmp_path / f"predictions/trackrcnn/training/{name}.txt",
        )
    config = yaml.safe_load(data.dataset.read_text())
    config["splits"]["val"]["sequences"] = names
    data.dataset.write_text(yaml.safe_dump(config))
    captured: dict[str, Any] = {}

    def run(args: Any, **kwargs: Any) -> ValidationResult:
        captured["args"] = args
        return ValidationResult("kitti-mots-fusion", {}, "cls_comb_cls_av", {}, exp_dir=tmp_path / "results", args=args)

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    result = CliRunner().invoke(boxmot, ["eval", "--dataset", str(data.dataset), "--tracker", "eagermot", *options])

    assert result.exit_code == 0, (result.output, result.exception)
    selected = names[:2] if "--sequence" in options else names
    expected = min(len(selected), configured_workers or max(1, cpu_count - 2))
    if "--sequence-workers" in options:
        expected = min(9, int(options[1]))
    if "--show" in options:
        expected = 1
    assert captured["args"].sequence_workers == expected
    assert captured["args"].sequence_names == tuple(selected)


@pytest.mark.parametrize("use_folder", (False, True))
def test_sensor_eval_uses_shared_dispatch_and_portable_dataset_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_folder: bool
) -> None:
    """Both dataset spellings resolve through evaluator.main from any working directory."""
    dataset = sensor_dataset_fixture(tmp_path / "bundle").dataset
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _forbid_perception(monkeypatch)
    captured: dict[str, Any] = {}
    command = importlib.import_module("boxmot.engine.commands.eval")
    dispatch = command._dispatch_cli_workflow

    def shared_dispatch(ctx: Any, mode: str, module: str, payload: dict[str, Any]) -> Any:
        captured["entrypoint"] = (mode, module)
        return dispatch(ctx, mode, module, payload)

    def run(args: SimpleNamespace, **kwargs: Any) -> ValidationResult:
        captured["args"] = args
        return ValidationResult(
            "kitti-mots-fusion",
            {"cls_comb_cls_av": {"HOTA": 75}},
            "cls_comb_cls_av",
            {"HOTA": 75},
            exp_dir=tmp_path / "evaluation",
            args=args,
        )

    monkeypatch.setattr(command, "_dispatch_cli_workflow", shared_dispatch)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    result = CliRunner().invoke(
        boxmot, ["eval", "--dataset", str(dataset.parent if use_folder else dataset), "--tracker", "eagermot"]
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["entrypoint"] == ("eval", "boxmot.engine.eval.evaluator")
    args = captured["args"]
    assert args.dataset == dataset.resolve()
    assert args.split == "val"
    assert args.sequence_names == ("0002",)
    assert args.device == "cpu"
    assert args.sequence_workers == 1
    assert args.per_class is True
    assert args.eval_masks is True
    assert args.project == Path("runs/eagermot")
    assert "Results:" in result.output
    assert "evaluation" in result.output


@pytest.mark.parametrize(
    "options",
    (
        ["--tracker", "bytetrack"],
        ["--tracker-backend", "cpp"],
        ["--detector", "yolox"],
        ["--reid", "osnet"],
        ["--build", "existing-build"],
        ["--build-root", "builds"],
        ["--data-root", "override"],
        ["--experiment", "experiment.yaml"],
        ["--tracker-config", "override.yaml"],
        ["--variable-dt"],
        ["--fps", "5"],
        ["--compare-trackeval"],
        ["--allow-noncanonical-build"],
        ["--sequence-workers", "0"],
        ["--device", "cuda:0"],
        ["--name", "custom"],
        ["--exist-ok"],
    ),
)
def test_sensor_eval_rejects_unsupported_controls_before_loading_evaluators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    """Reject unsupported controls rather than discarding image workflow settings."""
    _forbid_perception(monkeypatch)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", None)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", None)

    result = CliRunner().invoke(boxmot, [*_arguments(tmp_path), *options])

    assert result.exit_code == 2, (result.output, result.exception)
    assert options[0] in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("option", ("--class-config", "--show-3d", "--eval-3d", "--eval-ap"))
def test_image_eval_rejects_sensor_only_options_before_preparing_a_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str
) -> None:
    """Keep sensor visualization and class profiles out of image replay workflows."""
    _forbid_perception(monkeypatch)
    profile = tmp_path / "best.yaml"
    profile.write_text("car: {}\npedestrian: {}\n", encoding="utf-8")
    flags = [option, str(profile)] if option == "--class-config" else [option, "--show"]
    if option == "--eval-ap":
        flags.append("--eval-3d")
    result = CliRunner().invoke(boxmot, ["eval", "--dataset", "mot17", "--tracker", "bytetrack", *flags])

    assert result.exit_code == 2, (result.output, result.exception)
    assert f"{option} requires a sensor dataset with --tracker eagermot" in result.output


@pytest.mark.parametrize("eval_ap", (False, True))
def test_sensor_eval_dispatches_optional_ap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_ap: bool) -> None:
    """Object annotations are selected only by the explicit AP flag."""
    from tests.unit.engine.eval.test_sensor_3d_inputs import _declare_3d_labels

    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset)
    data.ground_truth.rmdir()
    if not eval_ap:
        shutil.rmtree(tmp_path / "sequences/training/0002/object_labels")
    _forbid_perception(monkeypatch)
    captured: dict[str, Any] = {}

    def run(args: SimpleNamespace, **kwargs: Any) -> ValidationResult:
        captured["args"] = args
        return ValidationResult("kitti-mots-fusion", {}, "cls_comb_cls_av", {}, exp_dir=tmp_path / "results", args=args)

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    flags = ["--eval-3d", *(["--eval-ap"] if eval_ap else [])]
    result = CliRunner().invoke(boxmot, ["eval", "--dataset", str(data.dataset), "--tracker", "eagermot", *flags])

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"].eval_3d is True
    assert captured["args"].eval_masks is False
    assert captured["args"].eval_ap is eval_ap


def test_sensor_eval_ap_requires_3d_before_loading_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_perception(monkeypatch)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", None)
    result = CliRunner().invoke(boxmot, [*_arguments(tmp_path), "--eval-ap"])

    assert result.exit_code == 2, (result.output, result.exception)
    assert "--eval-ap requires --eval-3d" in result.output


def test_sensor_eval_ap_reports_missing_object_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.unit.engine.eval.test_sensor_3d_inputs import _declare_3d_labels

    data = sensor_dataset_fixture(tmp_path)
    _declare_3d_labels(data.dataset)
    config = yaml.safe_load(data.dataset.read_text())
    del config["modalities"]["ground_truth_objects"]
    data.dataset.write_text(yaml.safe_dump(config))
    _forbid_perception(monkeypatch)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.evaluator", None)
    result = CliRunner().invoke(
        boxmot, ["eval", "--dataset", str(data.dataset), "--tracker", "eagermot", "--eval-3d", "--eval-ap"]
    )

    assert result.exit_code == 2, (result.output, result.exception)
    assert "--eval-ap requires per-image KITTI object ground truth" in result.output
    assert "Add ground_truth_objects" in result.output


def test_sensor_eval_passes_class_profile_yaml_to_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    def run(args: SimpleNamespace, **kwargs: Any) -> ValidationResult:
        captured["args"] = args
        return ValidationResult(
            "kitti-mots-fusion",
            {"cls_comb_cls_av": {"HOTA": 75}},
            "cls_comb_cls_av",
            {"HOTA": 75},
            exp_dir=tmp_path / "evaluation",
            args=args,
        )

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    profiles = tmp_path / "best.yaml"
    profiles.write_text("car: {}\npedestrian: {}\n", encoding="utf-8")
    result = CliRunner().invoke(boxmot, [*_arguments(tmp_path), "--class-config", str(profiles)])

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
def test_sensor_eval_dispatches_preview_and_video_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, show: bool, save: bool, show_3d: bool
) -> None:
    captured = {}

    def run(args: SimpleNamespace, **kwargs: Any) -> ValidationResult:
        captured["args"] = args
        return ValidationResult(
            "kitti-mots-fusion",
            {"cls_comb_cls_av": {"HOTA": 75}},
            "cls_comb_cls_av",
            {"HOTA": 75},
            exp_dir=tmp_path / "evaluation",
            args=args,
        )

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    arguments = _arguments(tmp_path)
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


def test_sensor_eval_rejects_3d_overlay_without_preview_or_save(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_run(_args: SimpleNamespace) -> Path:
        pytest.fail("Invalid visualization flags must be rejected before loading sensor data.")

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=unexpected_run)
    )
    result = CliRunner().invoke(boxmot, [*_arguments(tmp_path), "--show-3d"])

    assert result.exit_code == 2
    assert "--show-3d requires --show or --save" in result.output


def test_sensor_eval_reports_video_writer_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    message = "Could not open MP4 video writer."

    def fail(_args: SimpleNamespace, **kwargs: Any) -> Path:
        raise OSError(message)

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=fail))
    result = CliRunner().invoke(boxmot, [*_arguments(tmp_path), "--save"])

    assert result.exit_code == 1
    assert message in result.output
    assert result.output.count("Traceback (most recent call last)") == 1


def test_sensor_eval_help_describes_preview_and_saved_videos() -> None:
    result = CliRunner().invoke(boxmot, ["eval", "--help"], terminal_width=120)

    assert result.exit_code == 0, result.output
    assert "--show" in result.output
    assert "tracked masks, IDs, and classes" in result.output
    assert "--save" in result.output
    assert "results/videos" in result.output
    assert "--show-3d" in result.output
    assert "estimated tracked 3D cuboids" in result.output
    assert "--eval-3d" in result.output
    assert "3D tracking metrics from KITTI tracking ground truth" in result.output
    assert "--eval-ap" in result.output
    assert "Add official KITTI 2D/3D AP40" in result.output


@pytest.mark.parametrize("use_directory", (False, True))
def test_sensor_eval_requires_an_existing_class_profile_file(tmp_path: Path, use_directory: bool) -> None:
    config = tmp_path if use_directory else tmp_path / "missing.yaml"
    result = CliRunner().invoke(boxmot, [*_arguments(tmp_path), "--class-config", str(config)])

    assert result.exit_code == 2
    assert "Invalid value for '--class-config'" in result.output
