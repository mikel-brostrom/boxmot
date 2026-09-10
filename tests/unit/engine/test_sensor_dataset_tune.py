"""Ingest portable KITTI fusion bundles through the standard tuning command."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config.runtime import BOXMOT_DEFAULTS
from boxmot.engine.eval.results import ValidationResult
from boxmot.engine.tuning.results import TuneResult, TuneTrialResult
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


def _manifest(root: Path) -> Path:
    """Declare sequence inputs and prediction sets relative to their manifests."""
    return sensor_dataset_fixture(root).dataset


def _arguments(dataset: Path) -> list[str]:
    """Select fusion tuning without requiring separate sensor-root options."""
    return ["tune", "--dataset", str(dataset), "--tracker", "eagermot"]


def _result(args: SimpleNamespace, output: Path) -> TuneResult:
    """Return enough trial metrics to exercise the shared command's final report."""
    trial = TuneTrialResult(
        index=1,
        config={"car": {}, "pedestrian": {}},
        metrics=ValidationResult(
            benchmark="kitti-mots-fusion",
            raw={"cls_comb_cls_av": {"HOTA": 75}},
            summary_label="cls_comb_cls_av",
            summary={"HOTA": 75},
            args=args,
        ),
        score=(75,),
    )
    return TuneResult(
        benchmark=trial.benchmark,
        tracker="eagermot",
        trials=[trial],
        best=trial,
        best_config=trial.config,
        best_yaml=output / "best.yaml",
    )


def _forbid_perception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if bundle routing attempts to build or replay image detections."""
    command = importlib.import_module("boxmot.engine.commands.tune")

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Sensor datasets must skip perception preparation and cached replay.")

    monkeypatch.setattr(command, "_prepare_replay_build", unexpected)


@pytest.mark.parametrize("use_folder", (False, True))
def test_bundle_paths_and_defaults_are_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_folder: bool
) -> None:
    """Both supported dataset spellings resolve inputs and provenance identically."""
    manifest = _manifest(tmp_path / "bundle")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    _forbid_perception(monkeypatch)
    captured: dict[str, Any] = {}

    def run(args: SimpleNamespace, *, pipeline: Any = None) -> TuneResult:
        captured["args"] = args
        return _result(args, tmp_path / "results")

    command = importlib.import_module("boxmot.engine.commands.tune")
    dispatch = command._dispatch_cli_workflow

    def shared_dispatch(ctx: Any, mode: str, module: str, payload: dict[str, Any]) -> Any:
        captured["entrypoint"] = (mode, module)
        return dispatch(ctx, mode, module, payload)

    monkeypatch.setattr(command, "_dispatch_cli_workflow", shared_dispatch)

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.tuning.eagermot_kitti", SimpleNamespace(run_eagermot_kitti_tuning=run)
    )
    result = CliRunner().invoke(boxmot, _arguments(manifest.parent if use_folder else manifest))

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["entrypoint"] == ("tune", "boxmot.engine.tuning.tuner")
    args = captured["args"]
    assert args.split == "val"
    assert args.sequence_names == ("0002",)
    assert args.dataset == manifest.resolve()
    assert args.tracker == "eagermot"
    assert args.tracker_backend == "python"
    assert args.n_trials == BOXMOT_DEFAULTS.tune.n_trials
    assert args.seed == 0
    assert args.project == Path("runs/eagermot-tune")
    assert args.device == "cpu"
    assert args.max_concurrent_trials == 1
    assert args.sequence_workers == 1
    assert args.objectives == ("HOTA",)
    assert args.maximize == ("HOTA",)
    assert args.per_class is True
    assert args.eval_masks is True
    assert "best.yaml" in result.output


@pytest.mark.parametrize("max_concurrent_trials", ("0", "1"))
def test_bundle_accepts_explicit_supported_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, max_concurrent_trials: str
) -> None:
    """Compatible normal tune options retain their explicit sensor-run meaning."""
    manifest = _manifest(tmp_path)
    _forbid_perception(monkeypatch)
    captured: dict[str, Any] = {}

    def run(args: SimpleNamespace, *, pipeline: Any = None) -> TuneResult:
        captured["args"] = args
        return _result(args, args.project / args.split)

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.tuning.eagermot_kitti", SimpleNamespace(run_eagermot_kitti_tuning=run)
    )
    result = CliRunner().invoke(
        boxmot,
        [
            *_arguments(manifest),
            "--tracker-backend",
            "python",
            "--split",
            "val",
            "--sequence",
            "0002",
            "--n-trials",
            "1",
            "--seed",
            "19",
            "--project",
            str(tmp_path / "custom-results"),
            "--search-alg",
            "optuna",
            "--objectives",
            "HOTA",
            "--maximize",
            "HOTA",
            "--max-concurrent-trials",
            max_concurrent_trials,
            "--sequence-workers",
            "8",
            "--device",
            "cpu",
            "--eval-masks",
            "--per-class",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    args = captured["args"]
    assert args.sequence_names == ("0002",)
    assert args.n_trials == 1
    assert args.seed == 19
    assert args.project == tmp_path / "custom-results"
    assert args.verbose is True
    assert args.sequence_workers == 1


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("--n-trials", "0"),
        ("--n-trials", "-1"),
        ("--n-trials", "1.5"),
        ("--seed", "-1"),
        ("--seed", str(2**32)),
    ),
)
def test_sensor_tune_rejects_invalid_sampling_controls_before_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: str, value: str
) -> None:
    """Validate sampling bounds through the same parser as image tuning."""
    monkeypatch.setitem(sys.modules, "boxmot.engine.tuning.tuner", None)
    result = CliRunner().invoke(boxmot, [*_arguments(tmp_path), option, value])

    assert result.exit_code == 2
    assert f"Invalid value for '{option}'" in result.output


@pytest.mark.parametrize("error_type", (ValueError, FileNotFoundError, ImportError))
def test_sensor_tune_reports_actionable_runner_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error_type: type[Exception]
) -> None:
    """The common workflow preserves concise sensor runtime and dependency errors."""
    manifest = _manifest(tmp_path)
    message = "Selected sequence lacks KITTI ground truth."

    def fail(_args: SimpleNamespace, *, pipeline: Any = None) -> None:
        raise error_type(message)

    monkeypatch.setitem(
        sys.modules, "boxmot.engine.tuning.eagermot_kitti", SimpleNamespace(run_eagermot_kitti_tuning=fail)
    )
    result = CliRunner().invoke(boxmot, _arguments(manifest))

    assert result.exit_code == 1, (result.output, result.exception)
    assert message in result.output
    assert not any(line.startswith("Error:") for line in result.output.splitlines())
    if error_type is ImportError:
        assert "--extra mots --extra evolve" in result.output


def test_tune_help_explains_sensor_class_profiles() -> None:
    """The common command documents the sensor objective beside its shared controls."""
    result = CliRunner().invoke(boxmot, ["tune", "--help"], terminal_width=120)

    assert result.exit_code == 0, result.output
    assert "car and pedestrian" in result.output
    assert "class-average KITTI mask HOTA" in result.output


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
        ["--calibrate-kf"],
        ["--variable-dt"],
        ["--fps", "5"],
        ["--resume-tune", "old-study"],
        ["--time-budget-s", "10"],
        ["--search-alg", "random"],
        ["--objectives", "HOTA", "IDF1"],
        ["--maximize", "IDF1"],
        ["--minimize", "IDSW_rate"],
        ["--max-concurrent-trials", "2"],
        ["--sequence-workers", "0"],
        ["--device", "cuda:0"],
        ["--name", "custom"],
        ["--exist-ok"],
    ),
)
def test_bundle_rejects_unsupported_explicit_options_before_loading_tuners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, options: list[str]
) -> None:
    """Never silently ignore options or load a tuner for an invalid invocation."""
    manifest = _manifest(tmp_path)
    _forbid_perception(monkeypatch)
    monkeypatch.setitem(sys.modules, "boxmot.engine.tuning.eagermot_kitti", None)
    monkeypatch.setitem(sys.modules, "boxmot.engine.tuning.tuner", None)

    result = CliRunner().invoke(boxmot, [*_arguments(manifest), *options])

    assert result.exit_code == 2, (result.output, result.exception)
    assert options[0] in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    "missing",
    (
        "dataset.yaml",
        "replay.yaml",
        "sequences/training/0002/images",
        "sequences/training/0002/ground_truth",
        "sequences/training/0002/calibration.txt",
        "sequences/training/0002/poses.npy",
        "predictions/trackrcnn/manifest.yaml",
        "predictions/trackrcnn/training/0002.txt",
    ),
)
def test_bundle_reports_missing_manifest_or_input_root_before_starting_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """Incomplete bundles report the path to fix without launching perception."""
    _manifest(tmp_path)
    target = tmp_path / missing
    target.unlink() if target.is_file() else target.rmdir()
    _forbid_perception(monkeypatch)
    monkeypatch.setitem(sys.modules, "boxmot.engine.tuning.eagermot_kitti", None)

    result = CliRunner().invoke(boxmot, _arguments(tmp_path))

    assert result.exit_code == 2, (result.output, result.exception)
    assert missing in result.output
    assert "Traceback" not in result.output


def test_ordinary_dataset_with_build_retains_cached_tuning_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The new route leaves standard image-dataset tuning on its existing path."""
    command = importlib.import_module("boxmot.engine.commands.tune")
    captured: dict[str, Any] = {}

    def prepare(_ctx: Any, **kwargs: Any) -> tuple[None, str, str]:
        captured["preparation"] = kwargs
        return None, "mot17", str(tmp_path / "build")

    def dispatch(_ctx: Any, mode: str, module: str, kwargs: dict[str, Any]) -> None:
        captured.update(mode=mode, module=module, args=kwargs)

    monkeypatch.setattr(command, "_prepare_replay_build", prepare)
    monkeypatch.setattr(command, "_dispatch_cli_workflow", dispatch)
    monkeypatch.setitem(sys.modules, "boxmot.engine.tuning.eagermot_kitti", None)

    result = CliRunner().invoke(
        boxmot,
        ["tune", "--dataset", "mot17", "--build", str(tmp_path / "build"), "--tracker", "bytetrack"],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["preparation"]["dataset"] == "mot17"
    assert captured["mode"] == "tune"
    assert captured["module"] == "boxmot.engine.tuning.tuner"
    assert captured["args"]["tracker"] == "bytetrack"
    assert captured["args"]["build"] == str(tmp_path / "build")


def test_standard_tune_replays_bundle_and_saves_best_profiles_with_dataset_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Moving the complete bundle preserves real mask scoring and input provenance."""
    pytest.importorskip("optuna")
    from boxmot.engine.eval.eagermot_kitti import KITTI_PROFILES, load_kitti_profiles
    from tests.unit.engine.eval.test_eagermot_kitti import _fixture

    data = _fixture(tmp_path / "original-bundle")
    relocated = tmp_path / "relocated-bundle"
    data.root.rename(relocated)
    manifest = relocated / "dataset.yaml"
    project = tmp_path / "tuning-results"
    assert not data.root.exists()
    _forbid_perception(monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = CliRunner().invoke(
        boxmot,
        [*_arguments(manifest.parent), "--n-trials", "1", "--project", str(project)],
    )

    assert result.exit_code == 0, (result.output, result.exception)
    output = project / "val"
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete"
    assert run["completed_trials"] == 1
    assert run["best_hota"] == pytest.approx(100)
    assert run["objective"] == "cls_comb_cls_av.HOTA"
    assert run["sequences"] == {"0002": 3}
    assert run["dataset_id"] == "kitti-mots-fusion"
    assert run["dataset_config"] == str(manifest.resolve())
    assert run["replay_config"] == str(relocated / "replay.yaml")
    assert set(run["prediction_manifests"]) == {"image", "car", "pedestrian"}
    assert all(Path(path).is_relative_to(relocated) for path in run["prediction_manifests"].values())
    assert all(Path(path).is_relative_to(relocated) for path in run["sequence_inputs"]["0002"].values())
    assert load_kitti_profiles(output / "best.yaml") == KITTI_PROFILES
    metrics = json.loads((output / "trials/0000/metrics.json").read_text())
    assert metrics["cls_comb_cls_av"]["HOTA"] == pytest.approx(100)
    assert (output / "study.sqlite3").is_file()
