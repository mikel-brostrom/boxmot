"""Exercise the shared tuning entrypoints without loading sensor or Ray runtimes."""

from __future__ import annotations

import sys
from pathlib import Path
from shutil import copy2, copytree
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from boxmot.engine.cli import boxmot
from boxmot.engine.config import runtime
from boxmot.engine.eval.results import ValidationResult
from boxmot.engine.tuning import tuner
from boxmot.engine.tuning.results import TuneResult, TuneTrialResult
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


def _forbid_sensor_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if rejected or image-only arguments start sensor optimization."""

    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("This invocation must not start sensor optimization.")

    monkeypatch.setattr(tuner, "_run_eagermot_tuning", unexpected)


def _arguments(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    """Use portable declared inputs with the minimal public API selection."""
    dataset = sensor_dataset_fixture(tmp_path / "dataset").dataset
    return SimpleNamespace(**{"dataset": dataset, "tracker": "eagermot", **overrides})


def _result(args: Any) -> TuneResult:
    """Represent the class-average result returned by a sensor optimizer."""
    trial = TuneTrialResult(
        index=1,
        config={"car": {"min_hits": 1}, "pedestrian": {"min_hits": 2}},
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
        best_yaml=args.project / "val" / "best.yaml",
    )


def _three_sequence_dataset(root: Path) -> Path:
    """Declare three aligned sequences without loading real perception data."""
    data = sensor_dataset_fixture(root)
    names = ("0002", "0006", "0007")
    payload = yaml.safe_load(data.dataset.read_text())
    payload["splits"]["val"]["sequences"] = list(names)
    data.dataset.write_text(yaml.safe_dump(payload))
    for name in names[1:]:
        copytree(root / "sequences/training/0002", root / "sequences/training" / name)
    for key in ("detections_2d", "car_detections_3d", "pedestrian_detections_3d"):
        source = data.reader_paths[key]
        for name in names[1:]:
            if source.is_file():
                copy2(source, source.with_name(f"{name}.txt"))
            else:
                copytree(source, source.with_name(name))
    return data.dataset


@pytest.mark.parametrize("entrypoint", ("python", "cli"))
@pytest.mark.parametrize(
    ("cpus", "workers", "selected", "expected"),
    (
        (8, None, (), 3),
        (3, None, (), 1),
        (1, None, (), 1),
        (8, None, ("0002", "0006"), 2),
        (8, 2, (), 2),
        (8, 50, (), 3),
        (8, 50, ("0002",), 1),
    ),
)
def test_sensor_tune_resolves_workers_after_sequence_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    cpus: int,
    workers: int | None,
    selected: tuple[str, ...],
    expected: int,
) -> None:
    dataset = _three_sequence_dataset(tmp_path / "dataset")
    captured: dict[str, Any] = {}
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: cpus)

    def run(args: Any, *, pipeline: Any = None) -> TuneResult:
        captured["args"] = args
        if pipeline is not None:
            pipeline.advance()
        return _result(args)

    monkeypatch.setattr(tuner, "_run_eagermot_tuning", run)
    project = tmp_path / "results"
    if entrypoint == "python":
        tuner.run_tune(
            SimpleNamespace(
                dataset=dataset,
                tracker="eagermot",
                sequence_names=selected,
                sequence_workers=workers,
                project=project,
            )
        )
    else:
        arguments = ["tune", "--dataset", str(dataset), "--tracker", "eagermot", "--project", str(project)]
        if workers is not None:
            arguments.extend(("--sequence-workers", str(workers)))
        for sequence in selected:
            arguments.extend(("--sequence", sequence))
        result = CliRunner().invoke(boxmot, arguments)
        assert result.exit_code == 0, (result.output, result.exception)
    assert captured["args"].sequence_workers == expected
    assert captured["args"].sequence_names == (selected or ("0002", "0006", "0007"))
    assert captured["args"].max_concurrent_trials == 1


@pytest.mark.parametrize(("explicit", "expected"), ((None, 2), (3, 3)))
def test_sensor_cli_honors_configured_worker_cap_and_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit: int | None, expected: int
) -> None:
    dataset = _three_sequence_dataset(tmp_path / "dataset")
    defaults = yaml.safe_load(runtime.RUNTIME_DEFAULTS_PATH.read_text())
    defaults["tune"]["sequence_workers"] = 2
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(yaml.safe_dump(defaults))
    monkeypatch.setattr(runtime, "RUNTIME_DEFAULTS_PATH", runtime_path)
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: 8)
    captured = {}

    def run(args: Any, *, pipeline: Any = None) -> TuneResult:
        captured["workers"] = args.sequence_workers
        if pipeline is not None:
            pipeline.advance()
        return _result(args)

    monkeypatch.setattr(tuner, "_run_eagermot_tuning", run)
    arguments = ["tune", "--dataset", str(dataset), "--tracker", "eagermot", "--project", str(tmp_path / "results")]
    if explicit is not None:
        arguments.extend(("--sequence-workers", str(explicit)))

    result = CliRunner().invoke(boxmot, arguments)

    assert result.exit_code == 0, (result.output, result.exception)
    assert captured["workers"] == expected


@pytest.mark.parametrize("entrypoint", (tuner.main, tuner.run_tune))
def test_shared_entrypoints_route_sensor_datasets_without_ray(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], entrypoint: Any
) -> None:
    args = _arguments(tmp_path, seed=None)
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Any] = {}

    def run(normalized: Any, *, pipeline: Any = None) -> TuneResult:
        captured["args"] = normalized
        captured["pipeline"] = pipeline
        if pipeline is not None:
            pipeline.advance()
        captured["result"] = _result(normalized)
        return captured["result"]

    monkeypatch.setitem(sys.modules, "ray", None)
    monkeypatch.setattr(tuner, "_run_eagermot_tuning", run)
    monkeypatch.setattr(tuner, "Tuner", lambda *_args, **_kwargs: pytest.fail("Sensor tuning must not initialize Ray."))

    result = entrypoint(args)

    assert result is captured["result"]
    normalized = captured["args"]
    assert normalized.dataset == args.dataset.resolve()
    assert normalized.tracker == "eagermot"
    assert normalized.tracker_backend == "python"
    assert normalized.seed == 0
    assert normalized.project == Path("runs/eagermot-tune")
    assert normalized.sequence_names == ("0002",)
    assert normalized.split == "val"
    assert normalized.per_class is True
    assert normalized.eval_masks is True
    assert vars(args) == {"dataset": args.dataset, "tracker": "eagermot", "seed": None}
    captured_output = capsys.readouterr()
    output = captured_output.out + captured_output.err
    if entrypoint is tuner.main:
        assert "Saved Artifacts" in output
        assert "best.yaml" in output
        assert captured["pipeline"] is not None
        assert result.workflow_rendered is True
    else:
        assert not output
        assert captured["pipeline"] is None
        assert result.workflow_rendered is False


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"tracker": "bytetrack"}, "does not use inputs required"),
        ({"tracker_backend": "cpp"}, "has no C\\+\\+ backend"),
        ({"build": "build"}, "does not support build"),
        ({"experiment": "experiment.yaml"}, "does not support experiment"),
        ({"detector": "yolov8n"}, "does not support detector"),
        ({"tracker_config": "tracker.yaml"}, "does not support tracker_config"),
        ({"calibrate_kf": True}, "--calibrate-kf requires 3D ground truth with track IDs"),
        ({"class_config": "missing-profiles.yaml"}, "class_config requires an existing file"),
        ({"resume_tune": "old-study"}, "does not support resume_tune"),
        ({"time_budget_s": 10}, "does not support time_budget_s"),
        ({"search_alg": "random"}, "search_alg"),
        ({"search_alg": "hyperopt"}, "search_alg"),
        ({"max_concurrent_trials": 2}, "max_concurrent_trials"),
        ({"sequence_workers": 0}, "sequence_workers"),
        ({"sequence_workers": -1}, "sequence_workers"),
        ({"sequence_workers": True}, "sequence_workers"),
        ({"device": "cuda:0"}, "device"),
        ({"objectives": ("HOTA", "IDF1")}, "objectives must be HOTA"),
        ({"maximize": ("IDF1",)}, "maximize must be HOTA"),
        ({"minimize": ("IDSW_rate",)}, "does not support minimize"),
        ({"n_trials": 0}, "n_trials must be"),
        ({"seed": -1}, "seed must be"),
    ),
)
def test_python_sensor_controls_are_validated_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], message: str
) -> None:
    args = _arguments(tmp_path, **overrides)
    _forbid_sensor_tuning(monkeypatch)

    with pytest.raises(ValueError, match=message):
        tuner.run_tune(args)


def test_sensor_baseline_override_is_rejected_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path)
    _forbid_sensor_tuning(monkeypatch)

    with pytest.raises(ValueError, match="does not support baseline_config"):
        tuner.run_tune(args, baseline_config={"min_hits": 1})


def test_sensor_missing_inputs_are_reported_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path)
    (args.dataset.parent / "sequences/training/0002/calibration.txt").unlink()
    _forbid_sensor_tuning(monkeypatch)

    with pytest.raises(ValueError, match="calibration.txt"):
        tuner.run_tune(args)


def test_main_formats_missing_sensor_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _arguments(tmp_path)
    monkeypatch.setitem(sys.modules, "optuna", None)

    with pytest.raises(ImportError, match="uv sync --extra cpu --extra mots --extra evolve") as raised:
        tuner.main(args)
    assert raised.value._workflow_rendered_error is True


def test_image_main_keeps_existing_tuner_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(tracker="bytetrack", dataset="mot17", build=tmp_path / "build")
    calls: list[Any] = []
    _forbid_sensor_tuning(monkeypatch)
    monkeypatch.setattr(tuner, "Tuner", lambda received: SimpleNamespace(fit=lambda: calls.append(received)))

    assert tuner.main(args) is None
    assert calls == [args]
