"""Validate shared evaluation entrypoints for saved KITTI sensor inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from boxmot.engine.eval import evaluator
from boxmot.engine.eval.results import ValidationResult
from tests.unit.engine._sensor_dataset_fixture import sensor_dataset_fixture


def _arguments(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    """Select a portable sensor dataset using the minimal public API arguments."""
    dataset = sensor_dataset_fixture(tmp_path / "dataset").dataset
    return SimpleNamespace(**{"dataset": dataset, "tracker": "eagermot", **overrides})


@pytest.mark.parametrize("entrypoint", (evaluator.main, evaluator.run_eval))
@pytest.mark.parametrize(("show", "save", "show_3d"), ((False, False, False), (True, False, True), (False, True, True)))
def test_shared_entrypoints_preserve_sensor_profiles_and_visualization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    entrypoint: Any,
    show: bool,
    save: bool,
    show_3d: bool,
) -> None:
    profiles = tmp_path / "classes.yaml"
    profiles.write_text("car: {}\npedestrian: {}\n", encoding="utf-8")
    args = _arguments(tmp_path, class_config=profiles, show=show, save=save, show_3d=show_3d)
    captured: dict[str, Any] = {}

    def run(normalized: Any, **kwargs: Any) -> ValidationResult:
        captured["args"] = normalized
        captured["result"] = ValidationResult(
            benchmark="kitti-mots-fusion",
            raw={"cls_comb_cls_av": {"HOTA": 75}},
            summary_label="cls_comb_cls_av",
            summary={"HOTA": 75},
            exp_dir=normalized.project / "val",
            args=normalized,
        )
        return captured["result"]

    def forbid_image_work(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("Sensor evaluation must not initialize image evaluation or replay.")

    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", SimpleNamespace(run_eagermot_kitti=run))
    monkeypatch.setattr(evaluator, "eval_setup", forbid_image_work)
    monkeypatch.setattr(evaluator, "replay_build", forbid_image_work)

    result = entrypoint(args)

    assert result is captured["result"]
    normalized = captured["args"]
    assert normalized.dataset == args.dataset.resolve()
    assert normalized.tracker == "eagermot"
    assert normalized.tracker_backend == "python"
    assert normalized.project == Path("runs/eagermot")
    assert normalized.sequence_names == ("0002",)
    assert normalized.split == "val"
    assert normalized.class_config == profiles.resolve()
    assert (normalized.show, normalized.save, normalized.show_3d) == (show, save, show_3d)
    assert normalized.per_class is True
    assert normalized.eval_masks is True
    assert not hasattr(args, "split")
    captured_output = capsys.readouterr()
    output = captured_output.out + captured_output.err
    assert f"Results: {result.exp_dir}" in output if entrypoint is evaluator.main else not output


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"tracker": "bytetrack"}, "does not use inputs required"),
        ({"tracker_backend": "cpp"}, "has no C\\+\\+ backend"),
        ({"experiment": "experiment.yaml"}, "does not support experiment"),
        ({"build": "build"}, "does not support build"),
        ({"detector": "yolov8n"}, "does not support detector"),
        ({"tracker_config": "tracker.yaml"}, "does not support tracker_config"),
        ({"calibrate_kf": True}, "--calibrate-kf requires 3D ground truth with track IDs"),
        ({"compare_trackeval": True}, "does not support compare_trackeval"),
        ({"fps": 10}, "does not support fps"),
        ({"variable_dt": True}, "does not support variable_dt"),
        ({"sequence_workers": 0}, "sequence_workers must be a positive integer"),
        ({"sequence_workers": True}, "sequence_workers must be a positive integer"),
        ({"device": "cuda:0"}, "device must be cpu"),
        ({"show_3d": True}, "--show-3d requires --show or --save"),
        ({"class_config": "missing-profiles.yaml"}, "class_config requires an existing file"),
    ),
)
def test_sensor_arguments_are_validated_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, Any], message: str
) -> None:
    args = _arguments(tmp_path, **overrides)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", None)

    with pytest.raises(ValueError, match=message):
        evaluator.run_eval(args)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"evolve_config": {"min_hits": 1}}, "does not support evolve_config"),
        ({"per_class_configs": {1: {"min_hits": 1}}}, "does not support per_class_configs"),
        ({"output_dir": Path("results")}, "does not support output_dir"),
        ({"setup": False}, "does not support setup=False"),
        ({"prepare_cache": True}, "does not support prepare_cache"),
    ),
)
def test_sensor_runtime_controls_are_not_silently_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], message: str
) -> None:
    args = _arguments(tmp_path)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", None)

    with pytest.raises(ValueError, match=message):
        evaluator.run_eval(args, **kwargs)


def test_sensor_missing_inputs_are_reported_before_optional_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _arguments(tmp_path)
    (args.dataset.parent / "sequences/training/0002/calibration.txt").unlink()
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", None)

    with pytest.raises(ValueError, match="calibration.txt"):
        evaluator.run_eval(args)


def test_main_formats_missing_sensor_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = _arguments(tmp_path)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", None)

    with pytest.raises(ImportError, match="uv sync --extra cpu --extra mots") as raised:
        evaluator.main(args)
    assert raised.value._workflow_rendered_error is True


@pytest.mark.parametrize("entrypoint", (evaluator.main, evaluator.run_eval))
@pytest.mark.parametrize("overrides", ({"class_config": "profiles.yaml"}, {"show_3d": True}))
def test_image_inputs_reject_sensor_options_before_replay(
    monkeypatch: pytest.MonkeyPatch, entrypoint: Any, overrides: dict[str, Any]
) -> None:
    args = SimpleNamespace(tracker="bytetrack", dataset="mot17", build="build", **overrides)
    monkeypatch.setitem(sys.modules, "boxmot.engine.eval.eagermot_kitti", None)
    monkeypatch.setattr(evaluator, "EvalWorkflowReporter", lambda *_args: pytest.fail("Unexpected image workflow."))

    with pytest.raises(ValueError, match="requires an EagerMOT Sensor dataset dataset"):
        entrypoint(args)


def test_public_eval_returns_real_sensor_metrics_and_preserves_previous_results(tmp_path: Path) -> None:
    """The common API exposes persisted metrics without changing replay or thread behavior."""
    import torch

    from tests.unit.engine.eval.test_eagermot_kitti import _fixture

    data = _fixture(tmp_path)
    args = SimpleNamespace(tracker="eagermot", dataset=data.dataset, project=data.project)
    original_threads = torch.get_num_threads()

    result = evaluator.run_eval(args, verbose=False, show_progress=False)

    assert isinstance(result, ValidationResult)
    assert result.exp_dir == data.project / "val"
    assert result.benchmark == "kitti-mots-fusion"
    assert result.summary_label == "cls_comb_cls_av"
    assert result.summary["HOTA"] == pytest.approx(100)
    assert result.raw == json.loads((result.exp_dir / "metrics.json").read_text())
    assert result.summary == result.raw["cls_comb_cls_av"]
    assert torch.get_num_threads() == original_threads
    manifest = json.loads((result.exp_dir / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["dataset_config"] == str(data.dataset.resolve())
    payload = result.to_dict(include_raw=True)
    assert json.loads(json.dumps(payload)) == payload
    assert "HOTA" in result.render()
    original_files = {
        path.relative_to(result.exp_dir): path.read_bytes() for path in result.exp_dir.rglob("*") if path.is_file()
    }

    second = evaluator.run_eval(args, verbose=False, show_progress=False)

    assert second.exp_dir == data.project / "val2"
    assert second.raw == result.raw
    assert all((result.exp_dir / path).read_bytes() == contents for path, contents in original_files.items())
    assert torch.get_num_threads() == original_threads
