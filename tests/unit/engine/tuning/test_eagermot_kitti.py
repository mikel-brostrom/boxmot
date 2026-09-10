"""Persist per-class sensor tuning and reproduce its selected mask-HOTA result."""

from __future__ import annotations

import importlib
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

from boxmot.engine.config import runtime
from boxmot.engine.eval.eagermot_kitti import (
    KITTI_PROFILES,
    load_kitti_profiles,
    run_eagermot_kitti,
)
from boxmot.engine.tuning.results import TuneResult
from boxmot.engine.tuning.tuner import run_tune
from boxmot.engine.ui.reporters.tune import TuneWorkflowReporter
from tests.unit.engine.eval.test_eagermot_kitti import _fixture

optuna = pytest.importorskip("optuna")
tuning = importlib.import_module("boxmot.engine.tuning.eagermot_kitti")


def _arguments(data: SimpleNamespace, *, n_trials: int = 2) -> SimpleNamespace:
    """Use both classes and the real three-frame saved-sensor fixture."""
    return SimpleNamespace(
        dataset=data.dataset,
        split=None,
        sequence_names=("0002",),
        project=data.project,
        class_config=None,
        n_trials=n_trials,
        seed=0,
        tracker="eagermot",
        tracker_backend="python",
    )


def _study(output: Path) -> Any:
    """Load the independently persisted Optuna history from this run's database."""
    return optuna.load_study(study_name=None, storage=f"sqlite:///{(output / 'study.sqlite3').as_uri()}?uri=true")


def test_two_real_trials_preserve_class_baselines_and_export_replayable_best_config(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    args = _arguments(data)
    original_threads = torch.get_num_threads()
    result = run_tune(args)
    assert isinstance(result, TuneResult)
    output = result.best_yaml.parent
    assert torch.get_num_threads() == original_threads
    assert output == data.project / "val"
    assert json.loads((output / "run.json").read_text())["status"] == "complete"

    study = _study(output)
    assert len(study.trials) == 2
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    baseline = study.trials[0]
    assert baseline.value == pytest.approx(100)
    for prefix, class_id in (("car", 1), ("pedestrian", 2)):
        for parameter in ("det_thresh", "min_hits", "distance_threshold"):
            assert baseline.params[f"{prefix}.{parameter}"] == KITTI_PROFILES[class_id][parameter]
    assert baseline.params["car.det_thresh"] != baseline.params["pedestrian.det_thresh"]
    assert {name.partition(".")[0] for name in study.trials[1].params} == {"car", "pedestrian"}

    for trial in study.trials:
        directory = output / "trials" / f"{trial.number:04d}"
        metrics = json.loads((directory / "metrics.json").read_text())
        assert trial.value == pytest.approx(metrics["cls_comb_cls_av"]["HOTA"])
        assert json.loads((directory / "run.json").read_text())["status"] == "complete"
        assert (directory / "mots/0002.txt").is_file()
    assert study.best_value >= baseline.value

    exported = yaml.safe_load((output / "best.yaml").read_text())
    assert set(exported) == {"car", "pedestrian"}
    assert result.best_config == result.best.config == exported
    assert result.tracker == "eagermot"
    assert result.benchmark == "kitti-mots-fusion"
    assert result.best.index == study.best_trial.number + 1
    assert result.baseline is result.trials[0]
    assert len(result.trials) == len(study.trials)
    assert result.summary_label == "cls_comb_cls_av"
    assert result.summary["HOTA"] == pytest.approx(study.best_value)
    assert result.best.score == (study.best_value,)
    assert result.exp_dir == output / "trials" / f"{study.best_trial.number:04d}"
    for captured, persisted in zip(result.trials, study.trials):
        assert captured.index == persisted.number + 1
        assert captured.config == persisted.user_attrs["profiles"]
        assert captured.score == (persisted.value,)
        assert captured.raw == json.loads((captured.exp_dir / "metrics.json").read_text())
    serialized = result.to_dict(include_trials=True, include_raw=True)
    assert serialized["best_config"] == exported
    assert serialized["best"]["score"] == [study.best_value]
    assert len(serialized["trials"]) == 2
    assert json.loads(json.dumps(serialized)) == serialized
    assert "HOTA" in result.render()
    profiles = load_kitti_profiles(output / "best.yaml")
    assert set(profiles) == {1, 2}
    # The baseline already achieves the maximum possible score, so the selected
    # complete profiles must retain its car/pedestrian differences and fixed values.
    assert profiles == KITTI_PROFILES
    replay_args = SimpleNamespace(
        **{**vars(args), "project": tmp_path / "replay", "class_config": output / "best.yaml"}
    )
    replay = run_eagermot_kitti(replay_args).exp_dir
    replay_metrics = json.loads((replay / "metrics.json").read_text())
    assert replay_metrics["cls_comb_cls_av"]["HOTA"] == pytest.approx(study.best_value)
    assert torch.get_num_threads() == original_threads


@pytest.mark.parametrize("project_name", ("results", "results?experiment={car} space"))
def test_repeated_tuning_allocates_a_new_directory_without_overwriting(tmp_path: Path, project_name: str) -> None:
    data = _fixture(tmp_path)
    data.project = tmp_path / project_name
    args = _arguments(data, n_trials=1)
    first = tuning.run_eagermot_kitti_tuning(args).best_yaml.parent
    assert (first / "study.sqlite3").is_file()
    assert _study(first).best_value == pytest.approx(100)
    original_files = {path.relative_to(first): path.read_bytes() for path in first.rglob("*") if path.is_file()}

    second = tuning.run_eagermot_kitti_tuning(args).best_yaml.parent

    assert second == data.project / "val2"
    assert first != second
    assert all((first / relative).read_bytes() == contents for relative, contents in original_files.items())
    assert (second / "study.sqlite3").is_file()
    assert _study(second).best_value == pytest.approx(100)
    assert json.loads((second / "run.json").read_text())["status"] == "complete"


def test_objective_uses_class_average_hota_without_averaging_existing_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct result fields catch selection of one class or double aggregation."""
    data = _fixture(tmp_path)
    evaluate = tuning.evaluate_eagermot_kitti

    def distinct_metrics(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        results = evaluate(*args, **kwargs)
        for name, value in (("car", 80), ("pedestrian", 20), ("cls_comb_cls_av", 50), ("cls_comb_det_av", 70)):
            results[name]["HOTA"] = value
        return results

    monkeypatch.setattr(tuning, "evaluate_eagermot_kitti", distinct_metrics)
    result = run_tune(_arguments(data, n_trials=1))
    output = result.best_yaml.parent

    assert _study(output).best_value == pytest.approx(50)
    assert result.summary["HOTA"] == pytest.approx(50)
    assert result.best.score == (50,)


def test_interrupted_trial_preserves_completed_best_and_restores_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interrupt actual replay after the baseline has already been checkpointed."""
    data = _fixture(tmp_path)
    evaluation = importlib.import_module("boxmot.engine.eval.eagermot_kitti")
    replay = evaluation._replay
    calls = 0

    def interrupted_replay(*args: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
        nonlocal calls
        calls += 1
        assert torch.get_num_threads() == 1
        if calls == 2:
            raise KeyboardInterrupt("Interrupted after the completed baseline")
        return replay(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_replay", interrupted_replay)
    original_threads = torch.get_num_threads()
    with pytest.raises(KeyboardInterrupt, match="completed baseline"):
        tuning.run_eagermot_kitti_tuning(_arguments(data))

    assert torch.get_num_threads() == original_threads
    output = data.project / "val"
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["completed_trials"] == 1
    assert manifest["best_hota"] == pytest.approx(100)
    assert load_kitti_profiles(output / "best.yaml") == KITTI_PROFILES
    assert json.loads((output / "trials/0001/run.json").read_text())["status"] == "interrupted"
    study = _study(output)
    assert [trial.state for trial in study.trials] == [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
    ]


@pytest.mark.parametrize("with_pipeline", (False, True))
@pytest.mark.parametrize(
    ("cpus", "workers", "expected"),
    (
        (8, None, 3),
        (8, 2, 2),
        (8, 50, 3),
        (1, None, 1),
    ),
)
def test_each_serial_optuna_trial_forwards_resolved_sequence_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_pipeline: bool,
    cpus: int,
    workers: int | None,
    expected: int,
) -> None:
    """Resolve selected-sequence concurrency once without multiplying concurrent search trials."""
    args = _arguments(SimpleNamespace(dataset=tmp_path / "dataset.yaml", project=tmp_path / "results"))
    args.sequence_workers = workers
    monkeypatch.setattr(runtime.os, "cpu_count", lambda: cpus)
    sequences = dict.fromkeys(("0002", "0006", "0007"))
    inputs = SimpleNamespace(
        sequences=sequences,
        manifest={"split": "val", "dataset_id": "kitti-mots-fusion", "sequences": dict.fromkeys(sequences, 3)},
    )
    monkeypatch.setattr(tuning, "prepare_eagermot_kitti", lambda _args: inputs)
    replay_workers = []

    def evaluate(_inputs: Any, _profiles: Any, _output: Any, **kwargs: Any) -> dict[str, dict[str, float]]:
        replay_workers.append(kwargs["sequence_workers"])
        if kwargs.get("on_evaluate") is not None:
            kwargs["on_evaluate"]()
        return {name: {"HOTA": 75.0} for name in ("car", "pedestrian", "cls_comb_cls_av")}

    monkeypatch.setattr(tuning, "evaluate_eagermot_kitti", evaluate)
    optimize = optuna.Study.optimize
    search_workers = []

    def optimize_serial(study: Any, objective: Any, **kwargs: Any) -> None:
        search_workers.append(kwargs["n_jobs"])
        optimize(study, objective, **kwargs)

    monkeypatch.setattr(optuna.Study, "optimize", optimize_serial)
    context = TuneWorkflowReporter(args, maximize=["HOTA"], minimize=[]).pipeline() if with_pipeline else nullcontext()
    with context as pipeline:
        result = tuning.run_eagermot_kitti_tuning(args, pipeline=pipeline)

    assert replay_workers == [expected, expected]
    assert search_workers == [1]
    assert len(result.trials) == 2
    manifest = json.loads((result.best_yaml.parent / "run.json").read_text())
    assert manifest["sequence_workers"] == expected
