"""Failed searches must retain their error and never report successful completion."""

from __future__ import annotations

import errno
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

import boxmot.engine.tuning.tuner as tuner_module
from boxmot.engine.cli import boxmot
from boxmot.engine.ui.workflow.pipeline import PipelineTracker
from tests.unit.engine.tuning.test_calibrated_tuner import fake_tuning as fake_tuning


def test_driver_continues_after_a_failed_trial_without_retrying(fake_tuning, monkeypatch) -> None:
    failure_config = Mock()
    monkeypatch.setattr(sys.modules["ray.tune"], "FailureConfig", failure_config)

    tuner_module.Tuner(fake_tuning.args(calibrate_kf=False)).fit()

    failure_config.assert_called_once_with(max_failures=0, fail_fast=False)


@pytest.mark.parametrize("partial_export_fails", [False, True])
def test_fit_failure_preserves_original_error_and_attempts_partial_export(
    fake_tuning, monkeypatch, partial_export_fails: bool,
) -> None:
    failure = OSError(errno.ENOSPC, "No space left on device", "experiment_state.json")
    completed = [SimpleNamespace(config={"track_high_thresh": 0.4})]
    fake_tuning.captured["saved_results"] = completed
    monkeypatch.setattr(fake_tuning.ray_tuner, "fit", Mock(side_effect=failure))
    exporter = Mock(side_effect=OSError(errno.ENOSPC, "Partial export also failed") if partial_export_fails else None)
    monkeypatch.setattr(tuner_module, "save_all_results", exporter)
    args = fake_tuning.args(calibrate_kf=False)

    with pytest.raises(OSError) as raised:
        tuner_module.Tuner(args).fit()

    assert raised.value is failure
    assert exporter.call_count == 1
    assert exporter.call_args.args[1] is completed
    fake_tuning.pipeline.finish.assert_not_called()
    fake_tuning.pipeline.complete_step.assert_not_called()
    assert fake_tuning.pipeline.__exit__.call_args.args[1] is failure
    assert not Path(args._tune_sequence_progress_dir).exists()


@pytest.mark.parametrize("outcome", ["disk_full", "empty", "all_failed"])
def test_failed_search_renders_failed_and_exits_cli_nonzero(fake_tuning, monkeypatch, outcome: str) -> None:
    from boxmot.engine.tuning.postprocessing import save_all_results

    failure = OSError(errno.ENOSPC, "No space left on device", "experiment_state.json")
    returned = [SimpleNamespace(error=failure)] if outcome == "all_failed" else []
    monkeypatch.setattr(
        fake_tuning.ray_tuner, "fit", Mock(side_effect=failure) if outcome == "disk_full" else lambda _self: returned
    )
    monkeypatch.setattr(tuner_module, "save_all_results", save_all_results)
    pipelines = []

    def pipeline(reporter, **kwargs):
        value = PipelineTracker(reporter.create(), **kwargs)
        pipelines.append(value)
        return value

    monkeypatch.setattr(tuner_module.TuneWorkflowReporter, "pipeline", pipeline)
    result = CliRunner().invoke(boxmot, [
        "tune", "--experiment", "mot17/ablation-yolox-lmbn.yaml", "--build", "existing",
        "--tracker", "botsort", "--n-trials", "2", "--sequence-workers", "1",
    ])

    assert result.exit_code == 1, (result.output, result.exception)
    assert "FAILED" in result.output
    assert "No space left on device" in result.output if outcome == "disk_full" else (
        "No successful tuning trials were produced" in result.output
    )
    assert "Optimize trials" in result.output
    assert pipelines[0].workflow.steps[-1][1] == "failed"


@pytest.mark.parametrize("wrapped", [False, True])
def test_objective_propagates_disk_full_and_closes_replay(monkeypatch, wrapped: bool) -> None:
    failure = OSError(errno.ENOSPC, "No space left on device", "masks.npy")

    def evaluate(*_args, **_kwargs):
        if wrapped:
            raise RuntimeError("Replay worker failed") from failure
        raise failure

    monkeypatch.setattr(tuner_module, "run_eval", evaluate)
    objective = tuner_module.TrackerObjective(SimpleNamespace(sequence_workers=1, seq_info={"seq": 2}))
    session = Mock()
    objective._session = session
    progress = []

    with pytest.raises(RuntimeError if wrapped else OSError) as raised:
        objective({}, progress_callback=progress.append)

    assert raised.value.__cause__ is failure if wrapped else raised.value is failure
    session.close.assert_called_once()
    assert objective._session is None
    assert len(progress) == 1 and progress[0].status == "failed"


def test_objective_propagates_invalid_configuration_without_metric_observation(monkeypatch) -> None:
    failure = ValueError("Invalid trial settings")
    monkeypatch.setattr(tuner_module, "run_eval", Mock(side_effect=failure))
    objective = tuner_module.TrackerObjective(SimpleNamespace(sequence_workers=1))
    session = Mock()
    objective._session = session

    with pytest.raises(ValueError) as raised:
        objective({})

    assert raised.value is failure
    session.close.assert_called_once()
    assert objective._session is None


@pytest.mark.parametrize("raw", [None, {}])
def test_objective_rejects_missing_validation_metrics(monkeypatch, raw) -> None:
    monkeypatch.setattr(tuner_module, "run_eval", Mock(return_value=SimpleNamespace(raw=raw)))
    objective = tuner_module.TrackerObjective(SimpleNamespace(sequence_workers=1, seq_info={"seq": 2}))
    session = Mock()
    objective._session = session
    progress = []

    with pytest.raises(RuntimeError, match="no validation metrics"):
        objective({}, progress_callback=progress.append)

    session.close.assert_called_once()
    assert objective._session is None
    assert len(progress) == 1 and progress[0].status == "failed"


def test_objective_preserves_genuine_evaluated_zero_metrics(monkeypatch) -> None:
    raw = {key: 0.0 for key in tuner_module.ALL_TUNE_METRICS}
    result = SimpleNamespace(
        raw=raw, benchmark="fixture", summary_label="all", summary=raw, timings={}, exp_dir=None
    )
    monkeypatch.setattr(tuner_module, "run_eval", Mock(return_value=result))
    objective = tuner_module.TrackerObjective(SimpleNamespace(sequence_workers=1))
    session = Mock()
    objective._session = session
    try:
        metrics = objective({})
        assert all(metrics[key] == 0.0 for key in tuner_module.ALL_TUNE_METRICS)
        assert metrics["_validation"]["raw"] == raw
        session.close.assert_not_called()
    finally:
        objective.close()


def test_failed_trial_cannot_win_a_minimized_metric(tmp_path) -> None:
    from boxmot.engine.tuning.postprocessing import best_trial_data, collect_trial_data

    failed = SimpleNamespace(
        error=ValueError("Invalid trial settings"), metrics={"trial_id": "failed", "IDSW": 0.0},
        config={}, path=str(tmp_path / "failed"),
    )
    completed = SimpleNamespace(
        error=None, metrics={"trial_id": "completed", "IDSW": 7.0},
        config={}, path=str(tmp_path / "completed"),
    )

    trials = collect_trial_data([failed, completed])

    assert [trial["trial_id"] for trial in trials] == ["completed"]
    assert best_trial_data(trials, maximize=[], minimize=["IDSW"])["trial_id"] == "completed"


def test_user_interrupt_still_returns_partial_results() -> None:
    partial = [object()]
    ray_tuner = SimpleNamespace(fit=Mock(side_effect=KeyboardInterrupt), get_results=lambda: partial)

    assert tuner_module.Tuner(SimpleNamespace())._execute_tuner(ray_tuner) == (partial, True)
