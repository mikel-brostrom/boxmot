from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from boxmot.engine.eval.results import ValidationResult
from boxmot.engine.tuning.results import TuneResult, TuneTrialResult


def _validation_result(value: float, *, exp_dir: str) -> ValidationResult:
    return ValidationResult(
        benchmark="mot17",
        raw={"HOTA": value},
        summary_label="single_class",
        summary={"HOTA": value},
        exp_dir=Path(exp_dir),
        timings={"frames": 3},
        args=object(),
    )


def _tune_result() -> TuneResult:
    baseline = TuneTrialResult(
        index=0,
        config={"track_thresh": 0.5},
        metrics=_validation_result(70.0, exp_dir="runs/tune/trial-0"),
        score=(70.0,),
    )
    best = TuneTrialResult(
        index=1,
        config={"track_thresh": 0.6},
        metrics=_validation_result(72.0, exp_dir="runs/tune/trial-1"),
        score=(72.0,),
    )
    return TuneResult(
        benchmark="mot17",
        tracker="botsort",
        trials=[baseline, best],
        best=best,
        best_config={"track_thresh": 0.6},
        best_yaml=Path("runs/tune/best.yaml"),
    )


def test_tuning_results_preserve_fields_slots_properties_and_serialization():
    result = _tune_result()
    trial = result.best

    assert [field.name for field in fields(TuneTrialResult)] == [
        "index",
        "config",
        "metrics",
        "score",
    ]
    assert [field.name for field in fields(TuneResult)] == [
        "benchmark",
        "tracker",
        "trials",
        "best",
        "best_config",
        "best_yaml",
        "workflow_rendered",
    ]
    assert not hasattr(trial, "__dict__")
    assert not hasattr(result, "__dict__")
    assert result.baseline is result.trials[0]
    assert result.summary == {"HOTA": 72.0}
    assert trial.benchmark == "mot17"
    assert trial.exp_dir == Path("runs/tune/trial-1")

    payload = result.to_dict(include_trials=True, include_raw=True)
    assert payload["best_yaml"] == "runs/tune/best.yaml"
    assert payload["best"]["score"] == [72.0]
    assert payload["best"]["metrics"]["raw"] == {"HOTA": 72.0}
    assert [item["index"] for item in payload["trials"]] == [0, 1]


def test_tune_result_delegates_rendering_lazily(monkeypatch):
    from boxmot.engine.ui.reporters import validation

    calls = {}

    def render_report(raw, **kwargs):
        calls["render"] = (raw, kwargs)
        return "rendered"

    def format_report(raw, **kwargs):
        calls["format"] = (raw, kwargs)
        return "formatted"

    def print_report(raw, **kwargs):
        calls["print"] = (raw, kwargs)

    monkeypatch.setattr(validation, "render_validation_cli_report", render_report)
    monkeypatch.setattr(validation, "format_validation_report", format_report)
    monkeypatch.setattr(validation, "print_validation_cli_report", print_report)

    result = _tune_result()

    assert result.render(include_sequences=False, include_timings=True) == "rendered"
    assert calls["render"][0] == {"HOTA": 72.0}
    assert calls["render"][1]["compare_raw"] == {"HOTA": 70.0}
    assert calls["render"][1]["title"] == validation.CLI_TUNE_BEST_SUMMARY_TITLE

    assert result.format_report(include_sequences=False) == "formatted"
    assert calls["format"][1]["title"] == validation.DEFAULT_TUNE_BEST_REPORT_TITLE

    result.print_best_report(include_timings=True)
    assert calls["print"][1]["include_timings"] is True
    assert calls["print"][1]["compare_args"] is result.baseline.args

    result.workflow_rendered = True
    assert str(result) == ""
