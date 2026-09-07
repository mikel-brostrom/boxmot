from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

from boxmot.engine.eval.results import ValidationResult


def _validation_result(**overrides) -> ValidationResult:
    values = {
        "benchmark": "mot17",
        "raw": {"HOTA": 71.5},
        "summary_label": "single_class",
        "summary": {"HOTA": 71.5},
        "exp_dir": Path("runs/mot17/exp"),
        "timings": {"frames": 3},
        "args": object(),
        "reference_raw": {"HOTA": 70.0},
        "reference_name": "TrackEval",
    }
    values.update(overrides)
    return ValidationResult(**values)


def test_validation_result_preserves_field_order_slots_and_serialization():
    result = _validation_result()

    assert [field.name for field in fields(ValidationResult)] == [
        "benchmark",
        "raw",
        "summary_label",
        "summary",
        "exp_dir",
        "timings",
        "args",
        "workflow_rendered",
        "reference_raw",
        "reference_name",
    ]
    assert not hasattr(result, "__dict__")
    assert repr(result) == (
        "ValidationResult(benchmark='mot17', summary={'HOTA': 71.5}, exp_dir=PosixPath('runs/mot17/exp'))"
    )
    assert result.to_dict() == {
        "benchmark": "mot17",
        "summary_label": "single_class",
        "summary": {"HOTA": 71.5},
        "timings": {"frames": 3},
        "exp_dir": "runs/mot17/exp",
    }
    assert result.to_dict(include_raw=True) == {
        "benchmark": "mot17",
        "summary_label": "single_class",
        "summary": {"HOTA": 71.5},
        "timings": {"frames": 3},
        "exp_dir": "runs/mot17/exp",
        "raw": {"HOTA": 71.5},
        "reference_raw": {"HOTA": 70.0},
        "reference_name": "TrackEval",
    }
    assert get_type_hints(ValidationResult.renderable)["return"] is not None


def test_validation_result_delegates_presentation_lazily(monkeypatch):
    from boxmot.engine.ui.reporters import validation

    calls = {}
    renderable = object()

    def render_report(raw, **kwargs):
        calls["render"] = (raw, kwargs)
        return "rendered"

    def build_renderable(raw, **kwargs):
        calls["renderable"] = (raw, kwargs)
        return renderable

    def format_report(raw, **kwargs):
        calls["format"] = (raw, kwargs)
        return "formatted"

    def print_report(raw, **kwargs):
        calls["print"] = (raw, kwargs)

    monkeypatch.setattr(validation, "render_validation_cli_report", render_report)
    monkeypatch.setattr(validation, "build_validation_cli_renderable", build_renderable)
    monkeypatch.setattr(validation, "format_validation_report", format_report)
    monkeypatch.setattr(validation, "print_validation_cli_report", print_report)

    result = _validation_result()

    assert result.render(include_sequences=False, include_timings=True) == "rendered"
    assert calls["render"][1]["title"] == "📊 BOXMOT vs TRACKEVAL"
    assert calls["render"][1]["compare_label"] == "Δ vs TrackEval"
    assert str(result) == "rendered"

    assert result.renderable(include_sequences=False) is renderable
    assert calls["renderable"][1]["compare_raw"] == {"HOTA": 70.0}
    assert calls["renderable"][1]["compare_args"] is result.args

    assert result.format_report(include_sequences=False) == "formatted"
    assert calls["format"][1]["title"] == validation.DEFAULT_VALIDATION_REPORT_TITLE

    result.print_report(include_timings=True)
    assert calls["print"][1]["include_timings"] is True
    assert calls["print"][1]["compare_label"] == "Δ vs TrackEval"

    result.workflow_rendered = True
    assert str(result) == ""
