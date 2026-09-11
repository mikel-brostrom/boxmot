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
        "detection_metrics",
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


def test_official_ap_is_serialized_and_displayed_separately_from_tracking():
    """Both terminal renderers and saved reports label geometry and difficulty."""
    from rich.console import Console

    from boxmot.engine.ui.core.ui import BOXMOT_THEME

    ap = {
        "2d": {"car": {"easy": 81.25, "moderate": 70.5, "hard": 61.0}},
        "3d": {"car": {"easy": 72.0, "moderate": 59.5, "hard": None}},
    }
    result = _validation_result(
        raw={"car": {"HOTA": 71.5, "MOTA": 70.0, "IDF1": 75.0}},
        args=None,
        reference_raw=None,
        reference_name=None,
        detection_metrics=ap,
    )
    assert result.to_dict()["detection_metrics"] == ap
    assert result.summary == {"HOTA": 71.5}
    console = Console(width=120, record=True, force_terminal=False, theme=BOXMOT_THEME)
    with console.capture() as captured:
        console.print(result.renderable(include_sequences=False))
    for report in (result.render(include_sequences=False), result.format_report(), captured.get()):
        assert "AP40" in report
        assert all(level in report for level in ("Easy", "Moderate", "Hard"))
        assert "2D" in report and "3D" in report
        assert "N/A" in report
        assert "2D tracking" in report
        assert report.index("AP40") < report.index("HOTA")
