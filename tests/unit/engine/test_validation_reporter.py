"""Tests for validation and tuning result presentation."""

from __future__ import annotations

from types import SimpleNamespace

from boxmot.engine.eval import results as eval_results
from boxmot.engine.ui.core.ui import capture_renderable
from boxmot.engine.ui.reporters import validation


def _metrics(**overrides: float | int) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "HOTA": 70.0,
        "MOTA": 80.0,
        "IDF1": 90.0,
        "AssA": 75.0,
        "AssRe": 85.0,
        "IDSW": 2,
        "IDs": 10,
    }
    metrics.update(overrides)
    return metrics


def test_validation_reporter_uses_canonical_metric_columns() -> None:
    assert validation.CORE_SUMMARY_COLUMNS is eval_results.CORE_SUMMARY_COLUMNS
    assert validation.SUMMARY_COLUMNS is eval_results.SUMMARY_COLUMNS
    assert validation.SUMMARY_INT_COLUMNS is eval_results.SUMMARY_INT_COLUMNS


def test_render_validation_cli_report_formats_comparison_deltas_and_colors() -> None:
    report = validation.render_validation_cli_report(
        _metrics(HOTA=72.0, MOTA=78.0, IDSW=1),
        compare_raw=_metrics(),
        compare_label="Δ vs baseline",
        colorize=True,
    )

    assert "Δ vs baseline" in report
    assert "\033[32m(+2.00)\033[0m" in report
    assert "\033[31m(-2.00)\033[0m" in report
    assert "\033[32m(-1)\033[0m" in report


def test_render_validation_cli_report_appends_timing_snapshot() -> None:
    report = validation.render_validation_cli_report(
        _metrics(),
        timings={
            "frames": 2,
            "totals_ms": {"det": 10.0, "track": 4.0, "total": 20.0},
        },
        include_sequences=False,
        include_timings=True,
        colorize=False,
    )

    assert "TIMING SUMMARY" in report
    assert "Frames" in report
    assert "2" in report


def test_build_validation_cli_renderable_includes_comparison_and_timing() -> None:
    renderable = validation.build_validation_cli_renderable(
        _metrics(HOTA=72.0),
        compare_raw=_metrics(),
        compare_label="Δ vs baseline",
        timings={
            "frames": 2,
            "totals_ms": {"det": 10.0, "track": 4.0, "total": 20.0},
            "avg_ms": {"det": 5.0, "track": 2.0, "total": 10.0},
            "fps": 100.0,
        },
        include_timings=True,
    )

    rendered = capture_renderable(renderable, width=160)
    assert "COMBINED (results)" in rendered
    assert "Δ vs baseline" in rendered
    assert "(+2.00)" in rendered
    assert "Stage" in rendered
    assert "Frames" in rendered


def test_supports_ansi_color_honors_terminal_and_environment() -> None:
    tty = SimpleNamespace(isatty=lambda: True)
    non_tty = SimpleNamespace(isatty=lambda: False)

    assert validation.supports_ansi_color(tty, environ={}) is True
    assert validation.supports_ansi_color(non_tty, environ={}) is False
    assert validation.supports_ansi_color(tty, environ={"NO_COLOR": "1"}) is False
    assert validation.supports_ansi_color(tty, environ={"TERM": "dumb"}) is False
