from __future__ import annotations

import math
import os
import sys
from importlib import import_module
from typing import Any, Sequence

from boxmot.utils.timing import TimingStats

SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
CORE_SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1")
DEFAULT_VALIDATION_REPORT_TITLE = "VAL RESULTS"
DEFAULT_TUNE_BEST_REPORT_TITLE = "TUNE BEST RESULTS"
CLI_RESULTS_SUMMARY_TITLE = "📊 RESULTS SUMMARY"
CLI_TUNE_BEST_SUMMARY_TITLE = "📊 BEST TRIAL SUMMARY"


def extract_summary(raw_results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results_module = import_module("boxmot.utils.evaluation.results")
    label, metrics = results_module._select_plot_metrics_data(raw_results)
    if not metrics and raw_results:
        first_value = next(iter(raw_results.values()), {})
        if isinstance(first_value, dict):
            metrics = first_value

    summary = {
        column: metrics.get(column, 0)
        for column in SUMMARY_COLUMNS
        if isinstance(metrics, dict)
    }
    return label, summary


def timing_summary_from_stats(timing_stats: TimingStats) -> dict[str, Any]:
    totals = dict(timing_stats.totals)
    total_ms = float(totals.get("total", 0.0) or 0.0)
    if total_ms == 0.0:
        total_ms = float(sum(totals.values()))

    frames = int(timing_stats.frames)
    avg_ms = {
        key: (float(value) / frames if frames else 0.0)
        for key, value in totals.items()
    }
    avg_total_ms = total_ms / frames if frames else 0.0
    fps = (1000.0 * frames / total_ms) if total_ms else 0.0

    return {
        "frames": frames,
        "totals_ms": {**{key: float(value) for key, value in totals.items()}, "total": total_ms},
        "avg_ms": {**avg_ms, "total": avg_total_ms},
        "fps": fps,
    }


def core_summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    return {
        metric: float(summary.get(metric, 0.0) or 0.0)
        for metric in CORE_SUMMARY_COLUMNS
    }


def format_core_summary(summary: dict[str, Any]) -> str:
    metrics = core_summary_metrics(summary)
    return " ".join(f"{metric}={value:.3f}" for metric, value in metrics.items())


def supports_ansi_color(
    stream: Any | None = None,
    *,
    sys_module=sys,
    environ: dict[str, str] | None = None,
) -> bool:
    output = sys_module.stdout if stream is None else stream
    env = os.environ if environ is None else environ
    if env.get("NO_COLOR"):
        return False
    if env.get("TERM", "").lower() == "dumb":
        return False
    return bool(hasattr(output, "isatty") and output.isatty())


def format_validation_report(
    raw: dict[str, Any],
    *,
    args: Any = None,
    title: str | None = None,
    include_sequences: bool = True,
) -> str:
    results_module = import_module("boxmot.utils.evaluation.results")
    cfg = results_module._load_report_cfg_from_args(args)
    parsed_results = results_module.normalize_report_results(raw, args, cfg)
    if not parsed_results:
        fallback_title = title or "Results"
        return f"{fallback_title}\n{format_core_summary(raw if isinstance(raw, dict) else {})}"

    return results_module.render_trackeval_report(
        parsed_results,
        args,
        cfg,
        title=title or "Results",
        include_sequences=include_sequences,
        always_include_combined=True,
        colorize=False,
    )


def timing_stats_from_snapshot(timings: dict[str, Any] | None) -> TimingStats | None:
    if not timings:
        return None

    totals_ms = timings.get("totals_ms")
    if not isinstance(totals_ms, dict):
        return None

    timing_stats = TimingStats()
    for key in timing_stats.totals:
        timing_stats.totals[key] = float(totals_ms.get(key, 0.0) or 0.0)
    timing_stats.frames = int(timings.get("frames", 0) or 0)
    return timing_stats


def render_validation_cli_report(
    raw: dict[str, Any],
    *,
    args: Any = None,
    timings: dict[str, Any] | None = None,
    title: str = CLI_RESULTS_SUMMARY_TITLE,
    include_sequences: bool = True,
    include_timings: bool = False,
    compare_raw: dict[str, Any] | None = None,
    compare_args: Any = None,
    colorize: bool | None = None,
    sys_module=sys,
    environ: dict[str, str] | None = None,
) -> str:
    if colorize is None:
        colorize = supports_ansi_color(sys_module=sys_module, environ=environ)

    results_module = import_module("boxmot.utils.evaluation.results")
    cfg = results_module._load_report_cfg_from_args(args)
    parsed_results = results_module.normalize_report_results(raw, args, cfg)
    if not parsed_results:
        return format_core_summary(raw if isinstance(raw, dict) else {})

    compare_cfg = results_module._load_report_cfg_from_args(compare_args) if compare_raw else {}
    compare_results = (
        results_module.normalize_report_results(compare_raw, compare_args, compare_cfg)
        if compare_raw
        else {}
    )

    blocks = [
        results_module.render_trackeval_report(
            parsed_results,
            args,
            cfg,
            title=title,
            include_sequences=include_sequences,
            compare_results=compare_results,
            colorize=bool(colorize),
        )
    ]

    if include_timings:
        timing_stats = timing_stats_from_snapshot(timings)
        if timing_stats is not None:
            timing_summary = timing_stats.format_summary()
            if timing_summary:
                blocks.append(timing_summary)

    return "\n".join(block for block in blocks if block)


def print_validation_cli_report(
    raw: dict[str, Any],
    *,
    args: Any = None,
    timings: dict[str, Any] | None = None,
    title: str = CLI_RESULTS_SUMMARY_TITLE,
    include_sequences: bool = True,
    include_timings: bool = False,
    print_fn=print,
    sys_module=sys,
    environ: dict[str, str] | None = None,
) -> None:
    report = render_validation_cli_report(
        raw,
        args=args,
        timings=timings,
        title=title,
        include_sequences=include_sequences,
        include_timings=include_timings,
        sys_module=sys_module,
        environ=environ,
    )
    if report:
        print_fn(report)


def format_remaining_time(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--"

    total_seconds = 0 if seconds <= 0 else int(math.ceil(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_tune_remaining(trial_durations: Sequence[float], remaining_trials: int) -> float | None:
    if remaining_trials <= 0:
        return 0.0
    if not trial_durations:
        return None
    avg_trial_seconds = sum(trial_durations) / len(trial_durations)
    return avg_trial_seconds * remaining_trials


def format_progress_bar(current: int, total: int, *, bar_width: int = 20) -> tuple[str, float]:
    if total <= 0:
        pct = 1.0 if current >= total else 0.0
    else:
        pct = min(max(current / total, 0.0), 1.0)

    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    return bar, pct


def format_named_progress(label: str, current: int, total: int, *, detail: str = "") -> str:
    bar, pct = format_progress_bar(current, total)
    message = f"  {label:<8s} {bar} {pct:>5.0%}  ({current}/{total})"
    if detail:
        message = f"{message}  {detail}"
    return message


def format_tune_progress(
    completed: int,
    total: int,
    summary: dict[str, Any] | None = None,
    *,
    current_trial: int | None = None,
    is_new_best: bool = False,
    remaining_seconds: float | None = None,
) -> str:
    remaining = format_remaining_time(remaining_seconds)
    if summary is None:
        running = current_trial if current_trial is not None else (completed + 1)
        return format_named_progress(
            "Tune",
            completed,
            total,
            detail=f"running trial {running}/{total}  remaining {remaining}",
        )

    summary_prefix = "last " if current_trial is not None and completed < total else ""
    summary_text = f"{summary_prefix}{format_core_summary(summary)}"
    status = "  best" if is_new_best else ""
    if current_trial is not None and completed < total:
        return format_named_progress(
            "Tune",
            completed,
            total,
            detail=(
                f"running trial {current_trial}/{total}  "
                f"{summary_text}{status}  remaining {remaining}"
            ),
        )

    return format_named_progress(
        "Tune",
        completed,
        total,
        detail=f"{summary_text}{status}  remaining {remaining}",
    )


def write_progress_line(
    message: str,
    previous_width: int,
    *,
    stream=None,
    final: bool = False,
    sys_module=sys,
) -> int:
    output = sys_module.stderr if stream is None else stream
    width = max(previous_width, len(message))
    is_tty = hasattr(output, "isatty") and output.isatty()
    rendered = f"\033[36m{message}\033[0m" if is_tty else message
    prefix = "\r\033[2K" if is_tty else "\r"
    output.write(prefix + rendered + (" " * (width - len(message))))
    if final:
        output.write("\n")
    output.flush()
    return width

