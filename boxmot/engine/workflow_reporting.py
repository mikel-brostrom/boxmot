from __future__ import annotations

import math
import os
import sys
from importlib import import_module
from typing import Any, Sequence

from rich.console import Group, RenderableType
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from boxmot.utils.timing import TimingStats
from boxmot.utils.rich.ui import (
    STYLE_ACCENT,
    STYLE_COMBINED_ROW,
    STYLE_MUTED,
    STYLE_RULE,
    STYLE_SUBTLE,
    STYLE_TABLE_HEADER,
    STYLE_TEXT,
    STYLE_TEXT_STRONG,
    print_text,
)

SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
CORE_SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1")
SUMMARY_INT_COLUMNS = {"IDSW", "IDs"}
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


def _format_metric_value(metric: str, value: Any) -> str:
    if metric in SUMMARY_INT_COLUMNS:
        return f"{int(value or 0)}"
    return f"{float(value or 0.0):.2f}"


def _metric_delta_style(metric: str, delta: float) -> str:
    if delta == 0:
        return STYLE_MUTED
    positive_is_better = metric not in SUMMARY_INT_COLUMNS
    improved = delta > 0 if positive_is_better else delta < 0
    return "green" if improved else "red"


def _format_metric_delta(metric: str, value: Any, baseline_value: Any | None) -> Text:
    if baseline_value is None:
        return Text("", style=STYLE_MUTED)

    if metric in SUMMARY_INT_COLUMNS:
        delta = int(value or 0) - int(baseline_value or 0)
        return Text(f"({delta:+d})", style=_metric_delta_style(metric, float(delta)))

    delta = float(value or 0.0) - float(baseline_value or 0.0)
    return Text(f"({delta:+.2f})", style=_metric_delta_style(metric, delta))


def _build_sequence_table(
    rows: Sequence[tuple[str, dict[str, Any]]],
    *,
    show_header: bool = True,
    name_header: str = "Sequence",
    compare_rows: Sequence[dict[str, Any] | None] | None = None,
) -> Table:
    table = Table(
        expand=True,
        box=None,
        show_header=show_header,
        header_style=STYLE_TABLE_HEADER,
        row_styles=["", STYLE_MUTED],
        pad_edge=False,
        show_edge=False,
        padding=(0, 2),
        collapse_padding=False,
    )
    table.add_column(name_header, style=STYLE_TEXT_STRONG, no_wrap=True, ratio=3)
    for column in SUMMARY_COLUMNS:
        table.add_column(column, justify="right", no_wrap=True, ratio=1)

    compare_values = list(compare_rows) if compare_rows is not None else []

    for index, (row_name, metrics) in enumerate(rows):
        compare_metrics = compare_values[index] if index < len(compare_values) else None
        style = STYLE_COMBINED_ROW if row_name.startswith("COMBINED") else None
        values = [_format_metric_value(column, metrics.get(column, 0)) for column in SUMMARY_COLUMNS]
        table.add_row(row_name, *values, style=style)
        if compare_metrics is not None:
            delta_values = [
                _format_metric_delta(column, metrics.get(column, 0), compare_metrics.get(column))
                for column in SUMMARY_COLUMNS
            ]
            table.add_row(Text("", style=STYLE_MUTED), *delta_values)

    return table


def _build_timing_renderable(timings: dict[str, Any] | None) -> RenderableType | None:
    if not isinstance(timings, dict):
        return None
    avg_ms = timings.get("avg_ms")
    if not isinstance(avg_ms, dict) or not avg_ms:
        return None

    table = Table.grid(expand=True)
    for _ in range(4):
        table.add_column(justify="center")

    frames = int(timings.get("frames", 0) or 0)
    fps = float(timings.get("fps", 0.0) or 0.0)
    cells = [
        Text.assemble(Text("Frames", style=STYLE_ACCENT), "  ", Text(str(frames), style=STYLE_TEXT)),
        Text.assemble(Text("FPS", style=STYLE_ACCENT), "  ", Text(f"{fps:.1f}", style=STYLE_TEXT)),
        Text.assemble(Text("Avg total", style=STYLE_ACCENT), "  ", Text(f"{float(avg_ms.get('total', 0.0) or 0.0):.2f} ms", style=STYLE_TEXT)),
        Text.assemble(Text("Assoc", style=STYLE_ACCENT), "  ", Text(f"{float(avg_ms.get('track', 0.0) or 0.0):.2f} ms", style=STYLE_TEXT)),
    ]
    table.add_row(*cells)
    return table


def build_validation_cli_renderable(
    raw: dict[str, Any],
    *,
    args: Any = None,
    timings: dict[str, Any] | None = None,
    title: str | None = None,
    include_sequences: bool = True,
    include_timings: bool = False,
    compare_raw: dict[str, Any] | None = None,
    compare_args: Any = None,
) -> RenderableType:
    results_module = import_module("boxmot.utils.evaluation.results")
    cfg = results_module._load_report_cfg_from_args(args)
    parsed_results = results_module.normalize_report_results(raw, args, cfg)
    if not parsed_results:
        return Text(format_core_summary(raw if isinstance(raw, dict) else {}), style=STYLE_TEXT_STRONG)

    compare_cfg = results_module._load_report_cfg_from_args(compare_args) if compare_raw else {}
    compare_results = (
        results_module.normalize_report_results(compare_raw, compare_args, compare_cfg)
        if compare_raw
        else {}
    )

    primary_keys, aggregate_keys = results_module._summary_sort_keys(parsed_results, args or object(), cfg)
    if not primary_keys and not aggregate_keys:
        primary_keys = list(parsed_results.keys())

    sections: list[RenderableType] = []
    if title:
        sections.append(Text(title, style=STYLE_ACCENT))

    if len(primary_keys) > 1:
        sections.append(
            Group(
                Text("Per-Class Combined Metrics", style=STYLE_TEXT_STRONG),
                _build_sequence_table(
                        [(results_module._display_summary_name(name), parsed_results[name]) for name in primary_keys],
                        compare_rows=[compare_results.get(name) for name in primary_keys],
                    name_header="Class",
                ),
            )
        )
        if aggregate_keys:
            sections.append(Rule(style=STYLE_RULE))
            sections.append(
                Group(
                    Text("Aggregate Groups", style=STYLE_TEXT_STRONG),
                    _build_sequence_table(
                        [(results_module._display_summary_name(name), parsed_results[name]) for name in aggregate_keys],
                        compare_rows=[compare_results.get(name) for name in aggregate_keys],
                        name_header="Group",
                    ),
                )
            )

    if len(primary_keys) > 1:
        detail_keys = primary_keys
    else:
        detail_keys = [*primary_keys, *aggregate_keys] if primary_keys else aggregate_keys or list(parsed_results.keys())

    for index, class_name in enumerate(detail_keys):
        metrics = parsed_results[class_name]
        compare_metrics = compare_results.get(class_name)
        display_name = results_module._display_summary_name(class_name)
        per_sequence = list(sorted(metrics.get("per_sequence", {}).items())) if include_sequences else []
        combined_row = (f"COMBINED ({display_name})", metrics)
        per_sequence_compares = [
            compare_metrics.get("per_sequence", {}).get(seq_name)
            if isinstance(compare_metrics, dict)
            else None
            for seq_name, _ in per_sequence
        ]

        header = Text.assemble(
            Text(display_name, style=STYLE_TEXT_STRONG),
            Text("  •  ", style=STYLE_SUBTLE),
            Text(
                f"{len(per_sequence)} sequences" if per_sequence else "combined view",
                style=STYLE_MUTED,
            ),
        )

        block: list[RenderableType] = []
        if sections or index > 0:
            block.append(Rule(style=STYLE_RULE))
        block.extend(
            [
                header,
            ]
        )
        if per_sequence:
            block.append(_build_sequence_table(per_sequence, compare_rows=per_sequence_compares))
            block.append(Rule(style=STYLE_RULE))
            block.append(
                _build_sequence_table(
                    [combined_row],
                    show_header=False,
                    compare_rows=[compare_metrics if isinstance(compare_metrics, dict) and compare_metrics else None],
                )
            )
        else:
            block.append(
                _build_sequence_table(
                    [combined_row],
                    compare_rows=[compare_metrics if isinstance(compare_metrics, dict) and compare_metrics else None],
                )
            )
        sections.append(Group(*block))

    if include_timings:
        timing_renderable = _build_timing_renderable(timings)
        if timing_renderable is not None:
            sections.append(Rule(style=STYLE_RULE))
            sections.append(timing_renderable)

    return Group(*sections)


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
        if print_fn is print:
            print_text(report)
        else:
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

    core = format_core_summary(summary)
    suffix = "  best" if is_new_best else ""
    if current_trial is not None and current_trial > completed:
        detail = f"running trial {current_trial}/{total}  last {core}{suffix}  remaining {remaining}"
        return format_named_progress("Tune", completed, total, detail=detail)

    detail = f"{core}{suffix}  remaining {remaining}"
    return format_named_progress("Tune", completed, total, detail=detail)


__all__ = (
    "CLI_RESULTS_SUMMARY_TITLE",
    "CLI_TUNE_BEST_SUMMARY_TITLE",
    "CORE_SUMMARY_COLUMNS",
    "DEFAULT_TUNE_BEST_REPORT_TITLE",
    "DEFAULT_VALIDATION_REPORT_TITLE",
    "SUMMARY_COLUMNS",
    "core_summary_metrics",
    "estimate_tune_remaining",
    "extract_summary",
    "format_core_summary",
    "format_named_progress",
    "format_progress_bar",
    "format_remaining_time",
    "format_tune_progress",
    "format_validation_report",
    "print_validation_cli_report",
    "render_validation_cli_report",
    "supports_ansi_color",
    "timing_stats_from_snapshot",
    "timing_summary_from_stats",
)
