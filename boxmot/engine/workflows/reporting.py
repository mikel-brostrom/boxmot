from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Any, Sequence

from rich.console import Group, RenderableType
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from boxmot.utils.rich.core.ui import (
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
from boxmot.utils.timing import TimingStats, build_timing_display_rows, derive_timing_breakdown

SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
CORE_SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1")
SUMMARY_INT_COLUMNS = {"IDSW", "IDs"}
DEFAULT_VALIDATION_REPORT_TITLE = "VAL RESULTS"
DEFAULT_TUNE_BEST_REPORT_TITLE = "TUNE BEST RESULTS"
CLI_RESULTS_SUMMARY_TITLE = "📊 RESULTS SUMMARY"
CLI_TUNE_BEST_SUMMARY_TITLE = "📊 BEST TRIAL SUMMARY"

def extract_summary(raw_results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results_module = import_module("boxmot.engine.eval.results")
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
    """Serialize a :class:`TimingStats` into a JSON-friendly summary dict."""
    return timing_stats.to_summary_dict()


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
    totals_ms = timings.get("totals_ms")
    avg_ms = timings.get("avg_ms")
    if not isinstance(avg_ms, dict) or not avg_ms or not isinstance(totals_ms, dict):
        return None

    frames = int(timings.get("frames", 0) or 0)
    breakdown = derive_timing_breakdown(totals_ms, frames, total_time_ms=totals_ms.get("total"))
    metadata = timings.get("metadata") if isinstance(timings.get("metadata"), dict) else {}

    table = Table(
        expand=True,
        box=None,
        show_header=True,
        header_style=STYLE_TABLE_HEADER,
        row_styles=["", STYLE_MUTED],
        pad_edge=False,
        show_edge=False,
        padding=(0, 2),
        collapse_padding=False,
    )
    table.add_column("Stage", style=STYLE_TEXT_STRONG, no_wrap=True, ratio=3)
    table.add_column("Total (ms)", justify="right", no_wrap=True, ratio=1)
    table.add_column("Avg (ms)", justify="right", no_wrap=True, ratio=1)
    table.add_column("FPS", justify="right", no_wrap=True, ratio=1)

    for entry in build_timing_display_rows(
        breakdown,
        frames,
        metadata=metadata,
        overall_avg_ms=float(avg_ms.get("total", 0.0) or 0.0),
        overall_fps=float(timings.get("fps", 0.0) or 0.0),
    ):
        if entry["kind"] == "group":
            table.add_row(Text(str(entry["label"]), style=STYLE_ACCENT), "", "", "")
            continue
        if entry["kind"] == "note":
            table.add_row(Text(str(entry["label"]), style=STYLE_MUTED), "", "", "")
            continue

        row_style = STYLE_TEXT_STRONG if bool(entry["strong"]) else None
        table.add_row(
            str(entry["label"]),
            f"{float(entry['total']):.1f}",
            f"{float(entry['avg']):.2f}",
            f"{float(entry['fps']):.1f}",
            style=row_style,
        )

    cells = [
        Text.assemble(Text("Frames", style=STYLE_ACCENT), "  ", Text(str(frames), style=STYLE_TEXT)),
    ]
    meta = Table.grid(expand=True)
    meta.add_column(justify="left")
    meta.add_row(*cells)
    return Group(meta, table)


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
    results_module = import_module("boxmot.engine.eval.results")
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
        detail_keys = (
            [*primary_keys, *aggregate_keys]
            if primary_keys
            else aggregate_keys or list(parsed_results.keys())
        )

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
    results_module = import_module("boxmot.engine.eval.results")
    cfg = results_module._load_report_cfg_from_args(args)
    parsed_results = results_module.normalize_report_results(raw, args, cfg)
    if not parsed_results:
        fallback_title = title or "Results"
        return f"{fallback_title}\n{format_core_summary(raw if isinstance(raw, dict) else {})}"

    return results_module.render_mot_report(
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
    if isinstance(timings.get("metadata"), dict):
        timing_stats.metadata = dict(timings["metadata"])
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

    results_module = import_module("boxmot.engine.eval.results")
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
        results_module.render_mot_report(
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


__all__ = (
    "CLI_RESULTS_SUMMARY_TITLE",
    "CLI_TUNE_BEST_SUMMARY_TITLE",
    "CORE_SUMMARY_COLUMNS",
    "DEFAULT_TUNE_BEST_REPORT_TITLE",
    "DEFAULT_VALIDATION_REPORT_TITLE",
    "SUMMARY_COLUMNS",
    "core_summary_metrics",
    "extract_summary",
    "format_core_summary",
    "format_validation_report",
    "print_validation_cli_report",
    "render_validation_cli_report",
    "supports_ansi_color",
    "timing_stats_from_snapshot",
    "timing_summary_from_stats",
)
