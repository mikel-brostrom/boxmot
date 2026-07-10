from __future__ import annotations

import argparse
import re
from typing import Any

from boxmot.data.benchmark import (
    COCO_CLASSES,
    _ordered_benchmark_eval_class_names,
    load_benchmark_cfg_from_args,
    resolve_eval_box_type,
    resolve_obb_classes_to_eval,
)
from boxmot.utils.rich.core.ui import print_text

SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
SUMMARY_INT_COLUMNS = {"IDSW", "IDs"}
MOT_REPORT_INTEGER_FIELDS = {
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "IDTP",
    "IDFN",
    "IDFP",
    "Dets",
    "GT_Dets",
    "IDs",
    "GT_IDs",
}
MOT_REPORT_METRIC_SPECS = {
    "HOTA": (
        "HOTA:",
        "HOTA",
        (
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
            "HOTA(0)",
            "LocA(0)",
            "HOTALocA(0)",
        ),
    ),
    "CLEAR": (
        "CLEAR:",
        "MOTA",
        (
            "MOTA",
            "MOTP",
            "MODA",
            "CLR_Re",
            "CLR_Pr",
            "MTR",
            "PTR",
            "MLR",
            "sMOTA",
            "CLR_TP",
            "CLR_FN",
            "CLR_FP",
            "IDSW",
            "MT",
            "PT",
            "ML",
            "Frag",
        ),
    ),
    "Identity": (
        "Identity:",
        "IDF1",
        ("IDF1", "IDR", "IDP", "IDTP", "IDFN", "IDFP"),
    ),
    "Count": (
        "Count:",
        "Dets",
        ("Dets", "GT_Dets", "IDs", "GT_IDs"),
    ),
}
SUMMARY_AGGREGATE_LABELS = {
    "cls_comb_det_av": "Class Avg (Det)",
    "cls_comb_cls_av": "Class Avg (Cls)",
    "HUMAN": "Human (Super)",
    "VEHICLE": "Vehicle (Super)",
    "BIKE": "Bike (Super)",
    "all": "All Classes",
}


def _match_header_class_name(raw_name: str, known_classes: list[str] | None = None) -> str:
    """Resolve a fixed-width metric header suffix to a class name."""
    if known_classes:
        exact_matches = [name for name in known_classes if raw_name.endswith(name)]
        if exact_matches:
            return max(exact_matches, key=len)

        normalized = raw_name.lower()
        folded_matches = [name for name in known_classes if normalized.endswith(name.lower())]
        if folded_matches:
            return max(folded_matches, key=len)

    if "-" in raw_name:
        return raw_name.split("-")[-1]
    return raw_name or "default"


def _extract_metric_header_tracker_class(content: str, header_token: str) -> str:
    """Return the tracker-class prefix without splitting multi-word class names."""
    content = content.strip()
    if not content:
        return ""

    match = re.search(rf"{re.escape(header_token)}(?=\s|$)", content)
    if match:
        tracker_class = content[: match.start()].rstrip()
        if tracker_class:
            return tracker_class

    tokens = content.split()
    if not tokens:
        return ""

    first_word = tokens[0]
    if len(tokens) > 1 and tokens[1] == header_token:
        return first_word
    if first_word.endswith(header_token):
        return first_word[: -len(header_token)]
    return first_word


def parse_mot_results(results: str, seq_names=None, known_classes: list[str] | None = None) -> dict:
    """Extract COMBINED and per-sequence fixed-width MOT summary metrics."""
    parsed_results: dict = {}
    sorted_names = sorted(seq_names, key=len, reverse=True) if seq_names else None
    current_class = None
    current_metric_type = None

    for line in results.splitlines():
        line = line.strip()
        if not line:
            continue

        is_header = False
        for metric_name, (prefix, header_token, _) in MOT_REPORT_METRIC_SPECS.items():
            if line.startswith(prefix):
                is_header = True
                current_metric_type = metric_name
                content = line[len(prefix):].strip()
                tracker_class = _extract_metric_header_tracker_class(content, header_token)
                if tracker_class:
                    current_class = _match_header_class_name(tracker_class, known_classes)
                    parsed_results.setdefault(current_class, {"per_sequence": {}})
                break

        if is_header:
            continue
        if not current_class or not current_metric_type:
            continue

        _, _, fields = MOT_REPORT_METRIC_SPECS[current_metric_type]
        col_name = 35
        col_val = 10

        def _parse_values(rest: str, name_len: int) -> list[str]:
            pad = max(0, col_name - name_len)
            value_part = rest[pad:]
            chunks = [value_part[i: i + col_val].strip() for i in range(0, len(value_part), col_val)]
            return [chunk for chunk in chunks if chunk]

        if line.startswith("COMBINED"):
            row_name = "COMBINED"
            values = _parse_values(line[len("COMBINED"):], len("COMBINED"))
        elif sorted_names is not None:
            row_name = None
            values = []
            for name in sorted_names:
                if line.startswith(name):
                    row_name = name
                    values = _parse_values(line[len(name):], len(name))
                    break
            if row_name is None:
                continue
        else:
            if len(line) <= col_name:
                continue
            row_name = line[:col_name].strip()
            if not row_name:
                continue
            values = _parse_values(line[col_name:], col_name)

        if not values:
            continue

        target = parsed_results[current_class] if row_name == "COMBINED" else parsed_results[current_class][
            "per_sequence"
        ].setdefault(row_name, {})
        for idx, key in enumerate(fields):
            if idx >= len(values):
                continue
            val = values[idx]
            target[key] = max(0, int(val) if key in MOT_REPORT_INTEGER_FIELDS else float(val))

    return parsed_results


def _extract_numeric_metrics(metrics: dict) -> dict:
    numeric_metrics: dict = {}
    for key, value in metrics.items():
        if key == "per_sequence" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            numeric_metrics[key] = int(value) if key in MOT_REPORT_INTEGER_FIELDS else float(value)
    return numeric_metrics


def build_mot_feedback(results: dict) -> dict:
    """Normalize MOT metric output into a stable payload for research/reflection."""
    summary_label, summary_metrics = _select_plot_metrics_data(results)
    summary = _extract_numeric_metrics(summary_metrics)

    if summary_label == "single_class" and isinstance(results, dict):
        selected_view = results
    elif isinstance(results, dict):
        selected_view = results.get(summary_label, {}) or {}
    else:
        selected_view = {}

    per_sequence_metrics = {}
    for seq_name, seq_metrics in selected_view.get("per_sequence", {}).items():
        if isinstance(seq_metrics, dict):
            per_sequence_metrics[seq_name] = _extract_numeric_metrics(seq_metrics)

    per_class_metrics = {}
    if isinstance(results, dict) and summary_label != "single_class":
        for class_name, class_metrics in results.items():
            if not isinstance(class_metrics, dict):
                continue
            numeric_metrics = _extract_numeric_metrics(class_metrics)
            if numeric_metrics:
                per_class_metrics[class_name] = numeric_metrics

    return {
        "summary_label": summary_label,
        "summary": summary,
        "per_sequence_metrics": per_sequence_metrics,
        "per_class_metrics": per_class_metrics,
    }


def filter_obb_mot_results(
    parsed_results: dict,
    args: argparse.Namespace,
    bench_cfg: dict,
) -> tuple[dict, bool]:
    """Keep selected OBB classes and append aggregate MMOT rows when relevant."""
    if not parsed_results:
        return parsed_results, False

    selected_classes = resolve_obb_classes_to_eval(args, bench_cfg)
    ordered: dict = {}
    for class_name in selected_classes:
        actual_key = class_name if class_name in parsed_results else next(
            (key for key in parsed_results if key.lower() == class_name.lower()),
            None,
        )
        if actual_key is not None:
            ordered[actual_key] = parsed_results[actual_key]

    if len(ordered) > 1:
        for name in ["cls_comb_det_av", "cls_comb_cls_av", "HUMAN", "VEHICLE", "BIKE", "all"]:
            if name in parsed_results and name not in ordered:
                ordered[name] = parsed_results[name]

    if ordered:
        return ordered, len(selected_classes) == 1 and len(ordered) == 1

    preferred_order = ["cls_comb_det_av", "cls_comb_cls_av", "HUMAN", "VEHICLE", "BIKE", "all"]
    fallback = {name: parsed_results[name] for name in preferred_order if name in parsed_results}
    if fallback:
        return fallback, "cls_comb_det_av" in fallback and len(fallback) == 1
    return parsed_results, len(parsed_results) == 1


def _display_summary_name(name: str) -> str:
    return SUMMARY_AGGREGATE_LABELS.get(name, name)


def _is_numeric_metric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _combined_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        column: metrics[column]
        for column in SUMMARY_COLUMNS
        if column in metrics and _is_numeric_metric(metrics[column])
    }


def _load_report_cfg_from_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    try:
        return load_benchmark_cfg_from_args(args) or {}
    except Exception:
        return {}


def _infer_single_class_report_name(args: Any, cfg: dict[str, Any] | None = None) -> str:
    if args is not None:
        remapped = getattr(args, "remapped_class_names", None)
        if remapped:
            return str(remapped[0])

        translated = getattr(args, "translated_benchmark_class_names", None)
        if translated:
            return str(translated[0])

        cfg = cfg or _load_report_cfg_from_args(args)
        bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
        eval_classes = bench_cfg.get("eval_classes")
        if isinstance(eval_classes, dict) and len(eval_classes) == 1:
            return str(next(iter(eval_classes.values())))
        if isinstance(eval_classes, (list, tuple)) and len(eval_classes) == 1:
            return str(eval_classes[0])

        class_indices = getattr(args, "classes", None)
        if class_indices is not None:
            indices = class_indices if isinstance(class_indices, list) else [class_indices]
            if len(indices) == 1:
                return str(COCO_CLASSES[int(indices[0])])
    return "results"


def normalize_report_results(
    raw: dict[str, Any],
    args: Any = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict) or not raw:
        return {}
    if _combined_summary_metrics(raw):
        return {_infer_single_class_report_name(args, cfg): raw}
    return {str(name): metrics for name, metrics in raw.items() if isinstance(metrics, dict)}


def _select_plot_metrics_data(results: dict) -> tuple[str, dict]:
    if not results:
        return "", {}

    flat_summary = _combined_summary_metrics(results)
    if flat_summary:
        return "single_class", flat_summary

    first_value = next(iter(results.values()))
    if isinstance(first_value, (int, float)):
        return "single_class", results

    for name in ("cls_comb_det_av", "cls_comb_cls_av", "all"):
        metrics = results.get(name)
        if isinstance(metrics, dict):
            return name, metrics

    if len(results) == 1:
        name, metrics = next(iter(results.items()))
        if isinstance(metrics, dict):
            return name, metrics
    return "", {}


def _format_summary_values(metrics: dict) -> list[str]:
    return [_format_summary_cell(key, metrics.get(key, 0)) for key in SUMMARY_COLUMNS]


def _format_summary_cell(column: str, value: Any) -> str:
    if column in SUMMARY_INT_COLUMNS:
        return f"{int(value or 0):>10}"
    return f"{float(value or 0):>10.2f}"


def _ansi_wrap(text: str, *codes: str, colorize: bool) -> str:
    if not colorize or not text:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


def _colorize_delta(text: str, column: str, delta: float, *, colorize: bool) -> str:
    if not colorize or delta == 0:
        return text
    positive_is_better = column not in SUMMARY_INT_COLUMNS
    is_improvement = delta > 0 if positive_is_better else delta < 0
    color_code = "32" if is_improvement else "31"
    return f"\033[{color_code}m{text}\033[0m"


def _format_summary_delta_only_cell(
    column: str,
    value: Any,
    baseline_value: Any | None = None,
    *,
    width: int,
    colorize: bool,
) -> str:
    if baseline_value is None:
        return " " * width
    if column in SUMMARY_INT_COLUMNS:
        current = int(value or 0)
        baseline = int(baseline_value or 0)
        delta = current - baseline
        delta_text = f"({delta:+d})"
        padded = f"{delta_text:>{width}}"
        return padded.replace(delta_text, _colorize_delta(delta_text, column, float(delta), colorize=colorize), 1)

    current = float(value or 0.0)
    baseline = float(baseline_value or 0.0)
    delta = current - baseline
    delta_text = f"({delta:+.2f})"
    padded = f"{delta_text:>{width}}"
    return padded.replace(delta_text, _colorize_delta(delta_text, column, delta, colorize=colorize), 1)


def _summary_sort_keys(parsed_results: dict, args: argparse.Namespace, cfg: dict) -> tuple[list[str], list[str]]:
    if not parsed_results:
        return [], []
    eval_box_type = resolve_eval_box_type(args, cfg)
    if eval_box_type != "obb":
        return list(parsed_results.keys()), []

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    primary_keys: list[str] = []
    seen: set[str] = set()
    for class_name in resolve_obb_classes_to_eval(args, bench_cfg):
        actual_key = class_name if class_name in parsed_results else next(
            (key for key in parsed_results if key.lower() == class_name.lower()),
            None,
        )
        if actual_key is not None and actual_key not in seen:
            primary_keys.append(actual_key)
            seen.add(actual_key)

    aggregate_keys = [key for key in parsed_results if key not in seen]
    if not primary_keys:
        return aggregate_keys, []
    return primary_keys, aggregate_keys


def _known_mot_class_names(args: argparse.Namespace, cfg: dict) -> list[str]:
    known: list[str] = []
    if getattr(args, "remapped_class_names", None):
        known.extend([str(name) for name in args.remapped_class_names])

    bench_cfg = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    known.extend(_ordered_benchmark_eval_class_names(bench_cfg))
    known.extend(["cls_comb_cls_av", "cls_comb_det_av", "HUMAN", "VEHICLE", "BIKE", "all"])

    deduped: list[str] = []
    seen: set[str] = set()
    for name in known:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def _render_summary_table(
    title: str,
    name_header: str,
    rows: list[tuple[str, dict[str, Any], dict[str, Any] | None]],
    *,
    total_width: int,
    name_width: int,
    colorize: bool,
) -> str:
    if not rows:
        return ""

    header_values = [name_header, *SUMMARY_COLUMNS]
    header_fmt = f"{{:<{name_width}}} " + " ".join(["{:>10}"] * len(SUMMARY_COLUMNS))
    compare_enabled = any(compare_metrics is not None for _, _, compare_metrics in rows)
    lines = [
        _ansi_wrap("=" * total_width, "36", colorize=colorize),
        _ansi_wrap(f"{title:^{total_width}}", "1", "36", colorize=colorize),
        _ansi_wrap("=" * total_width, "36", colorize=colorize),
        _ansi_wrap(header_fmt.format(*header_values), "1", "34", colorize=colorize),
        _ansi_wrap("-" * total_width, "36", colorize=colorize),
    ]
    for row_name, metrics, compare_metrics in rows:
        row_line = f"{row_name:<{name_width}} " + " ".join(_format_summary_values(metrics))
        if row_name.startswith("COMBINED"):
            row_line = _ansi_wrap(row_line, "1", "33", colorize=colorize)
        lines.append(row_line)
        if compare_enabled and compare_metrics is not None:
            delta_vals = " ".join(
                _format_summary_delta_only_cell(
                    column,
                    metrics.get(column, 0),
                    compare_metrics.get(column),
                    width=10,
                    colorize=colorize,
                )
                for column in SUMMARY_COLUMNS
            )
            lines.append(f"{'':<{name_width}} {delta_vals}")
    lines.append(_ansi_wrap("=" * total_width, "36", colorize=colorize))
    return "\n".join(lines)


def render_mot_report(
    parsed_results: dict[str, dict[str, Any]],
    args: Any = None,
    cfg: dict[str, Any] | None = None,
    *,
    title: str = "📊 RESULTS SUMMARY",
    include_sequences: bool = True,
    always_include_combined: bool = False,
    compare_results: dict[str, dict[str, Any]] | None = None,
    colorize: bool = False,
) -> str:
    if not parsed_results:
        return ""

    cfg = cfg or _load_report_cfg_from_args(args)
    compare_results = compare_results or {}
    primary_keys, aggregate_keys = _summary_sort_keys(parsed_results, args or object(), cfg)
    if not primary_keys and not aggregate_keys:
        primary_keys = list(parsed_results.keys())

    single_sequence = all(
        len(metrics.get("per_sequence", {})) <= 1
        for metrics in parsed_results.values()
        if isinstance(metrics, dict)
    )
    all_names = [_display_summary_name(name) for name in [*primary_keys, *aggregate_keys]]
    for class_metrics in parsed_results.values():
        all_names.extend(class_metrics.get("per_sequence", {}).keys())
    all_names.extend([f"COMBINED ({_display_summary_name(name)})" for name in primary_keys])
    name_width = max(18, max((len(name) for name in all_names), default=18) + 2)
    total_width = name_width + 1 + (10 * len(SUMMARY_COLUMNS)) + (len(SUMMARY_COLUMNS) - 1)

    blocks = [
        "\n".join(
            [
                _ansi_wrap("=" * total_width, "36", colorize=colorize),
                _ansi_wrap(f"{title:^{total_width}}", "1", "36", colorize=colorize),
                _ansi_wrap("=" * total_width, "36", colorize=colorize),
            ]
        )
    ]
    if len(primary_keys) > 1:
        class_rows = [
            (_display_summary_name(name), parsed_results[name], compare_results.get(name))
            for name in primary_keys
        ]
        blocks.append(
            _render_summary_table(
                "Per-Class Combined Metrics",
                "Class",
                class_rows,
                total_width=total_width,
                name_width=name_width,
                colorize=colorize,
            )
        )
        if aggregate_keys:
            aggregate_rows = [
                (_display_summary_name(name), parsed_results[name], compare_results.get(name))
                for name in aggregate_keys
            ]
            blocks.append(
                _render_summary_table(
                    "Aggregate Groups",
                    "Group",
                    aggregate_rows,
                    total_width=total_width,
                    name_width=name_width,
                    colorize=colorize,
                )
            )
        if include_sequences and (always_include_combined or not single_sequence):
            for class_name in primary_keys:
                compare_class_metrics = compare_results.get(class_name)
                per_sequence_rows = [
                    (
                        seq_name,
                        seq_metrics,
                        compare_class_metrics.get("per_sequence", {}).get(seq_name)
                        if isinstance(compare_class_metrics, dict)
                        else None,
                    )
                    for seq_name, seq_metrics in sorted(parsed_results[class_name].get("per_sequence", {}).items())
                ]
                per_sequence_rows.append(
                    (
                        f"COMBINED ({_display_summary_name(class_name)})",
                        parsed_results[class_name],
                        compare_class_metrics
                        if isinstance(compare_class_metrics, dict) and compare_class_metrics
                        else None,
                    )
                )
                blocks.append(
                    _render_summary_table(
                        f"Per-Sequence Details: {_display_summary_name(class_name)}",
                        "Sequence",
                        per_sequence_rows,
                        total_width=total_width,
                        name_width=name_width,
                        colorize=colorize,
                    )
                )
    else:
        detail_keys = primary_keys or aggregate_keys or list(parsed_results.keys())
        for class_name in detail_keys:
            compare_class_metrics = compare_results.get(class_name)
            per_sequence_rows = [
                (
                    seq_name,
                    seq_metrics,
                    compare_class_metrics.get("per_sequence", {}).get(seq_name)
                    if isinstance(compare_class_metrics, dict)
                    else None,
                )
                for seq_name, seq_metrics in sorted(parsed_results[class_name].get("per_sequence", {}).items())
            ]
            if not include_sequences:
                per_sequence_rows = []
            if always_include_combined or not single_sequence or not per_sequence_rows:
                per_sequence_rows.append(
                    (
                        f"COMBINED ({_display_summary_name(class_name)})",
                        parsed_results[class_name],
                        compare_class_metrics
                        if isinstance(compare_class_metrics, dict) and compare_class_metrics
                        else None,
                    )
                )
            blocks.append(
                _render_summary_table(
                    _display_summary_name(class_name),
                    "Sequence",
                    per_sequence_rows,
                    total_width=total_width,
                    name_width=name_width,
                    colorize=colorize,
                )
            )
    return "\n".join(block for block in blocks if block)


def log_mot_report(report: str) -> None:
    if report:
        print_text(report)


def _print_summary_table(
    title: str,
    name_header: str,
    rows: list[tuple[str, dict, bool]],
    total_w: int,
    name_w: int,
) -> None:
    report = _render_summary_table(
        title,
        name_header,
        [(row_name, metrics, None) for row_name, metrics, _highlight in rows],
        total_width=total_w,
        name_width=name_w,
        colorize=False,
    )
    log_mot_report(report)


__all__ = [
    "MOT_REPORT_INTEGER_FIELDS",
    "MOT_REPORT_METRIC_SPECS",
    "SUMMARY_COLUMNS",
    "SUMMARY_INT_COLUMNS",
    "_combined_summary_metrics",
    "_display_summary_name",
    "_known_mot_class_names",
    "_load_report_cfg_from_args",
    "_print_summary_table",
    "_select_plot_metrics_data",
    "_summary_sort_keys",
    "build_mot_feedback",
    "filter_obb_mot_results",
    "log_mot_report",
    "normalize_report_results",
    "parse_mot_results",
    "render_mot_report",
]
