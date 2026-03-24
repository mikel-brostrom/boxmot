from __future__ import annotations

import argparse
import re
from typing import Optional

from boxmot.utils import logger as LOGGER

from .benchmark import resolve_eval_box_type, resolve_obb_classes_to_eval


SUMMARY_COLUMNS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs")
SUMMARY_INT_COLUMNS = {"IDSW", "IDs"}
SUMMARY_AGGREGATE_LABELS = {
    "cls_comb_det_av": "Class Avg (Det)",
    "cls_comb_cls_av": "Class Avg (Cls)",
    "HUMAN": "Human (Super)",
    "VEHICLE": "Vehicle (Super)",
    "BIKE": "Bike (Super)",
    "all": "All Classes",
}


def _match_header_class_name(raw_name: str, known_classes: Optional[list[str]] = None) -> str:
    """Resolve a TrackEval header suffix to a class name, preserving hyphenated names."""
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
    """Return the TrackEval ``tracker-class`` prefix without splitting multi-word class names."""
    content = content.strip()
    if not content:
        return ""

    match = re.search(rf"{re.escape(header_token)}(?=\s|$)", content)
    if match:
        tracker_class = content[:match.start()].rstrip()
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


def _ordered_benchmark_eval_class_names(bench_cfg: dict) -> list[str]:
    """Return benchmark eval class names in config order without splitting embedded whitespace."""
    if not isinstance(bench_cfg, dict):
        return []

    eval_classes_cfg = bench_cfg.get("eval_classes")
    if isinstance(eval_classes_cfg, dict) and eval_classes_cfg:
        return [str(name) for _, name in sorted(eval_classes_cfg.items(), key=lambda kv: int(kv[0]))]
    if isinstance(eval_classes_cfg, (list, tuple)):
        return [str(name) for name in eval_classes_cfg]
    return []


def parse_mot_results(results: str, seq_names=None, known_classes: Optional[list[str]] = None) -> dict:
    """
    Extract COMBINED and per-sequence HOTA, MOTA, IDF1, AssA, AssRe, IDSW, and IDs.
    """
    metric_specs = {
        "HOTA": ("HOTA:", {"HOTA": 0, "AssA": 2, "AssRe": 5}),
        "MOTA": ("CLEAR:", {"MOTA": 0, "IDSW": 12}),
        "IDF1": ("Identity:", {"IDF1": 0}),
        "IDs": ("Count:", {"IDs": 2}),
    }

    int_fields = {"IDSW", "IDs"}
    parsed_results: dict = {}
    sorted_names = sorted(seq_names, key=len, reverse=True) if seq_names else None

    lines = results.splitlines()
    current_class = None
    current_metric_type = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_header = False
        for metric_name, (prefix, _) in metric_specs.items():
            if line.startswith(prefix):
                is_header = True
                current_metric_type = metric_name
                header_token = "Dets" if metric_name == "IDs" else metric_name

                content = line[len(prefix):].strip()
                tracker_class = _extract_metric_header_tracker_class(content, header_token)
                if tracker_class:
                    current_class = _match_header_class_name(tracker_class, known_classes)
                    if current_class not in parsed_results:
                        parsed_results[current_class] = {"per_sequence": {}}
                break

        if is_header:
            continue

        if not current_class or not current_metric_type:
            continue

        _, field_map = metric_specs[current_metric_type]
        col_name = 35
        col_val = 10

        def _parse_values(rest: str, name_len: int) -> list[str]:
            pad = max(0, col_name - name_len)
            value_part = rest[pad:]
            chunks = [value_part[i:i + col_val].strip() for i in range(0, len(value_part), col_val)]
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

        if row_name == "COMBINED":
            for key, idx in field_map.items():
                if idx < len(values):
                    val = values[idx]
                    parsed_results[current_class][key] = max(
                        0,
                        int(val) if key in int_fields else float(val),
                    )
            continue

        if row_name not in parsed_results[current_class]["per_sequence"]:
            parsed_results[current_class]["per_sequence"][row_name] = {}
        for key, idx in field_map.items():
            if idx < len(values):
                val = values[idx]
                parsed_results[current_class]["per_sequence"][row_name][key] = max(
                    0,
                    int(val) if key in int_fields else float(val),
                )

    return parsed_results


def _filter_obb_trackeval_results(
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


def _select_plot_metrics_data(results: dict) -> tuple[str, dict]:
    if not results:
        return "", {}

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
    values: list[str] = []
    for key in SUMMARY_COLUMNS:
        value = metrics.get(key, 0)
        if key in SUMMARY_INT_COLUMNS:
            values.append(f"{int(value):>10}")
        else:
            values.append(f"{float(value):>10.2f}")
    return values


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


def _known_trackeval_class_names(args: argparse.Namespace, cfg: dict) -> list[str]:
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


def _print_summary_table(
    title: str,
    name_header: str,
    rows: list[tuple[str, dict, bool]],
    total_w: int,
    name_w: int,
) -> None:
    if not rows:
        return

    header_values = [name_header, *SUMMARY_COLUMNS]
    header_fmt = f"{{:<{name_w}}} " + " ".join(["{:>10}"] * len(SUMMARY_COLUMNS))
    LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold><cyan>{title:^{total_w}}</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>{header_fmt.format(*header_values)}</bold>")
    LOGGER.opt(colors=True).info("<blue>" + "-" * total_w + "</blue>")

    for row_name, metrics, highlight in rows:
        name_col = f"{row_name:<{name_w}}"
        vals_str = " ".join(_format_summary_values(metrics))
        if highlight:
            LOGGER.opt(colors=True).info(f"<bold>{name_col} <cyan>{vals_str}</cyan></bold>")
        else:
            LOGGER.opt(colors=True).info(f"{name_col} <blue>{vals_str}</blue>")

    LOGGER.opt(colors=True).info("<blue>" + "=" * total_w + "</blue>")


__all__ = [
    "_display_summary_name",
    "_filter_obb_trackeval_results",
    "_known_trackeval_class_names",
    "_ordered_benchmark_eval_class_names",
    "_print_summary_table",
    "_select_plot_metrics_data",
    "_summary_sort_keys",
    "parse_mot_results",
]
