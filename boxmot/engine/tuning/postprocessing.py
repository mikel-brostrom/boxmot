"""Post-processing for tune results: trial collection, CSV, summary, Pareto."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.markup import escape as _escape_markup

from boxmot.engine.tuning.search_space import flatten_yaml_config, normalize_trial_config
from boxmot.engine.workflows.reporting import SUMMARY_COLUMNS
from boxmot.utils import logger as LOGGER

# Metrics that must be summed across classes (not averaged), because they are counts
METRIC_SUM = frozenset({"IDSW", "IDs"})
MAXIMIZE_TUNE_METRICS = ("HOTA", "MOTA", "IDF1", "AssA", "AssRe")
MINIMIZE_TUNE_METRICS = ("IDSW", "IDs", "IDSW_rate")
ALL_TUNE_METRICS = (*SUMMARY_COLUMNS, "IDSW_rate")


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def aggregate_results(results: dict) -> dict:
    """Aggregate per-class MOT metric results into a single flat dict.

    Ratio metrics are averaged; count metrics (IDSW, IDs) are summed.
    """
    values = list(results.values())

    if values and all(isinstance(v, dict) for v in values):
        class_dicts = [v for v in values if isinstance(v, dict)]
        aggregated = {}
        for k in SUMMARY_COLUMNS:
            if k in METRIC_SUM:
                aggregated[k] = sum(c.get(k, 0) for c in class_dicts)
            else:
                aggregated[k] = sum(c.get(k, 0) for c in class_dicts) / max(len(class_dicts), 1)
    else:
        aggregated = {k: results.get(k, 0) for k in SUMMARY_COLUMNS}

    aggregated["IDSW_rate"] = aggregated.get("IDSW", 0) / max(aggregated.get("IDs", 1), 1)
    return {k: max(0.0, aggregated.get(k, 0.0)) for k in ALL_TUNE_METRICS}


# ---------------------------------------------------------------------------
# Pareto front
# ---------------------------------------------------------------------------

def find_pareto_front(rows: list, maximize: list, minimize: list) -> list:
    """Return the subset of *rows* (list of dicts) that is Pareto-optimal."""
    pareto = []
    for i, mi in enumerate(rows):
        dominated = any(
            i != j
            and all(mj.get(m, 0) >= mi.get(m, 0) for m in maximize)
            and all(mj.get(m, float("inf")) <= mi.get(m, float("inf")) for m in minimize)
            and (
                any(mj.get(m, 0) > mi.get(m, 0) for m in maximize)
                or any(mj.get(m, float("inf")) < mi.get(m, float("inf")) for m in minimize)
            )
            for j, mj in enumerate(rows)
        )
        if not dominated:
            pareto.append(mi)
    return pareto


# ---------------------------------------------------------------------------
# YAML writing
# ---------------------------------------------------------------------------

def _format_yaml_value(v):
    """Format a value for YAML output, preserving Python-style bools and flow-style lists."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_format_yaml_value(x) for x in v) + "]"
    if v == "":
        return '""'
    return str(v)


def write_trial_yaml(yaml_cfg: dict, config: dict, path: Path):
    """Write a YAML config with ``default`` values replaced by trial values."""
    known_keys = ("type", "default", "range", "options", "choices", "values")

    def _append_entries(entries: dict, lines: list[str], *, indent: int) -> None:
        prefix = " " * indent
        child_prefix = " " * (indent + 2)
        for param, details in entries.items():
            if not isinstance(details, dict):
                lines.append(f"{prefix}{param}: {_format_yaml_value(details)}")
                continue

            lines.append(f"{prefix}{param}:")
            for key in known_keys:
                if key not in details:
                    continue
                value = config[param] if key == "default" and param in config else details[key]
                lines.append(f"{child_prefix}{key}: {_format_yaml_value(value)}")

            children = details.get("activates")
            if isinstance(children, dict):
                lines.append(f"{child_prefix}activates:")
                _append_entries(children, lines, indent=indent + 4)

            if indent == 0:
                lines.append("")

    lines: list[str] = []
    _append_entries(yaml_cfg, lines, indent=0)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Trial data collection
# ---------------------------------------------------------------------------

def collect_trial_data(results) -> list:
    """Extract trial_id, config, metrics from Ray Tune ResultGrid."""
    if results is None:
        return []
    trial_data = []
    try:
        results_iter = iter(results)
    except TypeError:
        LOGGER.warning("Tune results object is not iterable; cannot collect trial data.")
        return []
    for result in results_iter:
        try:
            if result.error or not result.metrics:
                continue
            trial_id = result.metrics.get("trial_id", "unknown")
            validation = result.metrics.get("_validation", {})
            trial_data.append({
                "trial_id": trial_id,
                "trial_dir": Path(result.path),
                "config": normalize_trial_config(result.config),
                "metrics": {k: result.metrics.get(k, 0.0) for k in ALL_TUNE_METRICS},
                "validation": validation if isinstance(validation, dict) else {},
            })
        except Exception as exc:
            LOGGER.debug(f"Skipping malformed trial result: {exc}")
            continue
    return trial_data


# ---------------------------------------------------------------------------
# Scoring / best trial
# ---------------------------------------------------------------------------

def score_summary(
    summary: dict[str, Any],
    *,
    maximize: list[str] | tuple[str, ...],
    minimize: list[str] | tuple[str, ...],
) -> tuple[float, ...]:
    """Score a metric summary for Pareto comparison."""
    score: list[float] = []
    for metric in maximize:
        score.append(float(summary.get(metric, float("-inf"))))
    for metric in minimize:
        score.append(-float(summary.get(metric, float("inf"))))
    return tuple(score)


def best_trial_data(trial_data: list, *, maximize: list[str], minimize: list[str]) -> dict | None:
    best_trial: dict | None = None
    best_score: tuple[float, ...] | None = None
    for trial in trial_data:
        score = score_summary(trial["metrics"], maximize=maximize, minimize=minimize)
        if best_score is None or score > best_score:
            best_trial = trial
            best_score = score
    return best_trial


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def save_results_csv(csv_path: Path, trial_data: list):
    """Write (or overwrite) a tidy CSV with one row per trial."""
    import csv

    if not trial_data:
        return
    metric_keys = list(ALL_TUNE_METRICS)
    config_keys = sorted({key for td in trial_data for key in td["config"].keys()})
    fieldnames = ["trial_id"] + metric_keys + config_keys

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for td in trial_data:
            row = {"trial_id": td["trial_id"]}
            row.update({k: td["metrics"].get(k, "") for k in metric_keys})
            row.update({k: td["config"].get(k, "") for k in config_keys})
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

def _convergence_label(search_range, top10_vals):
    if not isinstance(search_range, list) or len(search_range) < 2:
        return "—"
    lo, hi = float(search_range[0]), float(search_range[-1])
    search_span = hi - lo
    if search_span <= 0:
        return "fixed"

    t_lo, t_hi = min(top10_vals), max(top10_vals)
    top10_span = t_hi - t_lo
    ratio = top10_span / search_span

    near_lo = (t_lo - lo) / search_span < 0.15
    near_hi = (hi - t_hi) / search_span < 0.15

    if ratio < 0.20:
        label = "yes"
        if near_lo:
            label += " (at lower bound)"
        elif near_hi:
            label += " (at upper bound)"
    elif ratio > 0.60:
        label = "no — insensitive"
    else:
        label = "moderate"
        if near_lo:
            label += " (near lower bound)"
        elif near_hi:
            label += " (near upper bound)"
    return label


def _as_float_values(values: list[Any]) -> list[float] | None:
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def _format_markdown_value(value: Any) -> str:
    return str(value).replace("|", r"\|")


def _format_value_counts(values: list[Any]) -> str:
    from collections import Counter
    counts = Counter(values)
    return ", ".join(
        f"{_format_markdown_value(value)}: {count}"
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_summary(
    tune_dir: Path,
    trial_data: list,
    yaml_cfg: dict,
    tracker_name: str,
    maximize: list,
    minimize: list,
    args,
    *,
    emit_logs: bool = True,
) -> Path:
    """Generate ``summary.md`` in *tune_dir*."""
    primary = maximize[0]
    is_pareto = bool(minimize)
    sorted_data = sorted(trial_data, key=lambda t: t["metrics"].get(primary, 0), reverse=True)
    best = sorted_data[0]

    hotas = [t["metrics"].get("HOTA", 0) for t in trial_data]
    hotas_arr = np.array(hotas)

    lines = []

    # Header
    lines.append(f"# Tuning Summary: {tracker_name}\n")
    lines.append(f"- **Tracker:** {tracker_name}")
    lines.append(f"- **Detector:** {Path(args.detector[0]).stem}")
    lines.append(f"- **Benchmark:** {getattr(args, 'benchmark', getattr(args, 'data', ''))}")
    lines.append(f"- **Completed trials:** {len(trial_data)}")
    if is_pareto:
        lines.append(f"- **Optimize:** maximize {', '.join(maximize)} | minimize {', '.join(minimize)}")
    else:
        lines.append(f"- **Optimize:** maximize {', '.join(maximize)}")
    lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # Best Trial
    lines.append("---\n")
    lines.append(f"## Best Trial: `{best['trial_id']}`\n")
    metric_header = " | ".join(ALL_TUNE_METRICS)
    metric_sep = " | ".join("---" for _ in ALL_TUNE_METRICS)
    metric_vals = " | ".join(f"{best['metrics'].get(k, 0):.4f}" for k in ALL_TUNE_METRICS)
    lines.append(f"| {metric_header} |")
    lines.append(f"| {metric_sep} |")
    lines.append(f"| {metric_vals} |")
    lines.append(f"\nConfig saved to: `best_{tracker_name}.yaml`\n")

    # Pareto Front
    if is_pareto:
        metrics_for_pareto = [t["metrics"] for t in trial_data]
        pareto = find_pareto_front(metrics_for_pareto, maximize, minimize)
        pareto_sorted = sorted(pareto, key=lambda m: m.get(primary, 0), reverse=True)

        lines.append("---\n")
        opt_label = f"maximize {', '.join(maximize)} | minimize {', '.join(minimize)}"
        lines.append(f"## Pareto Front ({opt_label})\n")
        display_cols = list(dict.fromkeys(
            maximize + minimize + [c for c in ALL_TUNE_METRICS if c not in maximize + minimize]
        ))
        lines.append("| rank | " + " | ".join(display_cols) + " |")
        lines.append("| ---: | " + " | ".join("---:" for _ in display_cols) + " |")
        for i, m in enumerate(pareto_sorted, 1):
            vals = " | ".join(f"{m.get(c, 0):.4f}" for c in display_cols)
            lines.append(f"| {i} | {vals} |")
        lines.append(f"\n{len(pareto)} Pareto-optimal trial(s) out of {len(trial_data)}.\n")

    # Parameter Convergence
    top_n = min(10, len(sorted_data))
    top10 = sorted_data[:top_n]

    tunable_params = []
    flat_cfg = flatten_yaml_config(yaml_cfg)
    for param, details in flat_cfg.items():
        if not isinstance(details, dict):
            continue
        t = details.get("type", "")
        opts = details.get("options") or details.get("choices") or []
        rng = details.get("range", [])
        if t in ("uniform", "loguniform") and len(rng) >= 2 and rng[0] != rng[-1]:
            tunable_params.append(param)
        elif t in ("randint", "qrandint") and len(rng) >= 2 and rng[0] != rng[-1]:
            tunable_params.append(param)
        elif t == "choice" and len(opts) > 1:
            tunable_params.append(param)

    if tunable_params:
        lines.append("---\n")
        lines.append(f"## Parameter Convergence (Top-{top_n} trials)\n")
        lines.append("| Parameter | Search range | Top-N range | Top-N mean | Converged? |")
        lines.append("| --- | --- | --- | --- | --- |")

        for param in tunable_params:
            details = flat_cfg[param]
            top_vals = [t["config"].get(param) for t in top10 if param in t["config"]]
            if not top_vals:
                continue

            if isinstance(top_vals[0], bool):
                search_opts = details.get("options", details.get("choices", []))
                lines.append(f"| {param} | {search_opts} | {_format_value_counts(top_vals)} | — | — |")
            else:
                search_range = details.get("range", details.get("options", []))
                top_vals_f = _as_float_values(top_vals)
                if top_vals_f is None:
                    lines.append(
                        f"| {param} | {search_range} | {_format_value_counts(top_vals)} | — | categorical |"
                    )
                    continue
                t_lo, t_hi = min(top_vals_f), max(top_vals_f)
                t_mean = np.mean(top_vals_f)
                label = _convergence_label(search_range, top_vals_f)
                lines.append(
                    f"| {param} | [{search_range[0]}, {search_range[-1]}] "
                    f"| [{t_lo:.4f}, {t_hi:.4f}] | {t_mean:.4f} | {label} |"
                )

    # HOTA Distribution
    lines.append("\n---\n")
    lines.append("## HOTA Distribution\n")
    lines.append("| min | 25th | median | 75th | max | mean | std |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    lines.append(
        f"| {hotas_arr.min():.3f} | {np.percentile(hotas_arr, 25):.3f} "
        f"| {np.median(hotas_arr):.3f} | {np.percentile(hotas_arr, 75):.3f} "
        f"| {hotas_arr.max():.3f} | {hotas_arr.mean():.3f} | {hotas_arr.std():.3f} |"
    )
    lines.append("")

    summary_path = tune_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    if emit_logs:
        LOGGER.info(f"[bold]Summary saved to:[/bold] [cyan]{summary_path}[/cyan]")
    return summary_path


# ---------------------------------------------------------------------------
# Orchestrate all post-processing
# ---------------------------------------------------------------------------

def save_all_results(
    tune_dir: Path,
    results,
    yaml_cfg: dict,
    tracker_name: str,
    maximize: list,
    minimize: list,
    args,
    *,
    emit_logs: bool = True,
) -> dict | None:
    """Post-processing after tuner.fit(): per-trial YAMLs, CSV, best config, summary."""
    trial_data = collect_trial_data(results)
    if not trial_data:
        LOGGER.warning("No successful trials found.")
        return None

    tune_dir.mkdir(parents=True, exist_ok=True)

    # Per-trial YAML configs
    for td in trial_data:
        trial_dir = td["trial_dir"]
        if trial_dir.exists():
            try:
                yaml_path = trial_dir / f"{tracker_name}_{td['trial_id']}.yaml"
                write_trial_yaml(yaml_cfg, td["config"], yaml_path)
            except OSError as exc:
                LOGGER.debug(f"Failed to write trial YAML for {td['trial_id']}: {exc}")

    # results.csv
    csv_path = tune_dir / "results.csv"
    try:
        save_results_csv(csv_path, trial_data)
        if emit_logs:
            LOGGER.info(f"[bold]Results CSV:[/bold] [cyan]{_escape_markup(str(csv_path))}[/cyan]")
    except OSError as exc:
        LOGGER.warning(f"Failed to write results CSV: {exc}")
        csv_path = None

    # Best config
    best = best_trial_data(trial_data, maximize=maximize, minimize=minimize)
    if best is None:
        return None
    best_yaml_path = tune_dir / f"best_{tracker_name}.yaml"
    try:
        write_trial_yaml(yaml_cfg, best["config"], best_yaml_path)
        if emit_logs:
            LOGGER.info(f"[bold]Best config ({best['trial_id']}):[/bold] [cyan]{best_yaml_path}[/cyan]")
    except OSError as exc:
        LOGGER.warning(f"Failed to write best config YAML: {exc}")
        best_yaml_path = None

    # summary.md
    summary_path = None
    try:
        summary_path = generate_summary(
            tune_dir, trial_data, yaml_cfg, tracker_name,
            maximize, minimize, args, emit_logs=emit_logs,
        )
    except Exception as exc:
        LOGGER.warning(f"Failed to generate summary: {exc}")

    # Analysis plots
    try:
        from boxmot.engine.tuning.analysis import generate_tune_analysis
        generate_tune_analysis(tune_dir, tracker_name=tracker_name, n_trials=len(trial_data))
    except Exception as exc:
        LOGGER.debug(f"Analysis plot generation skipped: {exc}")

    return {
        "trial_data": trial_data,
        "csv_path": csv_path,
        "best_yaml_path": best_yaml_path,
        "summary_path": summary_path,
        "best_trial_id": best["trial_id"],
    }
