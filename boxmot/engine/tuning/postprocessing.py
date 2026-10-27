"""Post-processing for tune results: trial collection, CSV, summary, Pareto."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yaml
from rich.markup import escape as _escape_markup

from boxmot.engine.eval.results import SUMMARY_COLUMNS
from boxmot.engine.tuning.search_space import normalize_trial_config
from boxmot.engine.tuning.setup_report import CMCSetupReport, describe_cmc_setup
from boxmot.trackers.common.config import nest_tracker_options
from boxmot.utils import logger as LOGGER

if TYPE_CHECKING:
    from boxmot.engine.tuning.analysis_data import ParameterSamples, TrialAnalysis

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


def write_trial_yaml(
    yaml_cfg: dict,
    config: dict,
    path: Path,
    *,
    base_config: dict | None = None,
    tracker_name: str | None = None,
    cmc_report: CMCSetupReport | None = None,
) -> None:
    """Write a reusable runtime tracker config for one tuning trial.

    Search metadata is deliberately excluded. When supplied, ``base_config``
    provides the full runtime config and the trial values overlay it. Policy
    annotations expose effective CMC without changing the executable settings.
    """

    del yaml_cfg  # retained in the public signature for tuning callers
    resolved = normalize_trial_config(base_config)
    resolved.update(normalize_trial_config(config))
    if tracker_name is not None:
        resolved = {"tracker": tracker_name, **resolved}
    contents = yaml.safe_dump(nest_tracker_options(resolved), sort_keys=False, allow_unicode=True)
    report = cmc_report or describe_cmc_setup(resolved)
    if report is not None:
        contents = re.sub(
            r"^use_cmc: (?:false|true)$",
            r"\g<0>  # Base value; overridden by the CMC policy for each sequence.",
            contents,
            flags=re.MULTILINE,
        )
        contents = report.yaml_header() + contents
    path.write_text(contents, encoding="utf-8")


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
            trial_data.append(
                {
                    "trial_id": trial_id,
                    "trial_dir": Path(result.path),
                    "config": normalize_trial_config(result.config),
                    "metrics": {k: result.metrics.get(k, 0.0) for k in ALL_TUNE_METRICS},
                    "validation": validation if isinstance(validation, dict) else {},
                }
            )
        except Exception as exc:
            LOGGER.debug(f"Skipping malformed trial result: {exc}")
            continue
    # Restored ResultGrid iteration follows checkpoint order, which can differ
    # from submission order. Preserve the first-trial baseline and stable CSVs
    # only when Ray recorded unambiguous counters for every successful trial.
    analysis = getattr(results, "_experiment_analysis", None)
    order = {}
    for trial in getattr(analysis, "trials", ()):
        counter = str(getattr(trial, "experiment_tag", "")).partition("_")[0]
        if counter.isdecimal():
            order[str(trial.trial_id)] = int(counter)
    counters = [order.get(str(trial["trial_id"])) for trial in trial_data]
    if None not in counters and len(set(counters)) == len(counters):
        trial_data.sort(key=lambda trial: order[str(trial["trial_id"])])
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
# Parameter concentration helpers
# ---------------------------------------------------------------------------


def _format_markdown_value(value: Any) -> str:
    """Escape table separators in user-authored configuration values."""
    return str(value).replace("|", r"\|")


def _format_value_counts(values: list[Any]) -> str:
    """Format categorical observations with their sample counts."""
    from collections import Counter

    counts = Counter(values)
    return ", ".join(
        f"{_format_markdown_value(value)}: {count}"
        for value, count in sorted(counts.items(), key=lambda item: str(item[0]))
    )


def _numeric_concentration(values: list[float], bounds: tuple | None, scale: str) -> str:
    """Describe observed spread and proximity without inferring convergence."""
    if not values:
        return "inactive"
    if len(values) == 1:
        label = "one sample"
    elif min(values) == max(values):
        label = "one observed value"
    else:
        label = "observed range"
    if bounds is None or bounds[1] <= bounds[0]:
        return label
    lo, hi = bounds
    v_lo, v_hi = min(values), max(values)
    if scale == "log" and min(lo, v_lo) > 0:
        lo, hi, v_lo, v_hi = np.log([lo, hi, v_lo, v_hi])
    span = hi - lo
    ratio = (v_hi - v_lo) / span
    if len(values) > 1 and v_lo != v_hi:
        label = "narrow" if ratio < 0.2 else "broad" if ratio > 0.6 else "intermediate"
        label += f" ({ratio:.1%} of {'log ' if scale == 'log' else ''}domain)"
    near = []
    if (v_lo - lo) / span <= 0.15:
        near.append("lower")
    if (hi - v_hi) / span <= 0.15:
        near.append("upper")
    if near:
        label += f"; near {' and '.join(near)} bound{'s' if len(near) > 1 else ''}"
    return label


def _format_sample_subset(samples: pd.DataFrame, selected: pd.Index, kind: str) -> str:
    """Render only active observations within a globally selected trial set."""
    values = samples.loc[samples.index.isin(selected), "value"].tolist()
    if not values:
        return f"0/{len(selected)}; inactive"
    detail = f"[{min(values):.4f}, {max(values):.4f}]" if kind == "numeric" else _format_value_counts(values)
    return f"{len(values)}/{len(selected)}; {detail}"


def parameter_concentration_markdown(
    frame: pd.DataFrame,
    schema: dict,
    *,
    parameters: dict[str, ParameterSamples] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Summarize active search observations among the best HOTA trials."""
    from boxmot.engine.tuning.analysis_data import NEAR_BEST_DELTA, analyze_parameters, near_best_mask

    lines = ["## Parameter Concentration", ""]
    if "HOTA" not in frame:
        return "\n".join(lines + ["Omitted: HOTA scores are unavailable.", ""])
    hotas = pd.to_numeric(frame["HOTA"], errors="coerce")
    scored = frame.loc[np.isfinite(hotas)].copy()
    scored["HOTA"] = hotas.loc[scored.index]
    if scored.empty:
        return "\n".join(lines + ["Omitted: no finite HOTA scores are available.", ""])
    top = scored.sort_values("HOTA", ascending=False, kind="stable").head(10).index
    near = scored.index[near_best_mask(scored["HOTA"])]
    lines.extend(
        [
            f"Top-{len(top)} and **{len(near)} near-best trials** "
            f"(within {NEAR_BEST_DELTA:g} absolute HOTA points of {scored['HOTA'].max():.4f}).",
            "Counts include only active searchable values; inactive defaults are excluded. "
            "Conditional branches use separate rows and recorded domains.",
            "",
        ]
    )
    if not schema:
        return "\n".join(lines + ["Omitted: the recorded search schema is unavailable.", ""])
    notes = list(notes or ())
    if parameters is None:
        try:
            parameters = analyze_parameters(scored, schema, notes=notes)
        except (KeyError, TypeError, ValueError) as exc:
            return "\n".join(lines + [f"Omitted: the recorded search schema could not be resolved ({exc}).", ""])
    if not parameters:
        return "\n".join(lines + ["No comparable active searchable parameters are available.", "", *notes, ""])
    lines.extend(
        [
            "| Parameter | Branch | Recorded search domain | All active | "
            f"Top-{len(top)} active; range/counts | Near-best active; range/counts | Near-best concentration |",
            "| --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for name, parameter in parameters.items():
        for branch, samples in parameter.samples.groupby("branch", sort=True):
            bounds = parameter.branch_bounds.get(branch, parameter.bounds)
            domain = (
                f"[{bounds[0]:g}, {bounds[1]:g}]" + (" (log)" if parameter.scale == "log" else "")
                if parameter.kind == "numeric" and bounds is not None
                else _format_markdown_value(list(parameter.branch_options.get(branch, parameter.options)))
            )
            near_values = samples.loc[samples.index.isin(near), "value"].tolist()
            if parameter.kind == "numeric":
                concentration = _numeric_concentration(near_values, bounds, parameter.scale)
            elif not near_values:
                concentration = "inactive"
            else:
                categories = len(set(near_values))
                concentration = f"{categories} observed categor{'y' if categories == 1 else 'ies'}"
            lines.append(
                f"| {_format_markdown_value(name)} | {_format_markdown_value(branch)} | {domain} "
                f"| {len(samples)}/{len(scored)} "
                f"| {_format_sample_subset(samples, top, parameter.kind)} "
                f"| {_format_sample_subset(samples, near, parameter.kind)} | {concentration} |"
            )
    lines.extend(
        [
            "",
            *notes,
            *([""] if notes else []),
            "Spread describes sampled values, not parameter importance or convergence. "
            "Narrow/intermediate/broad mean less than 20%, 20–60%, or more than 60% of the domain; "
            "near a bound means within 15% of that domain (in log space for log searches). "
            "Unequal sampling and simultaneous parameter changes limit comparisons. "
            "These ranges guide the next experiment; they do not establish repeatability or superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_search_schema(tune_dir: Path, yaml_cfg: dict) -> dict:
    """Prefer this run's saved schema over the caller's current tracker YAML."""
    path = tune_dir / "search-space.json"
    if not path.exists():
        return yaml_cfg
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema", {})
        return schema if isinstance(schema, dict) else {}
    except (OSError, ValueError, AttributeError) as exc:
        LOGGER.warning(f"Cannot read recorded search schema for summary: {exc}")
        return {}


def _refresh_saved_cmc_setup(tune_dir: Path, summary: str) -> str:
    """Expose recorded CMC overrides in an existing best config and summary."""
    best_path = tune_dir / "best.yaml"
    if not best_path.is_file():
        return summary
    try:
        config = yaml.safe_load(best_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("best.yaml must contain a mapping")
        report = describe_cmc_setup(normalize_trial_config(config), tune_dir=tune_dir)
        if report is None:
            return summary
        write_trial_yaml({}, config, best_path, cmc_report=report)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        LOGGER.warning(f"Cannot refresh saved CMC setup: {exc}")
        return summary
    section = report.markdown().rstrip() + "\n\n"
    pattern = re.compile(r"^## Effective CMC Setup\n.*?(?=^---\s*$|^## |\Z)", re.MULTILINE | re.DOTALL)
    if pattern.search(summary):
        return pattern.sub(lambda _match: section, summary, count=1)
    marker = "## Parameter Concentration"
    if marker in summary:
        return summary.replace(marker, section + "---\n\n" + marker, 1)
    return summary.rstrip() + "\n\n---\n\n" + section


def update_analysis_summary(tune_dir: Path, *, analysis: TrialAnalysis | None = None) -> Path | None:
    """Refresh diagnostics from a shared or saved snapshot, preserving provenance."""
    from boxmot.engine.tuning.analysis_data import load_tune_analysis

    tune_dir = Path(tune_dir)
    path = tune_dir / "summary.md"
    if not path.exists():
        return None
    if analysis is None:
        analysis = load_tune_analysis(tune_dir)
    if analysis is None:
        return None
    section = parameter_concentration_markdown(
        analysis.trials,
        analysis.schema,
        parameters=analysis.parameters,
        notes=analysis.parameter_notes,
    )
    original = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^## Parameter (?:Convergence|Concentration)[^\n]*\n.*?(?=^---\s*$|^## |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    if pattern.search(original):
        updated = pattern.sub(lambda _match: section + "\n", original, count=1)
    else:
        updated = original.rstrip() + "\n\n---\n\n" + section
    path.write_text(_refresh_saved_cmc_setup(tune_dir, updated), encoding="utf-8")
    return path


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
    base_config: dict | None = None,
    emit_logs: bool = True,
    analysis: TrialAnalysis | None = None,
) -> Path:
    """Generate ``summary.md`` in *tune_dir*."""
    primary = maximize[0] if maximize else minimize[0]
    is_pareto = bool(minimize)
    best = best_trial_data(trial_data, maximize=maximize, minimize=minimize)
    if best is None:
        raise ValueError("Cannot summarize tuning without successful trials.")
    objective_label = " | ".join(
        f"{direction} {', '.join(metrics)}"
        for direction, metrics in (("maximize", maximize), ("minimize", minimize))
        if metrics
    )

    hotas = [t["metrics"].get("HOTA", 0) for t in trial_data]
    hotas_arr = np.array(hotas)

    lines = []

    # Header
    lines.append(f"# Tuning Summary: {tracker_name}\n")
    lines.append(f"- **Tracker:** {tracker_name}")
    experiment_id = getattr(args, "experiment_id", None) or getattr(args, "experiment", None)
    dataset_id = getattr(args, "dataset_id", None)
    build_path = getattr(args, "build_path", None) or getattr(args, "build", None)
    if experiment_id:
        lines.append(f"- **Experiment:** {_format_markdown_value(experiment_id)}")
    if dataset_id:
        lines.append(f"- **Dataset:** {_format_markdown_value(dataset_id)}")
    if build_path:
        lines.append(f"- **Build:** {_format_markdown_value(Path(build_path))}")
    lines.append(f"- **Completed trials:** {len(trial_data)}")
    lines.append(f"- **Optimize:** {objective_label}")
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
    lines.append("\nConfig saved to: `best.yaml`\n")
    cmc_report = describe_cmc_setup(
        {**normalize_trial_config(base_config), **normalize_trial_config(best["config"])},
        args=args,
        tune_dir=tune_dir,
    )
    if cmc_report is not None:
        lines.extend(["---\n", cmc_report.markdown(), ""])

    # Pareto Front
    if is_pareto:
        metrics_for_pareto = [t["metrics"] for t in trial_data]
        pareto = find_pareto_front(metrics_for_pareto, maximize, minimize)
        pareto_sorted = sorted(pareto, key=lambda m: m.get(primary, 0), reverse=bool(maximize))

        lines.append("---\n")
        lines.append(f"## Pareto Front ({objective_label})\n")
        display_cols = list(
            dict.fromkeys(maximize + minimize + [c for c in ALL_TUNE_METRICS if c not in maximize + minimize])
        )
        lines.append("| rank | " + " | ".join(display_cols) + " |")
        lines.append("| ---: | " + " | ".join("---:" for _ in display_cols) + " |")
        for i, m in enumerate(pareto_sorted, 1):
            vals = " | ".join(f"{m.get(c, 0):.4f}" for c in display_cols)
            lines.append(f"| {i} | {vals} |")
        lines.append(f"\n{len(pareto)} Pareto-optimal trial(s) out of {len(trial_data)}.\n")

    # Parameter concentration: use the run's schema and conditional activity.
    if analysis is None:
        frame = pd.DataFrame(
            [
                {**normalize_trial_config(trial["config"]), **trial["metrics"], "trial_id": trial["trial_id"]}
                for trial in trial_data
            ]
        )
        concentration = parameter_concentration_markdown(frame, _summary_search_schema(tune_dir, yaml_cfg))
    else:
        concentration = parameter_concentration_markdown(
            analysis.trials,
            analysis.schema,
            parameters=analysis.parameters,
            notes=analysis.parameter_notes,
        )
    lines.extend(["---\n", concentration])

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
    base_config: dict | None = None,
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
                report = describe_cmc_setup(
                    {**normalize_trial_config(base_config), **td["config"]},
                    args=args,
                    tune_dir=tune_dir,
                )
                write_trial_yaml(
                    yaml_cfg,
                    td["config"],
                    yaml_path,
                    base_config=base_config,
                    tracker_name=tracker_name,
                    cmc_report=report,
                )
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
    best_yaml_path = tune_dir / "best.yaml"
    try:
        report = describe_cmc_setup(
            {**normalize_trial_config(base_config), **best["config"]},
            args=args,
            tune_dir=tune_dir,
        )
        write_trial_yaml(
            yaml_cfg,
            best["config"],
            best_yaml_path,
            base_config=base_config,
            tracker_name=tracker_name,
            cmc_report=report,
        )
        if emit_logs:
            LOGGER.info(f"[bold]Best config ({best['trial_id']}):[/bold] [cyan]{best_yaml_path}[/cyan]")
    except OSError as exc:
        LOGGER.warning(f"Failed to write best config YAML: {exc}")
        best_yaml_path = None

    # Export reusable configs before optional analysis, then share the resolved
    # parameter samples across reports. Summary can fall back to collected trials.
    analysis = None
    if csv_path is not None:
        from boxmot.engine.tuning.analysis_data import load_tune_analysis

        try:
            analysis = load_tune_analysis(tune_dir, fallback_schema=yaml_cfg)
        except Exception as exc:
            LOGGER.warning(f"Failed to load tuning analysis: {exc}")

    # summary.md
    summary_path = None
    try:
        summary_path = generate_summary(
            tune_dir,
            trial_data,
            yaml_cfg,
            tracker_name,
            maximize,
            minimize,
            args,
            base_config=base_config,
            emit_logs=emit_logs,
            analysis=analysis,
        )
    except Exception as exc:
        LOGGER.warning(f"Failed to generate summary: {exc}")

    # Analysis plots
    try:
        from boxmot.engine.tuning.analysis import generate_tune_analysis

        if analysis is not None:
            generate_tune_analysis(tune_dir, tracker_name=tracker_name, n_trials=len(trial_data), data=analysis)
    except Exception as exc:
        LOGGER.debug(f"Analysis plot generation skipped: {exc}")

    return {
        "trial_data": trial_data,
        "csv_path": csv_path,
        "best_yaml_path": best_yaml_path,
        "summary_path": summary_path,
        "best_trial_id": best["trial_id"],
    }
