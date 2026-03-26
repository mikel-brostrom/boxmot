#!/usr/bin/env python3
"""
Hyperparameter tuning for multi-object trackers using Ray Tune + Optuna.

Supports single-objective (default) and multi-objective Pareto search:
  boxmot tune --tracker bytetrack --benchmark ...                        # maximize HOTA
  boxmot tune ... --maximize HOTA --minimize IDSW_rate                  # Pareto mode
  boxmot tune ... --maximize HOTA --maximize IDF1 --minimize IDSW_rate  # 3-objective Pareto
"""

import csv
import os

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"   # keep CWD constant for all trials

from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from boxmot.utils.evaluation.results import SUMMARY_COLUMNS
from boxmot.engine.evaluator import (eval_setup,
                                     run_generate_dets_embs,
                                     run_generate_mot_results, run_trackeval)
from boxmot.utils import TRACKER_CONFIGS
from boxmot.utils import logger as LOGGER

# Metrics that must be summed across classes (not averaged), because they are counts
METRIC_SUM = frozenset({"IDSW", "IDs"})

# All metrics returned from each trial (SUMMARY_COLUMNS + derived)
ALL_TUNE_METRICS = (*SUMMARY_COLUMNS, "IDSW_rate")


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml_config(tracking_method: str) -> dict:
    config_path = TRACKER_CONFIGS / f"{tracking_method}.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def yaml_to_search_space(config: dict, tune) -> dict:
    space = {}
    for param, details in config.items():
        t = details.get("type")
        if t == "uniform":
            space[param] = tune.uniform(*details["range"])
        elif t == "randint":
            space[param] = tune.randint(*details["range"])
        elif t == "qrandint":
            space[param] = tune.qrandint(*details["range"])
        elif t == "choice":
            space[param] = tune.choice(details.get("options"))
        elif t == "grid_search":
            space[param] = tune.choice(details["values"])
        elif t == "loguniform":
            space[param] = tune.loguniform(*details["range"])
    return space


def _format_yaml_value(v):
    """Format a value for YAML output, preserving Python-style bools and flow-style lists."""
    if isinstance(v, bool):
        return str(v)          # True/False, not true/false
    if isinstance(v, float):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_format_yaml_value(x) for x in v) + "]"
    return str(v)


def _write_trial_yaml(yaml_cfg: dict, config: dict, path: Path):
    """
    Write a YAML config identical to the original search-space YAML,
    but with ``default`` values replaced by the trial's chosen values.
    """
    # Keys to output in order (skip unknown keys to be safe)
    known_keys = ("type", "default", "range", "options", "choices", "values")
    lines = []
    for param, details in yaml_cfg.items():
        lines.append(f"{param}:")
        for key in known_keys:
            if key not in details:
                continue
            value = details[key]
            if key == "default" and param in config:
                value = config[param]
            lines.append(f"  {key}: {_format_yaml_value(value)}")
        lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _aggregate_results(results: dict) -> dict:
    """
    Aggregate per-class trackeval results into a single flat dict.

    Ratio metrics (HOTA, MOTA, ...) are averaged across classes.
    Count metrics (IDSW, IDs) are summed — summing counts is physically meaningful
    and ensures IDSW_rate = sum(IDSW) / sum(IDs) is weighted by class frequency.
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

def _find_pareto_front(rows: list, maximize: list, minimize: list) -> list:
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
# Post-processing: per-trial YAML, CSV, summary, best config
# ---------------------------------------------------------------------------

def _collect_trial_data(results) -> list:
    """Extract trial_id, config, metrics from Ray Tune ResultGrid."""
    trial_data = []
    for result in results:
        if result.error or not result.metrics:
            continue
        trial_id = result.metrics.get("trial_id", "unknown")
        trial_data.append({
            "trial_id": trial_id,
            "trial_dir": Path(result.path),
            "config": result.config,
            "metrics": {k: result.metrics.get(k, 0.0) for k in ALL_TUNE_METRICS},
        })
    return trial_data


def _save_results_csv(csv_path: Path, trial_data: list):
    """Write (or overwrite) a tidy CSV with one row per trial."""
    if not trial_data:
        return
    metric_keys = list(ALL_TUNE_METRICS)
    config_keys = list(trial_data[0]["config"].keys())
    fieldnames = ["trial_id"] + metric_keys + config_keys

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for td in trial_data:
            row = {"trial_id": td["trial_id"]}
            row.update({k: td["metrics"].get(k, "") for k in metric_keys})
            row.update({k: td["config"].get(k, "") for k in config_keys})
            writer.writerow(row)


def _convergence_label(search_range, top10_vals):
    """Determine whether a parameter converged based on top-10 values vs search range."""
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


def _generate_summary(
    tune_dir: Path,
    trial_data: list,
    yaml_cfg: dict,
    tracking_method: str,
    maximize: list,
    minimize: list,
    args,
):
    """Generate ``summary.md`` in *tune_dir*."""
    primary = maximize[0]
    is_pareto = bool(minimize)
    sorted_data = sorted(trial_data, key=lambda t: t["metrics"].get(primary, 0), reverse=True)
    best = sorted_data[0]

    hotas = [t["metrics"].get("HOTA", 0) for t in trial_data]
    hotas_arr = np.array(hotas)

    lines = []

    # --- Header ---
    lines.append(f"# Tuning Summary: {tracking_method}\n")
    lines.append(f"- **Tracker:** {tracking_method}")
    lines.append(f"- **Detector:** {Path(args.yolo_model[0]).stem}")
    lines.append(f"- **Benchmark:** {getattr(args, 'benchmark', getattr(args, 'data', ''))}")
    lines.append(f"- **Completed trials:** {len(trial_data)}")
    if is_pareto:
        lines.append(f"- **Optimize:** maximize {', '.join(maximize)} | minimize {', '.join(minimize)}")
    else:
        lines.append(f"- **Optimize:** maximize {', '.join(maximize)}")
    lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # --- Best Trial ---
    lines.append("---\n")
    lines.append(f"## Best Trial: `{best['trial_id']}`\n")
    metric_header = " | ".join(ALL_TUNE_METRICS)
    metric_sep = " | ".join("---" for _ in ALL_TUNE_METRICS)
    metric_vals = " | ".join(f"{best['metrics'].get(k, 0):.4f}" for k in ALL_TUNE_METRICS)
    lines.append(f"| {metric_header} |")
    lines.append(f"| {metric_sep} |")
    lines.append(f"| {metric_vals} |")
    lines.append(f"\nConfig saved to: `best_{tracking_method}.yaml`\n")

    # --- Pareto Front ---
    if is_pareto:
        metrics_for_pareto = [t["metrics"] for t in trial_data]
        pareto = _find_pareto_front(metrics_for_pareto, maximize, minimize)
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

    # --- Parameter Convergence ---
    top_n = min(10, len(sorted_data))
    top10 = sorted_data[:top_n]

    # Identify tunable params (those with a search range, not fixed single-choice)
    tunable_params = []
    for param, details in yaml_cfg.items():
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
            details = yaml_cfg[param]
            top_vals = [t["config"].get(param) for t in top10 if param in t["config"]]

            if not top_vals:
                continue

            if isinstance(top_vals[0], bool):
                from collections import Counter
                dist = dict(Counter(top_vals))
                search_opts = details.get("options", [])
                lines.append(f"| {param} | {search_opts} | {dist} | — | — |")
            else:
                search_range = details.get("range", details.get("options", []))
                top_vals_f = [float(v) for v in top_vals]
                t_lo, t_hi = min(top_vals_f), max(top_vals_f)
                t_mean = np.mean(top_vals_f)
                label = _convergence_label(search_range, top_vals_f)
                lines.append(
                    f"| {param} | [{search_range[0]}, {search_range[-1]}] "
                    f"| [{t_lo:.4f}, {t_hi:.4f}] | {t_mean:.4f} | {label} |"
                )

    # --- HOTA Distribution ---
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

    (tune_dir / "summary.md").write_text("\n".join(lines))
    LOGGER.opt(colors=True).info(
        f"<bold>Summary saved to:</bold> <cyan>{tune_dir / 'summary.md'}</cyan>"
    )


def _save_all_results(
    tune_dir: Path,
    results,
    yaml_cfg: dict,
    tracking_method: str,
    maximize: list,
    minimize: list,
    args,
):
    """
    Post-processing after tuner.fit():
      1. Per-trial YAML configs (in each trial folder)
      2. results.csv (all trials, one row each)
      3. best_<tracker>.yaml (copy of best trial config at tune root)
      4. summary.md
    """
    trial_data = _collect_trial_data(results)
    if not trial_data:
        LOGGER.warning("No successful trials found.")
        return

    # 1. Per-trial YAML configs
    for td in trial_data:
        trial_dir = td["trial_dir"]
        if trial_dir.exists():
            yaml_path = trial_dir / f"{tracking_method}_{td['trial_id']}.yaml"
            _write_trial_yaml(yaml_cfg, td["config"], yaml_path)

    # 2. results.csv
    csv_path = tune_dir / "results.csv"
    _save_results_csv(csv_path, trial_data)
    LOGGER.opt(colors=True).info(f"<bold>Results CSV:</bold> <cyan>{csv_path}</cyan>")

    # 3. Best config → tune root
    primary = maximize[0]
    best = max(trial_data, key=lambda t: t["metrics"].get(primary, 0))
    best_yaml_path = tune_dir / f"best_{tracking_method}.yaml"
    _write_trial_yaml(yaml_cfg, best["config"], best_yaml_path)
    LOGGER.opt(colors=True).info(
        f"<bold>Best config ({best['trial_id']}):</bold> <cyan>{best_yaml_path}</cyan>"
    )

    # 4. summary.md
    _generate_summary(tune_dir, trial_data, yaml_cfg, tracking_method, maximize, minimize, args)


# ---------------------------------------------------------------------------
# Per-trial callback: saves YAML config after each trial completes
# ---------------------------------------------------------------------------

try:
    from ray.tune import Callback as _RayCallback

    class TrialSaveCallback(_RayCallback):
        """
        Saves a ready-to-use YAML config (matching the original search-space format
        but with ``default`` updated to this trial's values) into the trial directory
        immediately after each trial completes.
        """

        def __init__(self, yaml_cfg: dict, tracking_method: str):
            self._yaml_cfg = yaml_cfg
            self._tracking_method = tracking_method

        def on_trial_complete(self, iteration, trials, trial, **info):
            trial_dir = Path(
                getattr(trial, "local_path", None) or getattr(trial, "logdir", "")
            )
            if trial_dir and trial_dir.exists():
                yaml_path = trial_dir / f"{self._tracking_method}_{trial.trial_id}.yaml"
                _write_trial_yaml(self._yaml_cfg, trial.config, yaml_path)

except ImportError:
    TrialSaveCallback = None  # ray not installed; callback won't be used


# ---------------------------------------------------------------------------
# Tracker (objective function)
# ---------------------------------------------------------------------------

class Tracker:
    def __init__(self, opt):
        self.opt = opt

    def objective_function(self, config: dict) -> dict:
        run_generate_mot_results(self.opt, config, quiet=True)
        results = run_trackeval(self.opt)

        if not results:
            return {k: 0.0 for k in ALL_TUNE_METRICS}

        if isinstance(results, dict) and "per_sequence" in results:
            results = {k: v for k, v in results.items() if k != "per_sequence"}

        return _aggregate_results(results)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args):
    from boxmot.utils.checks import RequirementsChecker
    checker = RequirementsChecker()
    checker.sync_extra(extra="evolve")

    from ray import tune
    from ray.tune import RunConfig
    from ray.tune.search.optuna import OptunaSearch

    args.yolo_model = [Path(y).resolve() for y in args.yolo_model]
    args.reid_model = [Path(r).resolve() for r in args.reid_model]

    # Resolve optimize targets
    maximize = list(args.maximize) if args.maximize else [args.objectives[0]]
    minimize = list(args.minimize)
    opt_metrics = maximize + minimize
    opt_modes   = ["max"] * len(maximize) + ["min"] * len(minimize)

    if len(opt_metrics) == 1:
        optuna_search = OptunaSearch(metric=opt_metrics[0], mode=opt_modes[0])
    else:
        optuna_search = OptunaSearch(metric=opt_metrics, mode=opt_modes)

    yaml_cfg = load_yaml_config(args.tracking_method)
    search_space = yaml_to_search_space(yaml_cfg, tune)
    tracker = Tracker(args)

    def tune_wrapper(cfg):
        return tracker.objective_function(cfg)

    tune_name = f"{args.tracking_method}_tune"
    results_dir = args.project / "ray"
    restore_path = results_dir / tune_name

    n_threads = int(args.n_threads)
    trainable = tune.with_resources(tune_wrapper, {"cpu": n_threads, "gpu": 0})

    LOGGER.opt(colors=True).info("<cyan>[1/3]</cyan> Setting up evaluation environment...")
    eval_setup(args)

    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "=" * 60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>BoxMOT Hyperparameter Tuning</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "=" * 60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>    <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>   <cyan>{args.yolo_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>       <cyan>{args.reid_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Trials:</bold>     <cyan>{args.n_trials}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Maximize:</bold>   <cyan>{', '.join(maximize)}</cyan>")
    if minimize:
        LOGGER.opt(colors=True).info(f"<bold>Minimize:</bold>   <cyan>{', '.join(minimize)}</cyan>")
        LOGGER.opt(colors=True).info(f"<bold>Mode:</bold>       <cyan>Pareto (multi-objective)</cyan>")
    else:
        LOGGER.opt(colors=True).info(f"<bold>Mode:</bold>       <cyan>Single-objective</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "=" * 60 + "</blue>")

    LOGGER.opt(colors=True).info("<cyan>[2/3]</cyan> Generating detections and embeddings...")
    run_generate_dets_embs(args)

    LOGGER.opt(colors=True).info("<cyan>[3/3]</cyan> Running hyperparameter optimization...")
    if tune.Tuner.can_restore(restore_path):
        LOGGER.opt(colors=True).info(f"<bold>Resuming tuning from:</bold> <cyan>{restore_path}</cyan>")
        tuner = tune.Tuner.restore(
            str(restore_path),
            trainable=trainable,
            resume_errored=True,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=args.n_trials,
                search_alg=optuna_search,
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
            run_config=RunConfig(
                storage_path=results_dir,
                name=tune_name,
                callbacks=[TrialSaveCallback(yaml_cfg, args.tracking_method)],
            ),
        )

    tuner.fit()

    # Post-processing: per-trial configs, CSV, best config, summary
    tune_dir = results_dir / tune_name
    _save_all_results(tune_dir, tuner.get_results(), yaml_cfg,
                      args.tracking_method, maximize, minimize, args)


if __name__ == "__main__":
    main()
