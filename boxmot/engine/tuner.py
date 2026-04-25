#!/usr/bin/env python3
from __future__ import annotations

"""
Hyperparameter tuning for multi-object trackers using Ray Tune + Optuna.

Supports single-objective (default) and multi-objective Pareto search:
  boxmot tune --tracker bytetrack --benchmark ...                        # maximize HOTA
  boxmot tune ... --maximize HOTA --minimize IDSW_rate                  # Pareto mode
  boxmot tune ... --maximize HOTA --maximize IDF1 --minimize IDSW_rate  # 3-objective Pareto
"""

import csv
import inspect
import logging
import os
import warnings

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"   # keep CWD constant for all trials

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml

from boxmot.engine.evaluator import eval_setup, run_eval, run_generate_dets_embs
from boxmot.utils.rich.tune_reporting import (
    TUNE_GENERATE_STEP,
    TUNE_OPTIMIZE_STEP,
    TUNE_SETUP_STEP,
    TuneSilentReporter,
    TuneWorkflowCallback,
    build_tune_artifacts_renderable,
    combine_tune_result_renderables,
    format_initial_tune_progress,
    log_tune_pipeline_intro,
    set_tune_progress_workflow,
)
from boxmot.engine.workflow_reporting import (
    CLI_TUNE_BEST_SUMMARY_TITLE,
    SUMMARY_COLUMNS,
)
from boxmot.engine.workflow_results import TuneResult, TuneTrialResult, ValidationResult
from boxmot.engine.workflow_support import score_summary, suppress_boxmot_logs
from boxmot.utils import TRACKER_CONFIGS
from boxmot.utils import logger as LOGGER
from boxmot.utils.misc import increment_path

# Metrics that must be summed across classes (not averaged), because they are counts
METRIC_SUM = frozenset({"IDSW", "IDs"})

# All metrics returned from each trial (SUMMARY_COLUMNS + derived)
ALL_TUNE_METRICS = (*SUMMARY_COLUMNS, "IDSW_rate")
_TUNE_WARNING_FILTER = "ignore:resource_tracker:UserWarning"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_yaml_config(tracker_name: str) -> dict:
    config_path = TRACKER_CONFIGS / f"{tracker_name}.yaml"
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
        validation = result.metrics.get("_validation", {})
        trial_data.append({
            "trial_id": trial_id,
            "trial_dir": Path(result.path),
            "config": result.config,
            "metrics": {k: result.metrics.get(k, 0.0) for k in ALL_TUNE_METRICS},
            "validation": validation if isinstance(validation, dict) else {},
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


def _best_trial_data(trial_data: list, *, maximize: list[str], minimize: list[str]) -> dict | None:
    best_trial: dict | None = None
    best_score: tuple[float, ...] | None = None
    for trial in trial_data:
        score = score_summary(trial["metrics"], maximize=maximize, minimize=minimize)
        if best_score is None or score > best_score:
            best_trial = trial
            best_score = score
    return best_trial


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
    tracker_name: str,
    maximize: list,
    minimize: list,
    args,
    *,
    emit_logs: bool = True,
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

    # --- Best Trial ---
    lines.append("---\n")
    lines.append(f"## Best Trial: `{best['trial_id']}`\n")
    metric_header = " | ".join(ALL_TUNE_METRICS)
    metric_sep = " | ".join("---" for _ in ALL_TUNE_METRICS)
    metric_vals = " | ".join(f"{best['metrics'].get(k, 0):.4f}" for k in ALL_TUNE_METRICS)
    lines.append(f"| {metric_header} |")
    lines.append(f"| {metric_sep} |")
    lines.append(f"| {metric_vals} |")
    lines.append(f"\nConfig saved to: `best_{tracker_name}.yaml`\n")

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

    summary_path = tune_dir / "summary.md"
    summary_path.write_text("\n".join(lines))
    if emit_logs:
        LOGGER.opt(colors=True).info(
            f"<bold>Summary saved to:</bold> <cyan>{summary_path}</cyan>"
        )
    return summary_path


def _save_all_results(
    tune_dir: Path,
    results,
    yaml_cfg: dict,
    tracker_name: str,
    maximize: list,
    minimize: list,
    args,
    *,
    emit_logs: bool = True,
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
        return None

    # 1. Per-trial YAML configs
    for td in trial_data:
        trial_dir = td["trial_dir"]
        if trial_dir.exists():
            yaml_path = trial_dir / f"{tracker_name}_{td['trial_id']}.yaml"
            _write_trial_yaml(yaml_cfg, td["config"], yaml_path)

    # 2. results.csv
    csv_path = tune_dir / "results.csv"
    _save_results_csv(csv_path, trial_data)
    if emit_logs:
        LOGGER.opt(colors=True).info(f"<bold>Results CSV:</bold> <cyan>{csv_path}</cyan>")

    # 3. Best config → tune root
    best = _best_trial_data(trial_data, maximize=maximize, minimize=minimize)
    if best is None:
        return None
    best_yaml_path = tune_dir / f"best_{tracker_name}.yaml"
    _write_trial_yaml(yaml_cfg, best["config"], best_yaml_path)
    if emit_logs:
        LOGGER.opt(colors=True).info(
            f"<bold>Best config ({best['trial_id']}):</bold> <cyan>{best_yaml_path}</cyan>"
        )

    # 4. summary.md
    summary_path = _generate_summary(
        tune_dir,
        trial_data,
        yaml_cfg,
        tracker_name,
        maximize,
        minimize,
        args,
        emit_logs=emit_logs,
    )
    return {
        "trial_data": trial_data,
        "csv_path": csv_path,
        "best_yaml_path": best_yaml_path,
        "summary_path": summary_path,
        "best_trial_id": best["trial_id"],
    }


# ---------------------------------------------------------------------------
# Tracker (objective function)
# ---------------------------------------------------------------------------

class TrackerObjective:
    def __init__(self, opt):
        self.opt = opt

    def objective_function(self, config: dict) -> dict:
        with suppress_boxmot_logs(enabled=not bool(getattr(self.opt, "verbose", False)), level="ERROR"):
            result = run_eval(
                self.opt,
                evolve_config=config,
                setup=False,
                prepare_cache=False,
                verbose=False,
                show_progress=False,
            )

        if not result.raw:
            return {k: 0.0 for k in ALL_TUNE_METRICS}

        payload = _aggregate_results(result.raw)
        payload["_validation"] = {
            "benchmark": result.benchmark,
            "raw": result.raw,
            "summary_label": result.summary_label,
            "summary": result.summary,
            "timings": result.timings,
            "exp_dir": None if result.exp_dir is None else str(result.exp_dir),
        }
        return payload


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _validation_result_from_trial(trial_data: dict, args) -> ValidationResult:
    validation_payload = trial_data.get("validation", {})
    raw = validation_payload.get("raw")
    summary = validation_payload.get("summary")
    if not isinstance(summary, dict):
        summary = {
            key: float(trial_data["metrics"].get(key, 0.0))
            for key in SUMMARY_COLUMNS
            if key in trial_data["metrics"]
        }
    return ValidationResult(
        benchmark=str(validation_payload.get("benchmark", getattr(args, "benchmark", getattr(args, "data", "")))),
        raw=raw if isinstance(raw, dict) else dict(summary),
        summary_label=str(validation_payload.get("summary_label", "all")),
        summary=dict(summary),
        exp_dir=Path(validation_payload["exp_dir"]) if validation_payload.get("exp_dir") else None,
        timings=dict(validation_payload.get("timings", {})),
        args=args,
    )


def _configure_tune_warning_filters() -> None:
    existing = os.environ.get("PYTHONWARNINGS", "")
    if _TUNE_WARNING_FILTER not in existing.split(","):
        os.environ["PYTHONWARNINGS"] = ",".join(filter(None, [existing, _TUNE_WARNING_FILTER]))
    os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")
    warnings.filterwarnings(
        "ignore",
        message=r"Tip: In future versions of Ray, Ray will no longer override accelerator visible devices env var.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The distribution is specified by .* and step=.*",
        category=UserWarning,
        module=r"optuna\.distributions",
    )


def _is_ray_pickle_safe(value: Any) -> bool:
    """Return True when Ray's cloudpickle can serialize ``value`` safely."""
    try:
        from ray import cloudpickle
    except Exception:
        try:
            import cloudpickle
        except Exception:
            return False

    try:
        cloudpickle.dumps(value)
    except Exception:
        return False
    return True


def _ray_safe_namespace(args: Any) -> SimpleNamespace:
    """Return an args namespace containing only fields Ray can serialize."""
    safe_values: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in vars(args).items():
        if _is_ray_pickle_safe(value):
            safe_values[key] = value
        else:
            skipped.append(key)

    if skipped:
        LOGGER.debug(
            "Skipping non-serializable tuning args before Ray trial dispatch: "
            + ", ".join(sorted(skipped))
        )

    return SimpleNamespace(**safe_values)


def _resolve_tune_dir(args, *, resume: bool = False) -> Path:
    results_dir = Path(args.project).resolve() / "ray"
    base_dir = results_dir / f"{args.tracker}_tune"
    if resume:
        return base_dir
    return increment_path(base_dir, sep="_", exist_ok=False)


def _ensure_ray_initialized(*, verbose: bool) -> None:
    import ray

    if ray.is_initialized():
        return

    init_kwargs: dict[str, Any] = {
        "include_dashboard": False,
    }
    if not verbose:
        init_kwargs["logging_level"] = logging.ERROR
        init_kwargs["log_to_driver"] = False

    ray.init(**init_kwargs)


def _execute_tune_search(
    args,
    *,
    baseline_config: dict | None = None,
):
    import ray
    from boxmot.utils.checks import RequirementsChecker
    checker = RequirementsChecker()
    checker.sync_extra(extra="evolve", verbose=bool(getattr(args, "verbose", False)))

    from ray import tune
    from ray.tune import RunConfig
    from ray.tune.search.optuna import OptunaSearch

    _configure_tune_warning_filters()
    try:
        import optuna.logging as optuna_logging

        optuna_logging.set_verbosity(optuna_logging.INFO if bool(getattr(args, "verbose", False)) else optuna_logging.WARNING)
    except Exception:
        pass

    args.detector = [Path(y).resolve() for y in args.detector]
    args.reid = [Path(r).resolve() for r in args.reid]
    args.show_progress = False
    resume_tune = bool(getattr(args, "resume_tune", False))
    _ensure_ray_initialized(verbose=bool(getattr(args, "verbose", False)))

    # Resolve optimize targets
    maximize = list(args.maximize) if args.maximize else [args.objectives[0]]
    minimize = list(args.minimize)
    opt_metrics = maximize + minimize
    opt_modes   = ["max"] * len(maximize) + ["min"] * len(minimize)
    optuna_kwargs = {
        "metric": opt_metrics[0] if len(opt_metrics) == 1 else opt_metrics,
        "mode": opt_modes[0] if len(opt_modes) == 1 else opt_modes,
    }
    seed = getattr(args, "seed", None)
    if seed is not None:
        optuna_kwargs["seed"] = seed
    if baseline_config:
        optuna_kwargs["points_to_evaluate"] = [baseline_config]

    optuna_search = OptunaSearch(**optuna_kwargs)

    yaml_cfg = load_yaml_config(args.tracker)
    search_space = yaml_to_search_space(yaml_cfg, tune)

    workflow = log_tune_pipeline_intro(args, maximize=maximize, minimize=minimize)

    n_threads = int(args.n_threads)
    setup_status_message: str | None = None

    class _TuneSetupStatus:
        def set_detail(self, _label: str, message: str, *, render: bool = True) -> None:
            del render
            nonlocal setup_status_message
            setup_status_message = str(message)

    tune_callback = TuneWorkflowCallback(total=int(args.n_trials), maximize=maximize, minimize=minimize)

    with suppress_boxmot_logs(enabled=not bool(getattr(args, "verbose", False)), level="ERROR"):
        eval_setup(args, workflow=_TuneSetupStatus())

    tune_dir = _resolve_tune_dir(args, resume=resume_tune)
    tune_name = tune_dir.name

    workflow.complete(TUNE_SETUP_STEP, render=False)
    workflow.activate(TUNE_GENERATE_STEP, render=False)
    workflow.set_detail(
        TUNE_GENERATE_STEP,
        setup_status_message or "Preparing benchmark cache...",
        render=False,
    )
    workflow.start()

    results_dir = tune_dir.parent
    restore_path = tune_dir
    restore_path_str = str(restore_path)
    results_dir_str = str(results_dir)

    with suppress_boxmot_logs(enabled=not bool(getattr(args, "verbose", False)), level="ERROR"):
        run_generate_dets_embs(args)
    workflow.complete(TUNE_GENERATE_STEP, render=False)
    workflow.activate(TUNE_OPTIMIZE_STEP, render=False)
    workflow.set_detail(
        TUNE_OPTIMIZE_STEP,
        format_initial_tune_progress(int(args.n_trials)),
        render=True,
    )

    tracker_objective = TrackerObjective(_ray_safe_namespace(args))

    def tune_wrapper(cfg):
        return tracker_objective.objective_function(cfg)

    trainable = tune.with_resources(tune_wrapper, {"cpu": n_threads, "gpu": 0})

    if resume_tune and tune.Tuner.can_restore(restore_path_str):
        tuner = tune.Tuner.restore(
            restore_path_str,
            trainable=trainable,
            resume_errored=True,
        )
    else:
        run_config_kwargs = {
            "storage_path": results_dir_str,
            "name": tune_name,
        }
        run_config_signature = inspect.signature(RunConfig)
        if "callbacks" in run_config_signature.parameters:
            run_config_kwargs["callbacks"] = [tune_callback]
        if "verbose" in run_config_signature.parameters:
            run_config_kwargs["verbose"] = 0
        if "progress_reporter" in run_config_signature.parameters:
            run_config_kwargs["progress_reporter"] = TuneSilentReporter()
        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                num_samples=args.n_trials,
                search_alg=optuna_search,
                trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
            ),
            run_config=RunConfig(**run_config_kwargs),
        )

    try:
        set_tune_progress_workflow(workflow)
        result_grid = tuner.fit()
        if result_grid is None and hasattr(tuner, "get_results"):
            result_grid = tuner.get_results()

        saved_artifacts = _save_all_results(
            tune_dir,
            result_grid,
            yaml_cfg,
            args.tracker,
            maximize,
            minimize,
            args,
            emit_logs=False,
        )

        workflow.complete(TUNE_OPTIMIZE_STEP, render=False)
        artifacts_renderable = build_tune_artifacts_renderable(saved_artifacts) if saved_artifacts else None
        baseline_raw = None
        compare_to_first_trial = bool(getattr(args, "compare_to_first_trial", False))
        if (baseline_config is not None or compare_to_first_trial) and saved_artifacts and saved_artifacts.get("trial_data"):
            baseline_raw = (saved_artifacts["trial_data"][0].get("validation") or {}).get("raw")
        if saved_artifacts and saved_artifacts.get("trial_data"):
            best_trial = _best_trial_data(saved_artifacts["trial_data"], maximize=maximize, minimize=minimize)
            if best_trial is not None:
                best_metrics = _validation_result_from_trial(best_trial, args)
                best_renderable = best_metrics.renderable(
                    title=CLI_TUNE_BEST_SUMMARY_TITLE,
                    compare_raw=baseline_raw,
                    compare_args=args if baseline_raw else None,
                )
                workflow.set_detail_renderable(
                    "Results",
                    combine_tune_result_renderables(best_renderable, artifacts_renderable),
                    render=False,
                )
            elif artifacts_renderable is not None:
                workflow.set_detail_renderable("Results", artifacts_renderable, render=False)
        elif artifacts_renderable is not None:
            workflow.set_detail_renderable("Results", artifacts_renderable, render=False)
        else:
            workflow.set_detail(TUNE_OPTIMIZE_STEP, "No successful trials were produced.", render=False)
        return result_grid, tune_dir, maximize, minimize
    finally:
        set_tune_progress_workflow(None)
        workflow.stop()


def run_tune(
    args,
    *,
    baseline_config: dict | None = None,
) -> TuneResult:
    result_grid, tune_dir, maximize, minimize = _execute_tune_search(
        args,
        baseline_config=baseline_config,
    )
    trial_data = _collect_trial_data(result_grid)
    if not trial_data:
        raise RuntimeError("No successful tuning trials were produced.")

    trials: list[TuneTrialResult] = []
    best: TuneTrialResult | None = None
    for index, trial in enumerate(trial_data, start=1):
        metrics = _validation_result_from_trial(trial, args)
        score = score_summary(metrics.summary, maximize=maximize, minimize=minimize)
        trial_result = TuneTrialResult(
            index=index,
            config=dict(trial["config"]),
            metrics=metrics,
            score=score,
        )
        trials.append(trial_result)
        if best is None or trial_result.score > best.score:
            best = trial_result

    if best is None:
        raise RuntimeError("No successful tuning trials were produced.")

    best_yaml = tune_dir / f"best_{args.tracker}.yaml"
    return TuneResult(
        benchmark=str(getattr(args, "benchmark", getattr(args, "data", ""))),
        tracker=str(args.tracker),
        trials=trials,
        best=best,
        best_config=dict(best.config),
        best_yaml=best_yaml,
    )


def main(args):
    _execute_tune_search(args)
    return None


if __name__ == "__main__":
    main()
