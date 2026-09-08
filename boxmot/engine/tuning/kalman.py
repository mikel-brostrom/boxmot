"""Bounded Kalman-only HOTA optimization over an immutable evaluation build.

This workflow fits covariance scales to the selected split's tracking metric.
It does not estimate detector noise from ground-truth motion, and its reported
scores are training scores until the saved configuration is evaluated elsewhere.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import time
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from boxmot.engine.eval.results import ValidationResult


def kalman_search_bounds(tracker_name: str) -> dict[str, tuple[float, float]]:
    """Use the same five noise dimensions as the tracker's YAML search space."""
    from boxmot.engine.tuning.search_space import flatten_yaml_config, load_yaml_config
    from boxmot.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS

    schema = flatten_yaml_config(load_yaml_config(tracker_name))
    bounds = {}
    for name in KALMAN_NOISE_OPTIONS:
        entry = schema[name]
        if entry.get("type") != "loguniform":
            raise ValueError(f"KF tuning requires a loguniform search range for {name}.")
        low, high = map(float, entry["range"])
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            raise ValueError(f"KF tuning requires finite positive ordered bounds for {name}.")
        bounds[name] = (low, high)
    return bounds


def validate_kf_tuning(tracker_name: str, backend: str = "python") -> None:
    """Reject unsupported targets or missing dependencies before materialization."""
    from boxmot.motion.kalman_filters.noise import KALMAN_TRACKER_NAMES

    if backend != "python" or tracker_name not in KALMAN_TRACKER_NAMES:
        raise ValueError("--kf-tuning requires a Python tracker with a Kalman filter.")
    if importlib.util.find_spec("optuna") is None:
        raise ValueError("--kf-tuning requires Optuna. Install the BoxMOT evolve extra: pip install 'boxmot[evolve]'.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Keep the last completed trial report readable if a run is interrupted."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class KalmanTuningResult:
    """Reusable configuration and transparent record of the fitting split."""

    config_path: Path
    report_path: Path
    baseline_hota: float
    best_hota: float

    @property
    def description(self) -> str:
        """Explain the score's scope alongside the final evaluation table."""
        return (
            f"KF tuning on this split: baseline HOTA {self.baseline_hota:.2f}, "
            f"selected {self.best_hota:.2f}. These are tuning scores; evaluate a held-out split separately.\n"
            f"Saved tracker configuration: {self.config_path}"
        )

    def record_final(self, result: ValidationResult) -> None:
        """Attach the independently replayed final summary to the trial report."""
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        report["final_summary"] = result.summary
        report["final_output_dir"] = str(result.exp_dir)
        _write_json(self.report_path, report)


def tune_kalman(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> KalmanTuningResult:
    """Select five noise scales by HOTA, retaining the baseline as a candidate.

    All association settings and the timing mode stay fixed. Trial zero uses
    the resolved runtime configuration; subsequent trials use seeded Optuna
    TPE. ``kf_trials`` counts the baseline and defaults to twenty replays.
    """
    from boxmot.engine.eval.evaluator import _ensure_setup, run_eval
    from boxmot.engine.tracker_config import resolve_tracker_options

    validate_kf_tuning(args.tracker, getattr(args, "tracker_backend", "python"))
    n_trials = getattr(args, "kf_trials", 20)
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("kf_trials must be a positive integer, including the baseline trial.")
    import optuna

    _ensure_setup(args)
    base_config = resolve_tracker_options(args, include_defaults=True, stamp_timing=True)
    search_bounds = kalman_search_bounds(args.tracker)
    base_scales = {key: float(base_config[key]) for key in search_bounds}
    base_config.update(base_scales)
    bounds = {
        key: (min(low, base_scales[key]), max(high, base_scales[key])) for key, (low, high) in search_bounds.items()
    }

    directory = Path(output_dir) / "kf-tuning"
    directory.mkdir(parents=True, exist_ok=True)
    report_path, config_path = directory / "trials.json", directory / "best.yaml"
    report: dict[str, Any] = {
        "version": 1,
        "status": "running",
        "objective": "HOTA",
        "direction": "maximize",
        "sampler": "Optuna TPE",
        "seed": 0,
        "score_scope": "tuned_on_selected_split",
        "tracker": args.tracker,
        "backend": getattr(args, "tracker_backend", "python"),
        "geometry": args.geometry,
        "per_class": bool(getattr(args, "per_class", False)),
        "class_ids": list(getattr(args, "tracker_class_ids", None) or ()),
        "class_names": dict(getattr(args, "tracker_class_names", ()) or ()),
        "dataset": args.dataset_id,
        "split": args.split,
        "sequences": list(args.sequence_names or args.seq_info),
        "build": str(args.build_path),
        "variable_dt": base_config["variable_dt"],
        "timing": {key: base_config[key] for key in ("variable_dt", "kf_time_unit", "kf_reference_dt_s")},
        "noise_model": "continuous_reference_density" if base_config["variable_dt"] else "discrete_per_frame",
        "baseline_config": base_config,
        "search_bounds": bounds,
        "requested_trials": n_trials,
        "trials": [],
    }
    _write_json(report_path, report)

    # Optuna's creation log is redundant with our workflow panel. ask/tell keeps
    # trial reporting under the same panel without starting a separate runtime.
    optuna_logger = logging.getLogger("optuna")
    previous_level = optuna_logger.level
    try:
        optuna_logger.setLevel(logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=5)
        )
    finally:
        optuna_logger.setLevel(previous_level)
    study.enqueue_trial(base_scales)
    best_hota = -math.inf
    baseline_hota = 0.0
    for number in range(n_trials):
        if progress is not None:
            best_detail = f"; best HOTA {best_hota:.2f}" if number else " (baseline)"
            progress(f"KF tuning: trial {number + 1}/{n_trials}{best_detail}. Fitting the selected split…")
        trial = study.ask()
        scales = {key: trial.suggest_float(key, low, high, log=True) for key, (low, high) in bounds.items()}
        config = {**base_config, **scales}
        trial_args = copy(args)
        trial_args.compare_trackeval = False
        trial_args.show = False
        trial_args.save = False
        started = time.perf_counter()
        result = run_eval(
            trial_args,
            evolve_config=config,
            setup=False,
            verbose=False,
            show_progress=False,
            output_dir=directory / "trials" / f"{number:04d}",
        )
        score = float(result.summary.get("HOTA", math.nan))
        if not math.isfinite(score):
            raise ValueError("KF tuning requires a finite HOTA score; check the selected split's ground truth.")
        study.tell(trial, score)
        if number == 0:
            baseline_hota = score
            report["baseline_hota"] = score
        report["trials"].append(
            {
                "number": number,
                "scales": scales,
                "summary": result.summary,
                "seconds": time.perf_counter() - started,
                "output_dir": str(result.exp_dir),
            }
        )
        if score > best_hota:
            best_hota = score
            report["best_trial"] = number
            report["best_hota"] = score
            temporary = config_path.with_suffix(".tmp")
            temporary.write_text(yaml.safe_dump({"tracker": args.tracker, **config}, sort_keys=False), encoding="utf-8")
            temporary.replace(config_path)
        _write_json(report_path, report)

    report["status"] = "complete"
    report["best_config"] = str(config_path)
    _write_json(report_path, report)
    if progress is not None:
        progress(f"KF tuning complete: HOTA {baseline_hota:.2f} → {best_hota:.2f}. Replaying selected configuration…")
    return KalmanTuningResult(config_path, report_path, baseline_hota, best_hota)
