#!/usr/bin/env python3
from __future__ import annotations

"""
Hyperparameter tuning orchestration for multi-object trackers.

Uses Ray Tune with pluggable search backends (Optuna, HyperOpt, random).
"""

import inspect
import logging
import os
import warnings
from difflib import get_close_matches
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"


def _configure_ray_environment() -> None:
    """Set stable Ray runtime defaults before importing or initializing Ray."""
    os.environ.setdefault("RAY_CHDIR_TO_TRIAL_DIR", "0")
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

from boxmot.engine.tuning.backends import (  # noqa: F401
    SEARCH_BACKENDS,
    build_search_backend,
    resolve_search_backend,
)
from boxmot.engine.tuning.backends import (  # noqa: F401
    resolve_search_backend as _resolve_search_backend,
)
from boxmot.engine.tuning.backends.hyperopt_backend import (  # noqa: F401
    _hyperopt_param,
    yaml_to_hyperopt_space,
)
from boxmot.engine.tuning.backends.optuna_backend import (  # noqa: F401
    _OptunaDefineSpace,
    _suggest_param,
    yaml_to_optuna_define_space,
)
from boxmot.engine.tuning.postprocessing import (  # noqa: F401
    ALL_TUNE_METRICS,
    MAXIMIZE_TUNE_METRICS,
    METRIC_SUM,
    MINIMIZE_TUNE_METRICS,
    aggregate_results,
    best_trial_data,
    collect_trial_data,
    generate_summary,
    save_all_results,
    score_summary,
)
from boxmot.engine.tuning.postprocessing import (  # noqa: F401
    aggregate_results as _aggregate_results,
)
from boxmot.engine.tuning.postprocessing import (  # noqa: F401
    best_trial_data as _best_trial_data,
)
from boxmot.engine.tuning.postprocessing import (  # noqa: F401
    collect_trial_data as _collect_trial_data,
)
from boxmot.engine.tuning.postprocessing import (  # noqa: F401
    find_pareto_front as _find_pareto_front,
)
from boxmot.engine.tuning.postprocessing import (
    save_all_results as _save_all_results,
)
from boxmot.engine.tuning.postprocessing import (  # noqa: F401
    save_results_csv as _save_results_csv,
)
from boxmot.engine.tuning.postprocessing import (  # noqa: F401
    write_trial_yaml as _write_trial_yaml,
)

# Re-exports for backward compatibility
from boxmot.engine.tuning.search_space import (  # noqa: F401
    conditional_yaml_tree as _conditional_yaml_tree,
)
from boxmot.engine.tuning.search_space import (
    default_tune_config,
    load_yaml_config,
    normalize_trial_config,
)
from boxmot.engine.tuning.search_space import (  # noqa: F401
    default_tune_config as _default_tune_config,
)
from boxmot.engine.tuning.search_space import (  # noqa: F401
    flatten_yaml_config as _flatten_yaml_config,
)
from boxmot.engine.tuning.search_space import (  # noqa: F401
    is_valid_search_param as _is_valid_search_param,
)
from boxmot.engine.tuning.search_space import (  # noqa: F401
    normalize_trial_config as _normalize_trial_config,
)
from boxmot.engine.tuning.search_space import (  # noqa: F401
    to_builtin_value as _to_builtin_value,
)
from boxmot.engine.tuning.search_space import (  # noqa: F401
    unpack_nested_dict as _unpack_nested_dict,
)
from boxmot.engine.tuning.search_space import (  # noqa: F401
    yaml_to_tune_space as yaml_to_search_space,
)
from boxmot.engine.workflows.reporting import (
    CLI_TUNE_BEST_SUMMARY_TITLE,
    SUMMARY_COLUMNS,
)
from boxmot.engine.workflows.results import TuneResult, TuneTrialResult, ValidationResult
from boxmot.utils import logger as LOGGER
from boxmot.utils.misc import suppress_boxmot_logs
from boxmot.utils.rich.reporters.tune import (
    TuneSilentReporter,
    TuneWorkflowCallback,
    TuneWorkflowReporter,
    build_tune_artifacts_renderable,
    build_tune_workflow_fields,
    combine_tune_result_renderables,
    format_initial_tune_progress,
    format_tune_progress,
    set_tune_progress_workflow,
)

_TUNE_WARNING_FILTER = "ignore:resource_tracker:UserWarning"
_TUNE_METRIC_ALIASES = {
    "hota": "HOTA",
    "mota": "MOTA",
    "idf1": "IDF1",
    "assa": "AssA",
    "assre": "AssRe",
    "idsw": "IDSW",
    "id_switch": "IDSW",
    "id_switches": "IDSW",
    "idswitch": "IDSW",
    "idswitches": "IDSW",
    "ids": "IDs",
    "idsw_rate": "IDSW_rate",
    "id_switch_rate": "IDSW_rate",
    "id_switches_rate": "IDSW_rate",
    "idswitch_rate": "IDSW_rate",
    "idswitches_rate": "IDSW_rate",
}


def eval_setup(*args: Any, **kwargs: Any) -> Any:
    """Lazily import evaluator setup when tuning actually starts."""
    from boxmot.engine.eval.evaluator import eval_setup as _eval_setup

    return _eval_setup(*args, **kwargs)


def run_generate_dets_embs(*args: Any, **kwargs: Any) -> Any:
    """Lazily import cache generation when tuning actually starts."""
    from boxmot.engine.eval.evaluator import run_generate_dets_embs as _run_generate_dets_embs

    return _run_generate_dets_embs(*args, **kwargs)


def run_eval(*args: Any, **kwargs: Any) -> Any:
    """Lazily import evaluation inside Ray trial execution."""
    from boxmot.engine.eval.evaluator import run_eval as _run_eval

    return _run_eval(*args, **kwargs)


# ---------------------------------------------------------------------------
# Metric validation helpers
# ---------------------------------------------------------------------------

def _normalize_metric_names(values: Any) -> list[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    metrics: list[str] = []
    for value in raw_values:
        for part in str(value).split(","):
            metric = part.strip()
            if metric:
                metrics.append(_TUNE_METRIC_ALIASES.get(metric.lower(), metric))
    return metrics


def _validate_tune_metrics(option_name: str, metrics: list[str], allowed_metrics: tuple[str, ...]) -> None:
    invalid = [m for m in metrics if m not in allowed_metrics]
    if not invalid:
        return
    suggestions = ", ".join(
        f"{m}{_suggest(m, allowed_metrics)}" for m in invalid
    )
    raise click.UsageError(
        f"Invalid value for {option_name}: {suggestions}\n"
        f"Available maximize metrics: {', '.join(MAXIMIZE_TUNE_METRICS)}\n"
        f"Available minimize metrics: {', '.join(MINIMIZE_TUNE_METRICS)}"
    )


def _suggest(metric: str, allowed: tuple[str, ...]) -> str:
    prefix = metric.lower().rstrip("s")
    matches = [c for c in allowed if c.lower().startswith(prefix)]
    suggestions = matches[:2] or get_close_matches(metric, allowed, n=2, cutoff=0.5)
    return f" (did you mean {', '.join(suggestions)}?)" if suggestions else ""


# ---------------------------------------------------------------------------
# Tuner class
# ---------------------------------------------------------------------------

class Tuner:
    """Orchestrates hyperparameter tuning via Ray Tune.

    Usage::

        tuner = Tuner(args)
        result_grid, tune_dir, maximize, minimize = tuner.fit()
    """

    def __init__(self, args, *, baseline_config: dict | None = None):
        self.args = args
        self.baseline_config = baseline_config
        self._yaml_cfg: dict | None = None
        self._maximize: list[str] = []
        self._minimize: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """Run the full tuning pipeline. Returns (result_grid, tune_dir, maximize, minimize)."""
        self._resolve_metrics()
        return self._run()

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _resolve_metrics(self):
        args = self.args
        objectives = _normalize_metric_names(getattr(args, "objectives", ()))
        self._maximize = _normalize_metric_names(getattr(args, "maximize", ())) or [
            objectives[0] if objectives else "HOTA"
        ]
        self._minimize = _normalize_metric_names(getattr(args, "minimize", ()))

        _validate_tune_metrics("--objectives", objectives, ALL_TUNE_METRICS)
        _validate_tune_metrics("--maximize", self._maximize, MAXIMIZE_TUNE_METRICS)
        _validate_tune_metrics("--minimize", self._minimize, MINIMIZE_TUNE_METRICS)

        args.objectives = tuple(objectives)
        args.maximize = tuple(self._maximize)
        args.minimize = tuple(self._minimize)

    def _setup_ray(self):
        _configure_ray_environment()
        import ray

        if ray.is_initialized():
            return

        verbose = bool(getattr(self.args, "verbose", False))
        init_kwargs: dict[str, Any] = {
            "include_dashboard": False,
            "configure_logging": True,
        }
        if not verbose:
            init_kwargs["logging_level"] = logging.ERROR
            init_kwargs["log_to_driver"] = False
        else:
            init_kwargs["logging_level"] = logging.WARNING
        ray.init(**init_kwargs)

    def _run(self):
        args = self.args
        maximize, minimize = self._maximize, self._minimize

        args.detector = [Path(y).resolve() for y in args.detector]
        args.reid = [Path(r).resolve() for r in args.reid]
        args.show_progress = False

        # Load tracker config and build search
        self._yaml_cfg = load_yaml_config(args.tracker)
        yaml_cfg = self._yaml_cfg
        search_backend = resolve_search_backend(args)
        setattr(args, "search_alg", search_backend)

        baseline = self.baseline_config
        if baseline is None:
            baseline = default_tune_config(yaml_cfg) or None
        else:
            baseline = normalize_trial_config(baseline)

        max_concurrent = int(getattr(args, "max_concurrent_trials", 0)) or None
        if max_concurrent is None:
            max_concurrent = min(4, os.cpu_count() or 4)

        opt_metrics = maximize + minimize
        opt_modes = ["max"] * len(maximize) + ["min"] * len(minimize)

        # Pipeline and callback
        pipeline = TuneWorkflowReporter(args, maximize=maximize, minimize=minimize).pipeline(auto_start=False)
        tune_callback = TuneWorkflowCallback(total=int(args.n_trials), maximize=maximize, minimize=minimize)
        set_tune_progress_workflow(pipeline.workflow)

        try:
            with pipeline:
                pipeline.update("Initializing tuning runtime...")
                pipeline.start()

                self._configure_warning_filters()
                _sync_tuning_requirements(verbose=bool(getattr(args, "verbose", False)))

                pipeline.update("Initializing Ray...")
                self._setup_ray()

                pipeline.update("Loading Ray Tune...")
                from ray import tune
                from ray.tune import RunConfig

                pipeline.update("Building search space...")
                search_alg, param_space = build_search_backend(
                    backend=search_backend,
                    yaml_cfg=yaml_cfg,
                    tune=tune,
                    opt_metrics=opt_metrics,
                    opt_modes=opt_modes,
                    baseline_config=baseline,
                    seed=getattr(args, "seed", None),
                    max_concurrent=max_concurrent,
                )

                pipeline.update("Preparing evaluation setup...")
                with suppress_boxmot_logs(enabled=not bool(getattr(args, "verbose", False)), level="ERROR"):
                    eval_setup(args, pipeline=pipeline)

                pipeline.refresh_fields(build_tune_workflow_fields(args, maximize=maximize, minimize=minimize))

                tune_dir = self._resolve_tune_dir()
                tune_name = tune_dir.name
                resume_tune = getattr(args, "resume_tune", None) or None

                ray_dir = tune_dir.parent
                if ray_dir.name != "ray" and ray_dir.parent.name == "ray":
                    ray_dir = ray_dir.parent
                if resume_tune and ray_dir.name == "ray":
                    inferred_project = ray_dir.parent
                    if inferred_project != Path(args.project).resolve():
                        args.project = str(inferred_project)

                pipeline.advance("Preparing benchmark cache...")

                with suppress_boxmot_logs(enabled=not bool(getattr(args, "verbose", False)), level="ERROR"):
                    try:
                        run_generate_dets_embs(args)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to prepare detection/embedding cache: {exc}"
                        ) from exc

                # KF calibration (once, before trials start)
                if getattr(args, "tune_kf", False) and not getattr(args, "kf_tuning", None):
                    from boxmot.motion.kalman_filters.calibration import run_kf_tuning, tracker_kf_type

                    pipeline.advance("Calibrating Kalman filter noise...")
                    kf_type = tracker_kf_type(str(getattr(args, "tracker", "")))
                    if kf_type:
                        kf_result, kf_log = run_kf_tuning(args, kf_type, capture=True)
                        if kf_result is not None:
                            kf_result["kf_type"] = kf_type
                            args.kf_tuning = kf_result
                            pipeline.update(
                                "KF tuning applied "
                                f"({kf_type}): "
                                f"std_pos={kf_result['std_weight_position']:.6f}, "
                                f"std_vel={kf_result['std_weight_velocity']:.6f}"
                            )
                        elif kf_log:
                            pipeline.update(f"KF tuning skipped or failed ({kf_type}).")
                        else:
                            pipeline.update(f"KF tuning skipped ({kf_type}).")
                    else:
                        pipeline.update(f"KF tuning skipped: tracker '{args.tracker}' has no registered KF type.")

                elif getattr(args, "tune_kf", False):
                    pipeline.advance("Kalman filter tuning already available.")
                    kf_result = getattr(args, "kf_tuning", None) or {}
                    kf_type = kf_result.get("kf_type", "unknown")
                    pipeline.update(
                        "KF tuning already applied "
                        f"({kf_type}): "
                        f"std_pos={kf_result.get('std_weight_position', 'n/a')}, "
                        f"std_vel={kf_result.get('std_weight_velocity', 'n/a')}"
                    )

                pipeline.advance(format_initial_tune_progress(int(args.n_trials)))

                objective = TrackerObjective(self._make_safe_namespace())

                def tune_wrapper(cfg):
                    return objective(normalize_trial_config(cfg))

                n_threads = int(args.n_threads)
                trainable = tune.with_resources(tune_wrapper, {"cpu": n_threads, "gpu": 0})

                # Build or restore the Ray Tuner
                tuner = self._build_or_restore_tuner(
                    trainable, tune, RunConfig, tune_callback, pipeline,
                    param_space, search_alg, tune_dir, tune_name, max_concurrent,
                )

                # Execute
                result_grid, interrupted = self._execute_tuner(tuner)

                # Post-process
                saved_artifacts = self._post_process(result_grid, tune_dir, yaml_cfg, maximize, minimize)

                # Final UI
                self._finalize_ui(pipeline, saved_artifacts, baseline, maximize, minimize, tune_dir, interrupted)

                return result_grid, tune_dir, maximize, minimize
        finally:
            set_tune_progress_workflow(None)

    def _build_or_restore_tuner(
        self, trainable, tune, RunConfig, tune_callback, pipeline,
        param_space, search_alg, tune_dir, tune_name, max_concurrent,
    ):
        args = self.args
        resume_tune = getattr(args, "resume_tune", None) or None
        restore_path_str = str(tune_dir)
        results_dir_str = str(tune_dir.parent)

        if resume_tune is not None and tune.Tuner.can_restore(restore_path_str):
            try:
                tuner = tune.Tuner.restore(restore_path_str, trainable=trainable, resume_errored=True)
                self._inject_callback_into_restored(tuner, tune_callback, pipeline, tune_dir)
                return tuner
            except Exception as exc:
                LOGGER.warning(f"Failed to restore tuner: {exc}. Starting fresh.")

        # Fresh tuner
        from ray.tune import CheckpointConfig, FailureConfig

        run_config_kwargs: dict[str, Any] = {"storage_path": results_dir_str, "name": tune_name}
        sig = inspect.signature(RunConfig)
        if "callbacks" in sig.parameters:
            run_config_kwargs["callbacks"] = [tune_callback]
        if "verbose" in sig.parameters:
            run_config_kwargs["verbose"] = 0
        if "progress_reporter" in sig.parameters:
            run_config_kwargs["progress_reporter"] = TuneSilentReporter()
        run_config_kwargs["failure_config"] = FailureConfig(max_failures=3)
        run_config_kwargs["checkpoint_config"] = CheckpointConfig(num_to_keep=1)

        tune_config_kwargs: dict[str, Any] = {
            "num_samples": args.n_trials,
            "max_concurrent_trials": max_concurrent,
            "trial_dirname_creator": lambda trial: f"trial_{trial.trial_id}",
        }
        if search_alg is not None:
            tune_config_kwargs["search_alg"] = search_alg
        time_budget = getattr(args, "time_budget_s", None)
        if time_budget is not None:
            tune_config_kwargs["time_budget_s"] = float(time_budget)

        return tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(**tune_config_kwargs),
            run_config=RunConfig(**run_config_kwargs),
        )

    def _inject_callback_into_restored(self, tuner, tune_callback, pipeline, tune_dir):
        completed = 0
        try:
            from ray.tune import ExperimentAnalysis
            df = ExperimentAnalysis(str(tune_dir)).dataframe()
            completed = len(df)
        except Exception:
            pass
        if completed == 0:
            try:
                completed = sum(
                    1 for d in tune_dir.iterdir()
                    if d.is_dir() and d.name.startswith("trial_") and (d / "result.json").exists()
                )
            except Exception:
                pass
        tune_callback.completed = completed
        tune_callback._trial_index_offset = completed
        tuner._local_tuner._run_config.callbacks = [tune_callback]
        tuner._local_tuner._run_config.verbose = 0
        tuner._local_tuner._run_config.progress_reporter = TuneSilentReporter()
        pipeline.advance(
            format_tune_progress(completed, int(self.args.n_trials), current_trial=completed + 1)
        )

    def _execute_tuner(self, tuner):
        result_grid = None
        interrupted = False
        try:
            result_grid = tuner.fit()
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.info("Tuning interrupted by user. Saving partial results...")
            try:
                if hasattr(tuner, "get_results"):
                    result_grid = tuner.get_results()
            except Exception:
                pass
        except Exception as exc:
            LOGGER.warning(f"tuner.fit() failed: {type(exc).__name__}: {exc}")
            try:
                if hasattr(tuner, "get_results"):
                    result_grid = tuner.get_results()
            except Exception:
                pass

        if result_grid is None and hasattr(tuner, "get_results"):
            try:
                result_grid = tuner.get_results()
            except Exception:
                pass
        return result_grid, interrupted

    def _post_process(self, result_grid, tune_dir, yaml_cfg, maximize, minimize):
        try:
            return _save_all_results(
                tune_dir, result_grid, yaml_cfg, self.args.tracker,
                maximize, minimize, self.args, emit_logs=False,
            )
        except Exception as exc:
            LOGGER.warning(f"Failed to save tune results: {type(exc).__name__}: {exc}")
            return None

    def _finalize_ui(self, pipeline, saved_artifacts, baseline, maximize, minimize, tune_dir, interrupted):
        args = self.args
        final_renderable = None
        try:
            artifacts_renderable = build_tune_artifacts_renderable(saved_artifacts) if saved_artifacts else None
            baseline_raw = None
            compare_first = bool(getattr(args, "compare_to_first_trial", False))
            if (baseline is not None or compare_first) and saved_artifacts and saved_artifacts.get("trial_data"):
                baseline_raw = (saved_artifacts["trial_data"][0].get("validation") or {}).get("raw")
            if saved_artifacts and saved_artifacts.get("trial_data"):
                best = best_trial_data(saved_artifacts["trial_data"], maximize=maximize, minimize=minimize)
                if best is not None:
                    best_metrics = _validation_result_from_trial(best, args)
                    best_renderable = best_metrics.renderable(
                        title=CLI_TUNE_BEST_SUMMARY_TITLE,
                        compare_raw=baseline_raw,
                        compare_args=args if baseline_raw else None,
                    )
                    final_renderable = combine_tune_result_renderables(best_renderable, artifacts_renderable)
                else:
                    final_renderable = artifacts_renderable
            elif artifacts_renderable is not None:
                final_renderable = artifacts_renderable
        except Exception as exc:
            LOGGER.debug(f"Failed to build results renderable: {exc}")

        if interrupted:
            n_saved = len(saved_artifacts.get("trial_data", [])) if saved_artifacts else 0
            if final_renderable is not None:
                pipeline.finish(final_renderable, title="Interrupted — Partial Results")
            else:
                pipeline.complete_step()
                pipeline.update(f"Tuning interrupted after {n_saved} trial(s). Saved to {tune_dir}")
        elif final_renderable is not None:
            pipeline.finish(final_renderable, title="Results")
        else:
            pipeline.complete_step()
            pipeline.update("No successful trials were produced.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_tune_dir(self) -> Path:
        return _resolve_tune_dir(self.args)

    @classmethod
    def _ray_dataset_name(cls, args) -> str:
        for attr in ("benchmark_id", "dataset_id", "benchmark", "data"):
            value = getattr(args, attr, None)
            if value:
                return cls._path_slug(value, fallback="dataset")
        return "dataset"

    @staticmethod
    def _path_slug(value: Any, *, fallback: str) -> str:
        raw = str(value).strip()
        if not raw:
            return fallback
        path = Path(raw)
        if path.suffix.lower() in {".yaml", ".yml"} or path.parent != Path("."):
            raw = path.stem
        slug = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
        return slug.strip("._-") or fallback

    def _make_safe_namespace(self) -> SimpleNamespace:
        try:
            from ray import cloudpickle
        except Exception:
            try:
                import cloudpickle  # noqa: F811
            except Exception:
                return SimpleNamespace(**vars(self.args))

        safe: dict[str, Any] = {}
        for key, value in vars(self.args).items():
            try:
                cloudpickle.dumps(value)
                safe[key] = value
            except Exception:
                pass
        return SimpleNamespace(**safe)

    @staticmethod
    def _configure_warning_filters():
        existing = os.environ.get("PYTHONWARNINGS", "")
        if _TUNE_WARNING_FILTER not in existing.split(","):
            os.environ["PYTHONWARNINGS"] = ",".join(filter(None, [existing, _TUNE_WARNING_FILTER]))
        os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "0")
        warnings.filterwarnings("ignore", message=r"Tip: In future versions of Ray.*", category=FutureWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"The distribution is specified by.*",
            category=UserWarning,
            module=r"optuna\.distributions",
        )

        try:
            import optuna.logging as optuna_logging
            optuna_logging.set_verbosity(optuna_logging.WARNING)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tracker objective (called inside each Ray trial)
# ---------------------------------------------------------------------------

class TrackerObjective:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, config: dict) -> dict:
        try:
            with suppress_boxmot_logs(enabled=not bool(getattr(self.opt, "verbose", False)), level="ERROR"):
                result = run_eval(
                    self.opt,
                    evolve_config=config,
                    setup=False,
                    prepare_cache=False,
                    verbose=False,
                    show_progress=False,
                )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            LOGGER.debug(f"Trial failed with {type(exc).__name__}: {exc}")
            return {k: 0.0 for k in ALL_TUNE_METRICS}

        if not result.raw:
            return {k: 0.0 for k in ALL_TUNE_METRICS}

        payload = aggregate_results(result.raw)
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
# Helpers
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


def _sync_tuning_requirements(*, verbose: bool) -> None:
    """Ensure tuning extras are available before running evolve/tune."""
    try:
        from boxmot.utils.checks import RequirementsChecker

        RequirementsChecker().sync_extra("evolve", verbose=verbose)
    except Exception as exc:
        LOGGER.debug(f"Could not sync evolve requirements: {exc}")


def _is_ray_pickle_safe(value: Any) -> bool:
    """Return True if *value* is serializable with Ray's cloudpickle."""
    try:
        from ray import cloudpickle

        cloudpickle.dumps(value)
        return True
    except Exception:
        try:
            import cloudpickle  # type: ignore

            cloudpickle.dumps(value)
            return True
        except Exception:
            return False


def _resolve_tune_dir(args, resume: bool = False) -> Path:
    """Backward-compatible module helper to resolve the tune directory."""
    explicit_resume = getattr(args, "resume_tune", None)
    resume_value = explicit_resume if explicit_resume else (resume if isinstance(resume, (str, Path)) else None)

    results_dir = Path(args.project).resolve() / "ray"
    dataset_dir = results_dir / Tuner._ray_dataset_name(args)
    if resume_value:
        resume_path = Path(resume_value)
        if resume_path.is_absolute():
            return resume_path
        cwd_candidate = Path(resume_value).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        if len(resume_path.parts) > 1:
            return (results_dir / resume_path).resolve()
        return (dataset_dir / resume_path.name).resolve()

    tracker_name = Tuner._path_slug(getattr(args, "tracker", "tracker"), fallback="tracker")
    for index in range(1, 10000):
        tune_dir = dataset_dir / f"{tracker_name}_{index}"
        if not tune_dir.exists():
            return tune_dir.resolve()
    raise RuntimeError(f"Could not allocate tune directory under {dataset_dir}")


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
) -> Path:
    """Backward-compatible alias for summary.md generation."""
    return generate_summary(
        tune_dir,
        trial_data,
        yaml_cfg,
        tracker_name,
        maximize,
        minimize,
        args,
        emit_logs=emit_logs,
    )


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------

def run_tune(args, *, baseline_config: dict | None = None) -> TuneResult:
    """Run tuning and return a structured TuneResult."""
    tuner = Tuner(args, baseline_config=baseline_config)
    result_grid, tune_dir, maximize, minimize = tuner.fit()

    trial_data = collect_trial_data(result_grid)
    if not trial_data:
        raise RuntimeError("No successful tuning trials were produced.")

    trials: list[TuneTrialResult] = []
    best: TuneTrialResult | None = None
    for index, trial in enumerate(trial_data, start=1):
        metrics = _validation_result_from_trial(trial, args)
        score = score_summary(metrics.summary, maximize=maximize, minimize=minimize)
        trial_result = TuneTrialResult(index=index, config=dict(trial["config"]), metrics=metrics, score=score)
        trials.append(trial_result)
        if best is None or trial_result.score > best.score:
            best = trial_result

    if best is None:
        raise RuntimeError("No successful tuning trials were produced.")

    return TuneResult(
        benchmark=str(getattr(args, "benchmark", getattr(args, "data", ""))),
        tracker=str(args.tracker),
        trials=trials,
        best=best,
        best_config=dict(best.config),
        best_yaml=tune_dir / f"best_{args.tracker}.yaml",
    )


def main(args):
    """CLI entry point."""
    tuner = Tuner(args)
    tuner.fit()


if __name__ == "__main__":
    main()
