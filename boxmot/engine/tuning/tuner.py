#!/usr/bin/env python3
from __future__ import annotations

"""
Hyperparameter tuning orchestration for multi-object trackers.

Uses Ray Tune with pluggable search backends (Optuna, HyperOpt, random).
"""

import inspect
import json
import logging
import os
import warnings
from copy import deepcopy
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


from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.eval.results import SUMMARY_COLUMNS, ValidationResult
from boxmot.engine.tuning.backends import build_search_backend, resolve_search_backend
from boxmot.engine.tuning.postprocessing import (
    ALL_TUNE_METRICS,
    MAXIMIZE_TUNE_METRICS,
    MINIMIZE_TUNE_METRICS,
    aggregate_results,
    best_trial_data,
    collect_trial_data,
    save_all_results,
    score_summary,
)
from boxmot.engine.tuning.results import TuneResult, TuneTrialResult
from boxmot.engine.tuning.search_space import (
    default_tune_config,
    flatten_yaml_config,
    load_yaml_config,
    normalize_trial_config,
)
from boxmot.engine.ui.logging import suppress_boxmot_logs
from boxmot.engine.ui.reporters.tune import (
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
from boxmot.engine.ui.reporters.validation import CLI_TUNE_BEST_SUMMARY_TITLE
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_TIMING_OPTIONS, KALMAN_TRACKER_NAMES
from boxmot.utils import logger as LOGGER

_TUNE_WARNING_FILTER = "ignore:resource_tracker:UserWarning"


def eval_setup(*args: Any, **kwargs: Any) -> Any:
    """Lazily import evaluator setup when tuning actually starts."""
    from boxmot.engine.eval.evaluator import eval_setup as _eval_setup

    return _eval_setup(*args, **kwargs)


def run_eval(*args: Any, **kwargs: Any) -> Any:
    """Lazily import evaluation inside Ray trial execution."""
    from boxmot.engine.eval.evaluator import run_eval as _run_eval

    return _run_eval(*args, **kwargs)


# ---------------------------------------------------------------------------
# Metric validation helpers
# ---------------------------------------------------------------------------


def _parse_metric_names(values: Any) -> list[str]:
    if values is None:
        return []
    raw_values = [values] if isinstance(values, str) else list(values)
    metrics: list[str] = []
    for value in raw_values:
        for part in str(value).split(","):
            metric = part.strip()
            if metric:
                metrics.append(metric)
    return metrics


def _validate_tune_metrics(option_name: str, metrics: list[str], allowed_metrics: tuple[str, ...]) -> None:
    invalid = [m for m in metrics if m not in allowed_metrics]
    if not invalid:
        return
    suggestions = ", ".join(f"{m}{_suggest(m, allowed_metrics)}" for m in invalid)
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
        self._calibrated_fixed_options: dict[str, Any] = {}
        self._calibration_config_path: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self):
        """Run the full tuning pipeline. Returns (result_grid, tune_dir, maximize, minimize)."""
        self._resolve_metrics()
        if getattr(self.args, "calibrate_kf", False):
            from boxmot.engine.calibration.kalman import validate_kf_calibration

            validate_kf_calibration(self.args.tracker, getattr(self.args, "tracker_backend", "python"))
            if getattr(self.args, "resume_tune", None):
                raise ValueError("--calibrate-kf cannot be combined with --resume-tune; resume the saved calibration.")
        return self._run()

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _resolve_metrics(self):
        args = self.args
        objectives = _parse_metric_names(getattr(args, "objectives", ()))
        self._maximize = _parse_metric_names(getattr(args, "maximize", ())) or [objectives[0] if objectives else "HOTA"]
        self._minimize = _parse_metric_names(getattr(args, "minimize", ()))

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
        _require_tuning_requirements()
        maximize, minimize = self._maximize, self._minimize

        args.show_progress = False

        # Load tracker config and build search
        self._yaml_cfg = load_yaml_config(args.tracker)
        yaml_cfg = self._yaml_cfg
        search_backend = resolve_search_backend(args)
        setattr(args, "search_alg", search_backend)

        baseline_overlay = normalize_trial_config(self.baseline_config)
        runtime_config = resolve_tracker_options(args, baseline_overlay, include_defaults=True, stamp_timing=True)

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

                pipeline.update("Preparing evaluation setup...")
                with suppress_boxmot_logs(enabled=not bool(getattr(args, "verbose", False)), level="ERROR"):
                    eval_setup(args, pipeline=pipeline)
                if getattr(args, "seq_info", None) is not None:
                    from boxmot.engine.config.runtime import resolve_sequence_workers

                    args.sequence_workers = resolve_sequence_workers(
                        len(args.seq_info), getattr(args, "sequence_workers", None)
                    )

                tune_dir = self._resolve_tune_dir()
                self._prepare_evaluation_mode(tune_dir)
                tune_name = tune_dir.name
                resume_tune = getattr(args, "resume_tune", None) or None

                ray_dir = tune_dir.parent
                if ray_dir.name != "ray" and ray_dir.parent.name == "ray":
                    ray_dir = ray_dir.parent
                if resume_tune and ray_dir.name == "ray":
                    inferred_project = ray_dir.parent
                    if inferred_project != Path(args.project).resolve():
                        args.project = str(inferred_project)

                runtime_config = self._prepare_kalman_calibration(runtime_config, tune_dir, pipeline)
                # The local schema controls every backend's search dimensions.
                # Preserve the selected KF model while optimizing association
                # and track lifecycle parameters, without editing tracker YAMLs.
                yaml_cfg = self._freeze_calibrated_schema(yaml_cfg)
                self._yaml_cfg = yaml_cfg
                self._runtime_config = runtime_config
                flat_schema = flatten_yaml_config(yaml_cfg)
                fixed_options = {
                    parameter: value
                    for parameter, value in runtime_config.items()
                    if parameter not in flat_schema or set(flat_schema[parameter]) == {"default"}
                }
                fixed_options.update(self._calibrated_fixed_options)
                if "variable_dt" in runtime_config:
                    args.variable_dt = runtime_config["variable_dt"]
                    fixed_options["variable_dt"] = args.variable_dt
                baseline = default_tune_config(yaml_cfg, defaults=runtime_config) or None

                self._configure_warning_filters()

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

                # Runtime-only settings are recorded in every trial, never searched.
                param_space.update(fixed_options)

                pipeline.refresh_fields(build_tune_workflow_fields(args, maximize=maximize, minimize=minimize))

                pipeline.advance()
                tune_callback.set_workflow_detail_renderable(format_initial_tune_progress(int(args.n_trials)))

                from boxmot.engine.tuning.trainable import build_tracker_trainable

                sequence_workers = int(args.sequence_workers)
                trainable = tune.with_resources(
                    build_tracker_trainable(tune, self._make_safe_namespace()),
                    {"cpu": sequence_workers, "gpu": 0},
                )

                # Build or restore the Ray Tuner
                tuner = self._build_or_restore_tuner(
                    trainable,
                    tune,
                    RunConfig,
                    tune_callback,
                    pipeline,
                    param_space,
                    search_alg,
                    tune_dir,
                    tune_name,
                    max_concurrent,
                )

                # Execute
                result_grid, interrupted = self._execute_tuner(tuner)

                # Post-process
                saved_artifacts = self._post_process(
                    result_grid,
                    tune_dir,
                    yaml_cfg,
                    maximize,
                    minimize,
                    base_config=runtime_config,
                )

                # Final UI
                self._finalize_ui(pipeline, saved_artifacts, baseline, maximize, minimize, tune_dir, interrupted)

                return result_grid, tune_dir, maximize, minimize
        finally:
            set_tune_progress_workflow(None)

    def _prepare_evaluation_mode(self, tune_dir: Path) -> None:
        """Keep KITTI box and segmentation scores separate across tuning resumes."""
        if getattr(self.args, "evaluation_config", {}).get("layout") != "kitti-mots":
            return
        path = tune_dir / "evaluation.json"
        eval_masks = bool(getattr(self.args, "eval_masks", False))
        if getattr(self.args, "resume_tune", None):
            if not path.is_file():
                raise ValueError("Saved tuning run lacks evaluation mode metadata; start a new tuning run.")
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("Saved tuning evaluation mode metadata is invalid; start a new tuning run.") from exc
            if not isinstance(saved, dict) or type(saved.get("eval_masks")) is not bool:
                raise ValueError("Saved tuning evaluation mode metadata is invalid; start a new tuning run.")
            if saved["eval_masks"] != eval_masks:
                flag = "include --eval-masks" if saved["eval_masks"] else "omit --eval-masks"
                raise ValueError(
                    f"Resuming tuning requires the same evaluation mode as the saved run: {flag}. "
                    "Start a new tuning run to change between box and mask evaluation."
                )
            return
        tune_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"eval_masks": eval_masks}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _prepare_kalman_calibration(self, runtime_config: dict, tune_dir: Path, pipeline: Any) -> dict:
        """Fit KF noise once before search, or restore its fixed saved profile."""
        from boxmot.engine.tuning.calibration_profile import (
            CALIBRATED_KF_OPTIONS,
            load_tuning_calibration,
            record_tuning_calibration,
        )

        if getattr(self.args, "calibrate_kf", False):
            from boxmot.engine.calibration.kalman import calibrate_kalman

            calibration = calibrate_kalman(
                self.args,
                output_dir=tune_dir,
                progress=pipeline.update,
                tracker_options=runtime_config,
            )
            self.args.tracker_config = str(calibration.config_path)
            runtime_config = resolve_tracker_options(self.args, include_defaults=True, stamp_timing=True)
            self._calibrated_fixed_options = {
                key: runtime_config[key] for key in CALIBRATED_KF_OPTIONS if key in runtime_config
            }
            record_tuning_calibration(calibration.report_path, self._calibrated_fixed_options)
            self._calibration_config_path = calibration.config_path
            pipeline.update(f"{calibration.description}\nKeeping calibrated KF settings fixed during tracker tuning.")
        elif getattr(self.args, "resume_tune", None):
            restored = load_tuning_calibration(self.args, tune_dir, overrides=self.baseline_config)
            if restored is not None:
                runtime_config, self._calibrated_fixed_options = restored
                self._calibration_config_path = tune_dir / "kf-tuning" / "calibrated.yaml"
                if getattr(self.args, "tracker_config", None) is None:
                    self.args.tracker_config = str(self._calibration_config_path)
                pipeline.update(f"Restored fixed KF calibration: {self._calibration_config_path}")
        return runtime_config

    def _freeze_calibrated_schema(self, schema: dict) -> dict:
        """Remove calibrated parameters from all search backends' local schema."""
        if not self._calibrated_fixed_options:
            return schema
        fixed_schema = {}
        for key, entry in schema.items():
            if key in self._calibrated_fixed_options:
                fixed_schema[key] = {"default": self._calibrated_fixed_options[key]}
            elif isinstance(entry, dict) and isinstance(entry.get("activates"), dict):
                fixed_schema[key] = {**entry, "activates": self._freeze_calibrated_schema(entry["activates"])}
            else:
                fixed_schema[key] = entry
        return fixed_schema

    def _validate_resumed_timing(self, results) -> None:
        """Keep timing units and the reference prior basis fixed on resume."""
        if self.args.tracker not in KALMAN_TRACKER_NAMES:
            return
        expected = getattr(self, "_runtime_config", None)
        if expected is None:
            expected = resolve_tracker_options(
                self.args, self.baseline_config, include_defaults=True, stamp_timing=True
            )
        keys = ("variable_dt", *KALMAN_TIMING_OPTIONS)
        for result in results:
            saved = normalize_trial_config(result.config)
            if any(key not in saved for key in keys):
                raise ValueError("Saved Kalman trials lack explicit timing units/reference; start a new tuning run.")
            if any(saved[key] != expected[key] for key in keys):
                raise ValueError(
                    "Resuming tuning requires the same variable_dt, kf_time_unit and kf_reference_dt_s as saved trials."
                )
            if any(saved.get(key) != value for key, value in self._calibrated_fixed_options.items()):
                raise ValueError("Saved trials do not match the fixed KF calibration; start a new tuning run.")

    def _build_or_restore_tuner(
        self,
        trainable,
        tune,
        RunConfig,
        tune_callback,
        pipeline,
        param_space,
        search_alg,
        tune_dir,
        tune_name,
        max_concurrent,
    ):
        args = self.args
        resume_tune = getattr(args, "resume_tune", None) or None
        restore_path_str = str(tune_dir)
        results_dir_str = str(tune_dir.parent)

        if resume_tune is not None and tune.Tuner.can_restore(restore_path_str):
            try:
                tuner = tune.Tuner.restore(restore_path_str, trainable=trainable, resume_errored=True)
                self._inject_callback_into_restored(tuner, tune_callback, pipeline, tune_dir)
            except Exception as exc:
                LOGGER.warning(f"Failed to restore tuner: {exc}. Starting fresh.")
            else:
                self._validate_resumed_timing(tuner.get_results())
                return tuner

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
        # One step completes the entire evaluation. Ray defaults class actors
        # to checkpointing at completion, but trackers have no incremental
        # trial state to restore; interrupted evaluations restart from frame 0.
        run_config_kwargs["checkpoint_config"] = CheckpointConfig(num_to_keep=1, checkpoint_at_end=False)

        tune_config_kwargs: dict[str, Any] = {
            "num_samples": args.n_trials,
            "max_concurrent_trials": max_concurrent,
            "trial_dirname_creator": lambda trial: f"trial_{trial.trial_id}",
            "reuse_actors": True,
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
                    1
                    for d in tune_dir.iterdir()
                    if d.is_dir() and d.name.startswith("trial_") and (d / "result.json").exists()
                )
            except Exception:
                pass
        tune_callback.completed = completed
        tune_callback._trial_index_offset = completed
        tuner._local_tuner._run_config.callbacks = [tune_callback]
        tuner._local_tuner._run_config.verbose = 0
        tuner._local_tuner._run_config.progress_reporter = TuneSilentReporter()
        tune_callback.set_workflow_detail_renderable(format_tune_progress(completed, int(self.args.n_trials)))

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

    def _post_process(
        self,
        result_grid,
        tune_dir,
        yaml_cfg,
        maximize,
        minimize,
        *,
        base_config=None,
    ):
        try:
            return save_all_results(
                tune_dir,
                result_grid,
                yaml_cfg,
                self.args.tracker,
                maximize,
                minimize,
                self.args,
                base_config=base_config,
                emit_logs=False,
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

        if self._calibration_config_path is not None and final_renderable is not None:
            from rich.console import Group
            from rich.text import Text

            final_renderable = Group(
                Text(f"Fixed KF calibration: {self._calibration_config_path}"),
                final_renderable,
            )

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
        args = self.args
        results_dir = Path(args.project).resolve() / "ray"
        dataset_dir = results_dir / self._ray_dataset_name(args)
        resume_value = getattr(args, "resume_tune", None)
        if resume_value:
            resume_path = Path(resume_value)
            if resume_path.is_absolute():
                return resume_path
            cwd_candidate = resume_path.resolve()
            if cwd_candidate.exists():
                return cwd_candidate
            if len(resume_path.parts) > 1:
                return (results_dir / resume_path).resolve()
            return (dataset_dir / resume_path.name).resolve()

        tracker_name = self._path_slug(getattr(args, "tracker", "tracker"), fallback="tracker")
        for index in range(1, 10000):
            tune_dir = dataset_dir / f"{tracker_name}_{index}"
            if not tune_dir.exists():
                return tune_dir.resolve()
        raise RuntimeError(f"Could not allocate tune directory under {dataset_dir}")

    @classmethod
    def _ray_dataset_name(cls, args) -> str:
        for attr in ("experiment_id", "dataset_id", "benchmark", "experiment"):
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
        safe: dict[str, Any] = {}
        for key, value in vars(self.args).items():
            try:
                _ray_pickle_dumps(value)
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
    """Evaluate independent trials with one lazily started replay session."""

    def __init__(self, opt: SimpleNamespace) -> None:
        self.opt = opt
        self._session = None

    def close(self) -> None:
        """Discard the worker pool and retained inputs on actor cleanup/failure."""
        session, self._session = self._session, None
        if session is not None:
            session.close()

    def __call__(self, config: dict) -> dict:
        from boxmot.engine.eval.session import ReplaySession

        if self._session is None:
            self._session = ReplaySession(
                int(self.opt.sequence_workers), cache_inputs=bool(getattr(self.opt, "cache_inputs", False))
            )
        try:
            with suppress_boxmot_logs(enabled=not bool(getattr(self.opt, "verbose", False)), level="ERROR"):
                result = run_eval(
                    deepcopy(self.opt),
                    evolve_config=config,
                    setup=False,
                    prepare_cache=False,
                    verbose=False,
                    show_progress=False,
                    replay_session=self._session,
                )
        except KeyboardInterrupt:
            self.close()
            raise
        except Exception as exc:
            self.close()
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
            key: float(trial_data["metrics"].get(key, 0.0)) for key in SUMMARY_COLUMNS if key in trial_data["metrics"]
        }
    return ValidationResult(
        benchmark=str(validation_payload.get("benchmark", getattr(args, "benchmark", getattr(args, "experiment", "")))),
        raw=raw if isinstance(raw, dict) else dict(summary),
        summary_label=str(validation_payload.get("summary_label", "all")),
        summary=dict(summary),
        exp_dir=Path(validation_payload["exp_dir"]) if validation_payload.get("exp_dir") else None,
        timings=dict(validation_payload.get("timings", {})),
        args=args,
    )


def _require_tuning_requirements() -> None:
    """Require tuning dependencies without installing or hiding missing packages."""
    from boxmot.utils.dependencies import require_extra

    require_extra("evolve", purpose="Tracker tuning")


def _is_ray_pickle_safe(value: Any) -> bool:
    """Return True if *value* is serializable with Ray's cloudpickle."""
    try:
        _ray_pickle_dumps(value)
        return True
    except Exception:
        return False


def _ray_pickle_dumps(value: Any) -> bytes:
    """Serialize with Ray/cloudpickle when available, otherwise stdlib pickle."""
    try:
        from ray import cloudpickle as serializer
    except ImportError:
        try:
            import cloudpickle as serializer  # type: ignore[no-redef]
        except ImportError:
            import pickle as serializer  # type: ignore[no-redef]
    return serializer.dumps(value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _run_sensor_tuning(
    args: Any, *, baseline_config: dict | None = None, render_cli: bool = False
) -> TuneResult | None:
    """Validate declared sensor inputs before lazily loading their optimizer."""
    from boxmot.datasets.kitti_fusion_config import load_kitti_fusion_dataset, resolve_kitti_fusion_config_path
    from boxmot.trackers.common.specs import parse_tracker_spec

    reference = getattr(args, "dataset", None)
    path = resolve_kitti_fusion_config_path(reference) if reference else None
    if path is None:
        return None

    spec = parse_tracker_spec(getattr(args, "tracker", ""), default_backend=getattr(args, "tracker_backend", "python"))
    if spec.name != "eagermot" or spec.backend != "python":
        raise ValueError("KITTI fusion tuning requires --tracker eagermot --tracker-backend python.")
    if baseline_config is not None:
        raise ValueError("KITTI fusion tuning uses separate class profiles and does not support baseline_config.")
    unsupported = (
        "experiment",
        "build",
        "build_ref",
        "build_root",
        "detector",
        "reid",
        "data_root",
        "tracker_config",
        "class_config",
        "calibrate_kf",
        "resume_tune",
        "time_budget_s",
        "fps",
        "variable_dt",
        "cache_inputs",
    )
    for name in unsupported:
        value = getattr(args, name, None)
        if value is not None and value is not False and value != "":
            raise ValueError(f"KITTI fusion tuning does not support {name}; inputs come from the dataset manifest.")
    for name, allowed in {
        "search_alg": ("optuna",),
        "max_concurrent_trials": (0, 1),
        "device": ("cpu",),
    }.items():
        value = getattr(args, name, allowed[-1])
        if value not in allowed:
            raise ValueError(f"KITTI fusion tuning runs serial Optuna trials on CPU; {name} must be one of {allowed}.")
    for name in ("objectives", "maximize"):
        if _parse_metric_names(getattr(args, name, ())) not in ([], ["HOTA"]):
            raise ValueError(f"KITTI fusion tuning optimizes class-average mask HOTA; {name} must be HOTA.")
    if _parse_metric_names(getattr(args, "minimize", ())):
        raise ValueError("KITTI fusion tuning optimizes class-average mask HOTA and does not support minimize.")

    from boxmot.engine.config.runtime import BOXMOT_DEFAULTS, resolve_sequence_workers

    n_trials = getattr(args, "n_trials", BOXMOT_DEFAULTS.tune.n_trials)
    seed = getattr(args, "seed", None)
    seed = 0 if seed is None else seed
    if isinstance(n_trials, bool) or not isinstance(n_trials, int) or n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer within [0, 2**32).")
    dataset = load_kitti_fusion_dataset(
        path,
        split=getattr(args, "split", None) or None,
        sequence_names=getattr(args, "sequence_names", ()),
    )
    sequence_workers = resolve_sequence_workers(len(dataset.sequence_names), getattr(args, "sequence_workers", None))
    normalized = SimpleNamespace(
        **{
            **vars(args),
            "dataset": dataset.config_path,
            "dataset_id": dataset.id,
            "tracker": spec.name,
            "tracker_backend": spec.backend,
            "split": dataset.split,
            "sequence_names": dataset.sequence_names,
            "n_trials": n_trials,
            "seed": seed,
            "project": Path(getattr(args, "project", None) or "runs/eagermot-tune"),
            "device": "cpu",
            "sequence_workers": sequence_workers,
            "max_concurrent_trials": 1,
            "search_alg": "optuna",
            "objectives": ("HOTA",),
            "maximize": ("HOTA",),
            "minimize": (),
            "per_class": True,
            "eval_masks": True,
        }
    )
    if not render_cli:
        from boxmot.engine.tuning.eagermot_kitti import run_eagermot_kitti_tuning

        return run_eagermot_kitti_tuning(normalized)

    pipeline = TuneWorkflowReporter(normalized, maximize=["HOTA"], minimize=[]).pipeline()
    with pipeline, suppress_boxmot_logs(enabled=not bool(getattr(normalized, "verbose", False)), level="ERROR"):
        pipeline.update("Loading KITTI sensor inputs and tuning runtime…")
        try:
            from boxmot.engine.tuning.eagermot_kitti import run_eagermot_kitti_tuning

            result = run_eagermot_kitti_tuning(normalized, pipeline=pipeline)
        except ImportError as exc:
            raise ImportError(
                f"KITTI fusion tuning requires the mots and evolve extras: {exc}\n"
                "Install with: uv sync --extra cpu --extra mots --extra evolve"
            ) from exc
        best_renderable = result.best.metrics.renderable(
            title=CLI_TUNE_BEST_SUMMARY_TITLE,
            compare_raw=result.baseline.raw,
            compare_args=result.baseline.args,
        )
        artifacts = build_tune_artifacts_renderable(
            {
                "best_trial_id": f"trial {result.best.index}",
                "best_yaml_path": result.best_yaml,
                "study_path": result.best_yaml.parent / "study.sqlite3",
                "manifest_path": result.best_yaml.parent / "run.json",
            }
        )
        pipeline.finish(
            combine_tune_result_renderables(best_renderable, artifacts),
            exp_dir=result.best_yaml.parent,
        )
        result.workflow_rendered = True
        return result


def run_tune(args, *, baseline_config: dict | None = None) -> TuneResult:
    """Run tuning and return a structured TuneResult."""
    sensor_result = _run_sensor_tuning(args, baseline_config=baseline_config)
    if sensor_result is not None:
        return sensor_result

    from boxmot.engine.config.trackers import validate_image_tracker

    if getattr(args, "tracker", None) is not None:
        validate_image_tracker(str(args.tracker))
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

    resolved_best_config = resolve_tracker_options(
        args,
        {**normalize_trial_config(baseline_config), **best.config},
        include_defaults=True,
        stamp_timing=True,
    )
    return TuneResult(
        benchmark=str(getattr(args, "benchmark", getattr(args, "experiment", ""))),
        tracker=str(args.tracker),
        trials=trials,
        best=best,
        best_config=resolved_best_config,
        best_yaml=tune_dir / "best.yaml",
    )


def main(args: Any) -> TuneResult | None:
    """Tune either perception builds or saved sensor inputs through one entry point."""
    try:
        sensor_result = _run_sensor_tuning(args, render_cli=True)
    except ImportError as exc:
        if getattr(exc, "_workflow_rendered_error", False):
            raise
        raise click.ClickException(
            f"KITTI fusion tuning requires the mots and evolve extras: {exc}\n"
            "Install with: uv sync --extra cpu --extra mots --extra evolve"
        ) from exc
    except (ValueError, OSError) as exc:
        if getattr(exc, "_workflow_rendered_error", False):
            raise
        raise click.ClickException(str(exc)) from exc
    if sensor_result is not None:
        return sensor_result

    from boxmot.engine.config.trackers import validate_image_tracker

    if getattr(args, "tracker", None) is not None:
        validate_image_tracker(str(args.tracker))
    tuner = Tuner(args)
    tuner.fit()


if __name__ == "__main__":
    main()
