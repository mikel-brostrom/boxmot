"""Tune separate EagerMOT class profiles against joint KITTI mask HOTA."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import optuna

from boxmot.engine.eval.eagermot_kitti import (
    KITTI_CLASSES,
    KITTI_PROFILES,
    evaluate_eagermot_kitti,
    load_kitti_profiles,
    prepare_eagermot_kitti,
    write_kitti_profiles,
)
from boxmot.engine.eval.output import increment_path
from boxmot.engine.tuning.backends.optuna_backend import yaml_to_optuna_define_space
from boxmot.engine.tuning.search_space import default_tune_config, load_yaml_config
from boxmot.utils import logger as LOGGER

# Spatial confidence decay does not affect image masks; the 2D affinity is fixed.
_FIXED_PARAMETERS = frozenset({"max_age_2d", "asso_func", "per_class"})
_OBJECTIVE = "cls_comb_cls_av.HOTA"


def _active_schema(schema: dict[str, Any], method: str) -> dict[str, Any]:
    """Keep only parameters that affect the selected association method."""
    inactive = "distance_threshold" if method == "iou_3d" else "iou_3d_threshold"
    return {name: details for name, details in schema.items() if name not in _FIXED_PARAMETERS and name != inactive}


def _sample_profiles(trial: optuna.Trial, schema: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Sample independent class parameters using the shared YAML distributions."""
    profiles = load_kitti_profiles()
    for class_id, name in KITTI_CLASSES.items():
        method_key = f"{name}.first_matching_method"
        yaml_to_optuna_define_space({method_key: schema["first_matching_method"]})(trial)
        active = _active_schema(schema, trial.params[method_key])
        yaml_to_optuna_define_space(
            {f"{name}.{key}": details for key, details in active.items() if key != "first_matching_method"}
        )(trial)
        profiles[class_id].update({key: trial.params[f"{name}.{key}"] for key in active})
    return profiles


def run_eagermot_kitti_tuning(args: Any) -> Path:
    """Run serial Optuna trials, saving the best reusable profiles after each trial.

    Each trial replays both classes together because mask overlap resolution can
    couple the class results. The objective is the class-average mask HOTA, in
    percent, over all selected sequences. The first trial evaluates the original
    KITTI class presets and counts toward ``n_trials``.
    """
    if isinstance(args.n_trials, bool) or not isinstance(args.n_trials, int) or args.n_trials < 1:
        raise ValueError("n_trials must be an integer >= 1.")
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or not 0 <= args.seed < 2**32:
        raise ValueError("seed must be an integer within [0, 2**32).")
    schema = load_yaml_config("eagermot")
    inputs = prepare_eagermot_kitti(args)
    output = increment_path(Path(args.project).expanduser().resolve() / inputs.manifest["split"], mkdir=True)
    manifest = {
        **inputs.manifest,
        "status": "running",
        "mode": "tune",
        "per_class": True,
        "objective": _OBJECTIVE,
        "direction": "maximize",
        "n_trials": args.n_trials,
        "seed": args.seed,
        "sampler": "Optuna TPESampler",
        "optuna_version": optuna.__version__,
        "search_schema": schema,
        "fixed_parameters": sorted(_FIXED_PARAMETERS),
        "conditional_parameters": {
            "distance_threshold": "first_matching_method != iou_3d",
            "iou_3d_threshold": "first_matching_method == iou_3d",
        },
        "baseline_profiles": KITTI_PROFILES,
        "completed_trials": 0,
    }

    def save_manifest() -> None:
        """Persist progress so interrupted studies retain their completed results."""
        (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def objective(trial: optuna.Trial) -> float:
        """Replay the trial and select the native class-average aggregate once."""
        profiles = _sample_profiles(trial, schema)
        trial_output = output / "trials" / f"{trial.number:04d}"
        trial.set_user_attr("profiles", {name: profiles[class_id] for class_id, name in KITTI_CLASSES.items()})
        trial.set_user_attr("output", str(trial_output))
        LOGGER.info(f"EagerMOT tuning: trial {trial.number + 1}/{args.n_trials}")
        metrics = evaluate_eagermot_kitti(inputs, profiles, trial_output)
        score = float(metrics["cls_comb_cls_av"]["HOTA"])
        if not math.isfinite(score):
            raise ValueError("EagerMOT tuning returned a non-finite mask HOTA.")
        trial.set_user_attr("class_hota", {name: float(metrics[name]["HOTA"]) for name in KITTI_CLASSES.values()})
        return score

    def save_best(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Checkpoint the best complete class configurations after every result."""
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        best = study.best_trial
        profiles = {class_id: best.user_attrs["profiles"][name] for class_id, name in KITTI_CLASSES.items()}
        temporary = output / "best.yaml.tmp"
        write_kitti_profiles(temporary, profiles)
        temporary.replace(output / "best.yaml")
        manifest.update(
            completed_trials=manifest["completed_trials"] + 1,
            best_trial=best.number,
            best_hota=best.value,
            best_class_hota=best.user_attrs["class_hota"],
            best_profiles="best.yaml",
            best_results=str(Path("trials") / f"{best.number:04d}"),
        )
        save_manifest()
        LOGGER.info(f"EagerMOT tuning: best mask HOTA {best.value:.2f} (trial {best.number + 1})")

    save_manifest()
    try:
        study = optuna.create_study(
            study_name="eagermot-kitti",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            # A SQLite file URI preserves URL/template characters in directory names.
            storage=f"sqlite:///{(output / 'study.sqlite3').as_uri()}?uri=true",
        )
        baseline = {
            f"{KITTI_CLASSES[class_id]}.{name}": value
            for class_id, profile in KITTI_PROFILES.items()
            for name, value in default_tune_config(
                _active_schema(schema, profile["first_matching_method"]), defaults=profile
            ).items()
        }
        study.enqueue_trial(baseline)
        study.optimize(objective, n_trials=args.n_trials, n_jobs=1, callbacks=[save_best])
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        save_manifest()
    return output


__all__ = ("run_eagermot_kitti_tuning",)
