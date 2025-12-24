#!/usr/bin/env python3
"""
This script runs a hyperparameter tuning process for a multi-object tracking (MOT) tracker using Ray Tune,
with support for resuming (restoring) previous tuning runs.
"""

import os

os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"   # keep CWD constant for all trials
from pathlib import Path

import yaml

from boxmot.engine.evaluator import (eval_init, run_generate_dets_embs,
                                     run_generate_mot_results, run_trackeval)
from boxmot.utils import NUM_THREADS, TRACKER_CONFIGS
from boxmot.utils import logger as LOGGER


def load_yaml_config(tracking_method: str) -> dict:
    """
    Loads the YAML configuration file for the given tracking method.
    """
    config_path = TRACKER_CONFIGS / f"{tracking_method}.yaml"
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def yaml_to_search_space(config: dict, tune) -> dict:
    """
    Converts a YAML configuration dictionary to a Ray Tune search space.
    
    Args:
        config: YAML configuration dictionary
        tune: Ray Tune module (passed to avoid import at module level)
    """
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
            space[param] = tune.choice(details["options"])
        elif t == "grid_search":
            # Optuna doesn't support grid_search directly in the same way as basic Tune
            # Mapping grid_search to choice for Optuna compatibility
            space[param] = tune.choice(details["values"])
        elif t == "loguniform":
            space[param] = tune.loguniform(*details["range"])
    return space


class Tracker:
    """
    Encapsulates the evaluation of a tracking configuration.
    """
    def __init__(self, opt):
        self.opt = opt

    def objective_function(self, config: dict) -> dict:
        # Generate MOT-compliant results
        run_generate_mot_results(self.opt, config)
        # Evaluate and extract objectives
        results = run_trackeval(self.opt)

        # If results are nested (multi-class), average the metrics
        if results and isinstance(next(iter(results.values())), dict):
            return {
                k: max(0, sum(c.get(k, 0) for c in results.values()) / len(results))
                for k in self.opt.objectives
            }

        return {k: max(0, results.get(k, 0)) for k in self.opt.objectives}


def main(args):
    # Install evolve dependencies only once in main process
    from boxmot.utils.checks import RequirementsChecker
    checker = RequirementsChecker()
    checker.sync_extra(extra="evolve")
    
    # Import ray after dependencies are installed
    from ray import tune
    from ray.tune import RunConfig
    from ray.tune.search.optuna import OptunaSearch

    # Print tuning pipeline header (blue palette)
    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>🔧 BoxMOT Hyperparameter Tuning</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>   <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>  <cyan>{args.yolo_model}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>      <cyan>{args.reid_model}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Trials:</bold>    <cyan>{args.n_trials}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Objectives:</bold> <cyan>{', '.join(args.objectives)}</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    
    # --- initial setup ---
    args.yolo_model = [Path(y).resolve() for y in args.yolo_model]
    args.reid_model = [Path(r).resolve() for r in args.reid_model]

    # Load search space
    yaml_cfg = load_yaml_config(args.tracking_method)
    search_space = yaml_to_search_space(yaml_cfg, tune)
    tracker = Tracker(args)

    # Optuna search setup
    primary_metric = args.objectives[0]
    optuna_search = OptunaSearch(metric=primary_metric, mode="max")

    def tune_wrapper(cfg):
        return tracker.objective_function(cfg)

    # Paths for storage and restore
    tune_name = f"{args.tracking_method}_tune"
    results_dir = args.project / "ray"
    restore_path = results_dir / tune_name

    # Define trainable
    trainable = tune.with_resources(tune_wrapper, {"cpu": NUM_THREADS, "gpu": 0})

    # Ensure evaluation tools are available
    LOGGER.opt(colors=True).info("<cyan>[1/3]</cyan> Setting up evaluation environment...")
    eval_init(args)
    
    LOGGER.opt(colors=True).info("<cyan>[2/3]</cyan> Generating detections and embeddings...")
    run_generate_dets_embs(args)

    # Check for existing run to resume
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
            ),
            run_config=RunConfig(
                storage_path=results_dir,
                name=tune_name,
            ),
        )

    # Run or resume
    tuner.fit()
    
    # Print results summary
    results = tuner.get_results()
    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>📊 Tuning Results</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.info(results)
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")


if __name__ == "__main__":
    main()
