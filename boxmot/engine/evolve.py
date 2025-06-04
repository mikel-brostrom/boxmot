#!/usr/bin/env python3
"""
This script runs a hyperparameter tuning process for a multi-object tracking (MOT) tracker using Ray Tune.
It loads the tracker configuration from a YAML file, sets up the search space for hyperparameters, and evaluates
the tracker to optimize selected metrics (e.g., MOTA, HOTA, IDF1).
"""

import os
from pathlib import Path

import yaml
from ray import tune
from ray.air import RunConfig

from boxmot.utils import EXAMPLES, NUM_THREADS, TRACKER_CONFIGS
from boxmot.engine.val import (
    download_mot_eval_tools,
    run_generate_dets_embs,
    run_generate_mot_results,
    run_trackeval,
)


class Tracker:
    """
    Encapsulates the evaluation of a tracking configuration.
    """

    def __init__(self, opt):
        self.opt = opt

    def objective_function(self, config: dict) -> dict:
        """
        Evaluates a given tracker configuration.

        Args:
            config (dict): A dictionary of tracker hyperparameters.

        Returns:
            dict: Combined evaluation metrics extracted from run_trackeval.
        """
        # Ensure evaluation tools are available
        download_mot_eval_tools(self.opt.val_tools_path)
        # Generate MOT-compliant results with the specified tracker parameters
        run_generate_mot_results(self.opt, config)
        # Retrieve evaluation metrics (e.g., MOTA, HOTA, IDF1)
        results = run_trackeval(self.opt)
        # Extract only the desired objective results
        combined_results = {key: results.get(key) for key in self.opt.objectives}
        return combined_results


def load_yaml_config(tracking_method: str) -> dict:
    """
    Loads the YAML configuration file for the given tracking method.

    Args:
        tracking_method (str): Name of the tracking method.

    Returns:
        dict: Configuration parameters loaded from the YAML file.
    """
    config_path = TRACKER_CONFIGS / f"{tracking_method}.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def yaml_to_search_space(config: dict) -> dict:
    """
    Converts a YAML configuration dictionary to a Ray Tune search space.

    Args:
        config (dict): YAML configuration parameters.

    Returns:
        dict: A dictionary representing the search space for hyperparameters.
    """
    search_space = {}
    for param, details in config.items():
        search_type = details.get("type")
        if search_type == "uniform":
            search_space[param] = tune.uniform(*details["range"])
        elif search_type == "randint":
            search_space[param] = tune.randint(*details["range"])
        elif search_type == "qrandint":
            search_space[param] = tune.qrandint(*details["range"])
        elif search_type == "choice":
            search_space[param] = tune.choice(details["options"])
        elif search_type == "grid_search":
            search_space[param] = tune.grid_search(details["values"])
        elif search_type == "loguniform":
            search_space[param] = tune.loguniform(*details["range"])
    return search_space


def main(opt):
    # Parse options and set necessary paths
    opt.val_tools_path = EXAMPLES / "val_utils"
    opt.source = Path(opt.source).resolve()
    opt.yolo_model = [Path(y).resolve() for y in opt.yolo_model]
    opt.reid_model = [Path(r).resolve() for r in opt.reid_model]

    # Load YAML configuration and convert it to a Ray Tune search space
    yaml_config = load_yaml_config(opt.tracking_method)
    search_space = yaml_to_search_space(yaml_config)

    # Create a Tracker instance
    tracker = Tracker(opt)

    # Generate detection and embedding files required for evaluation
    run_generate_dets_embs(opt)

    # Define a wrapper for the objective function for Ray Tune
    def tune_wrapper(config):
        return tracker.objective_function(config)

    results_dir = os.path.abspath("ray/")

    # Set up and run the hyperparameter tuning using Ray Tune
    tuner = tune.Tuner(
        tune.with_resources(tune_wrapper, {"cpu": NUM_THREADS, "gpu": 0}),
        param_space=search_space,
        tune_config=tune.TuneConfig(num_samples=opt.n_trials),
        run_config=RunConfig(storage_path=results_dir),
    )
    tuner.fit()

    # Print the tuning results
    print(tuner.get_results())


if __name__ == "__main__":
    main()
