
import os
import yaml
from pathlib import Path

from boxmot.utils.checks import RequirementsChecker
from boxmot.utils import EXAMPLES, TRACKER_CONFIGS
from tracking.val import (
    run_generate_dets_embs,
    run_generate_mot_results,
    run_trackeval,
    parse_opt as parse_optt,
    download_mot_eval_tools
)
from boxmot.utils import ROOT, NUM_THREADS

checker = RequirementsChecker()
checker.check_packages(('ray[tune]',))  # install

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig


class Tracker:
    def __init__(self, opt, parameters):
        self.opt = opt

    def objective_function(self, config):
        download_mot_eval_tools(self.opt.val_tools_path)
        # generate new set of mot challenge compliant results with
        # new set of generated tracker parameters
        run_generate_mot_results(self.opt, config)
        # get MOTA, HOTA, IDF1 results
        results = run_trackeval(self.opt)
        # Extract objective results
        combined_results = {key: results.get(key) for key in self.opt.objectives}
        return combined_results


def load_yaml_config(tracking_method):
    config_path = TRACKER_CONFIGS / f"{tracking_method}.yaml"  # Example: 'botsort_search_space.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Define the search space for hyperparameters
def yaml_to_search_space(config):
    search_space = {}
    for param, details in config.items():
        search_type = details['type']
        if search_type == 'uniform':
            search_space[param] = tune.uniform(*details['range'])
        elif search_type == 'randint':
            search_space[param] = tune.randint(*details['range'])
        elif search_type == 'qrandint':
            search_space[param] = tune.qrandint(*details['range'])
        elif search_type == 'choice':
            search_space[param] = tune.choice(details['options'])
        elif search_type == 'grid_search':
            search_space[param] = tune.grid_search(details['values'])
        elif search_type == 'loguniform':
            search_space[param] = tune.loguniform(*details['range'])
    return search_space

        
opt = parse_optt()
opt.val_tools_path = EXAMPLES / 'val_utils'
opt.source = Path(opt.source).resolve()
opt.yolo_model = [Path(y).resolve() for y in opt.yolo_model]
opt.reid_model = [Path(r).resolve() for r in opt.reid_model]

# Load the appropriate YAML configuration
yaml_config = load_yaml_config(opt.tracking_method)

# Convert YAML config to Ray Tune search space
search_space = yaml_to_search_space(yaml_config)

tracker = Tracker(opt, search_space)
run_generate_dets_embs(opt)

def _tune(config):
    return tracker.objective_function(config)

# Asynchronous Successive Halving Algorithm Scheduler
# particularly well-suited for distributed and parallelized environments
# it prunes poorly performing trials focusing on promising configurations
asha_scheduler = ASHAScheduler(
    metric="HOTA",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=3
)

results_dir = os.path.abspath("ray/")
# Run Ray Tune
tuner = tune.Tuner(
    tune.with_resources(_tune, {"cpu": NUM_THREADS, "gpu": 0}),  # Adjust resources as needed
    param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=opt.n_trials),
    run_config=RunConfig(storage_path=results_dir)
)

tuner.fit()

print(tuner.get_results())