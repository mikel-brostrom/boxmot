import subprocess
import ray
import os
import yaml
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig
from tracking.val import run_generate_mot_results, run_trackeval, parse_opt as parse_optt
from boxmot.utils import ROOT

class Tracker:
    def __init__(self, opt, parameters):
        self.opt = opt
        self.parameters = parameters

    def get_new_config(self, config):
        # Update the options with the current config
        self.parameters.update(config)
        # Overwrite the local file with the new parameters
        tracking_config = ROOT / 'boxmot' / 'configs' / (self.opt.tracking_method + '.yaml')
        with open(tracking_config, 'w') as f:
            yaml.dump(self.parameters, f)

    def objective_function(self, config):
        # Generate new set of params
        # self.get_new_config(config)
        # # Run trial, get HOTA, MOTA, IDF1 combined results
        # run_generate_mot_results(self.opt)
        # results = run_trackeval(self.opt)
        # # Extract objective results of the current trial
        # combined_results = {key: results.get(key) for key in self.opt['objectives']}
        #return combined_results
        return {"HOTA": 0.4, "MOTA": 0.3, "IDF1": 0.2}

# Define the search space for hyperparameters
search_space = {
    'iou_thresh': tune.uniform(0.1, 0.4),
    'ecc': tune.choice([True, False]),
    'ema_alpha': tune.uniform(0.7, 0.95),
    'max_dist': tune.uniform(0.1, 0.4),
    'max_iou_dist': tune.uniform(0.5, 0.95),
    'max_age': tune.quniform(10, 150, 10),
    'n_init': tune.quniform(1, 3, 1),
    'mc_lambda': tune.uniform(0.90, 0.999),
    'nn_budget': tune.choice([100]),
    'max_unmatched_preds': tune.choice([0])
}

opt = parse_optt()
tracker = Tracker(opt, search_space)

def train(config):
    return tracker.objective_function(config)

asha_scheduler = ASHAScheduler(
    metric="HOTA",
    mode="max",
    max_t=100,
    grace_period=10,
    reduction_factor=3
)

results_dir = os.path.abspath("results/")
# Run Ray Tune
tuner = tune.Tuner(
    tune.with_resources(train, {"cpu": 1, "gpu": 0}),  # Adjust resources as needed
    param_space=search_space,
    tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=10),
    run_config=RunConfig(storage_path=results_dir)
)

tuner.fit()