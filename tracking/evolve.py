import subprocess
import ray
import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import RunConfig
from tracking.val import run_generate_mot_results, run_trackeval, parse_opt as parse_optt

class Tracker:
    def __init__(self, opt):
        self.opt = opt

    def objective_function(self, config):
        # self.opt.update(config)  # Update the options with the current config
        # run_generate_mot_results(self.opt)
        # results = run_trackeval(self.opt)
        # # Extract objective results of the current trial
        # combined_results = [results.get(key) for key in self.opt['objectives']]
        # Combine results into a single score or return as a dictionary
        return {"HOTA": 0.5, "MOTA": 0.7, "IDF1": 0.7}

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

# Initialize tracker options
opt = {
    "objectives": ["HOTA", "MOTA", "IDF1"],
    # Add other default options here
}

tracker = Tracker(opt)

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
    tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=10),
    run_config=RunConfig(storage_path=results_dir)
)

tuner.fit()