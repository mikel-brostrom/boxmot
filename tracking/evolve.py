
import os
import yaml
from pathlib import Path

from boxmot.utils.checks import RequirementsChecker
from boxmot.utils import EXAMPLES
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

# Define the search space for hyperparameters
def get_search_space(tracking_method):
    if tracking_method == 'strongsort':
        search_space = {
            "iou_thresh": tune.uniform(0.1, 0.4),
            "ecc": tune.choice([True, False]),
            "ema_alpha": tune.uniform(0.7, 0.95),
            "max_dist": tune.uniform(0.1, 0.4),
            "max_iou_dist": tune.uniform(0.5, 0.95),
            "max_age": tune.randint(10, 151),  # The upper bound is exclusive in randint
            "n_init": tune.randint(1, 4),  # The upper bound is exclusive in randint
            "mc_lambda": tune.uniform(0.90, 0.999),
            "nn_budget": tune.choice([100]),
        }
    elif tracking_method == 'hybridsort':
        search_space = {
            "det_thresh": tune.uniform(0, 0.6),
            "max_age": tune.randint(10, 151, 10),  # The upper bound is exclusive in randint
            "min_hits": tune.randint(1, 6),  # The upper bound is exclusive in randint
            "delta_t": tune.randint(1, 6),  # The upper bound is exclusive in randint
            "asso_func": tune.choice(['iou', 'giou', 'diou']),
            "iou_thresh": tune.uniform(0.1, 0.4),
            "inertia": tune.uniform(0.1, 0.4),
            "TCM_first_step_weight": tune.uniform(0, 0.5),
            "longterm_reid_weight": tune.uniform(0, 0.5),
            "use_byte": tune.choice([True, False])
        }
    elif tracking_method == 'botsort':
        search_space = {
            "track_high_thresh": tune.uniform(0.3, 0.7),
            "track_low_thresh": tune.uniform(0.1, 0.3),
            "new_track_thresh": tune.uniform(0.1, 0.8),
            "track_buffer": tune.randint(20, 81, 10),  # The upper bound is exclusive in randint
            "match_thresh": tune.uniform(0.1, 0.9),
            "proximity_thresh": tune.uniform(0.25, 0.75),
            "appearance_thresh": tune.uniform(0.1, 0.8),
            "cmc_method": tune.choice(['sparseOptFlow']),
            "frame_rate": tune.choice([30]),
            "lambda_": tune.uniform(0.97, 0.995)
        }
    elif tracking_method == 'bytetrack':
        search_space = {
            "track_thresh": tune.uniform(0.4, 0.6),
            "track_buffer": tune.randint(10, 61, 10),  # The upper bound is exclusive in randint
            "match_thresh": tune.uniform(0.7, 0.9)
        }
    elif tracking_method == 'ocsort':
        search_space = {
            "det_thresh": tune.uniform(0, 0.6),
            "max_age": tune.grid_search([10, 20, 30, 40, 50, 60]),  # Since step is 10, using grid_search for these discrete values
            "min_hits": tune.grid_search([1, 2, 3, 4, 5]),  # Since step is 1, using grid_search for these discrete values
            "iou_thresh": tune.uniform(0.1, 0.4),
            "delta_t": tune.grid_search([1, 2, 3, 4, 5]),  # Since step is 1, using grid_search for these discrete values
            "asso_func": tune.choice(['iou', 'giou', 'centroid']),
            "use_byte": tune.choice([True, False]),
            "inertia": tune.uniform(0.1, 0.4),
            "Q_xy_scaling": tune.loguniform(0.01, 1),
            "Q_s_scaling": tune.loguniform(0.0001, 1)
        }
    elif tracking_method == 'deepocsort':
        search_space = {
            "det_thresh": tune.uniform(0.3, 0.6),  # Changed from int to uniform since it seems to be a float range
            "max_age": tune.randint(10, 61, 10),  # The upper bound is exclusive in randint
            "min_hits": tune.randint(1, 6),  # The upper bound is exclusive in randint
            "iou_thresh": tune.uniform(0.1, 0.4),
            "delta_t": tune.randint(1, 6),  # The upper bound is exclusive in randint
            "asso_func": tune.choice(['iou', 'giou']),
            "inertia": tune.uniform(0.1, 0.4),
            "w_association_emb": tune.uniform(0.5, 0.9),
            "alpha_fixed_emb": tune.uniform(0.9, 0.999),
            "aw_param": tune.uniform(0.3, 0.7),
            "embedding_off": tune.choice([True, False]),
            "cmc_off": tune.choice([True, False]),
            "aw_off": tune.choice([True, False]),
            "Q_xy_scaling": tune.uniform(0.01, 1),
            "Q_s_scaling": tune.uniform(0.0001, 1)
        }
    elif tracking_method == 'imprassoc':
        search_space = {
            "track_high_thresh": tune.uniform(0.3, 0.7),
            "track_low_thresh": tune.uniform(0.1, 0.3),
            "new_track_thresh": tune.uniform(0.1, 0.8),
            "track_buffer": tune.qrandint(20, 80, 10),  # The upper bound is exclusive in randint
            "match_thresh": tune.uniform(0.1, 0.9),
            "second_match_thresh": tune.uniform(0.1, 0.4),
            "overlap_thresh": tune.uniform(0.3, 0.6),
            "proximity_thresh": tune.uniform(0.1, 0.8),
            "appearance_thresh": tune.uniform(0.1, 0.8),
            "cmc_method": tune.choice(['sparseOptFlow']),
            "frame_rate": tune.choice([30]),
            "lambda_": tune.uniform(0.97, 0.995)
        }
    return search_space
        
opt = parse_optt()
opt.val_tools_path = EXAMPLES / 'val_utils'
opt.source = Path(opt.source).resolve()
opt.yolo_model = [Path(y).resolve() for y in opt.yolo_model]
opt.reid_model = [Path(r).resolve() for r in opt.reid_model]
search_space = get_search_space(opt.tracking_method)
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