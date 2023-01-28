#  Yolov5_StrongSORT_OSNet, GPL-3.0 license
"""
Evolve hyperparameters for the specific selected tracking method and a specific dataset.
The best set of hyperparameters is written to the config file of the selected tracker
(trackers/<tracking-method>/configs). Tracker parameter importance and pareto front plots
are generated as well.

Usage:

    $ python3 evolve.py --tracking-method strongsort --benchmark MOT17 --device 0,1,2,3 --n-trials 100
                        --tracking-method ocsort     --benchmark MOT16 --n-trials 1000
"""

import os
import sys
import logging
import argparse
import yaml
import re
from pathlib import Path
from val import Evaluator

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov8') not in sys.path:
    sys.path.append(str(ROOT / 'yolov8'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov8.ultralytics.yolo.utils import LOGGER
from yolov8.ultralytics.yolo.utils.checks import check_requirements, print_args
from track import run


class Objective(Evaluator):
    """Objective function to evolve best set of hyperparams for
    
    This object is passed to an objective function and provides interfaces to overwrite
    a tracker's config yaml file and the call to the objective function (evaluation on 
    a specific benchmark: MOT16, MOT17... and split) with a specifc set up harams.
    
    Note:
        The objective function inherits all the methods and properties from the Evaluator
        which let us evolve hparams genetically for a specific dataset. Split your dataset in
        half to speed up this process.

    Args:
        opts: the parsed script arguments

    Attributes:
        opts: the parsed script arguments

    """
    def __init__(self, opts):  
        self.opt = opts
                
    def get_new_config(self, trial):
        """Overwrites the tracking config by newly generated hparams

        Args:
            trial (type): represents the current process to evaluate on objective function.

        Returns:
            None
        """
        
        d = {}
        self.opt.conf_thres = trial.suggest_float("conf_thres", 0.35, 0.55)
        
        if self.opt.tracking_method == 'strongsort':
            
            iou_thresh = trial.suggest_float("iou_thresh", 0.1, 0.4)
            ecc = trial.suggest_categorical("ecc", [True, False])
            ema_alpha = trial.suggest_float("ema_alpha", 0.7, 0.95)
            max_dist = trial.suggest_float("max_dist", 0.1, 0.4)
            max_iou_dist = trial.suggest_float("max_iou_dist", 0.5, 0.95)
            max_age = trial.suggest_int("max_age", 10, 150, step=10)
            n_init = trial.suggest_int("n_init", 1, 3, step=1)
            mc_lambda = trial.suggest_float("mc_lambda", 0.90, 0.999)
            nn_budget = trial.suggest_categorical("nn_budget", [100])
            max_unmatched_preds = trial.suggest_categorical("max_unmatched_preds", [0])

            d['strongsort'] = \
                {
                    'ecc': ecc,
                    'mc_lambda': mc_lambda,
                    'ema_alpha': ema_alpha,
                    'max_dist':  max_dist,
                    'max_iou_dist': max_iou_dist,
                    'max_unmatched_preds': max_unmatched_preds,
                    'max_age': max_age,
                    'n_init': n_init,
                    'nn_budget': nn_budget
                }
                
        elif self.opt.tracking_method == 'bytetrack':

            track_thresh = trial.suggest_float("track_thresh", 0.4, 0.6)              
            track_buffer = trial.suggest_int("track_buffer", 10, 60, step=10)  
            match_thresh = trial.suggest_float("match_thresh", 0.7, 0.9)
            
            d['bytetrack'] = \
                {
                    'track_thresh': self.opt.conf_thres,
                    'match_thresh': match_thresh,
                    'track_buffer': track_buffer,
                    'frame_rate': 30
                }
                
        elif self.opt.tracking_method == 'ocsort':
            
            det_thresh = trial.suggest_int("det_thresh", 0.4, 0.6)
            max_age = trial.suggest_int("max_age", 10, 60, step=10)
            min_hits = trial.suggest_int("min_hits", 1, 5, step=1)
            iou_thresh = trial.suggest_float("iou_thresh", 0.1, 0.4)
            delta_t = trial.suggest_int("delta_t", 1, 5, step=1)
            asso_func = trial.suggest_categorical("asso_func", ['iou', 'giou'])
            inertia = trial.suggest_float("inertia", 0.1, 0.4)
            use_byte = trial.suggest_categorical("use_byte", [True, False])
            
            d['ocsort'] = \
                {
                    'det_thresh': det_thresh,
                    'max_age': max_age,
                    'min_hits': min_hits,
                    'iou_thresh': iou_thresh,
                    'delta_t': delta_t,
                    'asso_func': asso_func,
                    'inertia': inertia,
                    'use_byte': use_byte,
                }
                        
        # overwrite existing config for tracker
        with open(self.opt.tracking_config, 'w') as f:
            data = yaml.dump(d, f)   
    

    def __call__(self, trial):
        """Objective function to evolve best set of hyperparams for

        Args:
            trial (type): represents the current process to evaluate on objective function.

        Returns:
            float, float, float: HOTA, MOTA and IDF1 scores respectively
        """
        
        # generate new set of params
        self.get_new_config(trial)
        # run trial
        results = self.run(self.opt)
        # get HOTA, MOTA, IDF1 COMBINED string lines
        combined_results = results.split('COMBINED')[2:-1]
        # robust way of getting first ints/float in string
        combined_results = [float(re.findall("[-+]?(?:\d*\.*\d+)", f)[0]) for f in combined_results]
        # pack everything in dict
        combined_results = {key: value for key, value in zip(['HOTA', 'MOTA', 'IDF1'], combined_results)}
        # extract objective results of current trial
        combined_results = [combined_results.get(key) for key in self.opt.objectives]
        return combined_results
    

def print_best_trial_metric_results(study, objectives):
    """Print the main MOTA metric (HOTA, MOTA, IDF1) results

    Args:
        study : the complete hyperparameter search study

    Returns:
        None
    """
    for ob in enumerate(objectives):  
        trial_with_highest_ob = max(study.best_trials, key=lambda t: t.values[0])
        print(f"Trial with highest {ob}: ")
        print(f"\tnumber: {trial_with_highest_ob.number}")
        print(f"\tparams: {trial_with_highest_ob.params}")
        print(f"\tvalues: {trial_with_highest_ob.values}")

    
    
def save_plots(opt, study, objectives):
    """Print the main MOTA metric (HOTA, MOTA, IDF1) results

    Args:
        opt: the parsed script arguments
        study : the complete hyperparameter search study

    Returns:
        None
    """
    if len(objectives) > 1:
        fig = optuna.visualization.plot_pareto_front(study, target_names=objectives)
        fig.write_html("pareto_front_" + opt.tracking_method + ".html")
    else:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html("plot_optim_history_" + opt.tracking_method + ".html")
    
    for i, ob in enumerate(objectives):  
        if not opt.n_trials <= 1:  # more than one trial needed for parameter importance 
            fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[i], target_name=ob)
            fig.write_html(f"{ob}_param_importances_" + opt.tracking_method + ".html")
        
        
def write_best_HOTA_params_to_config(opt, study):
    """Overwrites the config file for the selected tracking method with the 
       hparams from the trial resulting in the best HOTA result

    Args:
        opt: the parsed script arguments
        study : the complete hyperparameter search study

    Returns:
        None
    """
    trial_with_highest_HOTA = max(study.best_trials, key=lambda t: t.values[0])
    d = {opt.tracking_method: trial_with_highest_HOTA.params}
    with open(opt.tracking_config, 'w') as f:
        f.write(f'# Trial number:      {trial_with_highest_HOTA.number}\n')
        f.write(f'# HOTA, MOTA, IDF1:  {trial_with_highest_HOTA.values}\n')
        data = yaml.dump(d, f)  

    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default=WEIGHTS / 'yolov8x.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=str, default=WEIGHTS / 'osnet_x1_0_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--project', default=ROOT / 'runs' / 'evolve', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--benchmark', type=str,  default='MOT17', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str,  default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--n-trials', type=int, default=10, help='nr of trials for evolution')
    parser.add_argument('--resume', action='store_true', help='resume hparam search')
    parser.add_argument('--processes-per-device', type=int, default=2, help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    parser.add_argument('--objectives', type=str, default='HOTA,MOTA,IDF1', help='set of objective metrics: HOTA,MOTA,IDF1')
    
    opt = parser.parse_args()
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    opt.objectives = opt.objectives.split(",")

    device = []
    
    for a in opt.device.split(','):
        try:
            a = int(a)
        except ValueError:
            pass
        device.append(a)
    opt.device = device
        
    print_args(vars(opt))
    return opt


class ContinuousStudySave:
    """Helper class for saving the study after each trial. This is to avoid
       loosing partial study results if the study is stopped before finishing

    Args:
        tracking_method: the tracking method name
    Attributes:
        tracking_method: the tracking method name
    """
    def __init__(self, tracking_method):
        self.tracking_method = tracking_method
        
    def __call__(self, study, trial):
        joblib.dump(study, opt.tracking_method + "_study.pkl")

    
if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(('optuna', 'plotly', 'kaleido', 'joblib', 'pycocotools'))
    import joblib
    import optuna

    if opt.resume:
        # resume from last saved study
        study = joblib.load(opt.tracking_method + "_study.pkl")
    else:
        # A fast and elitist multiobjective genetic algorithm: NSGA-II
        # https://ieeexplore.ieee.org/document/996017
        study = optuna.create_study(directions=['maximize']*len(opt.objectives))
        # first trial with params in yaml file, evolved for MOT17
        with open(opt.tracking_config, 'r') as f:
            params = yaml.load(f, Loader=yaml.loader.SafeLoader)
            study.enqueue_trial(params[opt.tracking_config.stem])
            print(study.trials)

    continuous_study_save_cb = ContinuousStudySave(opt.tracking_method)
    study.optimize(Objective(opt), n_trials=opt.n_trials, callbacks=[continuous_study_save_cb])
        
    # write the parameters to the config file of the selected tracking method
    write_best_HOTA_params_to_config(opt, study)
    
    # save hps study, all trial results are stored here, used for resuming
    joblib.dump(study, opt.tracking_method + "_study.pkl")
    
    # plots
    save_plots(opt, study, opt.objectives)
    print_best_trial_metric_results(study, opt.objectives)

        
