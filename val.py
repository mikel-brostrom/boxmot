import os
import sys
import torch
import logging
import subprocess
from subprocess import Popen
import argparse
from io import StringIO
import git
import joblib
import yaml
import optuna
import re
import pandas as pd
from git import Repo
import zipfile
from pathlib import Path
import shutil
import threading
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.utils.general import LOGGER, check_requirements, print_args, increment_path
from yolov5.utils.torch_utils import select_device
from track import run
    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default=WEIGHTS / 'crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=str, default=WEIGHTS / 'osnet_x1_0_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--benchmark', type=str,  default='MOT17', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str,  default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--evolve', action='store_true', help='evolve hparams of the trackers')
    parser.add_argument('--n-trials', type=int, default=10, help='nr of trials for evolution')
    parser.add_argument('--resume', action='store_true', help='resume hparam search')
    parser.add_argument('--processes-per-device', type=int, default=2, help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    
    opt = parser.parse_args()
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')

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


class Evaluator:
    def __init__(self, opts):  
        self.opt = opts
        
    def download_official_mot_eval_tool(self, val_tools_target_location):
        # source: https://github.com/JonathonLuiten/TrackEval#official-evaluation-code
        val_tools_url = "https://github.com/JonathonLuiten/TrackEval"
        try:
            Repo.clone_from(val_tools_url, val_tools_target_location)
            LOGGER.info('Official MOT evaluation repo downloaded')
        except git.exc.GitError as err:
            LOGGER.info('Eval repo already downloaded')
            
    def download_mot_dataset(self, val_tools_target_location, benchmark):
        
        # download and unzip ground truth
        url = 'https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip'
        zip_dst = val_tools_target_location / 'data.zip'
        
        # download and unzip if not already unzipped
        if not zip_dst.with_suffix('').exists():
            os.system(f"curl -# -L {url} -o {zip_dst} -# --retry 3 -C -")
            LOGGER.info(f'data.zip downloaded sucessfully')
        
            try:
                with zipfile.ZipFile(val_tools_target_location / 'data.zip', 'r') as zip_file:
                    for member in tqdm(zip_file.namelist(), desc=f'Extracting MOT ground truth'):
                        # extract only if file has not already been extracted
                        if os.path.exists(val_tools_target_location / member) or os.path.isfile(val_tools_target_location / member):
                            pass
                        else:
                            zip_file.extract(member, val_tools_target_location)
                LOGGER.info(f'data.zip unzipped sucessfully')
            except Exception as e:
                print('data.zip is corrupted. Try deleting the file and run the script again')
                sys.exit()

        # download and unzip the rest of MOTXX
        url = 'https://motchallenge.net/data/' + benchmark + '.zip'
        zip_dst = val_tools_target_location / (benchmark + '.zip')
        if not (val_tools_target_location / 'data' / benchmark).exists():
            os.system(f"curl -# -L {url} -o {zip_dst} -# --retry 3 -C -")
            LOGGER.info(f'{benchmark}.zip downloaded sucessfully')
        
            try:
                with zipfile.ZipFile((val_tools_target_location / (benchmark + '.zip')), 'r') as zip_file:
                    if opt.benchmark == 'MOT16':
                        # extract only if file has not already been extracted
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            if os.path.exists(val_tools_target_location / 'data' / 'MOT16' / member) or os.path.isfile(val_tools_target_location / 'data' / 'MOT16' / member):
                                pass
                            else:
                                zip_file.extract(member, val_tools_target_location / 'data' / 'MOT16')
                    else:
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            if os.path.exists(val_tools_target_location / 'data' / member) or os.path.isfile(val_tools_target_location / 'data' / member):
                                pass
                            else:
                                zip_file.extract(member, val_tools_target_location / 'data')
                LOGGER.info(f'{benchmark}.zip unzipped successfully')
            except Exception as e:
                print(f'{benchmark}.zip is corrupted. Try deleting the file and run the script again')
                sys.exit()
    
    def eval_setup(self, opt, val_tools_target_location):
        
        # set paths
        mot_seqs_path = val_tools_target_location / 'data' / opt.benchmark / opt.split
        
        if opt.benchmark == 'MOT17':
            # each sequences is present 3 times, one for each detector
            # (DPM, FRCNN, SDP). Keep only sequences from  one of them
            seq_paths = sorted([str(p / 'img1') for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()])
            seq_paths = [Path(p) for p in seq_paths if 'FRCNN' in p]
            with open(val_tools_target_location / "data/gt/mot_challenge/seqmaps/MOT17-train.txt", "r") as f:  # 
                lines = f.readlines()
            # overwrite MOT17 evaluation sequences to evaluate so that they are not duplicated
            with open(val_tools_target_location / "data/gt/mot_challenge/seqmaps/MOT17-train.txt", "w") as f:
                for line in seq_paths:
                    f.write(str(line.parent.stem) + '\n')
        else:
            # this is not the case for MOT16, MOT20 or your custom dataset
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]
        
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        MOT_results_folder = val_tools_target_location / 'data' / 'trackers' / 'mot_challenge' / Path(str(opt.benchmark) + '-' + str(opt.split)) / save_dir.name / 'data'
        (MOT_results_folder).mkdir(parents=True, exist_ok=True)  # make 
        return seq_paths, save_dir, MOT_results_folder


    def device_setup(self, opt, seq_paths):
        # extend devices to as many sequences are available
        if any(isinstance(i,int) for i in opt.device) and len(opt.device) > 1:
            devices = opt.device
            for a in range(0, len(opt.device) % len(seq_paths)):
                opt.device.extend(devices)
            opt.device = opt.device[:len(seq_paths)]
        free_devices = opt.device * opt.processes_per_device
        return free_devices
    
    def eval(self, opt, seq_paths, save_dir, MOT_results_folder, val_tools_target_location, free_devices):
        
        if not self.opt.eval_existing:
            processes = []
            
            busy_devices = []
            for i, seq_path in enumerate(seq_paths):
                # spawn one subprocess per GPU in increasing order.
                # When max devices are reached start at 0 again
                if i > 0 and len(free_devices) == 0:
                    if len(processes) == 0:
                        raise IndexError("No active processes and no devices available.")
                    
                    # Wait for oldest process to finish so we can get a free device
                    processes.pop(0).wait()
                    free_devices.append(busy_devices.pop(0))
                
                tracking_subprocess_device = free_devices.pop(0)
                busy_devices.append(tracking_subprocess_device)
            
                dst_seq_path = seq_path.parent / seq_path.parent.name
                if not dst_seq_path.is_dir():
                    src_seq_path = seq_path
                    shutil.move(str(src_seq_path), str(dst_seq_path))   
                
                p = subprocess.Popen([
                    sys.executable, "track.py",
                    "--yolo-weights", self.opt.yolo_weights,
                    "--reid-weights",  self.opt.reid_weights,
                    "--tracking-method", self.opt.tracking_method,
                    "--conf-thres", str(self.opt.conf_thres),
                    "--imgsz", str(self.opt.imgsz[0]),
                    "--classes", str(0),
                    "--name", save_dir.name,
                    "--project", self.opt.project,
                    "--device", str(tracking_subprocess_device),
                    "--source", dst_seq_path,
                    "--exist-ok",
                    "--save-txt",
                ])
                processes.append(p)
            
            for p in processes:
                p.wait()
                
        print_args(vars(self.opt))

        results = (save_dir.parent / self.opt.eval_existing / 'tracks' if self.opt.eval_existing else save_dir / 'tracks').glob('*.txt')
        for src in results:
            if self.opt.eval_existing:
                dst = MOT_results_folder.parent.parent / self.opt.eval_existing / 'data' / Path(src.stem + '.txt')
            else:  
                dst = MOT_results_folder / Path(src.stem + '.txt')
            dst.parent.mkdir(parents=True, exist_ok=True)  # make
            shutil.copyfile(src, dst)

        # run the evaluation on the generated txts
        p = subprocess.run(
            args=[
                sys.executable,  val_tools_target_location / "scripts/run_mot_challenge.py",
                "--BENCHMARK", self.opt.benchmark,
                "--TRACKERS_TO_EVAL",  self.opt.eval_existing if self.opt.eval_existing else MOT_results_folder.parent.name,
                "--SPLIT_TO_EVAL", "train",
                "--METRICS", "HOTA", "CLEAR", "Identity",
                "--USE_PARALLEL", "True",
                "--NUM_PARALLEL_CORES", "4"
            ],
            universal_newlines=True,
            stdout=subprocess.PIPE
        )
        print(p.stdout)
        return p.stdout

    
    def run(self, opt):
        e = Evaluator(opt)
        val_tools_target_location = ROOT / 'val_utils'
        e.download_official_mot_eval_tool(val_tools_target_location)
        if any(opt.benchmark == s for s in ['MOT16', 'MOT17', 'MOT20']):
            e.download_mot_dataset(val_tools_target_location, opt.benchmark)
        seq_paths, save_dir, MOT_results_folder = e.eval_setup(opt, val_tools_target_location)
        free_devices = e.device_setup(opt, seq_paths)
        return e.eval(opt, seq_paths, save_dir, MOT_results_folder, val_tools_target_location, free_devices)


class Objective(Evaluator):
    def __init__(self, opts, evaluator):  
        self.opt = opts
        self.evaluator = evaluator
                
    def get_new_config(self, trial):
        
        d = {}
                
        if self.opt.tracking_method == 'strongsort':
            
            self.opt.conf_thres = trial.suggest_float("conf_thres", 0.35, 0.55)
            iou_thresh = trial.suggest_float("iou_thresh", 0.1, 0.4)
            ecc = trial.suggest_categorical("ecc", [True, False])
            ema_alpha = trial.suggest_float("ema_alpha", 0.7, 0.95)
            max_dist = trial.suggest_float("max_dist", 0.1, 0.4)
            max_iou_dist = trial.suggest_float("max_iou_dist", 0.5, 0.9)
            max_age = trial.suggest_int("max_age", 10, 200, step=10)
            n_init = trial.suggest_int("n_init", 1, 3, step=1)

            d['STRONGSORT'] = \
                {
                    'ECC': ecc,
                    'MC_LAMBDA': 0.995,
                    'EMA_ALPHA': ema_alpha,
                    'MAX_DIST':  max_dist,
                    'MAX_IOU_DISTANCE': max_iou_dist,
                    'MAX_UNMATCHED_PREDS': 0,
                    'MAX_AGE': max_age,
                    'N_INIT': n_init,
                    'NN_BUDGET': 100
                }
                
        elif self.opt.tracking_method == 'bytetrack':
            
            self.opt.track_thres = trial.suggest_float("track_thres", 0.35, 0.55)
            track_buffer = trial.suggest_int("track_buffer", 10, 60, step=10)  
            match_thresh = trial.suggest_float("match_thresh", 0.7, 0.9)
            
            d['BYTETRACK'] = \
                {
                    'TRACK_THRESH': self.opt.conf_thres,
                    'MATCH_THRESH': match_thresh,
                    'TRACK_BUFFER': track_buffer,
                    'FRAME_RATE': 30
                }
                
        elif self.opt.tracking_method == 'ocsort':
            
            self.opt.conf_thres = trial.suggest_float("conf_thres", 0.35, 0.55)
            max_age = trial.suggest_int("max_age", 10, 60, step=10)
            min_hits = trial.suggest_int("min_hits", 1, 5, step=1)
            iou_thresh = trial.suggest_float("iou_thresh", 0.1, 0.4)
            delta_t = trial.suggest_int("delta_t", 1, 5, step=1)
            asso_func = trial.suggest_categorical("asso_func", ['iou', 'giou'])
            inertia = trial.suggest_float("inertia", 0.1, 0.4)
            use_byte = trial.suggest_categorical("use_byte", [True, False])
            
            d['OCSORT'] = \
                {
                    'DET_THRESH': self.opt.conf_thres,
                    'MAX_AGE': max_age,
                    'MIN_HITS': min_hits,
                    'IOU_THRESH': iou_thresh,
                    'DELTA_T': delta_t,
                    'ASSO_FUNC': asso_func,
                    'INERTIA': inertia,
                    'USE_BYTE': use_byte,
                }
                
        with open(self.opt.tracking_config, 'w') as f:
            data = yaml.dump(d, f)   
    

    def __call__(self, trial):
        
        # Calculate an objective value by using the extra arguments.
        self.get_new_config(trial)
        
        # download eval files
        val_tools_target_location = ROOT / 'val_utils'
        results= self.run(opt)
        
        # get HOTA, MOTA, IDF1 COMBINED results
        combined_results = results.split('COMBINED')[2:-1]
        # robust way of getting first ints/float in string
        combined_results = [float(re.findall("[-+]?(?:\d*\.*\d+)", f)[0]) for f in combined_results]
        # pack everything in dict
        combined_results = {key: value for key, value in zip(['HOTA', 'MOTA', 'IDF1'], combined_results)}
        return combined_results['HOTA'], combined_results['MOTA'], combined_results['IDF1']
    

def print_best_trial_metric_results(study):
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")
    trial_with_highest_HOTA = max(study.best_trials, key=lambda t: t.values[0])
    print(f"Trial with highest HOTA: ")
    print(f"\tnumber: {trial_with_highest_HOTA.number}")
    print(f"\tparams: {trial_with_highest_HOTA.params}")
    print(f"\tvalues: {trial_with_highest_HOTA.values}")
    trial_with_highest_MOTA = max(study.best_trials, key=lambda t: t.values[1])
    print(f"Trial with highest MOTA: ")
    print(f"\tnumber: {trial_with_highest_MOTA.number}")
    print(f"\tparams: {trial_with_highest_MOTA.params}")
    print(f"\tvalues: {trial_with_highest_MOTA.values}")
    trial_with_highest_IDF1 = max(study.best_trials, key=lambda t: t.values[2])
    print(f"Trial with highest IDF1: ")
    print(f"\tnumber: {trial_with_highest_IDF1.number}")
    print(f"\tparams: {trial_with_highest_IDF1.params}")
    print(f"\tvalues: {trial_with_highest_IDF1.values}")
    
    
def save_plots(study, opt):
    fig = optuna.visualization.plot_pareto_front(study, target_names=["HOTA", "MOTA", "IDF1"])
    fig.write_html("pareto_front_" + opt.tracking_method + ".html")
    if not opt.n_trials <= 1:  # more than one trial needed for parameter importance 
        fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[0], target_name="HOTA")
        fig.write_html("HOTA_param_importances_" + opt.tracking_method + ".html")
        fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[1], target_name="MOTA")
        fig.write_html("MOTA_param_importances_" + opt.tracking_method + ".html")
        fig = optuna.visualization.plot_param_importances(study, target=lambda t: t.values[2], target_name="IDF1")
        fig.write_html("IDF1_param_importances_" + opt.tracking_method + ".html")

        
if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    e = Evaluator(opt)
    if opt.evolve == False:
        e.run(opt)
    else:
        objective_num = 3
        if opt.resume:
            # resume from last saved study
            study = joblib.load(opt.tracking_method + "_study.pkl")
        else:
            # A fast and elitist multiobjective genetic algorithm: NSGA-II
            # https://ieeexplore.ieee.org/document/996017
            study = optuna.create_study(directions=['maximize']*objective_num)

        study.optimize(Objective(opt, e), n_trials=opt.n_trials)
        
        # save hps study, all trial results are stored here, used for resuming
        joblib.dump(study, opt.tracking_method + "_study.pkl")
        
        save_plots(study, opt)
        print_best_trial_metric_results(study)
            