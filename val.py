import os
import sys
import torch
import logging
import subprocess
from subprocess import Popen
import argparse
import git
from git import Repo
import zipfile
from pathlib import Path
import shutil
import threading


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
from track import run


def setup_evaluation(dst_val_tools_folder):
    
    # source: https://github.com/JonathonLuiten/TrackEval#official-evaluation-code
    LOGGER.info('Download official MOT evaluation repo')
    val_tools_url = "https://github.com/JonathonLuiten/TrackEval"
    try:
        Repo.clone_from(val_tools_url, dst_val_tools_folder)
    except git.exc.GitError as err:
        LOGGER.info('Eval repo already downloaded')
        
    LOGGER.info('Get ground-truth txts, meta-data and example trackers for all currently supported benchmarks')
    gt_data_url = 'https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip'
    subprocess.run(["wget", "-nc", gt_data_url, "-O", dst_val_tools_folder / 'data.zip']) # python module has no -nc nor -N flag
    if not (dst_val_tools_folder / 'data').is_dir():
        with zipfile.ZipFile(dst_val_tools_folder / 'data.zip', 'r') as zip_ref:
            zip_ref.extractall(dst_val_tools_folder)

    LOGGER.info('Download official MOT images')
    mot_gt_data_url = 'https://motchallenge.net/data/MOT16.zip'
    subprocess.run(["wget", "-nc", mot_gt_data_url, "-O", dst_val_tools_folder / 'MOT16.zip']) # python module has no -nc nor -N flag
    if not (dst_val_tools_folder / 'data' / 'MOT16').is_dir():
        with zipfile.ZipFile(dst_val_tools_folder / 'MOT16.zip', 'r') as zip_ref:
            zip_ref.extractall(dst_val_tools_folder / 'data' / 'MOT16')
        
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'mobilenetv2_x1_0_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='track/strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--benchmark', type=str,  default='MOT16', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str,  default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')

    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    
    # download eval files
    dst_val_tools_folder = ROOT / 'val_utils'
    setup_evaluation(dst_val_tools_folder)
    
    # set paths
    mot_seqs_path = dst_val_tools_folder / 'data' / opt.benchmark / opt.split
    seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if p.is_dir()]
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    MOT_results_folder = dst_val_tools_folder / 'data' / 'trackers' / 'mot_challenge' / Path(str(opt.benchmark) + '-' + str(opt.split)) / save_dir.name / 'data'
    (MOT_results_folder).mkdir(parents=True, exist_ok=True)  # make

    
    if not opt.eval_existing:

        processes = []
        nr_gpus = torch.cuda.device_count()
        
        for i, seq_path in enumerate(seq_paths):

            device = i % nr_gpus

            dst_seq_path = seq_path.parent / seq_path.parent.name
            if not dst_seq_path.is_dir():
                src_seq_path = seq_path
                shutil.move(str(src_seq_path), str(dst_seq_path))

            p = subprocess.Popen([
                "python", "track.py",\
                "--yolo-weights", "weights/crowdhuman_yolov5m.pt",\
                "--strong-sort-weights",  "weights/osnet_x1_0_dukemtmcreid.pt",\
                "--imgsz", str(1280),\
                "--classes", str(0),\
                "--name", save_dir.name,\
                "--project", opt.project,\
                "--device", str(device),\
                "--source", dst_seq_path,\
                "--exist-ok",\
                "--save-txt",\
                "--eval"
            ])
            processes.append(p)
        
        for p in processes:
            p.wait()

    results = (save_dir.parent / opt.eval_existing / 'tracks' if opt.eval_existing else save_dir / 'tracks').glob('*.txt')
    for src in results:
        if opt.eval_existing:
            dst = MOT_results_folder.parent.parent / opt.eval_existing / 'data' / Path(src.stem + '.txt')
        else:  
            dst = MOT_results_folder / Path(src.stem + '.txt')
        dst.parent.mkdir(parents=True, exist_ok=True)  # make
        shutil.copyfile(src, dst)

    # run the evaluation on the generated txts
    subprocess.run([
        "python",  dst_val_tools_folder / "scripts/run_mot_challenge.py",\
        "--BENCHMARK", "MOT16",\
        "--TRACKERS_TO_EVAL",  opt.eval_existing if opt.eval_existing else MOT_results_folder.parent.name,\
        "--SPLIT_TO_EVAL", "train",\
        "--METRICS", "HOTA", "CLEAR", "Identity",\
        "--USE_PARALLEL", "True",\
        "--NUM_PARALLEL_CORES", "4"\
    ])
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
