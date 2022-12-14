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


def download_official_mot_eval_tool(val_tools_target_location):
    # source: https://github.com/JonathonLuiten/TrackEval#official-evaluation-code
    val_tools_url = "https://github.com/JonathonLuiten/TrackEval"
    try:
        Repo.clone_from(val_tools_url, val_tools_target_location)
        LOGGER.info('Official MOT evaluation repo downloaded')
    except git.exc.GitError as err:
        LOGGER.info('Eval repo already downloaded')
        
def download_mot_dataset(val_tools_target_location, benchmark):
    
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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default=WEIGHTS / 'crowdhuman_yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=str, default=WEIGHTS / 'osnet_x1_0_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--benchmark', type=str,  default='MOT17', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str,  default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--processes-per-device', type=int, default=2, help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    
    opt = parser.parse_args()
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


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    
    # download eval files
    val_tools_target_location = ROOT / 'val_utils'
    download_official_mot_eval_tool(val_tools_target_location)
    
    if any(opt.benchmark == s for s in ['MOT16', 'MOT17', 'MOT20']):
        download_mot_dataset(val_tools_target_location, opt.benchmark)
    
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

    # extend devices to as many sequences are available
    if any(isinstance(i,int) for i in opt.device) and len(opt.device) > 1:
        devices = opt.device
        for a in range(0, len(opt.device) % len(seq_paths)):
            opt.device.extend(devices)
        opt.device = opt.device[:len(seq_paths)]
 
    if not opt.eval_existing:
        processes = []
        free_devices = opt.device * opt.processes_per_device
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
                "--yolo-weights", opt.yolo_weights,
                "--reid-weights",  opt.reid_weights,
                "--tracking-method", opt.tracking_method,
                "--conf-thres", str(opt.conf_thres),
                "--imgsz", str(opt.imgsz[0]),
                "--classes", str(0),
                "--name", save_dir.name,
                "--project", opt.project,
                "--device", str(tracking_subprocess_device),
                "--source", dst_seq_path,
                "--exist-ok",
                "--save-txt",
            ])
            processes.append(p)
        
        for p in processes:
            p.wait()
            
    print_args(vars(opt))

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
        sys.executable,  val_tools_target_location / "scripts/run_mot_challenge.py",
        "--BENCHMARK", opt.benchmark,
        "--TRACKERS_TO_EVAL",  opt.eval_existing if opt.eval_existing else MOT_results_folder.parent.name,
        "--SPLIT_TO_EVAL", "train",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "True",
        "--NUM_PARALLEL_CORES", "4"
    ])
    

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
