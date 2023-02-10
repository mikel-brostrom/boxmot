#  Yolov5_StrongSORT_OSNet, GPL-3.0 license
"""
Evaluate on the benchmark of your choice. MOT16, 17 and 20 are donwloaded and unpackaged automatically when selected.
Mimic the structure of either of these datasets to evaluate on your custom one

Usage:

    $ python3 val.py --tracking-method strongsort --benchmark MOT16
                     --tracking-method ocsort     --benchmark MOT17
                     --tracking-method ocsort     --benchmark <your-custom-dataset>
"""

import os
import sys
import torch
import subprocess
from subprocess import Popen
import argparse
import git
import re
import yaml
from git import Repo
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm

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

import numpy as np

from yolov8.ultralytics.yolo.utils import LOGGER
from yolov8.ultralytics.yolo.utils.checks import check_requirements, print_args
from yolov8.ultralytics.yolo.utils.files import increment_path

from torch.utils.tensorboard import SummaryWriter

from track import run
    

class Evaluator:
    """Evaluates a specific benchmark (MOT16, MOT17, MOT20) and split (train, val, test)
    
    This object provides interfaces to download: the official tools for MOT evaluation and the
    official MOT datasets. It also provides setup functionality to select which devices to run
    sequences on and configuration to enable evaluation on different MOT datasets.

    Args:
        opt: the parsed script arguments

    Attributes:
        opt: the parsed script arguments

    """
    def __init__(self, opts):  
        self.opt = opts
        

    def download_mot_eval_tools(self, val_tools_path):
        """Download officail evaluation tools for MOT metrics

        Args:
            val_tools_path (pathlib.Path): path to the val tool folder destination

        Returns:
            None
        """
        # source: https://github.com/JonathonLuiten/TrackEval#official-evaluation-code
        val_tools_url = "https://github.com/JonathonLuiten/TrackEval"
        try:
            Repo.clone_from(val_tools_url, val_tools_path)
            LOGGER.info('Official MOT evaluation repo downloaded')
        except git.exc.GitError as err:
            LOGGER.info('Eval repo already downloaded')


    def download_mot_dataset(self, val_tools_path, benchmark):
        """Download specific MOT dataset and unpack it

        Args:
            val_tools_path (pathlib.Path): path to destination folder of the downloaded MOT benchmark zip
            benchmark (str): the MOT benchmark to download

        Returns:
            None
        """

        # download and unzip the rest of MOTXX
        url = 'https://motchallenge.net/data/' + benchmark + '.zip'
        zip_dst = val_tools_path / (benchmark + '.zip')
        if not (val_tools_path / 'data' / benchmark).exists():
            os.system(f"curl -# -L {url} -o {zip_dst} -# --retry 3 -C -")
            LOGGER.info(f'{benchmark}.zip downloaded sucessfully')
        
            try:
                with zipfile.ZipFile((val_tools_path / (benchmark + '.zip')), 'r') as zip_file:
                    if opt.benchmark == 'MOT16':
                        # extract only if file has not already been extracted
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            if os.path.exists(val_tools_path / 'data' / 'MOT16' / member) or os.path.isfile(val_tools_path / 'data' / 'MOT16' / member):
                                pass
                            else:
                                zip_file.extract(member, val_tools_path / 'data' / 'MOT16')
                    else:
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            if os.path.exists(val_tools_path / 'data' / member) or os.path.isfile(val_tools_path / 'data' / member):
                                pass
                            else:
                                zip_file.extract(member, val_tools_path / 'data')
                LOGGER.info(f'{benchmark}.zip unzipped successfully')
            except Exception as e:
                print(f'{benchmark}.zip is corrupted. Try deleting the file and run the script again')
                sys.exit()
    
    def eval_setup(self, opt, val_tools_path):
        """Download specific MOT dataset and unpack it

        Args:
            opt: the parsed script arguments
            val_tools_path (pathlib.Path): path to destination folder of the downloaded MOT benchmark zip

        Returns:
            [Path], Path, Path: benchmark sequence paths, original tracking results destination, eval tracking result destination
        """
        
        # set paths
        gt_folder = val_tools_path / 'data' / self.opt.benchmark / self.opt.split
        mot_seqs_path = val_tools_path / 'data' / opt.benchmark / opt.split
        if opt.benchmark == 'MOT17':
            # each sequences is present 3 times, one for each detector
            # (DPM, FRCNN, SDP). Keep only sequences from  one of them
            seq_paths = sorted([str(p / 'img1') for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()])
            seq_paths = [Path(p) for p in seq_paths if 'FRCNN' in p]
        elif opt.benchmark == 'MOT16' or opt.benchmark == 'MOT20':
            # this is not the case for MOT16, MOT20 or your custom dataset
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]
        elif opt.benchmark == 'MOT17-mini':
            mot_seqs_path = Path('./assets') / self.opt.benchmark / self.opt.split
            gt_folder = Path('./assets') / self.opt.benchmark / self.opt.split
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]
        
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        MOT_results_folder = val_tools_path / 'data' / 'trackers' / 'mot_challenge' / opt.benchmark / save_dir.name / 'data'
        (MOT_results_folder).mkdir(parents=True, exist_ok=True)  # make
        return seq_paths, save_dir, MOT_results_folder, gt_folder


    def device_setup(self, opt, seq_paths):
        """Selects which devices (cuda:N, cpu) to run each sequence on

        Args:
            opt: the parsed script arguments
            seq_paths (list of Path): list of paths to each sequence in the benchmark to be evaluated

        Returns:
            list of str
        """
        # extend devices to as many sequences are available
        if any(isinstance(i,int) for i in opt.device) and len(opt.device) > 1:
            devices = opt.device
            for a in range(0, len(opt.device) % len(seq_paths)):
                opt.device.extend(devices)
            opt.device = opt.device[:len(seq_paths)]
        free_devices = opt.device * opt.processes_per_device
        return free_devices
    
    def eval(self, opt, seq_paths, save_dir, MOT_results_folder, val_tools_path, gt_folder, free_devices):
        """Benchmark evaluation
        
        Runns each benchmark sequence on the selected device configuration and moves the results to
        a unique eval folder

        Args:
            opt: the parsed script arguments
            seq_paths ([Path]): path to sequence folders in benchmark
            save_dir (Path): original tracking result destination
            MOT_results_folder (Path): evaluation trackinf result destination
            val_tools_path (pathlib.Path): path to destination folder of the downloaded MOT benchmark zip
            free_devices: [str]

        Returns:
            (str): the complete evaluation results generated by "scripts/run_mot_challenge.py"
        """
        
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
        d = [seq_path.parent.name for seq_path in seq_paths]
        p = subprocess.run(
            args=[
                sys.executable,  val_tools_path / 'scripts' / 'run_mot_challenge.py',
                "--GT_FOLDER", gt_folder,
                "--BENCHMARK", self.opt.benchmark,
                "--TRACKERS_TO_EVAL",  self.opt.eval_existing if self.opt.eval_existing else self.opt.benchmark,
                "--SPLIT_TO_EVAL", "train",
                "--METRICS", "HOTA", "CLEAR", "Identity",
                "--USE_PARALLEL", "True",
                "--TRACKER_SUB_FOLDER", str(Path(*Path(MOT_results_folder).parts[-2:])),
                "--NUM_PARALLEL_CORES", "4",
                "--SKIP_SPLIT_FOL", "True",
                "--SEQ_INFO"] + d,
            universal_newlines=True,
            stdout=subprocess.PIPE
        )
        
        print(p.stdout)
        
        # save MOT results in txt 
        with open(save_dir / 'MOT_results.txt', 'w') as f:
            f.write(p.stdout)
        # copy tracking method config to exp folder
        shutil.copyfile(opt.tracking_config, save_dir / opt.tracking_config.name)

        return p.stdout
    
    def parse_mot_results(self, results):
        """Extract the COMBINED HOTA, MOTA, IDF1 from the results generate by the
           run_mot_challenge.py script.

        Args:
            str: mot_results

        Returns:
            (dict): {'HOTA': x, 'MOTA':y, 'IDF1':z}
        """
        combined_results = results.split('COMBINED')[2:-1]
        # robust way of getting first ints/float in string
        combined_results = [float(re.findall("[-+]?(?:\d*\.*\d+)", f)[0]) for f in combined_results]
        # pack everything in dict
        combined_results = {key: value for key, value in zip(['HOTA', 'MOTA', 'IDF1'], combined_results)}
        return combined_results
    
    
    def run(self, opt):
        """Download all needed resources for evaluation, setup and evaluate
        
        Downloads evaluation tools and MOT dataset. Setup to make evaluation possible on different benchmarks
        and with custom devices configuration.

        Args:
            opt: the parsed script arguments

        Returns:
            (str): the complete evaluation results generated by "scripts/run_mot_challenge.py"
        """
        e = Evaluator(opt)
        val_tools_path = ROOT / 'val_utils'
        e.download_mot_eval_tools(val_tools_path)
        if any(opt.benchmark == s for s in ['MOT16', 'MOT17', 'MOT20']):
            e.download_mot_dataset(val_tools_path, opt.benchmark)
        seq_paths, save_dir, MOT_results_folder, gt_folder = e.eval_setup(opt, val_tools_path)
        free_devices = e.device_setup(opt, seq_paths)
        results = e.eval(opt, seq_paths, save_dir, MOT_results_folder, val_tools_path, gt_folder, free_devices)
        # extract main metric results: HOTA, MOTA, IDF1
        combined_results = self.parse_mot_results(results)
        
        # log them with tensorboard
        writer = SummaryWriter(save_dir)
        writer.add_scalar('HOTA', combined_results['HOTA'])
        writer.add_scalar('MOTA', combined_results['MOTA'])
        writer.add_scalar('IDF1', combined_results['IDF1'])
        
        return combined_results
        
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=str, default=WEIGHTS / 'osnet_x1_0_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--project', default=ROOT / 'runs' / 'val', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--benchmark', type=str,  default='MOT17', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str,  default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', type=str, default='', help='evaluate existing tracker results under mot_callenge/MOTXX-YY/...')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--processes-per-device', type=int, default=2, help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    
    opt = parser.parse_args()
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    with open(opt.tracking_config, 'r') as f:
        params = yaml.load(f, Loader=yaml.loader.SafeLoader)
        opt.conf_thres = params[opt.tracking_method]['conf_thres']    

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

        
if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    e = Evaluator(opt)
    e.run(opt)
