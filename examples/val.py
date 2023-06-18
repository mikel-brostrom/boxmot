#  Yolov5_StrongSORT_OSNet, GPL-3.0 license
"""
Evaluate on the benchmark of your choice. MOT16, 17 and 20 are donwloaded and unpackaged automatically when selected.
Mimic the structure of either of these datasets to evaluate on your custom one

Usage:

    $ python3 examples/val.py --benchmark MOT16
                              --benchmark MOT17
                              --benchmark <your-custom-dataset>
                              --benchmark VEHICLE    --split test --conf 0.3      --classes 0 2 7 --eval-existing --name exp227
                              --benchmark MOT17mini --conf 0.3   --eval-existing --name exp231
"""

import os
import sys
import torch
import glob
import subprocess
from subprocess import Popen
import argparse
import git
import re
import yaml
from collections import OrderedDict
from git import Repo
import zipfile
from pathlib import Path
import shutil
import motmetrics as mm
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from boxmot.utils import logger as LOGGER
from ultralytics.yolo.utils.checks import check_requirements, print_args
from ultralytics.yolo.utils.files import increment_path

from boxmot.utils import ROOT, WEIGHTS, EXAMPLES
from track import run
import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes


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
        self.val_tools_path = EXAMPLES / 'val_utils'
        self.seq_paths = None
        self.save_dir = None
        self.gt_folder = None

    def download_mot_dataset(self, benchmark):
        """Download specific MOT dataset and unpack it
        Args:
            val_tools_path (pathlib.Path): path to destination folder of the downloaded MOT benchmark zip
            benchmark (str): the MOT benchmark to download
        Returns:
            None
        """
        url = 'https://motchallenge.net/data/' + benchmark + '.zip'
        zip_dst = self.val_tools_path / (benchmark + '.zip')
        if not (self.val_tools_path / 'data' / benchmark).exists():
            os.system(f"curl -# -L {url} -o {zip_dst} -# --retry 3 -C -")
            LOGGER.info(f'{benchmark}.zip downloaded sucessfully')

            try:
                with zipfile.ZipFile((self.val_tools_path / (benchmark + '.zip')), 'r') as zip_file:
                    if self.opt.benchmark == 'MOT16':
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            member_path = self.val_tools_path / 'data' / 'MOT16' / member
                            if not member_path.exists() and not member_path.is_file():
                                zip_file.extract(member, self.val_tools_path / 'data' / 'MOT16')
                    else:
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            member_path = self.val_tools_path / 'data' / member
                            if not member_path.exists() and not member_path.is_file():
                                zip_file.extract(member, self.val_tools_path / 'data')
                LOGGER.info(f'{benchmark}.zip unzipped successfully')
            except Exception as e:
                LOGGER.error(f'{benchmark}.zip is corrupted. Try deleting the file and run the script again')
                sys.exit()

    def eval_setup(self):
        """Download specific MOT dataset and unpack it

        Args:
            opt: the parsed script arguments
            val_tools_path (pathlib.Path): path to destination folder of the downloaded MOT benchmark zip

        Returns:
            [Path], Path, Path: benchmark sequence paths, original tracking results destination, eval tracking result destination
        """

        # set paths
        gt_folder = self.val_tools_path / 'data' / self.opt.benchmark / self.opt.split
        mot_seqs_path = self.val_tools_path / 'data' / self.opt.benchmark / self.opt.split
        if self.opt.benchmark == 'MOT17':
            # each sequences is present 3 times, one for each detector
            # (DPM, FRCNN, SDP). Keep only sequences from  one of them
            seq_paths = sorted([str(p / 'img1') for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()])
            seq_paths = [Path(p) for p in seq_paths if 'FRCNN' in p]
        elif self.opt.benchmark == 'MOT17mini':
            mot_seqs_path = ROOT / 'assets' / self.opt.benchmark / self.opt.split
            gt_folder = ROOT / 'assets' / self.opt.benchmark / self.opt.split
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]
        else:
            # this is not the case for MOT16, MOT20 or your custom dataset
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]

        if self.opt.eval_existing and (Path(self.opt.project) / self.opt.name).exists():
            save_dir = Path(self.opt.project) / opt.name
            if not (Path(self.opt.project) / self.opt.name).exists():
                LOGGER.error(f'{save_dir} does not exist')
        else:
            save_dir = increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok)

        self.seq_paths = seq_paths
        self.save_dir = save_dir
        self.gt_folder = gt_folder

    def device_setup(self, opt, seq_paths):
        """Selects which devices (cuda:N, cpu) to run each sequence on

        Args:
            opt: the parsed script arguments
            seq_paths (list of Path): list of paths to each sequence in the benchmark to be evaluated

        Returns:
            list of str
        """
        # extend devices to as many sequences are available
        if any(isinstance(i, int) for i in opt.device) and len(opt.device) > 1:
            devices = opt.device
            for a in range(0, len(opt.device) % len(seq_paths)):
                opt.device.extend(devices)
            opt.device = opt.device[:len(seq_paths)]
        free_devices = opt.device * opt.processes_per_device
        return free_devices
    
    def evaluate(self):
        
        gttxtfiles = list(self.gt_folder.glob('*/gt/gt.txt'))
        # get sequences in the right order; strip letters, only sort by numbers
        gttxtfiles.sort(key=lambda x: re.sub(r'[^0-9]*', "", str(x)))
        tstxtfiles = [f for f in (self.save_dir / 'labels').glob('*.txt')]
        
        LOGGER.info(f"Found {len(gttxtfiles)} groundtruths and {len(tstxtfiles)} test files.")
        if len(tstxtfiles) != len(gttxtfiles):
            LOGGER.warning(f"The number of gt files and tracking results files differ.")
            LOGGER.warning(f"Proceeding with the calculation of partial results")
        LOGGER.info(f"Available LAP solvers {str(mm.lap.available_solvers)}")
        LOGGER.info(f"Default LAP solver \'{mm.lap.default_solver}\'")
        LOGGER.info(f'Loading files.')
        
        # load all data for all sequences
        gt = OrderedDict([
            (Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot16', min_confidence=1)) 
            for f in gttxtfiles
        ])
        ts = OrderedDict([
            (os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt='mot16'))
            for f in tstxtfiles
        ])
        
        # for each of the predicted classes  
        for c in self.opt.classes:
            
            gt_temp = {}
            ts_temp = {}
        
            # for each of the sequences in the dataset 
            for seq_name in gt.keys():

                # in official MOT datasets, cls follows one-based indexing
                if self.opt.benchmark in ['MOT16', 'MOT17', 'MOT17mini', 'MOT20']:
                    gt_temp[seq_name] = gt[seq_name].loc[gt[seq_name]['ClassId'] == int(c) + 1]
                else:
                    gt_temp[seq_name] = gt[seq_name].loc[gt[seq_name]['ClassId'] == int(c)]
                ts_temp[seq_name] = ts[seq_name].loc[ts[seq_name]['ClassId'] == int(c)]

            LOGGER.info(f'Running metrics on: {list(gt.keys())} for class {c}')
            mh = mm.metrics.create()
            accs, names = compare_dataframes(gt_temp, ts_temp)

            metrics = list(mm.metrics.motchallenge_metrics)

            summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
            LOGGER.success(f"\n{mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)}")
          
        LOGGER.info(f'Running metrics on: {list(gt.keys())} for ALL classes')  
        
        accs, names = compare_dataframes(gt, ts)
        metrics = list(mm.metrics.motchallenge_metrics)
        summary = mh.compute_many(accs, names=names, metrics=['mota', 'idf1'], generate_overall=True)
        strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        
        LOGGER.success(f"\n{strsummary}")

        results = {
            'MOTA': summary.loc['OVERALL', 'mota'],
            'IDF1': summary.loc['OVERALL', 'idf1']
        }
        
        return results


    def generate_tracks(self):
        """Benchmark evaluation
        
        Runns each benchmark sequence on the selected device configuration and moves the results to
        a unique eval folder

        Args:
            seq_paths ([Path]): path to sequence folders in benchmark
            save_dir (Path): original tracking result destination

        Returns:
            (str): the complete evaluation results generated by "scripts/run_mot_challenge.py"
        """
        
        free_devices = self.device_setup(self.opt, self.seq_paths)

        if not self.opt.eval_existing:
            processes = []

            busy_devices = []
            for i, seq_path in enumerate(self.seq_paths):
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
                
                LOGGER.info(f"Staring evaluation process on {dst_seq_path}")
                p = subprocess.Popen(
                    args=[
                        sys.executable, str(EXAMPLES / "track.py"),
                        "--yolo-model", self.opt.yolo_model,
                        "--reid-model", self.opt.reid_model,
                        "--tracking-method", self.opt.tracking_method,
                        "--conf", str(self.opt.conf),
                        "--imgsz", str(self.opt.imgsz[0]),
                        "--classes", *self.opt.classes,
                        "--name", self.save_dir.name,
                        "--save-txt",
                        "--project", self.opt.project,
                        "--device", str(tracking_subprocess_device),
                        "--source", dst_seq_path,
                        "--exist-ok",
                        "--save",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                processes.append(p)
                # Wait for the subprocess to complete and capture output
                stdout, stderr = p.communicate()
                
                # Check the return code of the subprocess
                if p.returncode != 0:
                    LOGGER.error(stderr)
                    LOGGER.error(stdout)
                    sys.exit(1)
                else:
                    LOGGER.success(f"{dst_seq_path} evaluation succeeded")

            for p in processes:
                p.wait()

        print_args(vars(self.opt))


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
        
        # download supported datasets
        if opt.benchmark in ['MOT16', 'MOT17', 'MOT20']:
            e.download_mot_dataset(opt.benchmark)
            
        # generate necessary paths
        e.eval_setup()
        # generate txt files for each sequence
        e.generate_tracks()
        # evaluate these sequences
        results = e.evaluate()

        # log MOTA and IDF1 on tensorboard
        writer = SummaryWriter(self.save_dir)
        writer.add_scalar('MOTA', results['MOTA'])
        writer.add_scalar('IDF1', results['IDF1'])

        return results


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=str, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=str, default=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='strongsort, ocsort')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--classes', nargs='+', type=str, default=['0'], help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=EXAMPLES / 'runs' / 'val', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--benchmark', type=str, default='MOT17mini', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str, default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', action='store_true', help='evaluate existing results under project/name/labels')
    parser.add_argument('--conf', type=float, default=0.45, help='confidence threshold')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--processes-per-device', type=int, default=2,
                        help='how many subprocesses can be invoked per GPU (to manage memory consumption)')

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


if __name__ == "__main__":
    opt = parse_opt()
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    e = Evaluator(opt)
    e.run(opt)
