# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

"""
Evaluate on the benchmark of your choice. MOT16, 17 and 20 are donwloaded and unpackaged automatically when selected.
Mimic the structure of either of these datasets to evaluate on your custom one

Usage:

    $ python3 val.py --tracking-method strongsort --benchmark MOT16
                     --tracking-method ocsort     --benchmark MOT17
                     --tracking-method ocsort     --benchmark <your-custom-dataset>
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from boxmot.utils.checks import TestRequirements

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

import git
from git import Repo
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from ultralytics.utils.checks import check_requirements, print_args
from ultralytics.utils.files import increment_path

from boxmot.utils import EXAMPLES, ROOT, WEIGHTS
from boxmot.utils import logger as LOGGER


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
        val_tools_url = "https://github.com/JonathonLuiten/TrackEval"
        try:
            Repo.clone_from(val_tools_url, val_tools_path)
            LOGGER.info('Official MOT evaluation repo downloaded')
        except git.exc.GitError as err:
            LOGGER.info(f'Eval repo already downloaded {err}')

    def download_mot_dataset(self, val_tools_path, benchmark):
        """Download specific MOT dataset and unpack it
        Args:
            val_tools_path (pathlib.Path): path to destination folder of the downloaded MOT benchmark zip
            benchmark (str): the MOT benchmark to download
        Returns:
            None
        """
        url = 'https://motchallenge.net/data/' + benchmark + '.zip'
        zip_dst = val_tools_path / (benchmark + '.zip')
        if not (val_tools_path / 'data' / benchmark).exists():
            os.system(f"curl -# -L {url} -o {zip_dst} -# --retry 3 -C -")
            LOGGER.info(f'{benchmark}.zip downloaded sucessfully')

            try:
                with zipfile.ZipFile((val_tools_path / (benchmark + '.zip')), 'r') as zip_file:
                    if self.opt.benchmark == 'MOT16':
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            member_path = val_tools_path / 'data' / 'MOT16' / member
                            if not member_path.exists() and not member_path.is_file():
                                zip_file.extract(member, val_tools_path / 'data' / 'MOT16')
                    else:
                        for member in tqdm(zip_file.namelist(), desc=f'Extracting {benchmark}'):
                            member_path = val_tools_path / 'data' / member
                            if not member_path.exists() and not member_path.is_file():
                                zip_file.extract(member, val_tools_path / 'data')
                LOGGER.info(f'{benchmark}.zip unzipped successfully')
            except Exception as e:
                LOGGER.error(f'{benchmark}.zip is corrupted. Try deleting the file and run the script again {e}')
                sys.exit()

    def eval_setup(self, opt, val_tools_path):
        """Download specific MOT dataset and unpack it

        Args:
            opt: the parsed script arguments
            val_tools_path (pathlib.Path): path to destination folder of the downloaded MOT benchmark zip

        Returns:
            [Path], Path, Path: benchmark sequence paths,
            original tracking results destination, eval tracking result destination
        """

        # set paths
        gt_folder = val_tools_path / 'data' / self.opt.benchmark / self.opt.split
        mot_seqs_path = val_tools_path / 'data' / opt.benchmark / opt.split
        if opt.benchmark == 'MOT17':
            # each sequences is present 3 times, one for each detector
            # (DPM, FRCNN, SDP). Keep only sequences from  one of them
            seq_paths = sorted([str(p / 'img1') for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()])
            seq_paths = [Path(p) for p in seq_paths if 'FRCNN' in p]
        elif opt.benchmark == 'MOT17-mini':
            mot_seqs_path = ROOT / 'assets' / self.opt.benchmark / self.opt.split
            gt_folder = ROOT / 'assets' / self.opt.benchmark / self.opt.split
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]
        else:
            # this is not the case for MOT16, MOT20 or your custom dataset
            seq_paths = [p / 'img1' for p in Path(mot_seqs_path).iterdir() if Path(p).is_dir()]

        if opt.eval_existing and (Path(opt.project) / opt.name).exists():
            save_dir = Path(opt.project) / opt.name
            if not (Path(opt.project) / opt.name).exists():
                LOGGER.error(f'{save_dir} does not exist')
        else:
            save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
        MOT_results_folder = (
            val_tools_path / 'data' / 'trackers' /
            'mot_challenge' / opt.benchmark / save_dir.name / 'data'
        )
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
        if any(isinstance(i, int) for i in opt.device) and len(opt.device) > 1:
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

                LOGGER.info(f"Staring evaluation process on {seq_path}")
                p = subprocess.Popen(
                    args=[
                        sys.executable, str(EXAMPLES / "track.py"),
                        "--yolo-model", self.opt.yolo_model,
                        "--reid-model", self.opt.reid_model,
                        "--tracking-method", self.opt.tracking_method,
                        "--conf", str(self.opt.conf),
                        "--imgsz", str(self.opt.imgsz[0]),
                        "--classes", *self.opt.classes,
                        "--name", save_dir.name,
                        "--save" if self.opt.save else ""
                        "--save-mot",
                        "--project", self.opt.project,
                        "--device", str(tracking_subprocess_device),
                        "--source", seq_path,
                        "--exist-ok",
                    ],
                )
                processes.append(p)
                # Wait for the subprocess to complete and capture output

            for p in processes:
                p.wait()

            LOGGER.success("Evaluation succeeded")

        print_args(vars(self.opt))

        if opt.gsi:
            # apply gaussian-smoothed interpolation
            from boxmot.postprocessing.gsi import gsi
            gsi(mot_results_folder=save_dir / 'mot')

        # run the evaluation on the generated txts
        d = [seq_path.parent.name for seq_path in seq_paths]
        p = subprocess.Popen(
            args=[
                sys.executable, val_tools_path / 'scripts' / 'run_mot_challenge.py',
                "--GT_FOLDER", gt_folder,
                "--BENCHMARK", "",
                "--TRACKERS_FOLDER", save_dir,   # project/name
                "--TRACKERS_TO_EVAL", "mot",  # project/name/mot
                "--SPLIT_TO_EVAL", "train",
                "--METRICS", "HOTA", "CLEAR", "Identity",
                "--USE_PARALLEL", "True",
                "--TRACKER_SUB_FOLDER", "",
                "--NUM_PARALLEL_CORES", "4",
                "--SKIP_SPLIT_FOL", "True",
                "--SEQ_INFO", *d
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Wait for the subprocess to complete and capture output
        stdout, stderr = p.communicate()

        # Check the return code of the subprocess
        if p.returncode != 0:
            LOGGER.error(stderr)
            LOGGER.error(stdout)
            sys.exit(1)

        LOGGER.info(stdout)

        # save MOT results in txt
        with open(save_dir / 'MOT_results.txt', 'w') as f:
            f.write(stdout)
        # copy tracking method config to exp folder
        tracking_config = \
            ROOT /\
            'boxmot' /\
            'configs' /\
            (opt.tracking_method + '.yaml')
        shutil.copyfile(tracking_config, save_dir / Path(tracking_config).name)

        return stdout

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
        val_tools_path = EXAMPLES / 'val_utils'
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
    parser.add_argument('--yolo-model', type=str, default=WEIGHTS / 'yolov8n.pt', help='model.pt path(s)')
    parser.add_argument('--reid-model', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='strongsort, ocsort')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--classes', nargs='+', type=str, default=['0'],
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'val',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--gsi', action='store_true',
                        help='apply gsi to results')
    parser.add_argument('--benchmark', type=str, default='MOT17-mini',
                        help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str, default='train',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--eval-existing', action='store_true',
                        help='evaluate existing results under project/name/mot')
    parser.add_argument('--conf', type=float, default=0.45,
                        help='confidence threshold')
    parser.add_argument('--imgsz', '--img-size', nargs='+', type=int, default=[1280],
                        help='inference size h,w')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
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
