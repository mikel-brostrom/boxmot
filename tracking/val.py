# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import queue
import select
import re
import os
import torch
from functools import partial
import threading
import sys

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES, DATA
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device

from ultralytics import YOLO
from ultralytics.data.loaders import LoadImages
from ultralytics.utils.files import increment_path
from ultralytics.data.utils import VID_FORMATS

from tracking.detectors import get_yolo_inferer
from tracking.utils import convert_to_mot_format, write_mot_results, download_mot_eval_tools, download_mot_dataset, unzip_mot_dataset, eval_setup
from boxmot.appearance.reid_auto_backend import ReidAutoBackend

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install


def prompt_overwrite(path_type, path, ci=False):
    if ci:
        print(f"{path_type} {path} already exists. Use existing due to no UI mode.")
        return False
    
    def input_with_timeout(prompt, timeout=3.0):
        print(prompt, end='', flush=True)
        inputs, _, _ = select.select([sys.stdin], [], [], timeout)
        if inputs:
            result = sys.stdin.readline().strip().lower()
            return result in ['y', 'yes']
        else:
            print("\nNo response, not proceeding with overwrite...")
            return False
    
    return input_with_timeout(f"{path_type} {path} already exists. Overwrite? [y/N]: ")


def generate_dets_embs(args, y):
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(
        y if 'yolov8' in str(y) else 'yolov8n.pt',
    )

    results = yolo(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        stream=True,
        device=args.device,
        verbose=False,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
    )

    if 'yolov8' not in str(y):
        # replace yolov8 model
        m = get_yolo_inferer(y)
        model = m(
            model=y,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    reids = []
    for r in args.reid_model:
        model = ReidAutoBackend(
            weights=args.reid_model, device=yolo.predictor.device, half=args.half
        ).model
        reids.append(model)
        embs_path = args.project / 'dets_n_embs' / y.stem / 'embs' / r.stem / (Path(args.source).parent.name + '.txt')
        embs_path.parent.mkdir(parents=True, exist_ok=True)
        embs_path.touch(exist_ok=True)
        
        # Truncate the embeddings file if it is not empty
        if os.path.getsize(embs_path) > 0:
            open(embs_path, 'w').close()

    # store custom args in predictor
    yolo.predictor.custom_args = args

    dets_path = args.project / 'dets_n_embs' / y.stem / 'dets' / (Path(args.source).parent.name + '.txt')
    
    # create parent folder and txt files
    dets_path.parent.mkdir(parents=True, exist_ok=True)
    dets_path.touch(exist_ok=True)
    
    # Truncate the detections file if it is not empty
    if os.path.getsize(dets_path) > 0:
        open(dets_path, 'w').close()
    
    with open(str(dets_path), 'ab+') as f:  # append binary mode
        np.savetxt(f, [], fmt='%f', header=str(args.source))  # save as ints instead of scientific notation

    for frame_idx, r in enumerate(tqdm(results, desc="Frames")):

        nr_dets = len(r.boxes)
        frame_idx = torch.full((1, 1), frame_idx + 1)
        frame_idx = frame_idx.repeat(nr_dets, 1)

        if r.boxes.data.is_cpu:
            dets = r.boxes.data[:, 0:4].numpy()
        else:
            dets = r.boxes.data[:, 0:4].cpu().numpy()
            
        img = r.orig_img
        
        dets = np.concatenate(
            [
                frame_idx,
                r.boxes.xyxy.to('cpu'),
                r.boxes.conf.unsqueeze(1).to('cpu'),
                r.boxes.cls.unsqueeze(1).to('cpu'),
            ], axis=1
        )

        with open(str(dets_path), 'ab+') as f:  # append binary mode
            np.savetxt(f, dets, fmt='%f')  # save as ints instead of scientific notation

        for reid, reid_model_name in zip(reids, args.reid_model):
            embs = reid.get_features(dets[:, 1:5], img)
            embs_path = args.project / "dets_n_embs" / y.stem / 'embs' / reid_model_name.stem / (Path(args.source).parent.name + '.txt')
            with open(str(embs_path), 'ab+') as f:  # append binary mode
                np.savetxt(f, embs, fmt='%f')  # save as ints instead of scientific notation


def generate_mot_results(args):
    args.device = select_device(args.device)
    tracker = create_tracker(
        args.tracking_method,
        TRACKER_CONFIGS / (args.tracking_method + '.yaml'),
        args.reid_model[0].with_suffix('.pt'),
        args.device,
        False,
        False
    )

    with open(args.dets_file_path, 'r') as file:
        args.source = file.readline().strip().replace("# ", "")  # .strip() removes leading/trailing whitespace and newline characters

    LOGGER.info(f"\nStarting tracking on:\n\t{args.source}\nwith preloaded dets\n\t({args.dets_file_path.relative_to(ROOT)})\nand embs\n\t({args.embs_file_path.relative_to(ROOT)})\nusing\n\t{args.tracking_method}")

    dets = np.loadtxt(args.dets_file_path, skiprows=1)  # skiprows=1 skips the header row
    embs = np.loadtxt(args.embs_file_path)  # skiprows=1 skips the header row

    dets_n_embs = np.concatenate([dets, embs], axis=1)

    print('args.source', args.source)
    dataset = LoadImages(args.source)

    txt_path = args.exp_folder_path / (Path(args.source).parent.name + '.txt')
    all_mot_results = []

    for frame_idx, d in enumerate(tqdm(dataset, desc="Frames")):

        # don't generate dets_n_emb for the last frame
        if (frame_idx + 1) == len(dataset):
            break

        im = d[1][0]

        # get dets and embedding associated to this frame
        frame_dets_n_embs = dets_n_embs[dets_n_embs[:, 0] == frame_idx + 1]

        # frame id, x1, y1, x2, y2, conf, cls
        dets = frame_dets_n_embs[:, 1:7]
        embs = frame_dets_n_embs[:, 7:]
        tracks = tracker.update(dets, im, embs)

        if tracks.size > 0:  # Check if tracks is not an empty array
            mot_results = convert_to_mot_format(tracks, frame_idx + 1)
            all_mot_results.append(mot_results)

    if all_mot_results:
        all_mot_results = np.vstack(all_mot_results)
        write_mot_results(txt_path, all_mot_results)


def parse_mot_results(results):
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
    results_dict = {}
    for key, value in zip(["HOTA", "MOTA", "IDF1"], combined_results):
        results_dict[key]= value 
    
    return results_dict


def trackeval(args, seq_paths, save_dir, MOT_results_folder, gt_folder, metrics=["HOTA", "CLEAR", "Identity"]):
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.
    
    Parameters:
        script_path (str): The path to the evaluation script to run.
        trackers_folder (str): The folder where tracker results are stored.
        metrics (list): A list of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].
        num_parallel_cores (int): The number of parallel cores to use for evaluation. Defaults to 4.
    
    Outputs:
        Prints the standard output and standard error from the evaluation script.
    """
    # Define paths
    d = [seq_path.parent.name for seq_path in seq_paths]
    # Prepare arguments for subprocess call
    args = [
        sys.executable, EXAMPLES / 'val_utils' / 'scripts' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", "",
        "--TRACKERS_FOLDER", args.exp_folder_path,
        "--TRACKERS_TO_EVAL", "",
        "--SPLIT_TO_EVAL", "train",
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--SEQ_INFO", *d
    ]

    # Execute the evaluation script
    p = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    if stderr:
        print("Standard Error:\n", stderr)
    return stdout


def run_generate_dets_embs(opt):
    print('opt.source', opt.source)
    mot_folder_paths = [item for item in Path(opt.source).iterdir()]
    for y in opt.yolo_model:
        for i, mot_folder_path in enumerate(mot_folder_paths):
            dets_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'dets' / (mot_folder_path.name + '.txt')
            embs_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'embs' / (opt.reid_model[0].stem) / (mot_folder_path.name + '.txt')
            if dets_path.exists() and embs_path.exists():
                if prompt_overwrite('Detections and Embeddings', dets_path, opt.ci):
                    LOGGER.info(f'Overwriting detections and embeddings for {mot_folder_path}...')
                else:
                    LOGGER.info(f'Skipping generation for {mot_folder_path} as they already exist.')
                    continue
            LOGGER.info(f'Generating detections and embeddings for data under {mot_folder_path} [{i + 1}/{len(mot_folder_paths)} seqs]')
            opt.source = mot_folder_path / 'img1'
            generate_dets_embs(opt, y)


def run_generate_mot_results(opt):
    for y in opt.yolo_model:
        exp_folder_path = opt.project / 'mot' / (str(y.stem) + "_" + str(opt.reid_model[0].stem) + "_" + str(opt.tracking_method))
        exp_folder_path = increment_path(path=exp_folder_path, sep="_", exist_ok=False)
        opt.exp_folder_path = exp_folder_path
        dets_file_paths = [item for item in (opt.project / "dets_n_embs" / y.stem / 'dets').glob('*.txt') if not item.name.startswith('.')]
        embs_file_paths = [item for item in (opt.project / "dets_n_embs" / y.stem / 'embs' /  opt.reid_model[0].stem).glob('*.txt') if not item.name.startswith('.')]
        for d, e in zip(dets_file_paths, embs_file_paths):
            mot_result_path = exp_folder_path / (d.stem + '.txt')
            if mot_result_path.exists():
                if prompt_overwrite('MOT Result', mot_result_path, opt.ci):
                    LOGGER.info(f'Overwriting MOT result for {d.stem}...')
                else:
                    LOGGER.info(f'Skipping MOT result generation for {d.stem} as it already exists.')
                    continue
            opt.dets_file_path = d
            opt.embs_file_path = e
            generate_mot_results(opt)


def run_trackeval(opt):
    val_tools_path = EXAMPLES / 'val_utils'
    download_mot_eval_tools(val_tools_path)
    zip_path = download_mot_dataset(val_tools_path, opt.benchmark)
    unzip_mot_dataset(zip_path, val_tools_path, opt.benchmark)
    seq_paths, save_dir, MOT_results_folder, gt_folder = eval_setup(opt, val_tools_path)
    trackeval_results = trackeval(opt, seq_paths, save_dir, MOT_results_folder, gt_folder)
    hota_mota_idf1 = parse_mot_results(trackeval_results)
    if opt.verbose:
        print(trackeval_results)
        with open(opt.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(hota_mota_idf1))
    print(json.dumps(hota_mota_idf1))
    return hota_mota_idf1

def run_all(opt):
    run_generate_dets_embs(opt)
    run_generate_mot_results(opt)
    run_trackeval(opt)
    

def parse_opt():
    parser = argparse.ArgumentParser()

    # Global arguments
    parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs', type=Path, help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--ci', action='store_true', help='Automatically reuse existing due to no UI in CI')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    parser.add_argument('--dets-file-path', type=Path, help='path to detections file')
    parser.add_argument('--embs-file-path', type=Path, help='path to embeddings file')
    parser.add_argument('--exp-folder-path', type=Path, help='path to experiment folder')
    parser.add_argument('--benchmark', type=str, default='MOT17', help='MOT16, MOT17, MOT20')
    parser.add_argument('--split', type=str, default='train', help='existing project/name ok, do not increment')
    parser.add_argument('--verbose', action='store_true', help='print results')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--n-trials', type=int, default=4, help='nr of trials for evolution')
    parser.add_argument('--resume', action='store_true', help='resume hparam search')
    parser.add_argument('--processes-per-device', type=int, default=2, help='how many subprocesses can be invoked per GPU (to manage memory consumption)')
    parser.add_argument('--objectives', type=str, nargs='+', default=["HOTA", "MOTA", "IDF1"], help='set of objective metrics: HOTA,MOTA,IDF1')

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for generate_dets_embs
    generate_dets_embs_parser = subparsers.add_parser('generate_dets_embs', help='Generate detections and embeddings')
    generate_dets_embs_parser.add_argument('--source', type=str, required=True, help='file/dir/URL/glob, 0 for webcam')
    generate_dets_embs_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_dets_embs_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_dets_embs_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    
    # Subparser for generate_mot_results
    generate_mot_results_parser = subparsers.add_parser('generate_mot_results', help='Generate MOT results')
    generate_mot_results_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_mot_results_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_mot_results_parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, imprassoc')
    generate_mot_results_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')

    # Subparser for trackeval
    trackeval_parser = subparsers.add_parser('trackeval', help='Evaluate tracking results')
    trackeval_parser.add_argument('--exp-folder-path', type=Path, required=True, help='path to experiment folder')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    if opt.command == 'generate_dets_embs':
        run_generate_dets_embs(opt)
    elif opt.command == 'generate_mot_results':
        run_generate_mot_results(opt)
    elif opt.command == 'trackeval':
        run_trackeval(opt)
    else:
        run_all(opt)
        