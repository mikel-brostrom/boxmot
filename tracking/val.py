# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import configparser
import shutil
import json
import queue
import select
import re
import os
import torch
from functools import partial
import threading
import sys
import copy
import concurrent.futures

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES, DATA
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.misc import increment_path
from boxmot.postprocessing.gsi import gsi

from ultralytics import YOLO
from ultralytics.data.loaders import LoadImagesAndVideos

from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)
from tracking.utils import convert_to_mot_format, write_mot_results, download_mot_eval_tools, download_mot_dataset, unzip_mot_dataset, eval_setup, split_dataset
from boxmot.appearance.reid.auto_backend import ReidAutoBackend

checker = RequirementsChecker()
checker.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install


def cleanup_mot17(data_dir, keep_detection='FRCNN'):
    """
    Cleans up the MOT17 dataset to resemble the MOT16 format by keeping only one detection folder per sequence.
    Skips sequences that have already been cleaned.

    Args:
    - data_dir (str): Path to the MOT17 train directory.
    - keep_detection (str): Detection type to keep (options: 'DPM', 'FRCNN', 'SDP'). Default is 'DPM'.
    """

    # Get all folders in the train directory
    all_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Identify unique sequences by removing detection suffixes
    unique_sequences = set(seq.split('-')[0] + '-' + seq.split('-')[1] for seq in all_dirs)

    for seq in unique_sequences:
        # Directory path to the cleaned sequence
        cleaned_seq_dir = os.path.join(data_dir, seq)

        # Skip if the sequence is already cleaned
        if os.path.exists(cleaned_seq_dir):
            print(f"Sequence {seq} is already cleaned. Skipping.")
            continue

        # Directories for each detection method
        seq_dirs = [os.path.join(data_dir, d)
                    for d in all_dirs if d.startswith(seq)]

        # Directory path for the detection folder to keep
        keep_dir = os.path.join(data_dir, f"{seq}-{keep_detection}")

        if os.path.exists(keep_dir):
            # Move the directory to a new name (removing the detection suffix)
            shutil.move(keep_dir, cleaned_seq_dir)
            print(f"Moved {keep_dir} to {cleaned_seq_dir}")

            # Remove other detection directories
            for seq_dir in seq_dirs:
                if os.path.exists(seq_dir) and seq_dir != keep_dir:
                    shutil.rmtree(seq_dir)
                    print(f"Removed {seq_dir}")
        else:
            print(f"Directory for {seq} with {keep_detection} detection does not exist. Skipping.")

    print("MOT17 Cleanup completed!")


def prompt_overwrite(path_type: str, path: str, ci: bool = True) -> bool:
    """
    Prompts the user to confirm overwriting an existing file.

    Args:
        path_type (str): Type of the path (e.g., 'Detections and Embeddings', 'MOT Result').
        path (str): The path to check.
        ci (bool): If True, automatically reuse existing file without prompting (for CI environments).

    Returns:
        bool: True if user confirms to overwrite, False otherwise.
    """
    if ci:
        LOGGER.debug(f"{path_type} {path} already exists. Use existing due to no UI mode.")
        return False

    def input_with_timeout(prompt, timeout=3.0):
        print(prompt, end='', flush=True)

        result = []
        input_received = threading.Event()

        def get_input():
            user_input = sys.stdin.readline().strip().lower()
            result.append(user_input)
            input_received.set()

        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True  # Ensure thread does not prevent program exit
        input_thread.start()
        input_thread.join(timeout)

        if input_received.is_set():
            return result[0] in ['y', 'yes']
        else:
            print("\nNo response, not proceeding with overwrite...")
            return False

    return input_with_timeout(f"{path_type} {path} already exists. Overwrite? [y/N]: ")


def generate_dets_embs(args: argparse.Namespace, y: Path, source: Path) -> None:
    """
    Generates detections and embeddings for the specified 
    arguments, YOLO model and source.

    Args:
        args (Namespace): Parsed command line arguments.
        y (Path): Path to the YOLO model file.
        source (Path): Path to the source directory.
    """
    WEIGHTS.mkdir(parents=True, exist_ok=True)

    if args.imgsz is None:
        args.imgsz = default_imgsz(y)

    yolo = YOLO(
        y if is_ultralytics_model(y)
        else 'yolov8n.pt',
    )

    results = yolo(
        source=source,
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

    if not is_ultralytics_model(y):
        m = get_yolo_inferer(y)
        yolo_model = m(model=y, device=yolo.predictor.device,
                       args=yolo.predictor.args)
        yolo.predictor.model = yolo_model

        # If current model is YOLOX, change the preprocess and postprocess
        if is_yolox_model(y):
            # add callback to save image paths for further processing
            yolo.add_callback("on_predict_batch_start",
                              lambda p: yolo_model.update_im_paths(p))
            yolo.predictor.preprocess = (
                lambda im: yolo_model.preprocess(im=im))
            yolo.predictor.postprocess = (
                lambda preds, im, im0s:
                yolo_model.postprocess(preds=preds, im=im, im0s=im0s))

    reids = []
    for r in args.reid_model:
        reid_model = ReidAutoBackend(weights=args.reid_model,
                                     device=yolo.predictor.device,
                                     half=args.half).model
        reids.append(reid_model)
        embs_path = args.project / 'dets_n_embs' / y.stem / 'embs' / r.stem / (source.parent.name + '.txt')
        embs_path.parent.mkdir(parents=True, exist_ok=True)
        embs_path.touch(exist_ok=True)

        if os.path.getsize(embs_path) > 0:
            open(embs_path, 'w').close()

    yolo.predictor.custom_args = args

    dets_path = args.project / 'dets_n_embs' / y.stem / 'dets' / (source.parent.name + '.txt')
    dets_path.parent.mkdir(parents=True, exist_ok=True)
    dets_path.touch(exist_ok=True)

    if os.path.getsize(dets_path) > 0:
        open(dets_path, 'w').close()

    with open(str(dets_path), 'ab+') as f:
        np.savetxt(f, [], fmt='%f', header=str(source))

    for frame_idx, r in enumerate(tqdm(results, desc="Frames")):
        nr_dets = len(r.boxes)
        frame_idx = torch.full((1, 1), frame_idx + 1).repeat(nr_dets, 1)
        img = r.orig_img

        dets = np.concatenate(
            [
                frame_idx,
                r.boxes.xyxy.to('cpu'),
                r.boxes.conf.unsqueeze(1).to('cpu'),
                r.boxes.cls.unsqueeze(1).to('cpu'),
            ], axis=1
        )

        # Filter dets with incorrect boxes: (x2 < x1 or y2 < y1)
        boxes = r.boxes.xyxy.to('cpu').numpy().round().astype(int)
        boxes_filter = ((np.maximum(0, boxes[:, 0]) < np.minimum(boxes[:, 2], img.shape[1])) &
                        (np.maximum(0, boxes[:, 1]) < np.minimum(boxes[:, 3], img.shape[0])))
        dets = dets[boxes_filter]

        with open(str(dets_path), 'ab+') as f:
            np.savetxt(f, dets, fmt='%f')

        for reid, reid_model_name in zip(reids, args.reid_model):
            embs = reid.get_features(dets[:, 1:5], img)
            embs_path = args.project / "dets_n_embs" / y.stem / 'embs' / reid_model_name.stem / (source.parent.name + '.txt')
            with open(str(embs_path), 'ab+') as f:
                np.savetxt(f, embs, fmt='%f')


def generate_mot_results(args: argparse.Namespace, config_dict: dict = None) -> dict[str, np.ndarray]:
    """
    Generates MOT results for the specified arguments and configuration.

    Args:
        args (Namespace): Parsed command line arguments.
        config_dict (dict, optional): Additional configuration dictionary.

    Returns:
        dict[str, np.ndarray]: {seq_name: array} with frame ids used for MOT
    """
    args.device = select_device(args.device)
    tracker = create_tracker(
        args.tracking_method,
        TRACKER_CONFIGS / (args.tracking_method + '.yaml'),
        args.reid_model[0].with_suffix('.pt'),
        args.device,
        False,
        False,
        config_dict
    )

    with open(args.dets_file_path, 'r') as file:
        source = Path(file.readline().strip().replace("# ", ""))

    dets = np.loadtxt(args.dets_file_path, skiprows=1)
    embs = np.loadtxt(args.embs_file_path)

    dets_n_embs = np.concatenate([dets, embs], axis=1)

    dataset = LoadImagesAndVideos(source)

    txt_path = args.exp_folder_path / (source.parent.name + '.txt')
    all_mot_results = []

    # Change FPS
    if args.fps:

        # Extract original FPS
        conf_path = source.parent / 'seqinfo.ini'
        conf = configparser.ConfigParser()
        conf.read(conf_path)

        orig_fps = int(conf.get("Sequence", "frameRate"))
    
        if orig_fps < args.fps:
            LOGGER.warning(f"Original FPS ({orig_fps}) is lower than "
                           f"requested FPS ({args.fps}) for sequence "
                           f"{source.parent.name}. Using original FPS.")
            target_fps = orig_fps
        else:
            target_fps = args.fps

        
        step = orig_fps/target_fps
    else:
        step = 1
    
    # Create list with frame numbers according to needed step
    frame_nums = np.arange(1, len(dataset) + 1, step).astype(int).tolist()

    seq_frame_nums = {source.parent.name: frame_nums.copy()}

    for frame_num, d in enumerate(tqdm(dataset, desc=source.parent.name), 1):
        # Filter using list with needed numbers
        if len(frame_nums) > 0:
            if frame_num < frame_nums[0]:
                continue
            else:
                frame_nums.pop(0)

        im = d[1][0]
        frame_dets_n_embs = dets_n_embs[dets_n_embs[:, 0] == frame_num]

        dets = frame_dets_n_embs[:, 1:7]
        embs = frame_dets_n_embs[:, 7:]
        tracks = tracker.update(dets, im, embs)

        if tracks.size > 0:
            mot_results = convert_to_mot_format(tracks, frame_num)
            all_mot_results.append(mot_results)

    if all_mot_results:
        all_mot_results = np.vstack(all_mot_results)
    else:
        all_mot_results = np.empty((0, 0))

    write_mot_results(txt_path, all_mot_results)

    return seq_frame_nums


def parse_mot_results(results: str) -> dict:
    """
    Extracts the COMBINED HOTA, MOTA, IDF1 from the results generated by the run_mot_challenge.py script.

    Args:
        results (str): MOT results as a string.

    Returns:
        dict: A dictionary containing HOTA, MOTA, and IDF1 scores.
    """
    combined_results = results.split('COMBINED')[2:-1]
    combined_results = [float(re.findall(r"[-+]?(?:\d*\.*\d+)", f)[0])
                        for f in combined_results]

    results_dict = {}
    for key, value in zip(["HOTA", "MOTA", "IDF1"], combined_results):
        results_dict[key] = value

    return results_dict


def trackeval(args: argparse.Namespace, seq_paths: list, save_dir: Path, MOT_results_folder: Path, gt_folder: Path, metrics: list = ["HOTA", "CLEAR", "Identity"]) -> str:
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.

    Args:
        seq_paths (list): List of sequence paths.
        save_dir (Path): Directory to save evaluation results.
        MOT_results_folder (Path): Folder containing MOT results.
        gt_folder (Path): Folder containing ground truth data.
        metrics (list, optional): List of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].

    Returns:
        str: Standard output from the evaluation script.
    """

    d = [seq_path.parent.name for seq_path in seq_paths]

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
        "--GT_LOC_FORMAT", "{gt_folder}/{seq}/gt/gt_temp.txt",
        "--SEQ_INFO", *d
    ]

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


def run_generate_dets_embs(opt: argparse.Namespace) -> None:
    """
    Runs the generate_dets_embs function for all YOLO models and source directories.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    mot_folder_paths = sorted([item for item in Path(opt.source).iterdir()])
    for y in opt.yolo_model:
        for i, mot_folder_path in enumerate(mot_folder_paths):
            dets_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'dets' / (mot_folder_path.name + '.txt')
            embs_path = Path(opt.project) / 'dets_n_embs' / y.stem / 'embs' / (opt.reid_model[0].stem) / (mot_folder_path.name + '.txt')
            if dets_path.exists() and embs_path.exists():
                if prompt_overwrite('Detections and Embeddings', dets_path, opt.ci):
                    LOGGER.debug(f'Overwriting detections and embeddings for {mot_folder_path}...')
                else:
                    LOGGER.debug(f'Skipping generation for {mot_folder_path} as they already exist.')
                    continue
            LOGGER.debug(f'Generating detections and embeddings for data under {mot_folder_path} [{i + 1}/{len(mot_folder_paths)} seqs]')
            generate_dets_embs(opt, y, source=mot_folder_path / 'img1')


def process_single_mot(opt: argparse.Namespace, d: Path, e: Path, evolve_config: dict):
    # Create a deep copy of opt so each task works independently
    new_opt = copy.deepcopy(opt)
    new_opt.dets_file_path = d
    new_opt.embs_file_path = e
    frames_dict = generate_mot_results(new_opt, evolve_config)
    return frames_dict

def run_generate_mot_results(opt: argparse.Namespace, evolve_config: dict = None) -> None:
    """
    Runs the generate_mot_results function for all YOLO models and detection/embedding files
    in parallel.
    """
    
    for y in opt.yolo_model:
        exp_folder_path = opt.project / 'mot' / (f"{y.stem}_{opt.reid_model[0].stem}_{opt.tracking_method}")
        exp_folder_path = increment_path(path=exp_folder_path, sep="_", exist_ok=False)
        opt.exp_folder_path = exp_folder_path

        mot_folder_names = [item.stem for item in Path(opt.source).iterdir()]
        
        dets_folder = opt.project / "dets_n_embs" / y.stem / 'dets'
        embs_folder = opt.project / "dets_n_embs" / y.stem / 'embs' / opt.reid_model[0].stem
        
        dets_file_paths = sorted([
            item for item in dets_folder.glob('*.txt')
            if not item.name.startswith('.') and item.stem in mot_folder_names
        ])
        embs_file_paths = sorted([
            item for item in embs_folder.glob('*.txt')
            if not item.name.startswith('.') and item.stem in mot_folder_names
        ])
        
        LOGGER.info(f"\nStarting tracking on:\n\t{opt.source}\nwith preloaded dets\n\t({dets_folder.relative_to(ROOT)})\nand embs\n\t({embs_folder.relative_to(ROOT)})\nusing\n\t{opt.tracking_method}")

        tasks = []
        # Create a thread pool to run each file pair in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for d, e in zip(dets_file_paths, embs_file_paths):
                mot_result_path = exp_folder_path / (d.stem + '.txt')
                if mot_result_path.exists():
                    if prompt_overwrite('MOT Result', mot_result_path, opt.ci):
                        LOGGER.info(f'Overwriting MOT result for {d.stem}...')
                    else:
                        LOGGER.info(f'Skipping MOT result generation for {d.stem} as it already exists.')
                        continue
                # Submit the task to process this file pair in parallel
                tasks.append(executor.submit(process_single_mot, opt, d, e, evolve_config))
            
            # Dict with {seq_name: [frame_nums]}
            seqs_frame_nums = {}
            # Wait for all tasks to complete and log any exceptions
            for future in concurrent.futures.as_completed(tasks):
                try:
                    seqs_frame_nums.update(future.result())
                except Exception as exc:
                    LOGGER.error(f'Error processing file pair: {exc}')
    
    # Postprocess data with gsi if requested
    if opt.gsi:
        gsi(mot_results_folder=opt.exp_folder_path)

    with open(opt.exp_folder_path / 'seqs_frame_nums.json', 'w') as f:
        json.dump(seqs_frame_nums, f)


def run_trackeval(opt: argparse.Namespace) -> dict:
    """
    Runs the trackeval function to evaluate tracking results.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    seq_paths, save_dir, MOT_results_folder, gt_folder = eval_setup(opt, opt.val_tools_path)
    trackeval_results = trackeval(opt, seq_paths, save_dir, MOT_results_folder, gt_folder)
    hota_mota_idf1 = parse_mot_results(trackeval_results)
    if opt.verbose:
        LOGGER.info(trackeval_results)
        with open(opt.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(hota_mota_idf1))
    LOGGER.info(json.dumps(hota_mota_idf1))
    return hota_mota_idf1


def run_all(opt: argparse.Namespace) -> None:
    """
    Runs all stages of the pipeline: generate_dets_embs, generate_mot_results, and trackeval.

    Args:
        opt (Namespace): Parsed command line arguments.
    """
    run_generate_dets_embs(opt)
    run_generate_mot_results(opt)
    run_trackeval(opt)


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Global arguments
    parser.add_argument('--yolo-model', nargs='+', type=Path, default=[WEIGHTS / 'yolov8n.pt'], help='yolo model path')
    parser.add_argument('--reid-model', nargs='+', type=Path, default=[WEIGHTS / 'osnet_x0_25_msmt17.pt'], help='reid model path')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=None, help='inference size h,w')
    parser.add_argument('--fps', type=int, default=None, help='video frame-rate')
    parser.add_argument('--conf', type=float, default=0.01, help='min confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs', type=Path, help='save results to project/name')
    parser.add_argument('--name', default='', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', default=True, help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--ci', action='store_true', help='Automatically reuse existing due to no UI in CI')
    parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, boosttrack')
    parser.add_argument('--dets-file-path', type=Path, help='path to detections file')
    parser.add_argument('--embs-file-path', type=Path, help='path to embeddings file')
    parser.add_argument('--exp-folder-path', type=Path, help='path to experiment folder')
    parser.add_argument('--verbose', action='store_true', help='print results')
    parser.add_argument('--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--gsi', action='store_true', help='apply Gaussian smooth interpolation postprocessing')
    parser.add_argument('--n-trials', type=int, default=4, help='nr of trials for evolution')
    parser.add_argument('--objectives', type=str, nargs='+', default=["HOTA", "MOTA", "IDF1"], help='set of objective metrics: HOTA,MOTA,IDF1')
    parser.add_argument('--val-tools-path', type=Path, default=EXAMPLES / 'val_utils', help='path to store trackeval repo in')
    parser.add_argument('--split-dataset', action='store_true', help='Use the second half of the dataset')

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for generate_dets_embs
    generate_dets_embs_parser = subparsers.add_parser('generate_dets_embs', help='Generate detections and embeddings')
    generate_dets_embs_parser.add_argument('--source', type=str, required=True, help='file/dir/URL/glob, 0 for webcam')
    generate_dets_embs_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_dets_embs_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_dets_embs_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    generate_dets_embs_parser.add_argument('--classes', nargs='+', type=int, default=0, help='filter by class: --classes 0, or --classes 0 2 3')

    # Subparser for generate_mot_results
    generate_mot_results_parser = subparsers.add_parser('generate_mot_results', help='Generate MOT results')
    generate_mot_results_parser.add_argument('--yolo-model', nargs='+', type=Path, default=WEIGHTS / 'yolov8n.pt', help='yolo model path')
    generate_mot_results_parser.add_argument('--reid-model', nargs='+', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt', help='reid model path')
    generate_mot_results_parser.add_argument('--tracking-method', type=str, default='deepocsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, boosttrack')
    generate_mot_results_parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')

    # Subparser for trackeval
    trackeval_parser = subparsers.add_parser('trackeval', help='Evaluate tracking results')
    trackeval_parser.add_argument('--source', type=str, required=True, help='file/dir/URL/glob, 0 for webcam')
    trackeval_parser.add_argument('--exp-folder-path', type=Path, required=True, help='path to experiment folder')

    opt = parser.parse_args()
    source_path = Path(opt.source)
    opt.benchmark, opt.split = source_path.parent.name, source_path.name

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    
    # download MOT benchmark
    download_mot_eval_tools(opt.val_tools_path)

    if not Path(opt.source).exists():
        zip_path = download_mot_dataset(opt.val_tools_path, opt.benchmark)
        unzip_mot_dataset(zip_path, opt.val_tools_path, opt.benchmark)

    if opt.benchmark == 'MOT17':
        cleanup_mot17(opt.source)

    if opt.split_dataset:
        opt.source, opt.benchmark = split_dataset(opt.source)

    if opt.command == 'generate_dets_embs':
        run_generate_dets_embs(opt)
    elif opt.command == 'generate_mot_results':
        run_generate_mot_results(opt)
    elif opt.command == 'trackeval':
        run_trackeval(opt)
    else:
        run_all(opt)
