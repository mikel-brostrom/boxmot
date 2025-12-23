# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import configparser
import shutil
import json
import yaml
import cv2
import re
import os
import torch
import threading
import sys
import copy
import concurrent.futures
import traceback

from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.utils import NUM_THREADS, ROOT, WEIGHTS, TRACKER_CONFIGS, DATASET_CONFIGS, logger as LOGGER, TRACKEVAL
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.plots import MetricsPlotter
from boxmot.utils.misc import increment_path, prompt_overwrite
from boxmot.utils.clean import cleanup_mot17
from typing import Optional, List, Dict, Generator, Union

from boxmot.utils.dataloaders.MOT17 import MOT17DetEmbDataset
from boxmot.postprocessing.gsi import gsi

from ultralytics import YOLO

from boxmot.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)
from boxmot.utils.mot_utils import convert_to_mot_format, write_mot_results
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.utils.download import download_eval_data, download_trackeval

checker = RequirementsChecker()
checker.check_packages(('ultralytics', ))  # install


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_dataset_cfg(name: str) -> dict:
    """Load the dict from boxmot/configs/datasets/{name}.yaml."""
    path = DATASET_CONFIGS / f"{name}.yaml"
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def eval_init(args,
              trackeval_dest: Path = TRACKEVAL,
              branch: str = "master",
              overwrite: bool = False
    ) -> None:
    """
    Common initialization: download TrackEval and (if needed) the MOT-challenge
    data for ablation runs, then canonicalize args.source.
    Modifies args in place.
    """
    # 1) download the TrackEval code
    download_trackeval(dest=trackeval_dest, branch=branch, overwrite=overwrite)

    # 2) if doing MOT17/20-ablation, pull down the dataset and rewire args.source/split
    if (DATASET_CONFIGS / f"{args.source}.yaml").exists():
        cfg = load_dataset_cfg(str(args.source))
        
        # Determine dataset destination
        if cfg["download"]["dataset_url"]:
            dataset_dest = TRACKEVAL / f"{Path(cfg['benchmark']['source']).name}.zip"
        else:
            # For custom datasets without URL, use the path from config if available, or default to assets
            dataset_dest = Path(cfg["download"].get("dataset_dest", f"assets/{Path(cfg['benchmark']['source']).name}"))

        download_eval_data(
            runs_url=cfg["download"]["runs_url"],
            dataset_url=cfg["download"]["dataset_url"],
            dataset_dest=dataset_dest,
            overwrite=overwrite
        )
        args.benchmark = Path(cfg["benchmark"]["source"]).name
        args.split = cfg["benchmark"]["split"]
        if cfg["download"]["dataset_url"]:
            args.source = TRACKEVAL / f"{args.benchmark}/{args.split}"
        elif "source" in cfg["benchmark"]:
            args.source = Path(cfg["benchmark"]["source"]) / args.split
        else:
            args.source = dataset_dest / args.split

    # 3) finally, make source an absolute Path everywhere
    args.source = Path(args.source).resolve()
    args.project = Path(args.project).resolve()
    args.project.mkdir(parents=True, exist_ok=True)


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
        reid_model = ReidAutoBackend(weights=r,
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


def parse_mot_results(results: str) -> dict:
    """
    Extracts COMBINED HOTA, MOTA, IDF1, AssA, AssRe, IDSW, and IDs from MOTChallenge evaluation output.
    Returns a dictionary keyed by class name.
    """
    metric_specs = {
        'HOTA':   ('HOTA:',      {'HOTA': 0, 'AssA': 2, 'AssRe': 5}),
        'MOTA':   ('CLEAR:',     {'MOTA': 0, 'IDSW': 12}),
        'IDF1':   ('Identity:',  {'IDF1': 0}),
        'IDs':    ('Count:',     {'IDs': 2}),
    }

    int_fields = {'IDSW', 'IDs'}
    parsed_results = {}
    
    lines = results.splitlines()
    current_class = None
    current_metric_type = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for header lines
        is_header = False
        for metric_name, (prefix, _) in metric_specs.items():
            if line.startswith(prefix):
                is_header = True
                current_metric_type = metric_name
                
                # Format: "HOTA: tracker-classHOTA ..."
                content = line[len(prefix):].strip()
                first_word = content.split()[0]
                if first_word.endswith(metric_name):
                    tracker_class = first_word[:-len(metric_name)]
                    if '-' in tracker_class:
                        current_class = tracker_class.split('-')[-1]
                    else:
                        current_class = 'default'
                    
                    if current_class not in parsed_results:
                        parsed_results[current_class] = {}
                break
        
        if is_header:
            continue
        
        # Check for COMBINED row
        if line.startswith('COMBINED') and current_class and current_metric_type:
            fields = line.split()
            if len(fields) > 1:
                values = fields[1:] # Skip 'COMBINED'
                _, field_map = metric_specs[current_metric_type]
                for key, idx in field_map.items():
                    if idx < len(values):
                        val = values[idx]
                        parsed_results[current_class][key] = max(0, int(val) if key in int_fields else float(val))

    return parsed_results


def trackeval(args: argparse.Namespace, seq_paths: list, save_dir: Path, gt_folder: Path, metrics: list = ["HOTA", "CLEAR", "Identity"]) -> str:
    """
    Executes a Python script to evaluate MOT challenge tracking results using specified metrics.

    Args:
        seq_paths (list): List of sequence paths.
        save_dir (Path): Directory to save evaluation results.
        gt_folder (Path): Folder containing ground truth data.
        metrics (list, optional): List of metrics to use for evaluation. Defaults to ["HOTA", "CLEAR", "Identity"].

    Returns:
        str: Standard output from the evaluation script.
    """

    d = [seq_path.parent.name for seq_path in seq_paths]

    # Determine classes to evaluate
    classes_to_eval = ['person']
    if hasattr(args, 'classes') and args.classes is not None:
        class_indices = args.classes if isinstance(args.classes, list) else [args.classes]
        classes_to_eval = [COCO_CLASSES[int(i)] for i in class_indices]

    # Filter classes based on benchmark config
    try:
        if hasattr(args, 'benchmark'):
            cfg = load_dataset_cfg(args.benchmark)
            if "benchmark" in cfg and "classes" in cfg["benchmark"]:
                bench_classes = cfg["benchmark"]["classes"].split()
                # Map 'people' to 'person'
                bench_classes = ['person' if c == 'people' else c for c in bench_classes]
                
                # Filter classes_to_eval
                classes_to_eval = [c for c in classes_to_eval if c in bench_classes]
    except FileNotFoundError:
        pass
    except Exception as e:
        LOGGER.warning(f"Error filtering classes: {e}")

    cmd_args = [
        sys.executable, ROOT / 'boxmot' / 'utils' / 'run_mot_challenge.py',
        "--GT_FOLDER", str(gt_folder),
        "--BENCHMARK", "",
        "--TRACKERS_FOLDER", str(args.exp_dir.parent),
        "--TRACKERS_TO_EVAL", args.exp_dir.name,
        "--SPLIT_TO_EVAL", args.split,
        "--METRICS", *metrics,
        "--USE_PARALLEL", "True",
        "--TRACKER_SUB_FOLDER", "",
        "--NUM_PARALLEL_CORES", str(4),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_LOC_FORMAT", "{gt_folder}/{seq}/gt/gt_temp.txt",
        "--CLASSES_TO_EVAL", *classes_to_eval,
        "--SEQ_INFO", *d
    ]

    p = subprocess.Popen(
        args=cmd_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = p.communicate()

    if stderr:
        print("Standard Error:\n", stderr)
    return stdout


def process_single_det_emb(y: Path, source_path: Path, opt: argparse.Namespace, lock):
    try:
        new_opt = copy.deepcopy(opt)
        # Check if img1 exists, otherwise use source_path directly
        img_source = source_path / 'img1'
        if not img_source.exists():
            img_source = source_path
            
        # Use lock to ensure model loading/downloading is thread-safe
        with lock:
            if is_ultralytics_model(y):
                YOLO(y)

        generate_dets_embs(new_opt, y, source=img_source)
    except Exception:
        traceback.print_exc()
        raise

def run_generate_dets_embs(opt: argparse.Namespace) -> None:
    mot_folder_paths = sorted([item for item in Path(opt.source).iterdir() if item.is_dir()])
    
    # Create a manager to share the lock across processes
    manager = mp.Manager()
    lock = manager.Lock()

    for y in opt.yolo_model:
        dets_folder = Path(opt.project) / 'dets_n_embs' / y.stem / 'dets'
        embs_folder = Path(opt.project) / 'dets_n_embs' / y.stem / 'embs' / opt.reid_model[0].stem

        # Filter out already processed sequences
        tasks = []
        for i, mot_folder_path in enumerate(mot_folder_paths):
            dets_path = dets_folder / (mot_folder_path.name + '.txt')
            embs_path = embs_folder / (mot_folder_path.name + '.txt')
            if dets_path.exists() and embs_path.exists():
                if not prompt_overwrite('Detections and Embeddings', dets_path, opt.ci):
                    LOGGER.debug(f"Skipping generation for {mot_folder_path} as they already exist.")
                    continue
            tasks.append((y, mot_folder_path))

        total_sequences = len(mot_folder_paths)
        if len(tasks) == 0:
            LOGGER.info(f"Detections and embeddings cached for all {total_sequences} sequences")
        else:
            LOGGER.info(f"Generating detections and embeddings for {len(tasks)}/{total_sequences} sequences with model {y.name}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS, initializer=_worker_init) as executor:
            futures = [executor.submit(process_single_det_emb, y, source_path, opt, lock) for y, source_path in tasks]

            for fut in concurrent.futures.as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    LOGGER.error(f"An error occurred during detection/embedding generation: {e}")
                    raise e


def process_sequence(seq_name: str,
                     mot_root: str,
                     project_root: str,
                     model_name: str,
                     reid_name: str,
                     tracking_method: str,
                     exp_folder: str,
                     target_fps: Optional[int],
                     device: str,
                     cfg_dict: Optional[Dict] = None,
                     ):

    device = select_device(device)
    tracker = create_tracker(
        tracker_type=tracking_method,
        tracker_config=TRACKER_CONFIGS / (tracking_method + ".yaml"),
        reid_weights=Path(reid_name + '.pt'),
        device=device,
        half=False,
        per_class=False,
        evolve_param_dict=cfg_dict,
    )

    # load with the user’s FPS
    dataset = MOT17DetEmbDataset(
        mot_root=mot_root,
        det_emb_root=str(Path(project_root) / 'dets_n_embs'),
        model_name=model_name,
        reid_name=reid_name,
        target_fps=target_fps
    )
    sequence = dataset.get_sequence(seq_name)

    all_tracks = []
    kept_frame_ids = []
    for frame in sequence:
        fid  = int(frame['frame_id'])
        dets = frame['dets']
        embs = frame['embs']
        img  = frame['img']

        kept_frame_ids.append(fid)

        if dets.size and embs.size:
            tracks = tracker.update(dets, img, embs)
            if tracks.size:
                all_tracks.append(convert_to_mot_format(tracks, fid))

    out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
    write_mot_results(Path(exp_folder) / f"{seq_name}.txt", out_arr)
    return seq_name, kept_frame_ids


from boxmot.utils import configure_logging as _configure_logging

def _worker_init():
    # each spawned process needs its own sinks
    _configure_logging()

def run_generate_mot_results(opt: argparse.Namespace, evolve_config: dict = None) -> None:
    # Prepare experiment folder
    base = opt.project / 'mot' / f"{opt.yolo_model[0].stem}_{opt.reid_model[0].stem}_{opt.tracking_method}"
    exp_dir = increment_path(base, sep="_", exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=True)
    opt.exp_dir = exp_dir

    # Just collect sequence names by scanning directory names
    sequence_names = sorted([
        d.name for d in Path(opt.source).iterdir()
        if d.is_dir() and (d / "img1").exists()
    ])

    # Build task arguments
    task_args = [
        (
            seq,
            str(opt.source),
            str(opt.project),
            opt.yolo_model[0].stem,
            opt.reid_model[0].stem,
            opt.tracking_method,
            str(exp_dir),
            getattr(opt, 'fps', None),
            opt.device,
            evolve_config,
        )
        for seq in sequence_names
    ]

    seq_frame_nums = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_THREADS, initializer=_worker_init) as executor:
        futures = {
            executor.submit(process_sequence, *args): args[0] for args in task_args
        }

        for fut in concurrent.futures.as_completed(futures):
            seq = futures[fut]
            try:
                seq_name, kept_ids = fut.result()
                seq_frame_nums[seq_name] = kept_ids
            except Exception:
                LOGGER.exception(f"Error processing {seq}")

    # Optional GSI postprocessing
    if getattr(opt, "postprocessing", "none") == "gsi":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GSI postprocessing...")
        from boxmot.postprocessing.gsi import gsi
        gsi(mot_results_folder=exp_dir)

    elif getattr(opt, "postprocessing", "none") == "gbrc":
        LOGGER.opt(colors=True).info("<cyan>[3b/4]</cyan> Applying GBRC postprocessing...")
        from boxmot.postprocessing.gbrc import gbrc
        gbrc(mot_results_folder=exp_dir)


def run_trackeval(opt: argparse.Namespace, verbose: bool = True) -> dict:
    """
    Runs the trackeval function to evaluate tracking results.

    Args:
        opt (Namespace): Parsed command line arguments.
        verbose (bool): Whether to print results summary. Default True.
    """
    gt_folder = opt.source
    seq_paths = [p / "img1" for p in opt.source.iterdir() if p.is_dir()]
    save_dir = Path(opt.project) / opt.name
    
    trackeval_results = trackeval(opt, seq_paths, save_dir, gt_folder)
    parsed_results = parse_mot_results(trackeval_results)

    # Load config to filter classes
    # Try to load config from benchmark name first, then fallback to source parent name
    cfg_name = getattr(opt, 'benchmark', str(opt.source.parent.name))
    try:
        cfg = load_dataset_cfg(cfg_name)
    except FileNotFoundError:
        # If config not found, try to find it by checking if source path ends with a known config name
        # This handles cases where source is a custom path
        found = False
        for config_file in DATASET_CONFIGS.glob("*.yaml"):
            if config_file.stem in str(opt.source):
                cfg = load_dataset_cfg(config_file.stem)
                found = True
                break
        
        if not found:
            LOGGER.warning(f"Could not find dataset config for {cfg_name}. Class filtering might be incorrect.")
            cfg = {}

    # Filter parsed_results based on user provided classes (opt.classes)
    single_class_mode = False

    # Priority 1: Benchmark config classes (overrides user classes)
    if "benchmark" in cfg and "classes" in cfg["benchmark"]:
        bench_classes = cfg["benchmark"]["classes"].split()
        parsed_results = {k: v for k, v in parsed_results.items() if k in bench_classes}
        if len(bench_classes) == 1:
            single_class_mode = True
    # Priority 2: User provided classes
    elif hasattr(opt, 'classes') and opt.classes is not None:
        class_indices = opt.classes if isinstance(opt.classes, list) else [opt.classes]
        user_classes = [COCO_CLASSES[int(i)] for i in class_indices]
        parsed_results = {k: v for k, v in parsed_results.items() if k in user_classes}
        if len(user_classes) == 1:
            single_class_mode = True

    if single_class_mode and len(parsed_results) > 0:
        final_results = list(parsed_results.values())[0]
    else:
        final_results = parsed_results
    
    # Print results summary
    if verbose:
        LOGGER.info("")
        LOGGER.opt(colors=True).info("<blue>" + "="*90 + "</blue>")
        LOGGER.opt(colors=True).info("<bold><cyan>📊 Results Summary</cyan></bold>")
        LOGGER.opt(colors=True).info("<blue>" + "="*90 + "</blue>")
        
        headers = ["Class", "HOTA", "MOTA", "IDF1", "AssA", "AssRe", "IDSW", "IDs"]
        header_str = "{:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(*headers)
        LOGGER.opt(colors=True).info(f"<bold>{header_str}</bold>")
        LOGGER.opt(colors=True).info("<blue>" + "-"*90 + "</blue>")
        
        for cls, metrics in parsed_results.items():
            row = [
                cls,
                f"{metrics.get('HOTA', 0):.2f}%",
                f"{metrics.get('MOTA', 0):.2f}%",
                f"{metrics.get('IDF1', 0):.2f}%",
                f"{metrics.get('AssA', 0):.2f}%",
                f"{metrics.get('AssRe', 0):.2f}%",
                f"{metrics.get('IDSW', 0)}",
                f"{metrics.get('IDs', 0)}"
            ]
            row_str = "{:<15} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8} {:<8}".format(*row)
            LOGGER.opt(colors=True).info(row_str)
            
        LOGGER.opt(colors=True).info("<blue>" + "="*90 + "</blue>")

    if opt.ci:
        with open(opt.tracking_method + "_output.json", "w") as outfile:
            outfile.write(json.dumps(final_results))
    
    return final_results


def main(args):
    # Print evaluation pipeline header (blue palette)
    LOGGER.info("")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info("<bold><cyan>🚀 BoxMOT Evaluation Pipeline</cyan></bold>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    LOGGER.opt(colors=True).info(f"<bold>Detector:</bold>  <cyan>{args.yolo_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>ReID:</bold>      <cyan>{args.reid_model[0]}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Tracker:</bold>   <cyan>{args.tracking_method}</cyan>")
    LOGGER.opt(colors=True).info(f"<bold>Benchmark:</bold> <cyan>{args.source}</cyan>")
    LOGGER.opt(colors=True).info("<blue>" + "="*60 + "</blue>")
    
    # Step 1: Download TrackEval
    LOGGER.opt(colors=True).info("<cyan>[1/4]</cyan> Setting up TrackEval...")
    eval_init(args)

    # Step 2: Generate detections and embeddings
    LOGGER.opt(colors=True).info("<cyan>[2/4]</cyan> Generating detections and embeddings...")
    run_generate_dets_embs(args)
    
    # Step 3: Generate MOT results
    LOGGER.opt(colors=True).info("<cyan>[3/4]</cyan> Running tracker...")
    run_generate_mot_results(args)
    
    # Step 4: Evaluate with TrackEval
    LOGGER.opt(colors=True).info("<cyan>[4/4]</cyan> Evaluating results...")
    results = run_trackeval(args)
    
    # Only plot if we have results for a single class or handle multi-class plotting differently
    # For now, let's just skip the radar chart if we have multiple classes or complex structure
    # Or pick the first class found?
    # The original code expected a flat dict of metrics. Now we have {class: {metric: val}}
    
    # Let's try to plot for each class or just skip for now to avoid breaking
    # If 'pedestrian' is in results, use that, otherwise use the first key
    
    # Check if results is flat (single class backward compatibility) or nested
    is_flat = False
    if results and isinstance(list(results.values())[0], (int, float)):
        is_flat = True
        metrics_data = results
        plot_class = 'single_class'
    else:
        plot_class = 'pedestrian'
        if plot_class not in results and len(results) > 0:
            plot_class = list(results.keys())[0]
        metrics_data = results.get(plot_class, {})

    if metrics_data:
        plotter = MetricsPlotter(args.exp_dir)
        
        # Filter only the metrics we want to plot
        plot_metrics = ['HOTA', 'MOTA', 'IDF1']
        plot_values = [metrics_data.get(m, 0) for m in plot_metrics]

        plotter.plot_radar_chart(
            {args.tracking_method: plot_values},
            plot_metrics,
            title=f"MOT metrics radar Chart ({plot_class})",
            ylim=(0, 100),
            yticks=[20, 40, 60, 80, 100],
            ytick_labels=['20', '40', '60', '80', '100']
        )
        

if __name__ == "__main__":
    main()
