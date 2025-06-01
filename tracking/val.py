# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
import configparser
import shutil
import json
import cv2
import re
import os
import torch
import threading
import sys
import copy
import concurrent.futures

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS, TRACKER_CONFIGS, logger as LOGGER, EXAMPLES
from boxmot.utils.checks import RequirementsChecker
from boxmot.utils.torch_utils import select_device
from boxmot.utils.misc import increment_path, prompt_overwrite
from boxmot.utils.clean import cleanup_mot17
from typing import Optional, List, Dict, Generator, Union

from boxmot.utils.dataloaders.MOT17 import MOT17DetEmbDataset
from boxmot.postprocessing.gsi import gsi

from ultralytics import YOLO
from ultralytics.data.build import load_inference_source

from tracking.detectors import (get_yolo_inferer, default_imgsz,
                                is_ultralytics_model, is_yolox_model)
from tracking.utils import convert_to_mot_format, write_mot_results, download_mot_eval_tools, download_mot_dataset, unzip_mot_dataset, eval_setup, split_dataset
from boxmot.appearance.reid.auto_backend import ReidAutoBackend

checker = RequirementsChecker()
checker.check_packages(('ultralytics', ))  # install


class YoloTrackingPipeline:
    """
    Encapsulates the end-to-end boxmot tracking pipeline:
    1. Generating detections
    2. Generating embeddings
    3. Running the tracker over sequences
    4. Evaluating results with TrackEval
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def generate_dets(
        self,
        opt: argparse.Namespace,
        y: Path,
        source: Path
    ) -> List[tuple]:
        """
        Runs YOLO on a single sequence to produce detections and save them to disk.

        Args:
            opt (Namespace): Parsed command line arguments (possibly a deep copy).
            y (Path): Path to the YOLO model file.
            source (Path): Path to the sequence folder (contains 'img1').

        Returns:
            List[tuple]: A list of (frame_idx, dets_array, img_array) for all frames.
        """
        WEIGHTS.mkdir(parents=True, exist_ok=True)

        if opt.imgsz is None:
            opt.imgsz = default_imgsz(y)

        # Initialize YOLO model (fallback to yolov8n.pt if not Ultralytics format)
        yolo = YOLO(
            y if is_ultralytics_model(y)
            else 'yolov8n.pt',
        )

        # Run inference
        results = yolo(
            source=source,
            conf=opt.conf,
            iou=opt.iou,
            agnostic_nms=opt.agnostic_nms,
            stream=True,
            device=opt.device,
            verbose=False,
            exist_ok=opt.exist_ok,
            project=opt.project,
            name=opt.name,
            classes=opt.classes,
            imgsz=opt.imgsz,
            vid_stride=opt.vid_stride,
        )

        # If using a non-Ultralytics checkpoint (e.g., custom YOLOX), swap in custom inferer
        if not is_ultralytics_model(y):
            m = get_yolo_inferer(y)
            yolo_model = m(model=y, device=yolo.predictor.device,
                           args=yolo.predictor.args)
            yolo.predictor.model = yolo_model

            if is_yolox_model(y):
                yolo.add_callback(
                    "on_predict_batch_start",
                    lambda p: yolo_model.update_im_paths(p)
                )
                yolo.predictor.preprocess = lambda im: yolo_model.preprocess(im=im)
                yolo.predictor.postprocess = (
                    lambda preds, im, im0s: yolo_model.postprocess(
                        preds=preds, im=im, im0s=im0s
                    )
                )

        # Prepare dets output path
        dets_path = opt.project / 'dets_n_embs' / y.stem / 'dets' / (source.parent.name + '.txt')
        dets_path.parent.mkdir(parents=True, exist_ok=True)
        dets_path.touch(exist_ok=True)
        if os.path.getsize(dets_path) > 0:
            open(dets_path, 'w').close()

        # Write header (sequence identifier) to dets file
        with open(str(dets_path), 'ab+') as f:
            np.savetxt(f, [], fmt='%f', header=str(source))

        # Collect all detections in memory for downstream embedding
        detections_list: List[tuple] = []

        # Process each frame: extract boxes, write to file, and store in memory
        for frame_idx, r in enumerate(tqdm(results, desc="Frames")):
            nr_dets = len(r.boxes)
            frame_idx_arr = torch.full((1, 1), frame_idx + 1).repeat(nr_dets, 1)
            img = r.orig_img  # numpy array

            # Construct detections array: [frame_idx, x1, y1, x2, y2, conf, cls]
            dets = np.concatenate(
                [
                    frame_idx_arr,
                    r.boxes.xyxy.to('cpu'),
                    r.boxes.conf.unsqueeze(1).to('cpu'),
                    r.boxes.cls.unsqueeze(1).to('cpu'),
                ],
                axis=1
            )

            # Filter out invalid boxes (x2 < x1 or y2 < y1 / outside image)
            boxes = r.boxes.xyxy.to('cpu').numpy().round().astype(int)
            boxes_filter = (
                (np.maximum(0, boxes[:, 0]) < np.minimum(boxes[:, 2], img.shape[1])) &
                (np.maximum(0, boxes[:, 1]) < np.minimum(boxes[:, 3], img.shape[0]))
            )
            dets = dets[boxes_filter]

            # Append valid detections to the file
            with open(str(dets_path), 'ab+') as f:
                np.savetxt(f, dets, fmt='%f')

            # Store (frame_idx, dets, image) for later embedding
            detections_list.append((frame_idx + 1, dets, img))

        return detections_list

    def generate_embs(
        self,
        opt: argparse.Namespace,
        y: Path,
        source: Path,
        detections_list: List[tuple]
    ) -> None:
        """
        Given a list of detections per frame, computes embeddings for each detection
        and writes them to disk.

        Args:
            opt (Namespace): Parsed command line arguments.
            y (Path): Path to the YOLO model file (used for folder naming).
            source (Path): Path to the sequence folder (contains 'img1').
            detections_list (List[tuple]): Output of generate_dets(),
                                            each entry is (frame_idx, dets_array, img_array).
        """
        # Load all ReID backends once
        reid_backends = []
        for r in opt.reid_model:
            reid_model = ReidAutoBackend(
                weights=opt.reid_model,
                device=select_device('cpu') if opt.device == '' else select_device(opt.device),
                half=opt.half
            ).model
            reid_backends.append((r.stem, reid_model))

        # Prepare embedding file paths for each ReID model
        embs_paths: Dict[str, Path] = {}
        for reid_name, _ in reid_backends:
            embs_path = (
                opt.project
                / 'dets_n_embs'
                / y.stem
                / 'embs'
                / reid_name
                / (source.parent.name + '.txt')
            )
            embs_path.parent.mkdir(parents=True, exist_ok=True)
            embs_path.touch(exist_ok=True)
            if os.path.getsize(embs_path) > 0:
                open(embs_path, 'w').close()
            embs_paths[reid_name] = embs_path

        # Iterate over each frame's detections and compute embeddings
        for frame_idx, dets, img in tqdm(detections_list, desc="Embedding frames"):
            if dets.size == 0:
                continue

            # For each ReID backend, compute embeddings and append to its file
            for reid_name, backend in reid_backends:
                # The dets array columns 1:5 correspond to x1,y1,x2,y2
                embs = backend.get_features(dets[:, 1:5], img)
                with open(str(embs_paths[reid_name]), 'ab+') as f:
                    np.savetxt(f, embs, fmt='%f')

    @staticmethod
    def parse_mot_results(results: str) -> dict:
        """
        Extracts the COMBINED HOTA, MOTA, IDF1 from the results generated by the run_mot_challenge.py script.

        Args:
            results (str): MOT results as a string.

        Returns:
            dict: A dictionary containing HOTA, MOTA, and IDF1 scores.
        """
        combined_results = results.split('COMBINED')[2:-1]
        combined_results = [
            float(re.findall(r"[-+]?(?:\d*\.*\d+)", f)[0])
            for f in combined_results
        ]

        results_dict = {}
        for key, value in zip(["HOTA", "MOTA", "IDF1"], combined_results):
            results_dict[key] = value

        return results_dict

    def trackeval(
        self,
        seq_paths: List[Path],
        save_dir: Path,
        MOT_results_folder: Path,
        gt_folder: Path,
        metrics: List[str] = ["HOTA", "CLEAR", "Identity"]
    ) -> str:
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
            sys.executable,
            str(EXAMPLES / 'val_utils' / 'scripts' / 'run_mot_challenge.py'),
            "--GT_FOLDER", str(gt_folder),
            "--BENCHMARK", "",
            "--TRACKERS_FOLDER", self.args.exp_folder_path,
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

    def process_single_det_emb(self, y: Path, source_path: Path):
        """
        Helper for multiprocessing: clones args and invokes generate_dets and generate_embs on a single sequence.
        """
        new_opt = copy.deepcopy(self.args)
        # Step 1: generate detections (in-memory + file)
        detections = self.generate_dets(new_opt, y, source=source_path / 'img1')
        # Step 2: generate embeddings based on those detections
        self.generate_embs(new_opt, y, source=source_path / 'img1', detections_list=detections)

    def run_generate_dets_embs(self) -> None:
        """
        Iterates over YOLO models and MOT folders to generate detections and embeddings
        using multiprocessing for speed.
        """
        opt = self.args
        mot_folder_paths = sorted([item for item in Path(opt.source).iterdir()])

        for y in opt.yolo_model:
            dets_folder = Path(opt.project) / 'dets_n_embs' / y.stem / 'dets'
            embs_folder = Path(opt.project) / 'dets_n_embs' / y.stem / 'embs' / opt.reid_model[0].stem

            # Filter out already processed sequences
            tasks = []
            for mot_folder_path in mot_folder_paths:
                dets_path = dets_folder / (mot_folder_path.name + '.txt')
                embs_path = embs_folder / (mot_folder_path.name + '.txt')
                if dets_path.exists() and embs_path.exists():
                    if not prompt_overwrite('Detections and Embeddings', dets_path, opt.ci):
                        LOGGER.debug(f"Skipping generation for {mot_folder_path} as they already exist.")
                        continue
                tasks.append((y, mot_folder_path))

            LOGGER.info(f"Generating detections and embeddings for {len(tasks)} sequences with model {y.name}")

            max_workers = torch.cuda.device_count() or os.cpu_count()
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.process_single_det_emb, y, source_path)
                    for y, source_path in tasks
                ]

                for fut in concurrent.futures.as_completed(futures):
                    try:
                        fut.result()
                    except Exception as exc:
                        LOGGER.error(f"Error in det/emb task: {exc}")

    def process_sequence(
        self,
        seq_name: str,
        mot_root: str,
        project_root: str,
        model_name: str,
        reid_name: str,
        tracking_method: str,
        exp_folder: str,
        target_fps: Optional[int]
    ):
        """
        Processes a single sequence: loads detections+embeddings, runs tracker, writes MOT results.
        """
        device = select_device('cpu')
        tracker = create_tracker(
            tracking_method,
            TRACKER_CONFIGS / (tracking_method + ".yaml"),
            Path(reid_name + '.pt'),
            device,
            False,
            False,
            None,
        )

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
            fid = int(frame['frame_id'])
            dets = frame['dets']
            embs = frame['embs']
            img = frame['img']

            kept_frame_ids.append(fid)

            if dets.size and embs.size:
                tracks = tracker.update(dets, img, embs)
                if tracks.size:
                    all_tracks.append(convert_to_mot_format(tracks, fid))

        out_arr = np.vstack(all_tracks) if all_tracks else np.empty((0, 0))
        write_mot_results(Path(exp_folder) / f"{seq_name}.txt", out_arr)
        return seq_name, kept_frame_ids

    def run_generate_mot_results(self, evolve_config: dict = None) -> None:
        """
        Creates an experiment directory, dispatches sequence-level tracking jobs, 
        and optionally runs GSI postprocessing.
        """
        opt = self.args
        base = opt.project / 'mot' / f"{opt.yolo_model[0].stem}_{opt.reid_model[0].stem}_{opt.tracking_method}"
        exp_dir = increment_path(base, sep="_", exist_ok=False)
        exp_dir.mkdir(parents=True, exist_ok=True)
        opt.exp_folder_path = exp_dir

        sequence_names = sorted([
            d.name for d in Path(opt.source).iterdir()
            if d.is_dir() and (d / "img1").exists()
        ])

        task_args = [
            (
                seq,
                str(opt.source),
                str(opt.project),
                opt.yolo_model[0].stem,
                opt.reid_model[0].stem,
                opt.tracking_method,
                str(exp_dir),
                getattr(opt, 'fps', None)
            )
            for seq in sequence_names
        ]

        seq_frame_nums = {}
        max_workers = min(len(task_args), os.cpu_count() or 4)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.process_sequence, *args): args[0] for args in task_args
            }

            for fut in concurrent.futures.as_completed(futures):
                seq = futures[fut]
                try:
                    seq_name, kept_ids = fut.result()
                    seq_frame_nums[seq_name] = kept_ids
                except Exception as e:
                    LOGGER.error(f"Error processing {seq}: {e}")

        if getattr(opt, 'gsi', False):
            gsi(mot_results_folder=exp_dir)

    def run_trackeval(self) -> Dict[str, float]:
        """
        Prepares data for TrackEval, runs evaluation, and returns a dict of HOTA, MOTA, IDF1.
        """
        opt = self.args
        seq_paths, save_dir, MOT_results_folder, gt_folder = eval_setup(opt, opt.val_tools_path)
        trackeval_results = self.trackeval(seq_paths, save_dir, MOT_results_folder, gt_folder)
        hota_mota_idf1 = self.parse_mot_results(trackeval_results)
        if opt.verbose:
            LOGGER.info(trackeval_results)
            with open(opt.tracking_method + "_output.json", "w") as outfile:
                outfile.write(json.dumps(hota_mota_idf1))
        LOGGER.info(json.dumps(hota_mota_idf1))
        return hota_mota_idf1

def run_generate_dets_embs(args):
    pipeline = YoloTrackingPipeline(args)
    pipeline.run_generate_dets_embs()

def run_generate_mot_results(args):
    pipeline = YoloTrackingPipeline(args)
    pipeline.run_generate_mot_results()
    
def run_all(args):
    download_mot_eval_tools(args.val_tools_path)
    pipeline = YoloTrackingPipeline(args)
    pipeline.run_generate_dets_embs()
    pipeline.run_generate_mot_results()
    return pipeline.run_trackeval()
    

def main(args: argparse.Namespace):
    """
    Entry point when running as a script. Handles dataset download, splitting, cleanup,
    and dispatches to the appropriate pipeline stage based on args.command.
    """
    # download MOT benchmark
    download_mot_eval_tools(args.val_tools_path)

    if not Path(args.source).exists():
        zip_path = download_mot_dataset(args.val_tools_path, args.benchmark)
        unzip_mot_dataset(zip_path, args.val_tools_path, args.benchmark)

    if args.benchmark == 'MOT17':
        cleanup_mot17(args.source)

    if args.split_dataset:
        args.source, args.benchmark = split_dataset(args.source)

    pipeline = YoloTrackingPipeline(args)

    if args.command == 'generate_dets_embs':
        pipeline.run_generate_dets_embs()
    elif args.command == 'generate_mot_results':
        pipeline.run_generate_mot_results()
    elif args.command == 'trackeval':
        pipeline.run_trackeval()
    else:
        pipeline.run_generate_dets_embs()
        pipeline.run_generate_mot_results()
        pipeline.run_trackeval()


if __name__ == "__main__":
    main()
