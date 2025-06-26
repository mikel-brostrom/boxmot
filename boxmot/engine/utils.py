# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import json
import shutil
import time
import re

import zipfile
from pathlib import Path
from typing import Union, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from git import Repo, exc
from tqdm import tqdm
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from boxmot.utils import ROOT
from boxmot.utils import logger as LOGGER


def split_dataset(src_fldr: Path, percent_to_delete: float = 0.5) -> Tuple[Path, str]:
    """
    Copies the dataset to a new location and removes a specified percentage of images and annotations,
    adjusting the frame index to start at 1. Works for MOT17, MOT20, etc.

    Args:
        src_fldr (Path): Source folder (e.g. /…/MOT20/train or /…/MOT20/test)
        percent_to_delete (float): Fraction of the frames to drop (0.5 → drop 50%)

    Returns:
        dst_fldr (Path): The root of the new, smaller split (e.g. …/MOT20-50/train)
        new_benchmark_name (str): e.g. "MOT20-50"
    """
    src_fldr = Path(src_fldr)

    # --- detect the "MOTxx" part in the path ---
    m = re.search(r"(MOT\d+)", str(src_fldr))
    if not m:
        raise ValueError(f"Could not find MOT benchmark in path: {src_fldr}")
    benchmark = m.group(1)

    # build the new benchmark name
    new_benchmark_name = f"{benchmark}-ablation"
    dst_fldr = Path(str(src_fldr).replace(benchmark, new_benchmark_name))

    # copy entire folder tree if not already done
    if not dst_fldr.exists():
        for item in src_fldr.rglob("*"):
            target = dst_fldr / item.relative_to(src_fldr)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.write_bytes(item.read_bytes())

    # iterate every sequence under dst_fldr
    for seq_path in dst_fldr.iterdir():
        if not seq_path.is_dir():
            continue

        gt_path = seq_path / "gt" / "gt.txt"
        if not gt_path.exists():
            LOGGER.warning(f"Skipping `{seq_path}` – no gt.txt found")
            continue

        # load and compute split point
        df = pd.read_csv(gt_path, header=None)
        max_frame = int(df[0].max())
        split_frame = int(max_frame * (1 - percent_to_delete))

        if split_frame >= max_frame:
            LOGGER.info(f"`{seq_path}` already ≤ split size, skipping.")
            continue

        LOGGER.info(f"{seq_path.name}: keeping frames {split_frame+1}-{max_frame}")

        # filter and re‐index gt
        df = df[df[0] > split_frame].copy()
        df[0] = df[0] - split_frame
        df.to_csv(gt_path, header=False, index=False)

        # delete early images
        img_folder = seq_path / "img1"
        for img in img_folder.glob("*.jpg"):
            if int(img.stem) <= split_frame:
                img.unlink()

        # rename rest to 000001…000xxx
        remaining = sorted(img_folder.glob("*.jpg"))
        for idx, img in enumerate(remaining, start=1):
            img.rename(img_folder / f"{idx:06}.jpg")

        LOGGER.info(f"{seq_path.name}: now {len(remaining)} images")

    return dst_fldr, new_benchmark_name


def eval_setup(opt, val_tools_path):
    """
    Initializes and sets up evaluation paths for MOT challenge datasets.

    This function prepares the directories and paths needed for evaluating
    object tracking algorithms on MOT datasets like MOT17 or custom datasets like MOT17-mini.
    It filters sequence paths based on the detector (for MOT17), sets up the ground truth,
    sequences, and results directories according to the provided options.

    Parameters:
    - opt: An object with attributes that include benchmark (str), split (str),
      eval_existing (bool), project (str), and name (str). These options dictate
      the dataset to use, the split of the dataset, whether to evaluate on an
      existing setup, and the naming for the project and evaluation results directory.
    - val_tools_path: A string or Path object pointing to the base directory where
      the validation tools and datasets are located.

    Returns:
    - seq_paths: A list of Path objects pointing to the sequence directories to be evaluated.
    - save_dir: A Path object pointing to the directory where evaluation results will be saved.
    - MOT_results_folder: A Path object pointing to the directory where MOT challenge
      formatted results should be placed.
    - gt_folder: A Path object pointing to the directory where ground truth data is located.
    """

    # Convert val_tools_path to Path object if it's not already one
    val_tools_path = Path(val_tools_path)

    # Initial setup for paths based on benchmark and split options
    mot_seqs_path = val_tools_path / "data" / opt.benchmark / opt.split
    gt_folder = mot_seqs_path  # Assuming gt_folder is the same as mot_seqs_path initially
    
    # Handling different benchmarks
    if opt.benchmark == "MOT17":
        # Filter for FRCNN sequences in MOT17
        seq_paths = [p / "img1" for p in mot_seqs_path.iterdir() if p.is_dir()]
    elif opt.benchmark == "MOT17-mini":
        # Adjust paths for MOT17-mini
        base_path = ROOT / "assets" / opt.benchmark / opt.split
        mot_seqs_path = gt_folder = base_path
        seq_paths = [p / "img1" for p in mot_seqs_path.iterdir() if p.is_dir()]
    else:
        # Default handling for other datasets
        seq_paths = [p / "img1" for p in mot_seqs_path.iterdir() if p.is_dir()]

    # Determine save directory
    save_dir = Path(opt.project) / opt.name

    # Setup MOT results folder
    MOT_results_folder = (
        val_tools_path
        / "data"
        / "trackers"
        / "mot_challenge"
        / opt.benchmark
        / save_dir.name
        / "data"
    )
    MOT_results_folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    return seq_paths, save_dir, MOT_results_folder, gt_folder


def convert_to_mot_format(
    results: Union[Results, np.ndarray], frame_idx: int
) -> np.ndarray:
    """
    Converts tracking results for a single frame into MOT challenge format.

    This function supports inputs as either a custom object with a 'boxes' attribute or a numpy array.
    For custom object inputs, 'boxes' should contain 'id', 'xyxy', 'conf', and 'cls' sub-attributes.
    For numpy array inputs, the expected format per row is: (xmin, ymin, xmax, ymax, id, conf, cls).

    Parameters:
    - results (Union[Results, np.ndarray]): Tracking results for the current frame.
    - frame_idx (int): The zero-based index of the frame being processed.

    Returns:
    - np.ndarray: An array containing the MOT formatted results for the frame.
    """

    # Check if results are not empty
    if results.size != 0:
        if isinstance(results, np.ndarray):
            # Convert numpy array results to MOT format
            tlwh = ops.xyxy2ltwh(results[:, 0:4])
            frame_idx_column = np.full((results.shape[0], 1), frame_idx, dtype=np.int32)
            mot_results = np.column_stack((
                frame_idx_column, # frame index
                results[:, 4].astype(np.int32),  # track id
                tlwh.round().astype(np.int32),  # top,left,width,height
                np.ones((results.shape[0], 1), dtype=np.int32),  # "not ignored"
                results[:, 6].astype(np.int32),  # class
                results[:, 5],  # confidence (float)
            ))
            return mot_results
        else:
            # Convert ultralytics results to MOT format
            num_detections = len(results.boxes)
            frame_indices = torch.full((num_detections, 1), frame_idx + 1, dtype=torch.int32)
            not_ignored = torch.ones((num_detections, 1), dtype=torch.int32)

            mot_results = torch.cat([
                frame_indices, # frame index
                results.boxes.id.unsqueeze(1).astype(np.int32), # track id
                ops.xyxy2ltwh(results.boxes.xyxy).astype(np.int32),  ## top,left,width,height
                not_ignored, # "not ignored"
                results.boxes.cls.unsqueeze(1).astype(np.int32), # class
                results.boxes.conf.unsqueeze(1).astype(np.float32), # confidence (float)
            ], dim=1)

            return mot_results.numpy()


def write_mot_results(txt_path: Path, mot_results: np.ndarray) -> None:
    """
    Writes the MOT challenge formatted results to a text file.

    Parameters:
    - txt_path (Path): The path to the text file where results are saved.
    - mot_results (np.ndarray): An array containing the MOT formatted results.

    Note: The text file will be created if it does not exist, and the directory
    path to the file will be created as well if necessary.
    """
    if mot_results is not None:
        # Ensure the parent directory of the txt_path exists
        txt_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure the file exists before opening
        txt_path.touch(exist_ok=True)

        if mot_results.size != 0:
            # Open the file in append mode and save the MOT results
            with open(str(txt_path), "a") as file:
                np.savetxt(file, mot_results, fmt="%d,%d,%d,%d,%d,%d,%d,%d,%.6f")


# new_folder, name = split_dataset(Path("./boxmot/engine/trackeval/data/MOT20/train"), percent_to_delete=0.5)
# print(new_folder, name)