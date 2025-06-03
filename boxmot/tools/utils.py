# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import Union

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


def split_dataset(src_fldr: Path, percent_to_delete: float = 0.5) -> None:
    """
    Copies the dataset to a new location and removes a specified percentage of images and annotations,
    adjusting the frame index to start at 1.

    Args:
        src_fldr (Path): Source folder containing the dataset.
        percent_to_delete (float): Percentage of images and annotations to remove.
    """
    # Ensure source path is a Path object
    src_fldr = Path(src_fldr)

    # Generate the destination path by replacing "MOT17" with "MOT17-half" in the source path
    new_benchmark_name = f"MOT17-{int(percent_to_delete * 100)}"
    dst_fldr = Path(str(src_fldr).replace("MOT17", new_benchmark_name))

    # Copy the dataset to a new location manually using pathlib if it doesn't already exist
    if not dst_fldr.exists():
        dst_fldr.mkdir(parents=True)
        for item in src_fldr.rglob("*"):
            if item.is_dir():
                (dst_fldr / item.relative_to(src_fldr)).mkdir(
                    parents=True, exist_ok=True
                )
            else:
                (dst_fldr / item.relative_to(src_fldr)).write_bytes(item.read_bytes())

    # List all sequences in the destination folder
    seq_paths = [f for f in dst_fldr.iterdir() if f.is_dir()]

    # Iterate over each sequence and remove a percentage of images and annotations
    for seq_path in seq_paths:
        seq_gt_path = seq_path / "gt" / "gt.txt"

        # Check if the gt.txt file exists
        if not seq_gt_path.exists():
            print(f"Ground truth file not found for {seq_path}. Skipping...")
            continue

        df = pd.read_csv(seq_gt_path, sep=",", header=None)
        nr_seq_imgs = df[0].unique().max()
        split = int(nr_seq_imgs * (1 - percent_to_delete))

        # Check if the sequence is already split
        if nr_seq_imgs <= split:
            print(f"Sequence {seq_path} already split. Skipping...")
            continue
        
        print(f'Number of annotated frames in {seq_path}: Keeping from frame {split + 1} to {nr_seq_imgs}')

        # Keep rows from the ground truth file beyond the split point
        df = df[df[0] > split]

        # Adjust the frame indices to start from 1
        df[0] = df[0] - split

        df.to_csv(seq_gt_path, header=None, index=None, sep=",")

        # Remove images before the split point using pathlib
        jpg_folder_path = seq_path / "img1"
        jpg_paths = list(jpg_folder_path.glob("*.jpg"))
        for jpg_path in jpg_paths:
            # Extract frame number from image file name (e.g., '000300.jpg' -> 300)
            frame_number = int(jpg_path.stem)
            # Check if this frame number is in the removed range
            if frame_number <= split:
                jpg_path.unlink()

        # Rename the remaining images to have a continuous sequence starting from 1
        remaining_jpg_paths = sorted(jpg_folder_path.glob("*.jpg"))
        for new_index, jpg_path in enumerate(remaining_jpg_paths, start=1):
            new_jpg_name = f"{new_index:06}.jpg"  # zero-padded to 6 digits
            jpg_path.rename(jpg_folder_path / new_jpg_name)

        remaining_images = len(list(jpg_folder_path.glob("*.jpg")))
        print(f"Number of images in {seq_path} after delete: {remaining_images}")

    return dst_fldr, new_benchmark_name


def download_mot_eval_tools(val_tools_path):
    """
    Download the official evaluation tools for MOT metrics from the GitHub repository.

    Parameters:
        val_tools_path (Path): Path to the destination folder where the evaluation tools will be downloaded.

    Returns:
        None. Clones the evaluation tools repository and updates deprecated numpy types.
    """
    val_tools_url = "https://github.com/JonathonLuiten/TrackEval"

    try:
        # Clone the repository
        Repo.clone_from(val_tools_url, val_tools_path)
        LOGGER.debug("Official MOT evaluation repo downloaded successfully.")
    except exc.GitError as err:
        LOGGER.debug(f"Evaluation repo already downloaded or an error occurred: {err}")

    # Fix deprecated np.float, np.int & np.bool by replacing them with native Python types
    deprecated_types = {"np.float": "float", "np.int": "int", "np.bool": "bool"}

    for file_path in val_tools_path.rglob("*"):
        if file_path.suffix in {".py", ".txt"}:  # only consider .py and .txt files
            try:
                content = file_path.read_text(encoding="utf-8")
                updated_content = content
                for old_type, new_type in deprecated_types.items():
                    updated_content = updated_content.replace(old_type, new_type)

                if updated_content != content:  # Only write back if there were changes
                    file_path.write_text(updated_content, encoding="utf-8")
                    LOGGER.info(f"Replaced deprecated types in {file_path}.")
            except Exception as e:
                LOGGER.error(f"Error processing {file_path}: {e}")


def download_mot_dataset(val_tools_path, benchmark, max_retries=5, backoff_factor=2):
    """
    Download a specific MOT dataset zip file with resumable support and retry logic.

    Parameters:
        val_tools_path (Path): Path to the destination folder where the MOT benchmark zip will be downloaded.
        benchmark (str): The MOT benchmark to download (e.g., 'MOT20', 'MOT17').
        max_retries (int): Maximum number of retries for the download in case of failure.
        backoff_factor (int): Exponential backoff factor for delays between retries.

    Returns:
        Path: The path to the downloaded zip file.
    """
    url = f"https://motchallenge.net/data/{benchmark}.zip"
    zip_dst = val_tools_path / f"{benchmark}.zip"

    retries = 0  # Initialize retry counter

    response = None
    while retries <= max_retries:
        try:
            response = requests.head(url, allow_redirects=True)
            # Consider any status code less than 400 (e.g., 200, 302) as indicating that the resource exists
            if response.status_code < 400:
                # Get the total size of the file from the server
                total_size_in_bytes = int(response.headers.get("content-length", 0))

                # Check if there is already a partially or fully downloaded file
                if zip_dst.exists():
                    current_size = zip_dst.stat().st_size

                    # If the file is fully downloaded, skip the download
                    if current_size >= total_size_in_bytes:
                        LOGGER.info(f"{benchmark}.zip is already fully downloaded.")
                        return zip_dst

                    # If the file is partially downloaded, set the range header to resume
                    resume_header = {"Range": f"bytes={current_size}-"}
                    LOGGER.info(
                        f"Resuming download for {benchmark}.zip from byte {current_size}..."
                    )
                else:
                    current_size = 0
                    resume_header = {}

                # Start or resume the download
                response = requests.get(url, headers=resume_header, stream=True)
                response.raise_for_status()  # Check for HTTP request errors

                with open(zip_dst, "ab") as file, tqdm(
                    desc=zip_dst.name,
                    total=total_size_in_bytes,
                    initial=current_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)

                LOGGER.info(f"{benchmark}.zip downloaded successfully.")
                return zip_dst  # If download is successful, return the path

            else:
                LOGGER.warning(f"{benchmark} is not downloadable from {url}")
                return None

        except (requests.HTTPError, requests.ConnectionError) as e:
            if response and response.status_code == 416:  # Handle "Requested Range Not Satisfiable" error
                LOGGER.info(f"{benchmark}.zip is already fully downloaded.")
                return zip_dst
            LOGGER.error(f"Error occurred while downloading {benchmark}.zip: {e}")
            retries += 1
            wait_time = backoff_factor**retries
            LOGGER.info(
                f"Retrying download in {wait_time} seconds... (Attempt {retries} of {max_retries})"
            )
            time.sleep(wait_time)  # Exponential backoff delay

        except Exception as e:
            LOGGER.error(f"An unexpected error occurred: {e}")
            retries += 1
            wait_time = backoff_factor**retries
            LOGGER.info(
                f"Retrying download in {wait_time} seconds... (Attempt {retries} of {max_retries})"
            )
            time.sleep(wait_time)  # Exponential backoff delay

    LOGGER.error(f"Failed to download {benchmark}.zip after {max_retries} retries.")
    return None


def unzip_mot_dataset(zip_path, val_tools_path, benchmark):
    """
    Unzip a downloaded MOT dataset zip file into the specified directory.

    Parameters:
        zip_path (Path): Path to the downloaded MOT benchmark zip file.
        val_tools_path (Path): Base path to the destination folder where the dataset will be unzipped.
        benchmark (str): The MOT benchmark that was downloaded (e.g., 'MOT20', 'MOT17').

    Returns:
        None
    """
    if zip_path is None:
        LOGGER.warning("No zip file. Skipping unzipping")
        return None

    extract_path = val_tools_path / "data" / benchmark
    if not extract_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # folder will be called as the original fetched file
                zip_ref.extractall(val_tools_path / "data")

            LOGGER.info(f"{benchmark}.zip unzipped successfully.")
        except zipfile.BadZipFile:
            LOGGER.error(
                f"{zip_path.name} is corrupted. Try deleting the file and run the script again."
            )
        except Exception as e:
            LOGGER.error(f"An error occurred while unzipping {zip_path.name}: {e}")
    else:
        LOGGER.info(f"{benchmark} folder already exists.")
        return extract_path


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
