# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from typing import Union
from pathlib import Path
import os
import sys
import git
import requests
import zipfile
import subprocess
from git import Repo, exc
from boxmot.utils import logger as LOGGER
from tqdm import tqdm
from boxmot.utils import EXAMPLES, ROOT


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
        LOGGER.info('Official MOT evaluation repo downloaded successfully.')
    except exc.GitError as err:
        LOGGER.info(f'Evaluation repo already downloaded or an error occurred: {err}')

    # Fix deprecated np.float, np.int & np.bool by replacing them with native Python types
    deprecated_types = {'np.float': 'float', 'np.int': 'int', 'np.bool': 'bool'}
    for old_type, new_type in deprecated_types.items():
        # check if there are any occurrences of the old_type
        grep_cmd = ["grep", "-rl", old_type, str(val_tools_path)]
        grep_result = subprocess.run(grep_cmd, stdout=subprocess.PIPE, text=True)

        # if occurrences are found, use sed to replace them
        if grep_result.stdout:
            cmd = f"grep -rl {old_type} {val_tools_path} | xargs sed -i 's/{old_type}/{new_type}/g'"
            try:
                subprocess.run(cmd, shell=True, check=True)
                LOGGER.info(f'Replaced all occurrences of {old_type} with {new_type}.')
            except subprocess.CalledProcessError as e:
                LOGGER.error(f'Error occurred while trying to replace {old_type} with {new_type}: {e}')
        else:
            LOGGER.info(f'No occurrences of {old_type} found in TrackEval to replace with {new_type}.')


def download_mot_dataset(val_tools_path, benchmark):
    """
    Download a specific MOT dataset zip file.
    
    Parameters:
        val_tools_path (Path): Path to the destination folder where the MOT benchmark zip will be downloaded.
        benchmark (str): The MOT benchmark to download (e.g., 'MOT20', 'MOT17').
    
    Returns:
        Path: The path to the downloaded zip file.
    """
    url = f'https://motchallenge.net/data/{benchmark}.zip'
    zip_dst = val_tools_path / f'{benchmark}.zip'

    if not zip_dst.exists():
        try:
            response = requests.head(url, allow_redirects=True)
            # Consider any status code less than 400 (e.g., 200, 302) as indicating that the resource exists
            if response.status_code < 400:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for HTTP request errors
                total_size_in_bytes = int(response.headers.get('content-length', 0))

                with open(zip_dst, 'wb') as file, tqdm(
                    desc=zip_dst.name,
                    total=total_size_in_bytes,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)
                LOGGER.info(f'{benchmark}.zip downloaded successfully.')
            else:
                LOGGER.warning(f'{benchmark} is not downloadeable from {url}')
                zip_dst = None
        except requests.HTTPError as e:
            LOGGER.error(f'HTTP Error occurred while downloading {benchmark}.zip: {e}')
        except Exception as e:
            LOGGER.error(f'An error occurred: {e}')
    else:
        LOGGER.info(f'{benchmark}.zip already exists.')
    return zip_dst


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
        LOGGER.warning(f'No zip file. Skipping unzipping')
        return None

    extract_path = val_tools_path / 'data' / benchmark
    if not extract_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            LOGGER.info(f'{benchmark}.zip unzipped successfully.')
        except zipfile.BadZipFile:
            LOGGER.error(f'{zip_path.name} is corrupted. Try deleting the file and run the script again.')
        except Exception as e:
            LOGGER.error(f'An error occurred while unzipping {zip_path.name}: {e}')
    else:
        LOGGER.info(f'{benchmark} folder already exists.')
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
    mot_seqs_path = val_tools_path / 'data' / opt.benchmark / opt.split
    gt_folder = mot_seqs_path  # Assuming gt_folder is the same as mot_seqs_path initially
    
    # Handling different benchmarks
    if opt.benchmark == 'MOT17':
        # Filter for FRCNN sequences in MOT17
        seq_paths = [p / 'img1' for p in mot_seqs_path.iterdir() if p.is_dir() and 'FRCNN' in str(p)]
    elif opt.benchmark == 'MOT17-mini':
        # Adjust paths for MOT17-mini
        base_path = ROOT / 'assets' / opt.benchmark / opt.split
        mot_seqs_path = gt_folder = base_path
        seq_paths = [p / 'img1' for p in mot_seqs_path.iterdir() if p.is_dir()]
    else:
        # Default handling for other datasets
        seq_paths = [p / 'img1' for p in mot_seqs_path.iterdir() if p.is_dir()]

    # Determine save directory
    save_dir = Path(opt.project) / opt.name


    # Setup MOT results folder
    MOT_results_folder = val_tools_path / 'data' / 'trackers' / 'mot_challenge' / opt.benchmark / save_dir.name / 'data'
    MOT_results_folder.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    
    return seq_paths, save_dir, MOT_results_folder, gt_folder



def convert_to_mot_format(results: Union[Results, np.ndarray], frame_idx: int) -> np.ndarray:
    """
    Converts tracking results for a single frame into MOT challenge format.

    Parameters:
    - results (object): An object containing detection results for the current frame.
                        Must have 'boxes' attribute with 'id', 'xyxy', 'conf', and 'cls' sub-attributes.
    - frame_idx (int): The zero-based index of the frame being processed.

    Returns:
    - torch.Tensor: A tensor containing the MOT formatted results for the frame.
    """
    """
    Converts tracking results for a single frame into MOT challenge format.
    
    This function supports inputs as either a custom object with a 'boxes' attribute or a numpy array.
    For custom object inputs, 'boxes' should contain 'id', 'xyxy', 'conf', and 'cls' sub-attributes.
    For numpy array inputs, the expected format per row is: (xmin, ymin, xmax, ymax, id, conf, cls).

    Parameters:
    - results (Union[Results, np.ndarray]): Tracking results for the current frame.
    - frame_idx (int): The zero-based index of the frame being processed.

    Returns:
    - Union[torch.Tensor, np.ndarray]: A tensor or array containing the MOT formatted results for the frame.
    """

    # do not try to safe if array is empty
    if results.size != 0:
        if isinstance(results, np.ndarray):
            # convert numpy array results to MOT format
            tlwh = ops.xyxy2ltwh(results[:, 0:4])
            frame_idx_column = np.full((results.shape[0], 1), frame_idx + 1)
            mot_results = np.column_stack((
                frame_idx_column,
                results[:, 4],  # track id
                tlwh,  # top,left,bottom, right
                results[:, 5],  # confidence
                results[:, 6],  # class
            ))
            return mot_results
        else:
            # convert ultralytics results to MOT format
            num_detections = len(results.boxes)
            frame_indices = torch.full((num_detections, 1), frame_idx + 1, dtype=torch.float32)
            dont_care_values = torch.full((num_detections, 1), -1, dtype=torch.float32)
            
            mot_results = torch.cat([
                frame_indices,
                results.boxes.id.unsqueeze(1).float(),
                ops.xyxy2ltwh(results.boxes.xyxy).float(),  # Convert bbox format
                results.boxes.conf.unsqueeze(1),
                results.boxes.cls.unsqueeze(1).float(),
                dont_care_values
            ], dim=1)

            return mot_results.numpy()


def write_mot_results(txt_path: Path, mot_results: torch.Tensor) -> None:
    """
    Writes the MOT challenge formatted results to a text file.

    Parameters:
    - txt_path (Path): The path to the text file where results are saved.
    - mot_results (torch.Tensor): A tensor containing the MOT formatted results.

    Note: The text file will be created if it does not exist, and the directory
    path to the file will be created as well if necessary.
    """
    if mot_results is not None:
        if mot_results.size != 0:
            # Ensure the parent directory of the txt_path exists
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure the file exists before opening
            txt_path.touch(exist_ok=True)

            # Open the file in append binary mode and save the MOT results
            with open(str(txt_path), 'ab+') as file:
                np.savetxt(file, mot_results, fmt='%.6f')
