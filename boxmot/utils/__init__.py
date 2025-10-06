# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import sys
import threading
from pathlib import Path

import numpy as np
import multiprocessing as mp

# global logger
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TOML = ROOT / "pyproject.toml"

BOXMOT     = ROOT / "boxmot"
CONFIGS    = BOXMOT / "configs"
TRACKER_CONFIGS   = CONFIGS / "trackers"
DATASET_CONFIGS   = CONFIGS / "datasets"

ENGINE   = BOXMOT / "engine"
WEIGHTS  = ENGINE / "weights"
TRACKEVAL  = ENGINE / "trackeval"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads

def _is_main_process(record):
    # Works correctly even with enqueue=True
    return record["process"].name == "MainProcess"

def configure_logging(main_only: bool = True):
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        backtrace=True,
        diagnose=True,
        enqueue=True,  # safe with ProcessPool / spawn
        filter=_is_main_process if main_only else None,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> "
            "| {process.name}/{thread.name} "
            "| <level>{level: <8}</level> "
            "| <cyan>{file.path}</cyan>:<cyan>{line}</cyan> "
            "| {function} - <level>{message}</level>"
        ),
    )
    return logger
    
configure_logging()