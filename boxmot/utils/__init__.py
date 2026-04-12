# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import multiprocessing as mp
import os
import sys
import threading
from pathlib import Path

import numpy as np
# global logger
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]

# If running from a cloned repository, prefer the local path
# This handles the case where the package is installed in site-packages
# but the user is running from the source root and expects local data/configs.
_local_root = Path.cwd()
if (_local_root / "pyproject.toml").is_file() and (_local_root / "boxmot").is_dir():
    ROOT = _local_root

DATA = ROOT / "data"
TOML = ROOT / "pyproject.toml"

BOXMOT     = ROOT / "boxmot"
CONFIGS    = BOXMOT / "configs"
TRACKER_CONFIGS   = CONFIGS / "trackers"
BENCHMARK_CONFIGS = CONFIGS / "benchmarks"
DATASET_CONFIGS   = CONFIGS / "datasets"
DETECTOR_CONFIGS  = CONFIGS / "detectors"
REID_CONFIGS      = CONFIGS / "reid"

ENGINE   = BOXMOT / "engine"
WEIGHTS  = ROOT / "models"
TRACKEVAL  = ENGINE / "trackeval"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads

def _is_main_process(record):
    # Works correctly even with enqueue=True
    return record["process"].name == "MainProcess"

def _is_main_process_main_thread(record):
    return _is_main_process(record) and record["thread"].name == threading.main_thread().name


def configure_logging(main_only: bool = True, main_thread_only: bool = False):
    log_filter = None
    if main_thread_only:
        log_filter = _is_main_process_main_thread
    elif main_only:
        log_filter = _is_main_process

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # safe with ProcessPool / spawn
        filter=log_filter,
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )
    return logger
    
configure_logging()
