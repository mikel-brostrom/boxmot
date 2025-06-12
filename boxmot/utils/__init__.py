# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import sys
import threading
from pathlib import Path

import numpy as np
import multiprocessing as mp

# global logger
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory

DATA = ROOT / "data"
BOXMOT = ROOT / "boxmot"
TOML = ROOT / "pyproject.toml"
TRACKER_CONFIGS = ROOT / "boxmot" / "configs"

EXAMPLES = BOXMOT / "engine"
WEIGHTS = BOXMOT / "engine" / "weights"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))

def _is_main_process(record):
    return mp.current_process().name == "MainProcess"

def configure_logging():
    # this will remove *all* existing handlers and then add yours
    logger.configure(handlers=[
        {
            "sink": sys.stderr,
            "level":    "INFO",
            "filter":   _is_main_process,
        }
    ])
    
configure_logging()