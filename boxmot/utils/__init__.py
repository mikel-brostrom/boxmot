# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import os
import sys
import threading
from pathlib import Path

import numpy as np

# global logger
from loguru import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
DATA = ROOT / 'data'
BOXMOT = ROOT / "boxmot"
EXAMPLES = ROOT / "tracking"
TRACKER_CONFIGS = ROOT / "boxmot" / "configs"
WEIGHTS = ROOT / "tracking" / "weights"
REQUIREMENTS = ROOT / "requirements.txt"

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of BoxMOT multiprocessing threads


def only_main_thread(record):
    # Check if the current thread is the main thread
    return threading.current_thread().name == "MainThread"

logger.remove()
logger.add(sys.stderr, filter=only_main_thread, colorize=True, level="INFO")