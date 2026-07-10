# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import logging
import multiprocessing as mp
import os
import threading
from pathlib import Path

from rich.logging import RichHandler

from boxmot.utils.rich.core.ui import get_console

ROOT = Path(__file__).resolve().parents[2]

# If running from a cloned repository, prefer the local path
# This handles the case where the package is installed in site-packages
# but the user is running from the source root and expects local data/configs.
_local_root = Path.cwd()
if (_local_root / "pyproject.toml").is_file() and (_local_root / "boxmot").is_dir():
    ROOT = _local_root

TOML = ROOT / "pyproject.toml"

BOXMOT     = ROOT / "boxmot"
ENGINE     = BOXMOT / "engine"
CONFIGS    = BOXMOT / "configs"
TRACKER_CONFIGS   = CONFIGS / "trackers"
BENCHMARK_CONFIGS = CONFIGS / "benchmarks"
DATASETS = BOXMOT / "datasets"
MOT_DATASETS = DATASETS / "mot"
REID_DATASETS = DATASETS / "reid"

WEIGHTS  = ROOT / "models"
DATA = ROOT / "data"
BENCHMARK_DATA = MOT_DATASETS

NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of multiprocessing threads


# ----------------------------------------------------------------------------
# Logging — Rich-based.
# ----------------------------------------------------------------------------


_BOXMOT_LOGGER_NAME = "boxmot"
_stdlib_logger = logging.getLogger(_BOXMOT_LOGGER_NAME)


def _is_main_process() -> bool:
    return mp.current_process().name == "MainProcess"


def _is_main_process_main_thread() -> bool:
    return _is_main_process() and threading.current_thread() is threading.main_thread()


class _ProcessFilter(logging.Filter):
    def __init__(self, *, main_thread_only: bool) -> None:
        super().__init__()
        self._main_thread_only = main_thread_only

    def filter(self, record: logging.LogRecord) -> bool:
        if self._main_thread_only:
            return _is_main_process_main_thread()
        return _is_main_process()


def configure_logging(main_only: bool = True, main_thread_only: bool = False):
    """Configure the boxmot logger with a single Rich handler.

    Subsequent calls fully replace any previously installed handlers so the
    logger keeps a single output destination.
    """
    _stdlib_logger.handlers.clear()
    _stdlib_logger.setLevel(logging.INFO)
    _stdlib_logger.propagate = False

    handler = RichHandler(
        level=logging.INFO,
        console=get_console(stderr=True),
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    if main_only or main_thread_only:
        handler.addFilter(_ProcessFilter(main_thread_only=main_thread_only))
    _stdlib_logger.addHandler(handler)
    return logger


logger = _stdlib_logger
configure_logging()
