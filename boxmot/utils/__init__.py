# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import logging
import multiprocessing as mp
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np

from rich.logging import RichHandler

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


# ----------------------------------------------------------------------------
# Logging — Rich-based, with a thin loguru-compatible wrapper.
# ----------------------------------------------------------------------------

_LOGURU_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")


def _strip_loguru_markup(msg: object) -> object:
    """Remove loguru ``<tag>`` markup from messages so plain Rich shows clean text."""
    if isinstance(msg, str) and "<" in msg and ">" in msg:
        return _LOGURU_TAG_RE.sub("", msg)
    return msg


class _BoxmotLogger:
    """Thin compatibility wrapper around a stdlib :class:`logging.Logger`.

    Provides the loguru API surface the codebase historically relied on
    (``opt``, ``add``, ``remove``, ``success``) while routing all output
    through a single :class:`rich.logging.RichHandler`. Loguru-style
    ``<tag>`` colour markup is stripped from messages — Rich's own markup
    is not enabled to keep log output unambiguous.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    # ---- Standard log levels ------------------------------------------------
    def debug(self, msg, *args, **kwargs):
        self._logger.debug(_strip_loguru_markup(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._logger.info(_strip_loguru_markup(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(_strip_loguru_markup(msg), *args, **kwargs)

    warn = warning

    def error(self, msg, *args, **kwargs):
        self._logger.error(_strip_loguru_markup(msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(_strip_loguru_markup(msg), *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        self._logger.exception(_strip_loguru_markup(msg), *args, **kwargs)

    def success(self, msg, *args, **kwargs):
        # loguru's SUCCESS level (25) — degrade to INFO for stdlib.
        self._logger.info(_strip_loguru_markup(msg), *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        self._logger.log(level, _strip_loguru_markup(msg), *args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)

    @property
    def level(self) -> int:
        return self._logger.level

    def setLevel(self, level) -> None:
        self._logger.setLevel(level)

    # ---- Loguru compatibility shims ----------------------------------------
    def opt(self, *_args, **_kwargs) -> "_BoxmotLogger":
        """No-op compatibility shim for ``loguru``'s ``logger.opt(colors=...)``."""
        return self

    def bind(self, **_kwargs) -> "_BoxmotLogger":
        return self

    def add(self, *_args, **kwargs) -> int:  # noqa: D401 — mimic loguru signature
        """Compatibility shim for loguru's ``logger.add``.

        Only the ``level`` argument is honoured (applied to the underlying
        stdlib logger). All other loguru-specific options are ignored.
        """
        level = kwargs.get("level")
        if level is not None:
            try:
                self._logger.setLevel(level)
            except (TypeError, ValueError):
                pass
        return 0

    def remove(self, *_args, **_kwargs) -> None:
        """Compatibility shim for loguru's ``logger.remove`` — no-op."""
        return None


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


_BOXMOT_LOGGER_NAME = "boxmot"
_stdlib_logger = logging.getLogger(_BOXMOT_LOGGER_NAME)


def configure_logging(main_only: bool = True, main_thread_only: bool = False):
    """Configure the boxmot logger with a single Rich handler.

    Subsequent calls fully replace any previously installed handlers so the
    logger keeps a single output destination, matching the previous loguru
    behaviour.
    """
    _stdlib_logger.handlers.clear()
    _stdlib_logger.setLevel(logging.INFO)
    _stdlib_logger.propagate = False

    handler = RichHandler(
        level=logging.INFO,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    if main_only or main_thread_only:
        handler.addFilter(_ProcessFilter(main_thread_only=main_thread_only))
    _stdlib_logger.addHandler(handler)
    return logger


logger = _BoxmotLogger(_stdlib_logger)
configure_logging()
