"""Logging controls owned by engine workflows."""

from __future__ import annotations

import logging
import multiprocessing as mp
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from rich.logging import RichHandler


class _ProcessFilter(logging.Filter):
    """Keep CLI output on the owning process and optionally its main thread."""

    def __init__(self, *, main_thread_only: bool) -> None:
        super().__init__()
        self._main_thread_only = main_thread_only

    def filter(self, record: logging.LogRecord) -> bool:
        """Return whether this process and thread own terminal output."""
        is_main_process = mp.current_process().name == "MainProcess"
        return is_main_process and (
            not self._main_thread_only or threading.current_thread() is threading.main_thread()
        )


class _DashboardLogFilter(logging.Filter):
    """Keep engine log output inside an active dashboard's log drawer."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Capture a formatted record, falling back to normal terminal output."""
        from boxmot.engine.ui.dashboard import capture_dashboard_log

        message = record.getMessage()
        if record.exc_info:
            message += "\n" + logging.Formatter().formatException(record.exc_info)
        return not capture_dashboard_log(message, record.levelname)


def configure_engine_logging(
    main_only: bool = True,
    main_thread_only: bool = False,
) -> logging.Logger:
    """Configure CLI logs on the stderr console shared by engine Live UIs.

    This explicit application boundary replaces existing BoxMOT handlers. It
    is never called by domain imports or standalone component construction.
    """

    from boxmot.engine.ui.core.ui import get_console

    logger = logging.getLogger("boxmot")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

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
    handler.addFilter(_DashboardLogFilter())
    logger.addHandler(handler)
    return logger


@contextmanager
def suppress_boxmot_logs(enabled: bool, *, level: str = "WARNING") -> Iterator[None]:
    """Temporarily raise the BoxMOT logger level for quiet engine operations."""

    if not enabled:
        yield
        return

    boxmot_logger = logging.getLogger("boxmot")
    previous_level = boxmot_logger.level
    try:
        boxmot_logger.setLevel(getattr(logging, str(level).upper(), logging.WARNING))
        yield
    finally:
        boxmot_logger.setLevel(previous_level)


__all__ = ("configure_engine_logging", "suppress_boxmot_logs")
