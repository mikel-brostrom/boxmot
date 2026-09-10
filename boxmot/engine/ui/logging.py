"""Logging controls owned by engine workflows."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager


def configure_engine_logging(
    main_only: bool = True,
    main_thread_only: bool = False,
) -> logging.Logger:
    """Route BoxMOT logs through the same stderr console as engine Live UIs."""

    from boxmot.engine.ui.core.ui import get_console
    from boxmot.utils import configure_logging

    return configure_logging(
        main_only=main_only,
        main_thread_only=main_thread_only,
        console=get_console(stderr=True),
    )


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
