# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""Hardened callback wrappers for multiprocessing-safe progress reporting."""

from __future__ import annotations

import logging
from typing import Callable

_LOGGER = logging.getLogger(__name__)


def safe_progress_callback(
    callback: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    """Wrap a string progress callback so exceptions never crash the pipeline.

    Returns None if the input is None, otherwise returns a wrapper that
    catches and logs any exception raised by the underlying callback.
    """
    if callback is None:
        return None

    def _safe(message: str) -> None:
        try:
            callback(message)
        except Exception:
            _LOGGER.debug("Progress callback raised; suppressed.", exc_info=True)

    return _safe


def safe_seq_progress_callback(
    callback: "Callable[[str, int, int], None] | None",
) -> "Callable[[str, int, int], None] | None":
    """Wrap a per-sequence progress callback so exceptions never crash the pipeline.

    The callback signature is (seq_name, current, total).
    Returns None if the input is None.
    """
    if callback is None:
        return None

    def _safe(seq_name: str, current: int, total: int) -> None:
        try:
            callback(seq_name, current, total)
        except Exception:
            _LOGGER.debug("Seq progress callback raised; suppressed.", exc_info=True)

    return _safe
