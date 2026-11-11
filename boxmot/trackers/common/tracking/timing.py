"""Capture timestamp validation shared by Python and native tracker adapters."""

from __future__ import annotations

from numbers import Real

import numpy as np

from boxmot.structures import Frame


def resolve_frame_timing(
    frame: Frame | np.ndarray | None,
    timestamp_s: float | None,
    *,
    variable_dt: bool,
    last_timestamp_s: float | None,
) -> tuple[float | None, float | None]:
    """Validate capture time and derive elapsed seconds without advancing state."""
    if not variable_dt:
        return None, None
    frame_timestamp = frame.timestamp_s if isinstance(frame, Frame) else None
    if timestamp_s is not None and frame_timestamp is not None:
        raise ValueError("Supply timestamp_s either in Frame or as an update argument, not both.")
    timestamp_s = frame_timestamp if timestamp_s is None else timestamp_s
    if timestamp_s is None:
        raise ValueError("variable_dt=True requires timestamp_s on every frame, including the first.")
    if isinstance(timestamp_s, (bool, np.bool_)) or not isinstance(timestamp_s, Real):
        raise ValueError("timestamp_s must be a finite real scalar.")
    try:
        timestamp_s = float(timestamp_s)
    except OverflowError as error:
        raise ValueError("timestamp_s must be a finite real scalar.") from error
    if not np.isfinite(timestamp_s):
        raise ValueError("timestamp_s must be a finite real scalar.")
    dt = None
    if last_timestamp_s is not None:
        dt = timestamp_s - last_timestamp_s
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("timestamp_s must increase by a finite positive interval; reset for a new sequence.")
    return timestamp_s, dt
