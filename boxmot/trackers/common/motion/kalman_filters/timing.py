"""Elapsed intervals and continuous constant-velocity motion integration."""

from __future__ import annotations

from numbers import Real

import numpy as np


def validate_dt(dt: float | None) -> float | None:
    """Validate an elapsed interval without accepting booleans or arrays."""
    if dt is None:
        return None
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real):
        raise ValueError("dt must be a finite, strictly positive real scalar")
    try:
        dt = float(dt)
    except OverflowError as error:
        raise ValueError("dt must be a finite, strictly positive real scalar") from error
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite, strictly positive real scalar")
    return dt


def elapsed_motion(
    motion_mat: np.ndarray, density: np.ndarray, dt: float, *, reference_dt: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate constant-velocity F and continuous noise density over ``dt``.

    The transition defines ``A = (F - I) / reference_dt``, with ``A @ A = 0``.
    Noise density must already use the same units as the state and elapsed
    interval. The exact integral preserves covariance across subdivided gaps;
    either matrix may have leading batch dimensions.
    """
    dt = validate_dt(dt)
    if dt is None:
        raise ValueError("elapsed_motion requires an explicit dt")
    reference_dt = validate_dt(reference_dt)
    if reference_dt is None:
        raise ValueError("elapsed_motion requires an explicit reference_dt")
    identity = np.eye(motion_mat.shape[-1])
    generator = (motion_mat - identity) / reference_dt
    try:
        with np.errstate(over="raise", invalid="raise"):
            left = generator @ density
            transpose = generator.swapaxes(-1, -2)
            noise = dt * density + (dt**2 / 2.0) * (left + density @ transpose) + (dt**3 / 3.0) * (left @ transpose)
            transition = identity + dt * generator
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("dt produces an unrepresentable motion covariance") from error
    return transition, noise
