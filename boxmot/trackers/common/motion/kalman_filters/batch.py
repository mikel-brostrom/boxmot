"""Batched Kalman arithmetic shared by image and spatial motion models."""

from __future__ import annotations

import numpy as np


def diagonal(values: np.ndarray) -> np.ndarray:
    """Build diagonal matrices from the last axis without looping over tracks."""
    values = np.asarray(values)
    result = np.zeros((*values.shape, values.shape[-1]), dtype=values.dtype)
    indices = np.arange(values.shape[-1])
    result[..., indices, indices] = values
    return result


def predict(
    mean: np.ndarray,
    covariance: np.ndarray,
    transition: np.ndarray,
    noise: np.ndarray,
    *,
    alpha_sq: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict row states using shared or per-track transition/noise matrices."""
    projected_mean = (transition @ mean[..., None])[..., 0]
    projected_covariance = transition @ covariance @ np.swapaxes(transition, -1, -2)
    alpha = np.asarray(alpha_sq)
    if alpha.ndim:
        alpha = alpha[..., None, None]
    return projected_mean, alpha * projected_covariance + noise


def _cholesky(covariance: np.ndarray, stabilize: bool) -> np.ndarray:
    """Factor a batch, repairing individual matrices only after a failure."""
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        if not stabilize:
            raise

    # Numerical repairs are rare and must use each track's own scale. Import
    # lazily to reuse the scalar policy without a module-level import cycle.
    from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter

    return np.asarray([np.tril(BaseKalmanFilter._safe_cho_factor(matrix)[0]) for matrix in covariance])


def correct(
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    observation: np.ndarray,
    noise: np.ndarray,
    *,
    joseph: bool = False,
    stabilize: bool = False,
    return_inverse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Correct row states with batched Cholesky solves and optional Joseph form.

    Returns posterior means/covariances, gain, innovation, projected covariance,
    and optionally its stabilized inverse. Matrices may be shared or batched.
    """
    observation_t = np.swapaxes(observation, -1, -2)
    cross_covariance = covariance @ observation_t
    projected_covariance = observation @ cross_covariance + noise
    if stabilize:
        projected_covariance = 0.5 * (projected_covariance + np.swapaxes(projected_covariance, -1, -2))
    factor = _cholesky(projected_covariance, stabilize)
    factor_t = np.swapaxes(factor, -1, -2)
    gain = np.swapaxes(
        np.linalg.solve(factor_t, np.linalg.solve(factor, np.swapaxes(cross_covariance, -1, -2))), -1, -2
    )
    innovation = measurement - (observation @ mean[..., None])[..., 0]
    new_mean = mean + (gain @ innovation[..., None])[..., 0]
    gain_t = np.swapaxes(gain, -1, -2)
    if joseph:
        residual = np.eye(mean.shape[-1]) - gain @ observation
        new_covariance = residual @ covariance @ np.swapaxes(residual, -1, -2) + gain @ noise @ gain_t
        new_covariance = 0.5 * (new_covariance + np.swapaxes(new_covariance, -1, -2))
    else:
        new_covariance = covariance - gain @ projected_covariance @ gain_t
    inverse = None
    if return_inverse:
        identity = np.broadcast_to(np.eye(measurement.shape[-1]), projected_covariance.shape)
        inverse = np.linalg.solve(factor_t, np.linalg.solve(factor, identity))
    return new_mean, new_covariance, gain, innovation, projected_covariance, inverse
