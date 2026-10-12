"""Bounded batches of independent open-loop Kalman selection forecasts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np

from boxmot.trackers.common.motion.models import MotionModelAdapter

FORECAST_BATCH_SIZE = 256


def iter_forecasts(
    *,
    model: MotionModelAdapter,
    kalman: Any,
    means: np.ndarray,
    covariances: np.ndarray,
    origins: np.ndarray,
    frame_indices: np.ndarray,
    horizons: Sequence[int],
    camera_warps: np.ndarray | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield horizon, origin rows, target rows, and predicted state means.

    The caller supplies posterior states at matched observations and a fixed
    timestep filter without adaptive noise. Forecasts advance one delivered
    frame at a time, retaining native size constraints, angle wrapping, and
    covariance validation even at intermediate, unscored horizons. Optional
    camera warps are indexed by delivered frame and applied after prediction.
    These become causal image-assisted detection-gap replays: later images
    provide motion estimates while intermediate detections remain withheld.
    Each row
    stops at the first annotation discontinuity or the end of the trajectory.

    Batches limit forecast work arrays independently of trajectory length.
    Results are emitted in batch/horizon order; original row indices allow
    callers to retain their original scoring and aggregation order.
    """
    if not horizons or not len(origins):
        return
    scored = frozenset(horizons)
    last_horizon = max(scored)
    for start in range(0, len(origins), FORECAST_BATCH_SIZE):
        stop = start + FORECAST_BATCH_SIZE
        active = np.asarray(origins[start:stop], dtype=np.intp).copy()
        predicted = np.asarray(means[start:stop]).copy()
        uncertainty = np.asarray(covariances[start:stop]).copy()
        for horizon in range(1, last_horizon + 1):
            targets = active + horizon
            available = targets < len(frame_indices)
            available[available] &= frame_indices[targets[available]] == frame_indices[targets[available] - 1] + 1
            if not np.any(available):
                break
            active = active[available]
            targets = targets[available]
            predicted, uncertainty = model.multi_predict(kalman, predicted[available], uncertainty[available])
            if camera_warps is not None:
                # Every row targets a different frame. Group equal transforms
                # so stationary sequences and disabled policy intervals stay
                # cheap, while sharing native covariance propagation.
                matrices, groups = np.unique(
                    camera_warps[np.asarray(frame_indices[targets], dtype=np.intp)].reshape(-1, 6),
                    axis=0,
                    return_inverse=True,
                )
                for group, matrix in enumerate(matrices):
                    selected = groups == group
                    predicted[selected], uncertainty[selected] = model.transform_states(
                        predicted[selected], uncertainty[selected], matrix.reshape(2, 3)
                    )
            if not np.isfinite(predicted).all() or not np.isfinite(uncertainty).all():
                raise ValueError("Kalman prediction produced a non-finite state or covariance.")
            if horizon in scored:
                yield horizon, active, targets, predicted
