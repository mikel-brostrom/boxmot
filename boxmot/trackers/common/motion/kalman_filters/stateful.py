"""Batch numerical steps while retaining observation-centric filter histories."""

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from boxmot.trackers.common.motion.kalman_filters import batch

if TYPE_CHECKING:
    from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter


def matrix_overrides(value, count: int) -> list:
    """Resolve a shared matrix/scalar or one optional override per filter."""
    if value is None or np.isscalar(value):
        return [value] * count
    if isinstance(value, np.ndarray) and value.ndim == 2:
        return [value] * count
    if len(value) != count:
        raise ValueError("Expected one matrix override per filter")
    return list(value)


def predict_many(
    filters: Sequence["BaseKalmanFilter"],
    *,
    dt: float | None = None,
    Q=None,
    F=None,
    score_state: bool = False,
) -> None:
    """Predict independent matrix-state filters using batched linear algebra.

    Q and F may be shared matrices or sequences of per-filter overrides.
    Explicit matrices are already discrete, matching ``predict_state``.
    History bookkeeping stays instance-local, including measured-time replay.
    """
    filters = list(filters)
    noises = matrix_overrides(Q, len(filters))
    transitions = matrix_overrides(F, len(filters))
    if len(filters) == 1:
        filters[0].predict(dt=dt, Q=noises[0], F=transitions[0])
        return
    groups = defaultdict(list)
    for kf, supplied_q, supplied_f in zip(filters, noises, transitions):
        interval = kf._validate_prediction_dt(dt)
        noise = kf.Q if supplied_q is None else supplied_q
        if np.isscalar(noise):
            noise = np.eye(kf.dim_x) * float(noise)
        if supplied_f is not None and supplied_q is not None:
            transition, integrated = supplied_f, noise
        else:
            transition, integrated = kf._elapsed_motion(noise, interval, motion_mat=kf.F)
        transition = supplied_f if supplied_f is not None else (kf.F if interval is None else transition)
        noise = noise if supplied_q is not None else integrated
        if interval is not None:
            kf._time_aware = True
            if kf._unrecorded_prediction:
                kf._prediction_history_overflowed = True
        if kf._time_aware:
            kf._record_prediction(kf.dt if interval is None else interval, transition, noise, None)
        else:
            kf._unrecorded_prediction = kf._last_observed_measurement is not None
        groups[kf.dim_x, kf.dim_z].append((kf, transition, noise))

    for group in groups.values():
        means, covariances = batch.predict(
            np.stack([kf.x[:, 0] for kf, _, _ in group]),
            np.stack([kf.P for kf, _, _ in group]),
            np.stack([transition for _, transition, _ in group]),
            np.stack([noise for _, _, noise in group]),
            alpha_sq=np.array([kf._alpha_sq for kf, _, _ in group]),
        )
        constrained_means = means.copy()
        positive_indices = (2, 4) if score_state else (2, 3)
        constrained_means[:, positive_indices] = np.maximum(constrained_means[:, positive_indices], 1e-6)
        if not score_state and group[0][0]._is_obb:
            constrained_means[:, 4] = group[0][0]._wrap_angle(constrained_means[:, 4])
        constrained_covariances = 0.5 * (covariances + covariances.swapaxes(-1, -2))
        for index, (kf, _, _) in enumerate(group):
            kf.x = constrained_means[index, :, None].copy()
            kf.P = constrained_covariances[index].copy()
            kf.x_prior = means[index, :, None].copy()
            kf.P_prior = covariances[index].copy()


def update_many(
    filters: Sequence["BaseKalmanFilter"],
    measurements: Sequence[np.ndarray | None],
    *,
    R=None,
    H=None,
    score_state: bool = False,
) -> None:
    """Correct observed filters in a batch, retaining scalar gap reconstruction.

    Missing observations and recovery interpolate each track's own history.
    Their scalar path remains intact; ordinary matched observations share the
    matrix solve and Joseph covariance update.
    """
    filters, measurements = list(filters), list(measurements)
    if len(measurements) != len(filters):
        raise ValueError("Expected one measurement per filter")
    noises = matrix_overrides(R, len(filters))
    observations = matrix_overrides(H, len(filters))
    if len(filters) == 1:
        filters[0].update(measurements[0], R=noises[0], H=observations[0])
        return
    groups = defaultdict(list)
    for kf, z, supplied_r, supplied_h in zip(filters, measurements, noises, observations):
        if z is None or not kf.observed:
            kf.update(z, R=supplied_r, H=supplied_h)
            continue
        if score_state:
            measurement = kf._prepare_measurement(z)
        else:
            measurement = kf._prepare_measurement(z, reference_state=kf._measurement_reference_state())
        kf.history_obs.append(measurement.copy())
        if not score_state and kf._time_aware:
            measurement = kf._prepare_measurement(measurement, reference_state=kf._measurement_reference_state())
        noise = kf.noise_config.measurement_covariance(kf.R) if supplied_r is None else supplied_r
        if np.isscalar(noise):
            noise = np.eye(kf.dim_z) * float(noise)
        observation = kf.H if supplied_h is None else supplied_h
        groups[kf.dim_x, kf.dim_z].append((kf, measurement, noise, observation))

    for group in groups.values():
        means, covariances, gains, innovations, projected, inverses = batch.correct(
            np.stack([kf.x[:, 0] for kf, _, _, _ in group]),
            np.stack([kf.P for kf, _, _, _ in group]),
            np.stack([measurement[:, 0] for _, measurement, _, _ in group]),
            np.stack([observation for _, _, _, observation in group]),
            np.stack([noise for _, _, noise, _ in group]),
            joseph=True,
            stabilize=True,
            return_inverse=True,
        )
        for index, (kf, measurement, _, _) in enumerate(group):
            kf.x = means[index, :, None].copy()
            kf.P = covariances[index].copy()
            kf.K, kf.y = gains[index].copy(), innovations[index, :, None].copy()
            kf.S, kf.SI = projected[index].copy(), inverses[index].copy()
            kf.z = measurement.copy()
            kf.x_post, kf.P_post = kf.x.copy(), kf.P.copy()
            if not score_state and kf._is_obb and kf.dim_x >= 9:
                kf.x = kf._damp_theta_velocity(kf.x, damping=0.8)
            kf._enforce_state_constraints()
            kf._remember_observation(measurement)
            if not score_state and not kf._time_aware:
                kf.history_obs.append(kf.z.copy())
