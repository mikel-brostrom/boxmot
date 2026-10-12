"""Model-aware Kalman operations for externally and independently owned state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from boxmot.trackers.common.geometry.obb import align_obb_measurement
from boxmot.trackers.common.motion.kalman_filters import batch
from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter
from boxmot.trackers.common.motion.kalman_filters.stateful import constrain_area_state
from boxmot.trackers.common.motion.kalman_filters.xyhr import KalmanFilterXYHR
from boxmot.trackers.common.motion.models import MotionModelKind

if TYPE_CHECKING:
    from boxmot.trackers.common.motion.models import MotionModelAdapter


_MATRIX_MODELS = (MotionModelKind.XYSR, MotionModelKind.XYSCR)
_MEAN_MODELS = (MotionModelKind.XYAH, MotionModelKind.XYWH)


def _constrain(model: MotionModelAdapter, mean: np.ndarray) -> np.ndarray:
    """Apply positive geometry and angular wrapping in native coordinates."""
    if model.kind is MotionModelKind.XYHR:
        return KalmanFilterXYHR.constrain_state(mean, model.dim_z)
    if model.kind in _MATRIX_MODELS:
        return constrain_area_state(
            mean,
            dim_z=model.dim_z,
            ratio_index=4 if model.kind is MotionModelKind.XYSCR else 3,
            angle_index=model.angle_measurement_index,
        )
    result = np.asarray(mean, dtype=float).copy()
    result[..., 2:4] = np.maximum(result[..., 2:4], 1e-4)
    if model.is_obb:
        angle = model.angle_measurement_index
        result[..., angle] = BaseKalmanFilter._wrap_angle(result[..., angle])
    return result


def _measurement(model: MotionModelAdapter, kalman: Any, measurement: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Align an observation without changing either caller-owned array."""
    result = np.asarray(measurement, dtype=float).reshape(-1).copy()
    if model.kind is MotionModelKind.XYSR:
        result = kalman._prepare_measurement(result, reference_state=mean if model.is_obb else None).reshape(-1)
    elif model.kind is MotionModelKind.XYSCR:
        result = kalman._prepare_measurement(result, reference_state=mean if model.is_obb else None).reshape(-1)
    elif model.is_obb:
        # Express equivalent rectangles through geometry rather than relying
        # on the particular order of ratio and size in the selected filter.
        aligned = align_obb_measurement(model.to_box(result)[0], model.to_box(mean)[0])
        result = model.to_measurement(aligned, column=False)
        result[4] = BaseKalmanFilter._align_angle_to_reference(result[4], mean[4])
    return result


def _motion(
    model: MotionModelAdapter, kalman: Any, mean: np.ndarray, dt: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve native transition/noise for prediction and replay recording."""
    if model.kind in _MATRIX_MODELS:
        return kalman._elapsed_motion(kalman.Q, dt, motion_mat=kalman.F)
    if model.kind is MotionModelKind.XYHR:
        return kalman._elapsed_motion(kalman.cov_update_policy.get_q(dt=dt), dt)
    position, velocity = kalman._get_process_noise_std(np.asarray(mean).reshape(-1))
    return kalman._elapsed_motion(np.diag(np.square(np.r_[position, velocity])), dt)


def predict(
    model: MotionModelAdapter, kalman: Any, mean: np.ndarray, covariance: np.ndarray, *, dt: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a flat mean with native model equations and independent state."""
    mean = np.asarray(mean).reshape(-1)
    if model.kind not in _MATRIX_MODELS:
        if model.kind is MotionModelKind.XYHR:
            kalman._last_motion_mat, kalman._last_process_noise = _motion(model, kalman, mean, dt)
            kalman._last_prediction_dt = dt
            predicted, uncertainty = kalman.predict(mean, covariance, dt=dt)
            return _constrain(model, predicted), 0.5 * (uncertainty + uncertainty.T)
        return kalman.predict(mean, covariance, dt=dt)
    transition, noise = _motion(model, kalman, mean, dt)
    predicted, covariance = batch.predict(mean, covariance, transition, noise, alpha_sq=kalman._alpha_sq)
    return _constrain(model, predicted), 0.5 * (covariance + covariance.T)


def multi_predict(
    model: MotionModelAdapter, kalman: Any, mean: np.ndarray, covariance: np.ndarray, *, dt: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Predict rows together, keeping native stateless fast paths unchanged."""
    if model.kind not in _MATRIX_MODELS:
        if model.kind is MotionModelKind.XYHR and len(mean):
            kalman._last_motion_mat, kalman._last_process_noise = _motion(model, kalman, mean[0], dt)
            kalman._last_prediction_dt = dt
            predicted, uncertainty = kalman.multi_predict(mean, covariance, dt=dt)
            return _constrain(model, predicted), 0.5 * (uncertainty + uncertainty.swapaxes(-1, -2))
        return kalman.multi_predict(mean, covariance, dt=dt)
    if not len(mean):
        return mean.copy(), covariance.copy()
    transition, noise = _motion(model, kalman, mean[0], dt)
    predicted, covariance = batch.predict(mean, covariance, transition, noise, alpha_sq=kalman._alpha_sq)
    return _constrain(model, predicted), 0.5 * (covariance + covariance.swapaxes(-1, -2))


def project(
    model: MotionModelAdapter, kalman: Any, mean: np.ndarray, covariance: np.ndarray, confidence: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Project using matrix-state or box-scaled measurement covariance."""
    mean = np.asarray(mean).reshape(-1)
    if model.kind in _MATRIX_MODELS:
        noise = kalman.noise_config.measurement_covariance(kalman.R, confidence=confidence)
        return kalman.H @ mean, kalman.H @ covariance @ kalman.H.T + noise
    if model.kind is MotionModelKind.XYHR:
        if kalman.noise_config.measurement_confidence_floor:
            noise = kalman.noise_config.measurement_covariance(kalman.cov_update_policy.get_r(), confidence=confidence)
            observation = kalman._update_mat
            return observation @ mean, observation @ covariance @ observation.T + noise
        projected, uncertainty = kalman.project(mean, covariance)
        if confidence:
            noise = kalman.noise_config.measurement_covariance(kalman.cov_update_policy.get_r())
            uncertainty = uncertainty + ((1.0 - confidence) ** 2 - 1.0) * noise
        return projected, uncertainty
    return kalman.project(mean, covariance, confidence)


def update(
    model: MotionModelAdapter,
    kalman: Any,
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    confidence: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct a flat state while retaining the model's noise and constraints."""
    if model.kind in _MEAN_MODELS:
        return kalman.update(mean, covariance, np.asarray(measurement).reshape(-1), confidence)
    result, uncertainty = multi_update(
        model,
        kalman,
        np.asarray(mean).reshape(1, -1),
        np.asarray(covariance)[None],
        np.asarray(measurement).reshape(1, -1),
        confidence,
    )
    return result[0], uncertainty[0]


def multi_update(
    model: MotionModelAdapter,
    kalman: Any,
    mean: np.ndarray,
    covariance: np.ndarray,
    measurement: np.ndarray,
    confidence: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct native matrix-state models without recording shared histories."""
    if model.kind in _MEAN_MODELS:
        return kalman.multi_update(mean, covariance, measurement, confidence)
    if not len(mean):
        return mean.copy(), covariance.copy()
    measurements = np.asarray(
        [_measurement(model, kalman, z, state) for z, state in zip(measurement, mean, strict=True)]
    )
    observation = kalman.H if model.kind in _MATRIX_MODELS else kalman._update_mat
    raw_noise = kalman.R if model.kind in _MATRIX_MODELS else kalman.cov_update_policy.get_r()
    confidence = np.broadcast_to(np.asarray(confidence), (len(mean),))
    noise = kalman.noise_config.measurement_covariance(raw_noise, confidence=confidence)
    corrected, uncertainty, gains, innovations, projected, _ = batch.correct(
        mean,
        covariance,
        measurements,
        observation,
        noise,
        joseph=True,
        stabilize=True,
    )
    if model.kind is MotionModelKind.XYHR:
        if len(mean) > 1 and hasattr(kalman.cov_update_policy, "_innovations"):
            raise ValueError("Adaptive XYHR requires an independent filter per track")
        for index in range(len(mean)):
            kalman.cov_update_policy.observe_innovation(
                innovations[index],
                gains[index],
                covariance[index],
                kalman._last_motion_mat,
                dt=kalman._last_prediction_dt,
                process_noise=kalman._last_process_noise,
                innovation_covariance=projected[index],
            )
    if model.is_obb:
        corrected[:, -1] *= 0.8
    return _constrain(model, corrected), uncertainty


def gating_distance(
    model: MotionModelAdapter,
    kalman: Any,
    mean: np.ndarray,
    covariance: np.ndarray,
    measurements: np.ndarray,
    only_position: bool = False,
    metric: str = "maha",
) -> np.ndarray:
    """Compute squared gating distances with geometry-aware angular residuals."""
    if model.kind in _MEAN_MODELS:
        return kalman.gating_distance(mean, covariance, measurements, only_position, metric)
    mean = np.asarray(mean).reshape(-1)
    measurements = np.asarray([_measurement(model, kalman, z, mean) for z in measurements]).reshape(-1, model.dim_z)
    projected, uncertainty = project(model, kalman, mean, covariance)
    if only_position:
        projected, uncertainty, measurements = projected[:2], uncertainty[:2, :2], measurements[:, :2]
    return BaseKalmanFilter._gating_from_residuals(measurements - projected, uncertainty, metric)


def predict_filter(
    model: MotionModelAdapter, kalman: Any, *, dt: float | None = None, observation_centric: bool = False
) -> None:
    """Advance an owned filter; record alternate-model gap origins and timing."""
    if model.kind in _MATRIX_MODELS:
        kalman.predict(dt=dt)
        return
    if observation_centric:
        transition, noise = _motion(model, kalman, kalman.x, dt)
        kalman._record_prediction(kalman.dt if dt is None else dt, transition, noise, None)
        kalman._time_aware = kalman._time_aware or dt is not None
    if model.kind is MotionModelKind.XYHR:
        kalman.predict(dt=dt)
        return
    else:
        shape = kalman.x.shape
        mean, kalman.P = predict(model, kalman, kalman.x, kalman.P, dt=dt)
        kalman.x = mean.reshape(shape)
    kalman.x_prior, kalman.P_prior = kalman.x.copy(), kalman.P.copy()


def _correct_filter(model: MotionModelAdapter, kalman: Any, measurement: np.ndarray, alpha: float) -> None:
    """Correct without history bookkeeping so replay cannot recurse."""
    old = kalman.x.copy()
    if model.kind in _MATRIX_MODELS:
        measurement = _measurement(model, kalman, measurement, np.asarray(kalman.x).reshape(-1))
        kalman._correct_observation(measurement[:, None])
        if alpha != 1.0:
            kalman.x = old + alpha * (kalman.x - old)
    elif model.kind is MotionModelKind.XYHR:
        measurement = _measurement(model, kalman, measurement, kalman.x)
        kalman.update(measurement, alpha=alpha)
    else:
        mean, kalman.P = update(model, kalman, kalman.x.reshape(-1), kalman.P, measurement)
        kalman.x = old + alpha * (mean.reshape(old.shape) - old)
    kalman.x_post, kalman.P_post = kalman.x.copy(), kalman.P.copy()


def _interpolate(model: MotionModelAdapter, previous: np.ndarray, current: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate physical box dimensions and optional confidence or angle."""
    previous, current = np.asarray(previous).reshape(-1), np.asarray(current).reshape(-1)
    box1, box2 = model.to_box(previous)[0], model.to_box(current)[0]
    if model.is_obb:
        box2 = align_obb_measurement(box2, box1)
    box = box1 + fraction * (box2 - box1)
    measurement = model.to_measurement(box, column=False)
    if model.kind is MotionModelKind.XYSCR:
        measurement[3] = previous[3] + fraction * (current[3] - previous[3])
    return measurement


def update_filter(
    model: MotionModelAdapter,
    kalman: Any,
    measurement: np.ndarray | None,
    *,
    observation_centric: bool = False,
    alpha: float = 1.0,
    confidence: float = 0.0,
) -> None:
    """Correct owned state, reconstructing intervening geometry when requested."""
    if model.kind is MotionModelKind.XYHR and not observation_centric:
        if measurement is not None:
            kalman.update(measurement, alpha=alpha)
        return
    if model.kind in _MATRIX_MODELS and observation_centric:
        if alpha != 1.0:
            raise ValueError("Gain suppression cannot be combined with observation-centric replay")
        kalman.update(measurement)
        return
    if measurement is None:
        kalman.history_obs.append(None)
        kalman.observed = False
        return
    measurement = _measurement(model, kalman, measurement, np.asarray(kalman.x).reshape(-1))
    if (
        observation_centric
        and not kalman.observed
        and kalman._prediction_origin is not None
        and len(kalman._prediction_steps) > 1
    ):
        kalman.x, kalman.P = (value.copy() for value in kalman._prediction_origin)
        total = sum(step[0] for step in kalman._prediction_steps)
        elapsed = 0.0
        for index, (interval, transition, noise, control) in enumerate(kalman._prediction_steps):
            mean = transition @ kalman.x.reshape(-1)
            if control is not None:
                mean += np.asarray(control).reshape(-1)
            kalman.x = _constrain(model, mean).reshape(kalman.x.shape)
            kalman.P = transition @ kalman.P @ transition.T + noise
            if model.kind is MotionModelKind.XYHR:
                kalman._last_motion_mat, kalman._last_process_noise = transition, noise
                # A discrete prediction uses the policy's discrete-noise window.
                kalman._last_prediction_dt = interval if kalman._time_aware else None
            elapsed += interval
            if index < len(kalman._prediction_steps) - 1:
                _correct_filter(
                    model,
                    kalman,
                    _interpolate(model, kalman._last_observed_measurement, measurement, elapsed / total),
                    alpha,
                )
    _correct_filter(model, kalman, measurement, alpha)
    kalman.observed = True
    kalman.history_obs.append(measurement.copy())
    kalman._remember_observation(measurement)


def predict_filters(
    model: MotionModelAdapter, filters: Sequence[Any], *, dt: float | None = None, observation_centric: bool = False
) -> None:
    """Retain optimized native matrix prediction and track-local replay state."""
    filters = list(filters)
    if model.kind in _MATRIX_MODELS or (model.kind is MotionModelKind.XYHR and not observation_centric):
        if filters:
            type(filters[0]).predict_many(filters, dt=dt)
        return
    for kalman in filters:
        predict_filter(model, kalman, dt=dt, observation_centric=observation_centric)


def update_filters(
    model: MotionModelAdapter,
    filters: Sequence[Any],
    measurements: Sequence[np.ndarray | None],
    *,
    observation_centric: bool = False,
    alpha: float | Sequence[float] = 1.0,
    confidence: float | Sequence[float] = 0.0,
) -> None:
    """Correct filters without sharing noise adaptation or recovery records."""
    filters, measurements = list(filters), list(measurements)
    if len(filters) != len(measurements):
        raise ValueError("Expected one measurement per filter")
    alphas = np.broadcast_to(np.asarray(alpha, dtype=float), (len(filters),))
    confidences = np.broadcast_to(np.asarray(confidence, dtype=float), (len(filters),))
    if filters and model.kind in _MATRIX_MODELS and observation_centric and np.all(alphas == 1.0):
        type(filters[0]).update_many(filters, measurements)
    elif (
        filters
        and model.kind is MotionModelKind.XYHR
        and not observation_centric
        and all(z is not None for z in measurements)
    ):
        type(filters[0]).update_many(filters, measurements, alpha=alphas)
    else:
        for kalman, measurement, gain_scale, score in zip(filters, measurements, alphas, confidences, strict=True):
            update_filter(model, kalman, measurement, observation_centric=observation_centric,
                          alpha=float(gain_scale), confidence=float(score))


def transform_states(
    model: MotionModelAdapter,
    means: np.ndarray,
    covariances: np.ndarray,
    transform: np.ndarray,
    *,
    return_jacobian: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp states with shape-preserving similarities for AABB area/XYHR models."""
    from boxmot.trackers.common.geometry import xywh2xyxy, xyxy2xywh
    from boxmot.trackers.common.motion.cmc.state import (
        is_orientation_preserving_similarity,
        transform_aabb_kalman_states,
        transform_obb_kalman_states,
        transform_similarity_kalman_states,
    )

    if (
        not model.is_obb
        and model.kind in (*_MATRIX_MODELS, MotionModelKind.XYHR)
        and is_orientation_preserving_similarity(transform)
    ):
        return transform_similarity_kalman_states(
            means,
            covariances,
            transform,
            measurement_size=model.dim_z,
            velocity_measurement_indices=model.velocity_measurement_indices,
            size_scale_exponent=1 if model.kind is MotionModelKind.XYHR else 2,
            return_jacobian=return_jacobian,
        )

    warp = transform_obb_kalman_states if model.is_obb else transform_aabb_kalman_states
    options = {}
    if model.kind is MotionModelKind.XYSCR:
        options = {"measurement_size": model.dim_z, "positive_indices": (2, 4), "preserved_measurement_indices": (3,)}
        if model.is_obb:
            options["angle_measurement_index"] = model.angle_measurement_index

    def identity(rows: np.ndarray) -> np.ndarray:
        """Retain native XYWH OBB finite-difference conversion semantics."""
        return np.asarray(rows).copy()

    measurement_to_box, box_to_measurement = model.to_boxes, model.to_measurements
    if model.kind in (MotionModelKind.XYWH,):
        # Finite differences amplify even rounding changes in coordinate
        # conversion. Preserve the original XYWH arithmetic in both geometries.
        measurement_to_box = identity if model.is_obb else xywh2xyxy
        box_to_measurement = identity if model.is_obb else xyxy2xywh
    return warp(
        means,
        covariances,
        transform,
        measurement_to_box=measurement_to_box,
        box_to_measurement=box_to_measurement,
        velocity_measurement_indices=model.velocity_measurement_indices,
        return_jacobian=return_jacobian,
        **options,
    )
