"""Camera-aware GT secants and process moments using runtime warp geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from boxmot.engine.calibration.kalman_data import CalibrationTrack
    from boxmot.engine.calibration.kalman_model import CalibrationModel
    from boxmot.trackers.common.motion.kalman_filters.fitting import CoupledProcessNoiseMoments


_IDENTITY = np.eye(3, dtype=float)[:2]


def validate_camera_warps(
    tracks: Sequence[CalibrationTrack], camera_warps: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray] | None:
    """Validate delivered-frame coverage, collapsing all-identity input to off."""
    frame_limits = {}
    for track in tracks:
        if len(track.frame_indices):
            if np.any(track.frame_indices < 0) or np.any(track.frame_indices != np.floor(track.frame_indices)):
                raise ValueError("Camera calibration requires nonnegative integer delivered-frame indices.")
            frame_limits[track.sequence_id] = max(
                frame_limits.get(track.sequence_id, 0), int(track.frame_indices[-1]) + 1
            )
    result = {}
    has_motion = False
    for sequence, frame_limit in frame_limits.items():
        if sequence not in camera_warps:
            raise ValueError(f"Camera warps are missing calibration sequence {sequence!r}.")
        warps = np.asarray(camera_warps[sequence], dtype=float)
        if warps.ndim != 3 or warps.shape[1:] != (2, 3) or len(warps) < frame_limit:
            raise ValueError(
                f"Camera warps for {sequence!r} require (N, 2, 3) covering {frame_limit} delivered frames."
            )
        if not np.all(np.isfinite(warps)) or np.any(np.abs(np.linalg.det(warps[:, :, :2])) < 1e-10):
            raise ValueError(f"Camera warps for {sequence!r} must be finite, nonsingular affine transforms.")
        result[sequence] = warps
        has_motion |= bool(np.any(warps != _IDENTITY))
    return result if has_motion else None


def _difference(model: CalibrationModel, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Subtract measurements on the same angular branch."""
    difference = first - second
    if model.is_obb:
        angle = model.motion_model.angle_measurement_index
        difference[angle] = (difference[angle] + np.pi) % (2.0 * np.pi) - np.pi
    return difference


def compensated_secants(
    track: CalibrationTrack,
    model: CalibrationModel,
    truth: np.ndarray,
    intervals: np.ndarray,
    camera_warps: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate object derivatives in each interval's destination and origin.

    The destination secant subtracts the camera-warped previous box, so a
    stationary world object has zero velocity even during camera motion.
    A birth inferred from a later label needs that derivative transported
    back to its own frame; a previous-frame birth uses the destination value.
    """
    velocity = np.asarray(model.velocity_indices, dtype=int)
    coordinates = np.asarray(model.velocity_measurement_indices, dtype=int)
    derivatives = np.zeros((len(intervals), len(velocity)))
    origin_derivatives = np.zeros_like(derivatives)
    for index, interval in enumerate(intervals):
        if track.frame_indices[index + 1] != track.frame_indices[index] + 1:
            continue
        state = np.zeros(model.dim_x)
        state[: model.dim_z] = truth[index]
        warp = _IDENTITY if camera_warps is None else camera_warps[int(track.frame_indices[index + 1])]
        warped, jacobian = model.camera_transform(state, warp)
        delta = _difference(model, truth[index + 1], warped[: model.dim_z])
        derivatives[index] = delta[coordinates] / interval
        velocity_transform = jacobian[np.ix_(velocity, velocity)]
        origin_derivatives[index] = np.linalg.lstsq(velocity_transform, derivatives[index], rcond=None)[0]
    return derivatives, origin_derivatives


def collect_compensated_process_moments(
    track: CalibrationTrack,
    model: CalibrationModel,
    truth: np.ndarray,
    states: np.ndarray,
    intervals: np.ndarray,
    process: CoupledProcessNoiseMoments,
    *,
    variable_dt: bool,
    camera_warps: np.ndarray | None,
) -> None:
    """Project both interval increments through predict then camera transform.

    With H selecting observations and T the runtime camera Jacobian, the
    current residual receives H T Q T' H'. A GT secant approximates velocity:
    its error is L w, where L has zero position rows and velocity rows
    ``V - D H / previous_dt``. The previous increment therefore enters via
    ``H T_current F_current L T_previous``. Adjacent residuals share exactly
    this previous increment, which also supplies their signed lag moments.

    This is the same local linear approximation runtime CMC uses for its
    covariance, including rotation/scale and position/velocity cross terms.
    """
    measured = np.asarray(model.measurement_indices, dtype=int)
    velocity = np.asarray(model.velocity_indices, dtype=int)
    velocity_coordinates = np.asarray(model.velocity_measurement_indices, dtype=int)
    observation = np.eye(model.dim_x)[measured]
    adjacent = np.diff(track.frame_indices) == 1
    previous_index = -2
    previous_residual = previous_bases = previous_projection = previous_jacobian = None
    fixed_transition = None if variable_dt else model.transition()
    for index in range(1, len(truth) - 1):
        if not (adjacent[index - 1] and adjacent[index]):
            continue
        dt = float(intervals[index]) if variable_dt else None
        prior_dt = float(intervals[index - 1]) if variable_dt else None
        transition = model.transition(dt=dt) if variable_dt else fixed_transition
        warp = _IDENTITY if camera_warps is None else camera_warps[int(track.frame_indices[index + 1])]
        predicted, jacobian = model.camera_transform(transition @ states[index], warp)
        error = _difference(model, truth[index + 1], predicted[: model.dim_z])[measured]
        current_projection = observation @ jacobian
        current_bases = model.process_coordinate_covariance_bases(states[index], dt=dt)
        if previous_index == index - 1:
            prior_bases, prior_jacobian = previous_bases, previous_jacobian
        else:
            prior_transition = model.transition(dt=prior_dt) if variable_dt else fixed_transition
            prior_warp = _IDENTITY if camera_warps is None else camera_warps[int(track.frame_indices[index])]
            _, prior_jacobian = model.camera_transform(prior_transition @ states[index - 1], prior_warp)
            prior_bases = model.process_coordinate_covariance_bases(states[index - 1], dt=prior_dt)
        secant_error = np.zeros((model.dim_x, model.dim_x))
        secant_error[velocity, velocity] = 1.0
        secant_error[velocity, velocity_coordinates] = -1.0 / intervals[index - 1]
        prior_projection = current_projection @ transition @ secant_error @ prior_jacobian
        projected = (
            current_projection @ current_bases @ current_projection.T
            + prior_projection @ prior_bases @ prior_projection.T
        )
        process.add(error, np.diagonal(projected, axis1=-2, axis2=-1))
        if previous_index == index - 1:
            cross = previous_projection @ prior_bases @ prior_projection.T
            process.add_cross_covariance(previous_residual, error, np.diagonal(cross, axis1=-2, axis2=-1))
        previous_index, previous_residual = index, error
        previous_bases, previous_jacobian = current_bases, jacobian
        previous_projection = current_projection
