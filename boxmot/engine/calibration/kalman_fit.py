"""Fit supervised covariance moments from matched labelled trajectories.

The reference models supply the same covariance equations as runtime. Generic
moment accumulators live beside the filters; this module maps dataset residuals
onto those equations and retains configured priors when evidence is insufficient.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from boxmot.trackers.common.motion.kalman_filters.fitting import (
    CoupledProcessNoiseMoments,
    DiagonalConfidenceNoiseMoments,
    DiagonalNoiseMoments,
    DiagonalProcessNoiseMoments,
    ProcessNoiseMoments,
    ScalarNoiseMoments,
)
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_DIAGONAL_OPTIONS, KALMAN_NOISE_OPTIONS

if TYPE_CHECKING:
    from boxmot.engine.calibration.kalman_data import CalibrationTrack
    from boxmot.engine.calibration.kalman_model import CalibrationModel
    from boxmot.engine.calibration.kalman_model_3d import CalibrationModel3D


def _measurements(
    track: CalibrationTrack,
    model: CalibrationModel | CalibrationModel3D,
    *,
    camera_warps: np.ndarray | None = None,
) -> np.ndarray:
    """Align GT box branches to the preceding box in the current camera frame."""
    values = []
    for index, box in enumerate(track.gt_boxes):
        reference = values[-1] if index and track.frame_indices[index] == track.frame_indices[index - 1] + 1 else None
        if reference is not None and camera_warps is not None and model.is_obb:
            state = np.zeros(model.dim_x)
            state[: model.dim_z] = reference
            warped, _ = model.camera_transform(state, camera_warps[int(track.frame_indices[index])])
            reference = warped[: model.dim_z]
        values.append(model.to_measurement(box, score=1.0, reference=reference))
    return np.asarray(values)


def _collect_track_moments(
    track: CalibrationTrack,
    model: CalibrationModel | CalibrationModel3D,
    *,
    variable_dt: bool,
    measurement: ScalarNoiseMoments | DiagonalNoiseMoments | DiagonalConfidenceNoiseMoments,
    initial_position: ScalarNoiseMoments | DiagonalNoiseMoments,
    initial_velocity: ScalarNoiseMoments | DiagonalNoiseMoments,
    process: ProcessNoiseMoments | DiagonalProcessNoiseMoments | CoupledProcessNoiseMoments,
    camera_warps: np.ndarray | None = None,
) -> None:
    """Use labelled motion and matched detector errors without association trials."""
    truth = _measurements(track, model, camera_warps=camera_warps)
    measured = np.asarray(model.measurement_indices, dtype=int)
    velocity = np.asarray(model.velocity_indices, dtype=int)
    velocity_measurements = np.asarray(model.velocity_measurement_indices, dtype=int)
    observation = np.eye(model.dim_x)[measured]
    measured_rows = {column: row for row, column in enumerate(measured)}
    if variable_dt:
        if track.timestamps_s is None:
            raise ValueError("Variable-time KF calibration requires capture timestamps on every frame.")
        times = np.asarray(track.timestamps_s)
    else:
        times = np.asarray(track.frame_indices, dtype=float)
    intervals = np.diff(times)
    if np.any(~np.isfinite(times)) or np.any(intervals <= 0):
        raise ValueError("Calibration trajectory times must be finite and strictly increasing.")
    adjacent = np.diff(track.frame_indices) == 1
    derivatives = np.diff(truth[:, velocity_measurements], axis=0) / intervals[:, None]
    birth_derivatives = derivatives
    if isinstance(process, CoupledProcessNoiseMoments):
        from boxmot.engine.calibration.kalman_compensation import compensated_secants

        derivatives, birth_derivatives = compensated_secants(track, model, truth, intervals, camera_warps)
    states = np.zeros((len(truth), model.dim_x))
    states[:, : model.dim_z] = truth
    if len(truth) > 1:
        states[1:, velocity] = derivatives
        states[0, velocity] = birth_derivatives[0]
        if isinstance(process, CoupledProcessNoiseMoments):
            for index in range(1, len(truth) - 1):
                if not adjacent[index - 1] and adjacent[index]:
                    states[index, velocity] = birth_derivatives[index]

    birth_seen = False
    for index, box in enumerate(track.detection_boxes):
        if not np.all(np.isfinite(box)):
            continue
        score = float(track.scores[index])
        observed = model.to_measurement(box, score=score, reference=truth[index])
        error = observed - truth[index]
        if isinstance(measurement, DiagonalConfidenceNoiseMoments):
            reference_r = model.measurement_covariance(observed, score=0.0)
            measurement.add(error[measured], np.diag(reference_r)[measured], confidence=score)
        else:
            reference_r = model.measurement_covariance(observed, score=score)
            measurement.add(error[measured], np.diag(reference_r)[measured])
        if not birth_seen:
            birth_seen = True
            initial_mean, reference_p = model.initial_state(observed)
            # Initialization copies observed geometry but can wrap its angle.
            # The aligned measurement error avoids artificial 2*pi birth errors.
            initial_position.add(error[measured], np.diag(reference_p)[measured])
            # A birth's velocity is latent. Use a local GT secant only when an
            # adjacent delivered frame has a label; never invent a zero velocity.
            if index and adjacent[index - 1]:
                birth_velocity = derivatives[index - 1]
            elif index + 1 < len(truth) and adjacent[index]:
                birth_velocity = birth_derivatives[index]
            else:
                birth_velocity = None
            if birth_velocity is not None:
                initial_velocity.add(initial_mean[velocity] - birth_velocity, np.diag(reference_p)[velocity])

    if isinstance(process, CoupledProcessNoiseMoments):
        from boxmot.engine.calibration.kalman_compensation import collect_compensated_process_moments

        collect_compensated_process_moments(
            track, model, truth, states, intervals, process, variable_dt=variable_dt, camera_warps=camera_warps
        )
        return

    # A GT secant is a noisy proxy for velocity. Project the process noise from
    # BOTH adjacent intervals into the three-position prediction error instead
    # of treating that secant as an independently observed latent velocity.
    # Consecutive residuals share one process increment; their cross covariance
    # distinguishes position and velocity diffusion even at regular intervals.
    previous_residual = None
    previous_index = -2
    previous_bases = None
    fixed_transition = None if variable_dt else model.transition()
    for index in range(1, len(truth) - 1):
        if not (adjacent[index - 1] and adjacent[index]):
            continue
        dt = float(intervals[index]) if variable_dt else None
        prior_dt = float(intervals[index - 1]) if variable_dt else None
        transition = model.transition(dt=dt) if variable_dt else fixed_transition
        predicted = transition @ states[index]
        error = truth[index + 1, measured] - predicted[measured]
        prior_projection = np.zeros_like(observation)
        for position_index, velocity_index in zip(velocity_measurements, velocity, strict=True):
            row = measured_rows[position_index]
            prior_projection[row, position_index] = -intervals[index] / intervals[index - 1]
            prior_projection[row, velocity_index] = intervals[index]
        # Adjacent residuals share the same interval/state covariance bases.
        # A gap starts a new pair, so it must recompute the earlier interval.
        prior_bases = (
            previous_bases
            if previous_index == index - 1
            else model.process_covariance_bases(states[index - 1], dt=prior_dt)
        )
        current_bases = model.process_covariance_bases(states[index], dt=dt)
        residual_bases = tuple(
            np.diag(observation @ current @ observation.T + prior_projection @ prior @ prior_projection.T)
            for prior, current in zip(prior_bases, current_bases, strict=True)
        )
        process.add(error, *residual_bases)
        if previous_index == index - 1:
            cross_bases = tuple(np.diag(observation @ prior @ prior_projection.T) for prior in prior_bases)
            process.add_cross_covariance(previous_residual, error, *cross_bases)
        previous_residual, previous_index = error, index
        previous_bases = current_bases


def fit_kalman_noise(
    tracks: Sequence[CalibrationTrack],
    models: Mapping[int, CalibrationModel | CalibrationModel3D],
    baseline: Mapping[str, Any],
    *,
    variable_dt: bool = False,
    progress: Callable[[str], None] | None = None,
    camera_warps: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Fit native 2D covariance axes or the existing 3D scalar parameterization.

    Optional ``camera_warps`` maps each sequence to dense previous-to-current
    full-resolution affine matrices, indexed by zero-based delivered frame.
    Camera behavior changes the velocity/process fit; same-frame detector
    noise and initial position estimates retain their native coordinates.
    """
    from boxmot.engine.calibration.kalman_model import CalibrationModel

    if not models:
        raise ValueError("KF calibration requires at least one reference model.")
    is_box_model = isinstance(next(iter(models.values())), CalibrationModel)
    if camera_warps is not None and not is_box_model:
        raise ValueError("Camera-compensated calibration requires a 2D Kalman tracker.")
    if camera_warps is not None:
        from boxmot.engine.calibration.kalman_compensation import validate_camera_warps

        camera_warps = validate_camera_warps(tracks, camera_warps)
    if is_box_model:
        return _fit_diagonal_noise(
            tracks, models, baseline, variable_dt=variable_dt, progress=progress, camera_warps=camera_warps
        )
    measurement, initial_position, initial_velocity = (ScalarNoiseMoments() for _ in range(3))
    process = ProcessNoiseMoments()
    for number, track in enumerate(tracks):
        _collect_track_moments(
            track,
            models[track.class_id],
            variable_dt=variable_dt,
            measurement=measurement,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            process=process,
        )
        if progress is not None and (number + 1) % 100 == 0:
            progress(f"KF calibration: estimating noise from GT trajectory {number + 1}/{len(tracks)}…")
    process_position, process_velocity = process.estimate(
        (float(baseline["kalman.noise.process_position_scale"]), float(baseline["kalman.noise.process_velocity_scale"]))
    )
    parameters = {
        "kalman.noise.process_position_scale": process_position,
        "kalman.noise.process_velocity_scale": process_velocity,
        "kalman.noise.measurement_noise_scale": measurement.estimate(
            float(baseline["kalman.noise.measurement_noise_scale"])
        ),
        "kalman.noise.initial_position_scale": initial_position.estimate(
            float(baseline["kalman.noise.initial_position_scale"])
        ),
        "kalman.noise.initial_velocity_scale": initial_velocity.estimate(
            float(baseline["kalman.noise.initial_velocity_scale"])
        ),
    }
    return parameters, {"gt_transitions": process.events, "gt_lag_pairs": process.lag_pairs}


def _diagonal_estimate(coordinates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize serializable coordinate estimates without hiding partial fallback."""
    entries = [dict(item) for item in coordinates]
    statuses = {item["status"] for item in entries if item["status"] != "excluded"}
    status = next(iter(statuses)) if len(statuses) == 1 else "partial" if statuses else "excluded"
    return {
        "value": [item["value"] for item in entries],
        "baseline": [item["baseline"] for item in entries],
        "status": status,
        "events": max((item["events"] for item in entries), default=0),
        "components": sum(item["components"] for item in entries),
        "coordinates": entries,
    }


def _fit_diagonal_noise(
    tracks: Sequence[CalibrationTrack],
    models: Mapping[int, CalibrationModel],
    baseline: Mapping[str, Any],
    *,
    variable_dt: bool,
    progress: Callable[[str], None] | None,
    camera_warps: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Fit native geometry axes, accounting for camera-induced axis coupling."""
    model = next(iter(models.values()))
    measured, velocities = model.measurement_indices, model.velocity_indices
    if any((item.kind, item.dim_x, item.dim_z) != (model.kind, model.dim_x, model.dim_z) for item in models.values()):
        raise ValueError("Pooled KF calibration requires the same state coordinates for every class.")
    confidence_fit = model.tracker_name == "strongsort"
    measurement = (
        DiagonalConfidenceNoiseMoments(len(measured)) if confidence_fit else DiagonalNoiseMoments(len(measured))
    )
    initial_position = DiagonalNoiseMoments(len(measured))
    initial_velocity = DiagonalNoiseMoments(len(velocities))
    process = (
        CoupledProcessNoiseMoments(len(measured) + len(velocities))
        if camera_warps is not None
        else DiagonalProcessNoiseMoments(len(measured))
    )
    for number, track in enumerate(tracks):
        _collect_track_moments(
            track,
            models[track.class_id],
            variable_dt=variable_dt,
            measurement=measurement,
            initial_position=initial_position,
            initial_velocity=initial_velocity,
            process=process,
            camera_warps=None if camera_warps is None else camera_warps[track.sequence_id],
        )
        if progress is not None and (number + 1) % 100 == 0:
            progress(f"KF calibration: estimating coordinate noise from GT trajectory {number + 1}/{len(tracks)}…")

    coordinates = {}
    for diagonal, scalar in zip(KALMAN_DIAGONAL_OPTIONS, KALMAN_NOISE_OPTIONS, strict=True):
        velocity_option = "velocity" in diagonal
        names = model.velocity_names if velocity_option else model.measurement_names
        supplied = baseline.get(diagonal, ())
        values = np.asarray(supplied if len(supplied) else np.ones(len(names)), dtype=float)
        if values.shape != (len(names),):
            raise ValueError(f"{diagonal} requires {len(names)} coordinate multipliers.")
        values = values * float(baseline[scalar])
        allowed = {index - model.dim_z for index in velocities} if velocity_option else set(measured)
        coordinates[diagonal] = [
            {
                "index": index,
                "coordinate": name,
                "baseline": float(values[index]) if index in allowed else 1.0,
                "value": float(values[index]) if index in allowed else 1.0,
                "status": "retained" if index in allowed else "excluded",
                "reason": "insufficient residual events" if index in allowed else "not supervised by box ground truth",
                "events": 0,
                "components": 0,
            }
            for index, name in enumerate(names)
        ]
    confidence_option = "kalman.noise.measurement_confidence_floor"
    if confidence_fit:
        supplied = baseline.get(confidence_option, ())
        defaults = np.zeros(model.dim_z)
        values = np.asarray(supplied if len(supplied) else defaults, dtype=float)
        if values.shape != (model.dim_z,):
            raise ValueError(f"{confidence_option} requires {model.dim_z} coordinate values.")
        coordinates[confidence_option] = [
            {
                "index": index,
                "coordinate": name,
                "baseline": float(values[index]) if index in measured else 0.0,
                "value": float(values[index]) if index in measured else 0.0,
                "status": "retained" if index in measured else "excluded",
                "reason": "insufficient residual events" if index in measured else "not supervised by box ground truth",
                "events": 0,
                "components": 0,
            }
            for index, name in enumerate(model.measurement_names)
        ]

    def assign(option: str, index: int, estimate: Mapping[str, Any]) -> None:
        """Attach a numerical estimate to its explicit runtime coordinate."""
        target = coordinates[option][index]
        coordinates[option][index] = {"index": index, "coordinate": target["coordinate"], **estimate}

    for option, collection, indices in (
        ("kalman.noise.measurement_noise_diagonal", measurement, measured),
        ("kalman.noise.initial_position_diagonal", initial_position, measured),
        (
            "kalman.noise.initial_velocity_diagonal",
            initial_velocity,
            tuple(index - model.dim_z for index in velocities),
        ),
    ):
        for index, moments in zip(indices, collection.moments, strict=True):
            if isinstance(collection, DiagonalConfidenceNoiseMoments):
                scale, floor = moments.estimate(
                    coordinates[option][index]["baseline"], coordinates[confidence_option][index]["baseline"]
                )
                assign(option, index, scale)
                assign(confidence_option, index, floor)
            else:
                assign(option, index, moments.estimate(coordinates[option][index]["baseline"]))

    position_option, velocity_option = KALMAN_DIAGONAL_OPTIONS[:2]
    if isinstance(process, CoupledProcessNoiseMoments):
        options = [(position_option, index) for index in measured] + [
            (velocity_option, index - model.dim_z) for index in velocities
        ]
        baselines = np.asarray([coordinates[name][index]["baseline"] for name, index in options])
        for (option, index), estimate in zip(options, process.estimate(baselines), strict=True):
            assign(option, index, estimate)
    else:
        velocity_by_measurement = dict(zip(model.velocity_measurement_indices, velocities, strict=True))
        for index, moments in zip(measured, process.moments, strict=True):
            velocity_index = velocity_by_measurement.get(index)
            velocity_offset = velocity_index - model.dim_z if velocity_index is not None else None
            estimates = moments.estimate(
                (
                    coordinates[position_option][index]["baseline"],
                    coordinates[velocity_option][velocity_offset]["baseline"] if velocity_offset is not None else 1.0,
                ),
                has_velocity=velocity_offset is not None,
            )
            assign(position_option, index, estimates[0])
            if velocity_offset is not None:
                assign(velocity_option, velocity_offset, estimates[1])

    parameters = {
        option: {
            "value": 1.0,
            "baseline": float(baseline[option]),
            "status": "fixed",
            "reason": "independent coordinate multipliers carry the calibration",
            "events": 0,
            "components": 0,
        }
        for option in KALMAN_NOISE_OPTIONS
    }
    parameters.update({option: _diagonal_estimate(entries) for option, entries in coordinates.items()})
    return parameters, {"gt_transitions": process.events, "gt_lag_pairs": process.lag_pairs}


def apply_global_noise_fallback(
    parameters: Mapping[str, Mapping[str, Any]], global_parameters: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Use an identifiable pooled estimate when a class has too little evidence."""
    result = {}
    for name, estimate in parameters.items():
        pooled = global_parameters[name]
        if "coordinates" in estimate:
            entries = apply_global_noise_fallback(
                {str(item["index"]): item for item in estimate["coordinates"]},
                {str(item["index"]): item for item in pooled["coordinates"]},
            )
            result[name] = _diagonal_estimate(tuple(entries.values()))
            continue
        if estimate["status"] == "retained" and pooled["status"] == "fitted":
            result[name] = {
                **estimate,
                "value": pooled["value"],
                "status": "global_fallback",
                "source": "pooled_classes",
                "global_events": pooled["events"],
                "global_components": pooled["components"],
            }
        else:
            result[name] = {**estimate, "source": "class" if estimate["status"] == "fitted" else "configured"}
    return result
