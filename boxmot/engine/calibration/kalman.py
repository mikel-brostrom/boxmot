"""Supervised KF calibration from cached detections and ground-truth motion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from boxmot.trackers.common.motion.kalman_filters.fitting import (
    MIN_COVARIANCE_SCALE,
    ProcessNoiseMoments,
    ScalarNoiseMoments,
)
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS, KALMAN_NOISE_TRACKER_NAMES

if TYPE_CHECKING:
    from boxmot.engine.calibration.kalman_data import CalibrationTrack
    from boxmot.engine.calibration.kalman_model import CalibrationModel
    from boxmot.engine.calibration.kalman_model_3d import CalibrationModel3D
    from boxmot.engine.eval.results import ValidationResult


def validate_kf_calibration(tracker_name: str, backend: str = "python") -> None:
    """Reject unsupported filters before materialization; no search dependency."""
    if backend != "python" or tracker_name not in KALMAN_NOISE_TRACKER_NAMES:
        raise ValueError("--calibrate-kf requires a Python tracker with a Kalman filter.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish one complete calibration report."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class KalmanCalibrationResult:
    """Reusable calibrated settings and counts supporting their estimates."""

    config_path: Path
    report_path: Path
    matched_detections: int
    gt_transitions: int
    fitted_parameters: tuple[str, ...]
    parameter_count: int = 5

    @property
    def description(self) -> str:
        """Describe the calibration evidence alongside the one final evaluation."""
        return (
            f"KF calibration: {self.matched_detections} matched detections, {self.gt_transitions} GT transitions; "
            f"{len(self.fitted_parameters)}/{self.parameter_count} covariance scales fitted.\n"
            f"Saved tracker configuration: {self.config_path}"
        )

    def record_final(self, result: ValidationResult) -> None:
        """Attach the final evaluation score without using it to select parameters."""
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        report["final_summary"] = result.summary
        report["final_output_dir"] = str(result.exp_dir)
        _write_json(self.report_path, report)


def _measurements(track: CalibrationTrack, model: CalibrationModel | CalibrationModel3D) -> np.ndarray:
    """Keep box angles continuous within each annotated segment."""
    values = []
    for index, box in enumerate(track.gt_boxes):
        reference = values[-1] if index and track.frame_indices[index] == track.frame_indices[index - 1] + 1 else None
        values.append(model.to_measurement(box, score=1.0, reference=reference))
    return np.asarray(values)


def _collect_track_moments(
    track: CalibrationTrack,
    model: CalibrationModel | CalibrationModel3D,
    *,
    variable_dt: bool,
    measurement: ScalarNoiseMoments,
    initial_position: ScalarNoiseMoments,
    initial_velocity: ScalarNoiseMoments,
    process: ProcessNoiseMoments,
) -> None:
    """Use labelled motion and matched detector errors without association trials."""
    truth = _measurements(track, model)
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
    states = np.zeros((len(truth), model.dim_x))
    states[:, : model.dim_z] = truth
    if len(truth) > 1:
        states[1:, velocity] = derivatives
        states[0, velocity] = derivatives[0]

    birth_seen = False
    for index, box in enumerate(track.detection_boxes):
        if not np.all(np.isfinite(box)):
            continue
        score = float(track.scores[index])
        observed = model.to_measurement(box, score=score, reference=truth[index])
        error = observed - truth[index]
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
                birth_velocity = derivatives[index]
            else:
                birth_velocity = None
            if birth_velocity is not None:
                initial_velocity.add(initial_mean[velocity] - birth_velocity, np.diag(reference_p)[velocity])

    # A GT secant is a noisy proxy for velocity. Project the process noise from
    # BOTH adjacent intervals into the three-position prediction error instead
    # of treating that secant as an independently observed latent velocity.
    # Consecutive residuals share one process increment; their cross covariance
    # distinguishes position and velocity diffusion even at regular intervals.
    previous_residual = None
    previous_index = -2
    for index in range(1, len(truth) - 1):
        if not (adjacent[index - 1] and adjacent[index]):
            continue
        dt = float(intervals[index]) if variable_dt else None
        prior_dt = float(intervals[index - 1]) if variable_dt else None
        predicted = model.transition(dt=dt) @ states[index]
        error = truth[index + 1, measured] - predicted[measured]
        prior_projection = np.zeros_like(observation)
        for position_index, velocity_index in zip(velocity_measurements, velocity, strict=True):
            row = measured_rows[position_index]
            prior_projection[row, position_index] = -intervals[index] / intervals[index - 1]
            prior_projection[row, velocity_index] = intervals[index]
        prior_bases = model.process_covariance_bases(states[index - 1], dt=prior_dt)
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


def fit_kalman_noise(
    tracks: Sequence[CalibrationTrack],
    models: Mapping[int, CalibrationModel | CalibrationModel3D],
    baseline: Mapping[str, Any],
    *,
    variable_dt: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Fit the same five supervised covariance scales for 2D or 3D states."""
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
        (float(baseline["kf_process_position_scale"]), float(baseline["kf_process_velocity_scale"]))
    )
    parameters = {
        "kf_process_position_scale": process_position,
        "kf_process_velocity_scale": process_velocity,
        "kf_measurement_noise_scale": measurement.estimate(float(baseline["kf_measurement_noise_scale"])),
        "kf_initial_position_scale": initial_position.estimate(float(baseline["kf_initial_position_scale"])),
        "kf_initial_velocity_scale": initial_velocity.estimate(float(baseline["kf_initial_velocity_scale"])),
    }
    return parameters, {"gt_transitions": process.events, "gt_lag_pairs": process.lag_pairs}


def calibrate_kalman(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
    tracker_options: Mapping[str, Any] | None = None,
) -> KalmanCalibrationResult:
    """Calibrate five noise scales directly, leaving tracker evaluation to the caller.

    R uses matched detection/GT errors. Q fits the position and velocity process
    covariance bases to labelled constant-velocity prediction errors and their
    lag covariance, accounting for both adjacent intervals. P0 uses
    matched birth-position errors and local GT velocity proxies. No tracker
    search, metrics optimization, Ray runtime, or Optuna study is started.
    """
    from boxmot.engine.calibration.kalman_data import load_calibration_data
    from boxmot.engine.calibration.kalman_model import CalibrationModel, validate_calibration_options
    from boxmot.engine.config.trackers import resolve_tracker_options
    from boxmot.engine.eval.evaluator import _ensure_setup

    validate_kf_calibration(args.tracker, getattr(args, "tracker_backend", "python"))
    if args.tracker == "eagermot":
        from boxmot.datasets.inputs import load_dataset_inputs
        from boxmot.engine.calibration.kalman_sensor import calibrate_sensor_kalman
        from boxmot.engine.eval.eagermot_kitti import load_kitti_profiles

        if tracker_options or getattr(args, "tracker_config", None):
            raise ValueError("EagerMOT KF calibration uses --class-config for car and pedestrian profiles.")
        if getattr(args, "variable_dt", False):
            raise ValueError("EagerMOT 3D KF calibration uses fixed frame steps.")
        dataset = load_dataset_inputs(
            args.dataset, split=getattr(args, "split", None), sequence_names=getattr(args, "sequence_names", ()) or ()
        )
        with ExitStack() as resources:
            options = {}
            if getattr(args, "cache_inputs", False):
                from boxmot.datasets.sensor_cache import open_sensor_sequence, prepare_sensor_sequence

                sequences = {}
                for sequence_id in dataset.sequence_names:
                    path = prepare_sensor_sequence(dataset, sequence_id, progress=progress)
                    sequences[sequence_id] = open_sensor_sequence(path)
                    resources.callback(sequences[sequence_id].close)
                options["cached_sequences"] = sequences
            return calibrate_sensor_kalman(
                dataset,
                load_kitti_profiles(getattr(args, "class_config", None)),
                output_dir=output_dir,
                progress=progress,
                **options,
            )
    _ensure_setup(args)
    base_config = resolve_tracker_options(args, tracker_options, include_defaults=True, stamp_timing=True)
    validate_calibration_options(args.tracker, base_config)
    resolved_args = copy(args)
    resolved_args.variable_dt = base_config["variable_dt"]
    if progress is not None:
        progress("KF calibration: matching cached detections to ground truth…")
    data = load_calibration_data(resolved_args, progress=progress)
    if not data.statistics["matched"]:
        raise ValueError(
            "KF calibration requires detections matched to target ground truth; no valid matches were found."
        )
    models = {
        class_id: CalibrationModel(args.tracker, args.geometry, base_config, cls_id=class_id)
        for class_id in {track.class_id for track in data.tracks}
    }
    parameters, statistics = fit_kalman_noise(
        data.tracks, models, base_config, variable_dt=base_config["variable_dt"], progress=progress
    )
    config = {**base_config, **{name: estimate["value"] for name, estimate in parameters.items()}}
    directory = Path(output_dir) / "kf-tuning"
    directory.mkdir(parents=True, exist_ok=True)
    config_path, report_path = directory / "calibrated.yaml", directory / "calibration.json"
    report = {
        "version": 2,
        "status": "complete",
        "method": "supervised_covariance_moments",
        "score_scope": "calibrated_on_selected_split",
        "tracker": args.tracker,
        "geometry": args.geometry,
        "dataset": args.dataset_id,
        "split": args.split,
        "build": str(args.build_path),
        "sequences": list(args.sequence_names or args.seq_info),
        "per_class": bool(getattr(args, "per_class", False)),
        "class_ids": list(getattr(args, "tracker_class_ids", ()) or ()),
        "class_names": dict(getattr(args, "tracker_class_names", ()) or ()),
        "timing": {name: config[name] for name in ("variable_dt", "kf_time_unit", "kf_reference_dt_s")},
        "matching": {"method": "same_class_hungarian_iou", "minimum_iou": data.match_iou},
        "statistics": {**data.statistics, **statistics},
        "ground_truth_sources": list(data.ground_truth_sources),
        "parameters": parameters,
        "numerical_scale_floor": MIN_COVARIANCE_SCALE,
        "baseline_config": base_config,
        "calibrated_config": str(config_path),
        "limitations": [
            "GT annotation noise contributes to residual moments; initial velocities use local GT secant proxies.",
            "Process residuals use image coordinates, including camera motion; they do not isolate tracker CMC.",
            "Unlabelled confidence states are excluded; shared scales still affect the complete filter.",
            "Adaptive KF policies keep their configured online behavior after fixed reference calibration.",
            "Matched detections condition measurement estimates on the IoU gate and do not model false positives.",
            "Calibration does not optimize HOTA or establish accuracy on held-out sequences.",
        ],
    }
    temporary = config_path.with_suffix(".tmp")
    temporary.write_text(yaml.safe_dump({"tracker": args.tracker, **config}, sort_keys=False), encoding="utf-8")
    temporary.replace(config_path)
    _write_json(report_path, report)
    fitted = tuple(name for name in KALMAN_NOISE_OPTIONS if parameters[name]["status"] == "fitted")
    if progress is not None:
        progress(f"KF calibration complete: {len(fitted)}/5 scales fitted.")
    return KalmanCalibrationResult(
        config_path, report_path, data.statistics["matched"], statistics["gt_transitions"], fitted
    )
