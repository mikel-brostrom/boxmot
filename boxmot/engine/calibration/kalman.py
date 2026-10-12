"""Orchestrate supervised KF calibration from cached detections and ground truth."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from boxmot.engine.calibration.kalman_artifacts import KalmanCalibrationResult, write_json, write_yaml
from boxmot.engine.calibration.kalman_fit import apply_global_noise_fallback, fit_kalman_noise
from boxmot.trackers.common.config import flatten_tracker_options, nest_tracker_options
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.fitting import MIN_COVARIANCE_SCALE
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_TRACKER_NAMES
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_profile_signature

if TYPE_CHECKING:
    from boxmot.engine.calibration.kalman_data import CalibrationData


def validate_kf_calibration(tracker_name: str, backend: str = "python") -> None:
    """Reject unsupported filters before materialization; no search dependency."""
    if backend != "python" or tracker_name not in KALMAN_NOISE_TRACKER_NAMES:
        raise ValueError("--calibrate-kf requires a Python tracker with a Kalman filter.")


def calibrate_kalman(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
    tracker_options: Mapping[str, Any] | None = None,
    calibration_data: CalibrationData | None = None,
    profile_name: str | None = None,
    camera_warps: Mapping[str, np.ndarray] | None = None,
) -> KalmanCalibrationResult:
    """Calibrate coordinate noise directly, leaving tracker evaluation to the caller.

    R uses matched detection/GT errors. Q fits the position and velocity process
    covariance bases to labelled constant-velocity prediction errors and their
    lag covariance, accounting for both adjacent intervals. P0 uses
    matched birth-position errors and local GT velocity proxies. No tracker
    search, metrics optimization, Ray runtime, or Optuna study is started.
    Callers comparing box-state filters may share already matched trajectories
    through ``calibration_data`` and save each filter under ``profile_name``.
    ``camera_warps`` calibrates the actual predict-then-compensate behavior;
    transforms must come from images, never from the annotated object motion.
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

        if calibration_data is not None or profile_name is not None or camera_warps is not None:
            raise ValueError("Shared calibration trajectories and named profiles require a 2D Kalman tracker.")
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
    if base_config.get("calibration.cmc_policy") == "auto" and camera_warps is None:
        raise ValueError("Recalibrating a CMC-bound profile requires its camera warps; run joint KF/CMC selection.")
    if getattr(args, "per_class", False):
        base_config["per_class"] = True
    validate_calibration_options(args.tracker, base_config, geometry=args.geometry)
    if profile_name is not None:
        from boxmot.trackers.common.motion.models import MotionModelKind, resolve_motion_model_kind

        profile_name = MotionModelKind(profile_name).value
        if (
            profile_name
            != resolve_motion_model_kind(
                args.tracker, base_config.get("kalman.type"), is_obb=args.geometry == "obb"
            ).value
        ):
            raise ValueError("Named calibration profiles must match the configured Kalman filter.")
    resolved_args = copy(args)
    resolved_args.variable_dt = base_config["kalman.variable_dt"]
    data = calibration_data
    if data is None:
        if progress is not None:
            progress("KF calibration: matching cached detections to ground truth…")
        data = load_calibration_data(resolved_args, progress=progress)
    if not data.statistics["matched"]:
        raise ValueError(
            "KF calibration requires detections matched to target ground truth; no valid matches were found."
        )
    per_class = bool(base_config.get("per_class", False))
    class_ids = {track.class_id for track in data.tracks}
    if per_class:
        class_ids.update(getattr(args, "tracker_class_ids", ()) or ())
    models = {
        class_id: CalibrationModel(args.tracker, args.geometry, base_config, cls_id=class_id)
        for class_id in sorted(class_ids)
    }
    parameters, statistics = fit_kalman_noise(
        data.tracks,
        models,
        base_config,
        variable_dt=base_config["kalman.variable_dt"],
        progress=progress,
        camera_warps=camera_warps,
    )
    config = {**base_config, **{name: estimate["value"] for name, estimate in parameters.items()}}
    class_reports = {}
    if per_class:
        config["per_class"] = True
        noise = normalize_kalman_config(base_config, tracker_name=args.tracker).noise
        class_names = dict(getattr(args, "tracker_class_names", ()) or ())
        for class_id, model in models.items():
            tracks = tuple(track for track in data.tracks if track.class_id == class_id)
            baseline = {
                f"kalman.noise.{name}": value
                for name, value in noise.for_class(class_id).to_dict().items()
                if name != "by_class"
            }
            class_parameters, class_statistics = fit_kalman_noise(
                tracks,
                models,
                baseline,
                variable_dt=base_config["kalman.variable_dt"],
                progress=progress,
                camera_warps=camera_warps,
            )
            class_parameters = apply_global_noise_fallback(class_parameters, parameters)
            class_profile = {**baseline, **{name: estimate["value"] for name, estimate in class_parameters.items()}}
            config.update(
                {
                    f"kalman.noise.by_class.{class_id}.{name.removeprefix('kalman.noise.')}": value
                    for name, value in class_profile.items()
                }
            )
            class_reports[str(class_id)] = {
                "class_id": class_id,
                "class_name": class_names.get(class_id),
                "filter": model.kind.value,
                "state_dimensions": model.dim_x,
                "measurement_dimensions": model.dim_z,
                "timing": {
                    "kalman.variable_dt": base_config["kalman.variable_dt"],
                    "kalman.noise.time_unit": model.noise_config.time_unit,
                    "kalman.noise.reference_dt_s": model.noise_config.reference_dt_s,
                },
                "statistics": {
                    "trajectories": len(tracks),
                    "ground_truth": sum(len(track.gt_boxes) for track in tracks),
                    "matched": sum(int(np.all(np.isfinite(track.detection_boxes), axis=1).sum()) for track in tracks),
                    **class_statistics,
                },
                "parameters": class_parameters,
            }
    directory = Path(output_dir) / "kf-tuning"
    if profile_name is not None:
        directory /= profile_name
    directory.mkdir(parents=True, exist_ok=True)
    config_path, report_path = directory / "calibrated.yaml", directory / "calibration.json"
    config.update(
        flatten_tracker_options(
            {
                "calibration": {
                    **calibration_profile_signature(args.tracker, args.geometry, config),
                    "dataset": args.dataset_id,
                    "split": args.split,
                    "source": str(report_path.resolve()),
                }
            }
        )
    )
    report = {
        "version": 4,
        "status": "complete",
        "method": "supervised_covariance_moments",
        "parameterization": "coordinate_diagonal",
        "measurement_confidence_model": (
            "per_coordinate_nonnegative_intercept_plus_slope_times_one_minus_confidence_squared"
            if args.tracker == "strongsort"
            else "configured_runtime_measurement_noise"
        ),
        "score_scope": "calibrated_on_selected_split",
        "tracker": args.tracker,
        "geometry": args.geometry,
        "dataset": args.dataset_id,
        "split": args.split,
        "build": str(args.build_path),
        "sequences": list(args.sequence_names or args.seq_info),
        "per_class": per_class,
        "camera_compensated": camera_warps is not None,
        "class_ids": list(getattr(args, "tracker_class_ids", ()) or ()),
        "class_names": dict(getattr(args, "tracker_class_names", ()) or ()),
        "timing": {
            name: config[name]
            for name in ("kalman.variable_dt", "kalman.noise.time_unit", "kalman.noise.reference_dt_s")
        },
        "matching": {"method": "same_class_hungarian_iou", "minimum_iou": data.match_iou},
        "statistics": {**data.statistics, **statistics},
        "ground_truth_sources": list(data.ground_truth_sources),
        "input_sources": list(data.input_sources),
        "parameters": parameters,
        "parameter_scope": "pooled_classes",
        "classes": class_reports,
        "filters": {
            str(class_id): {
                "kind": model.kind.value,
                "state_dimensions": model.dim_x,
                "measurement_dimensions": model.dim_z,
            }
            for class_id, model in models.items()
        },
        "numerical_scale_floor": MIN_COVARIANCE_SCALE,
        "baseline_config": base_config,
        "calibrated_config": str(config_path),
        "limitations": [
            "GT annotation noise contributes to residual moments; initial velocities use local GT secant proxies.",
            (
                "Process residuals follow predict then supplied CMC, using the runtime local covariance Jacobian; "
                "nonlinear curvature and camera-estimation uncertainty are absorbed into the residual fit."
                if camera_warps is not None
                else "Process residuals use image coordinates, including camera motion; "
                "they do not isolate tracker CMC."
            ),
            "Unlabelled confidence and confidence velocity retain their original covariance priors.",
            "Adaptive KF policies keep their configured online behavior after fixed reference calibration.",
            "Matched detections condition measurement estimates on the IoU gate and do not model false positives.",
            "Calibration does not optimize HOTA or establish accuracy on held-out sequences.",
        ],
    }
    write_yaml(config_path, {"tracker": args.tracker, **nest_tracker_options(config)})
    write_json(report_path, report)
    estimates = {
        f"{name}[{entry['index']}]": entry
        for name, estimate in parameters.items()
        for entry in estimate.get("coordinates", ())
        if entry["status"] != "excluded"
    }
    estimates.update(
        {
            f"kalman.noise.by_class.{class_id}.{name.removeprefix('kalman.noise.')}[{entry['index']}]": entry
            for class_id, class_report in class_reports.items()
            for name, estimate in class_report["parameters"].items()
            for entry in estimate.get("coordinates", ())
            if entry["status"] != "excluded"
        }
    )
    fitted = tuple(name for name, entry in estimates.items() if entry["status"] == "fitted")
    parameter_count = len(estimates)
    if progress is not None:
        progress(f"KF calibration complete: {len(fitted)}/{parameter_count} scales fitted.")
    return KalmanCalibrationResult(
        config_path,
        report_path,
        data.statistics["matched"],
        statistics["gt_transitions"],
        fitted,
        parameter_count=parameter_count,
    )
