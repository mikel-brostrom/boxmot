"""Supervised covariance calibration for saved 3D sensor datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from boxmot.datasets.inputs import DatasetInputs
from boxmot.engine.calibration.kalman import (
    KalmanCalibrationResult,
    _write_json,
    apply_global_noise_fallback,
    fit_kalman_noise,
)
from boxmot.engine.calibration.kalman_model_3d import CalibrationModel3D
from boxmot.engine.calibration.kalman_sensor_data import load_sensor_calibration_data
from boxmot.trackers.common.config import flatten_tracker_options, nest_tracker_options
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.kalman_filters.fitting import MIN_COVARIANCE_SCALE
from boxmot.trackers.common.motion.kalman_filters.noise import (
    KALMAN_NOISE_OPTIONS,
    KALMAN_TIMING_OPTIONS,
)
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_profile_signature

if TYPE_CHECKING:
    from boxmot.datasets.sensor_cache import SensorReplaySequence


def calibrate_sensor_kalman(
    dataset: DatasetInputs,
    profiles: Mapping[int, Mapping[str, Any]],
    *,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
    cached_sequences: Mapping[str, SensorReplaySequence] | None = None,
) -> KalmanCalibrationResult:
    """Fit each class's 3D Q, R and P0 scales with the shared 2D estimators.

    Annotations and matched detections use EagerMOT's camera coordinates or
    its world coordinates when ego poses are declared. Pose covariance cannot
    be identified separately from object and annotation errors by this fit.
    Insufficient per-class evidence uses fitted pooled scales, or retains that
    class's configured scales when the pooled evidence is also insufficient.
    The resulting YAML is reusable with ``--class-config``.
    """
    class_names = {
        int(value["id"]): name for name, value in dataset.classes.items() if value.get("evaluation") == "target"
    }
    if not profiles or set(profiles) != set(class_names):
        raise ValueError("3D KF calibration requires one tracker profile for each target dataset class.")
    baselines = {
        class_id: {
            **dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0),
            "kalman.is_angular": False,
            **flatten_tracker_options(profile),
        }
        for class_id, profile in profiles.items()
    }
    for class_id, baseline in baselines.items():
        kalman = normalize_kalman_config(baseline, tracker_name="eagermot")
        noise = kalman.noise.for_class(class_id)
        baselines[class_id] = {
            **{key: value for key, value in baseline.items() if not key.startswith("kalman.noise.")},
            **{f"kalman.noise.{key}": value for key, value in noise.to_dict().items() if key != "by_class"},
            "kalman.variable_dt": kalman.variable_dt,
            "kalman.is_angular": kalman.is_angular,
        }
    models = {class_id: CalibrationModel3D(profile) for class_id, profile in baselines.items()}
    options = {} if cached_sequences is None else {"cached_sequences": cached_sequences}
    data = load_sensor_calibration_data(dataset, progress=progress, **options)
    if not data.statistics["matched"]:
        raise ValueError("3D KF calibration found no detections matched to target 3D ground truth (IoU >= 0.5).")
    global_parameters, global_statistics = fit_kalman_noise(
        data.tracks, models, dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0), progress=progress
    )
    calibrated: dict[str, dict[str, Any]] = {}
    class_reports: dict[str, dict[str, Any]] = {}
    fitted: list[str] = []
    totals = {"gt_transitions": 0, "gt_lag_pairs": 0}
    for class_id, baseline in baselines.items():
        name = class_names[class_id]
        tracks = tuple(track for track in data.tracks if track.class_id == class_id)
        if progress is not None:
            progress(f"KF calibration: fitting {name} 3D covariance scales…")
        parameters, statistics = fit_kalman_noise(tracks, models, baseline, progress=progress)
        parameters = apply_global_noise_fallback(parameters, global_parameters)
        calibrated[name] = {**baseline, **{key: estimate["value"] for key, estimate in parameters.items()}}
        class_reports[name] = {
            "class_id": class_id,
            "state_dimensions": models[class_id].dim_x,
            "filter": "box3d_angular" if baseline["kalman.is_angular"] else "box3d",
            "is_angular": baseline["kalman.is_angular"],
            "timing": {"kalman.variable_dt": False, **{key: baseline[key] for key in KALMAN_TIMING_OPTIONS}},
            "statistics": {
                "trajectories": len(tracks),
                "ground_truth": sum(len(track.gt_boxes) for track in tracks),
                "matched": sum(int(np.all(np.isfinite(track.detection_boxes), axis=1).sum()) for track in tracks),
                **statistics,
            },
            "parameters": parameters,
        }
        fitted.extend(f"{name}.{key}" for key in KALMAN_NOISE_OPTIONS if parameters[key]["status"] == "fitted")
        for key in totals:
            totals[key] += statistics[key]

    directory = Path(output_dir) / "kf-tuning"
    directory.mkdir(parents=True, exist_ok=True)
    config_path, report_path = directory / "calibrated.yaml", directory / "calibration.json"
    for class_id, name in class_names.items():
        calibrated[name].update(
            flatten_tracker_options(
                {
                    "calibration": {
                        **calibration_profile_signature("eagermot", "aabb", calibrated[name]),
                        "class_id": class_id,
                        "class_name": name,
                        "dataset": dataset.id,
                        "split": dataset.split,
                        "source": str(report_path.resolve()),
                    }
                }
            )
        )
    coordinate_frames = {
        source["sequence_id"]: source["value"] for source in data.input_sources if source["role"] == "coordinate_frame"
    }
    coordinate_values = set(coordinate_frames.values())
    report = {
        "version": 2,
        "status": "complete",
        "method": "supervised_covariance_moments",
        "score_scope": "calibrated_on_selected_split",
        "tracker": "eagermot",
        "geometry": "box3d",
        "coordinates": next(iter(coordinate_values))
        if len(coordinate_values) == 1
        else "mixed"
        if coordinate_values
        else None,
        "coordinate_frames": coordinate_frames,
        "box_order": ["x", "y", "z", "yaw", "length", "width", "height"],
        "dataset": dataset.id,
        "dataset_config": str(dataset.config_path) if dataset.config_path is not None else None,
        "split": dataset.split,
        "sequences": list(dataset.sequence_names),
        "per_class": True,
        "timing": {"kalman.variable_dt": False, "kalman.noise.time_unit": "frames", "fps": dataset.fps},
        "matching": {"method": "same_class_hungarian_3d_iou", "minimum_iou": data.match_iou, "coordinates": "camera"},
        "statistics": {**data.statistics, **totals},
        "ground_truth_sources": list(data.ground_truth_sources),
        "input_sources": list(data.input_sources),
        "classes": class_reports,
        "global": {"parameters": global_parameters, "statistics": global_statistics},
        "numerical_scale_floor": MIN_COVARIANCE_SCALE,
        "baseline_profiles": {class_names[class_id]: profile for class_id, profile in baselines.items()},
        "calibrated_config": str(config_path),
        "limitations": [
            "GT annotation noise contributes to residual moments; initial velocities use local GT secant proxies.",
            "Supplied ego poses are treated as exact; pose covariance is not fitted separately.",
            "Full rigid poses transform centers; box orientation follows the runtime yaw-only approximation.",
            "Yaw is pi-equivalent; angular velocity assumes less than pi/2 rotation between adjacent labels.",
            "Matched detections condition measurement estimates on the IoU gate and do not model false positives.",
            "Prediction advances one delivered frame; timestamp intervals do not change the 3D motion model.",
            "Calibration does not optimize HOTA or establish accuracy on held-out sequences.",
        ],
    }
    temporary = config_path.with_suffix(".tmp")
    temporary.write_text(
        yaml.safe_dump({name: nest_tracker_options(profile) for name, profile in calibrated.items()}, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(config_path)
    _write_json(report_path, report)
    result = KalmanCalibrationResult(
        config_path,
        report_path,
        data.statistics["matched"],
        totals["gt_transitions"],
        tuple(fitted),
        parameter_count=len(profiles) * len(KALMAN_NOISE_OPTIONS),
    )
    if progress is not None:
        progress(f"KF calibration complete: {len(fitted)}/{result.parameter_count} 3D scales fitted.")
    return result
