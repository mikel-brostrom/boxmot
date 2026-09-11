"""Supervised covariance calibration for saved 3D sensor datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from boxmot.datasets.inputs import DatasetInputs
from boxmot.engine.calibration.kalman import KalmanCalibrationResult, _write_json, fit_kalman_noise
from boxmot.engine.calibration.kalman_model_3d import CalibrationModel3D
from boxmot.engine.calibration.kalman_sensor_data import load_sensor_calibration_data
from boxmot.trackers.common.motion.kalman_filters.fitting import MIN_COVARIANCE_SCALE
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS


def calibrate_sensor_kalman(
    dataset: DatasetInputs,
    profiles: Mapping[int, Mapping[str, Any]],
    *,
    output_dir: Path,
    progress: Callable[[str], None] | None = None,
) -> KalmanCalibrationResult:
    """Fit each class's 3D Q, R and P0 scales with the shared 2D estimators.

    Annotations and matched detections use the exact ego-pose transform used
    by EagerMOT. Poses are treated as supplied inputs: their covariance cannot
    be identified separately from object and annotation errors by this fit.
    Insufficient per-class evidence retains that class's configured scales.
    The resulting YAML is reusable with ``--class-config``.
    """
    class_names = {
        int(value["id"]): name for name, value in dataset.classes.items() if value.get("evaluation") == "target"
    }
    if not profiles or set(profiles) != set(class_names):
        raise ValueError("3D KF calibration requires one tracker profile for each target dataset class.")
    baselines = {
        class_id: {**dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0), "is_angular": False, **profile}
        for class_id, profile in profiles.items()
    }
    models = {class_id: CalibrationModel3D(profile) for class_id, profile in baselines.items()}
    data = load_sensor_calibration_data(dataset, progress=progress)
    if not data.statistics["matched"]:
        raise ValueError("3D KF calibration found no detections matched to target 3D ground truth (IoU >= 0.5).")
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
        calibrated[name] = {**baseline, **{key: estimate["value"] for key, estimate in parameters.items()}}
        class_reports[name] = {
            "class_id": class_id,
            "state_dimensions": models[class_id].dim_x,
            "is_angular": baseline.get("is_angular", False),
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
    report = {
        "version": 2,
        "status": "complete",
        "method": "supervised_covariance_moments",
        "score_scope": "calibrated_on_selected_split",
        "tracker": "eagermot",
        "geometry": "box3d",
        "coordinates": "world",
        "box_order": ["x", "y", "z", "yaw", "length", "width", "height"],
        "dataset": dataset.id,
        "dataset_config": str(dataset.config_path) if dataset.config_path is not None else None,
        "split": dataset.split,
        "sequences": list(dataset.sequence_names),
        "per_class": True,
        "timing": {"variable_dt": False, "kf_time_unit": "frames", "fps": dataset.fps},
        "matching": {"method": "same_class_hungarian_3d_iou", "minimum_iou": data.match_iou, "coordinates": "camera"},
        "statistics": {**data.statistics, **totals},
        "ground_truth_sources": list(data.ground_truth_sources),
        "input_sources": list(data.input_sources),
        "classes": class_reports,
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
    temporary.write_text(yaml.safe_dump(calibrated, sort_keys=False), encoding="utf-8")
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
