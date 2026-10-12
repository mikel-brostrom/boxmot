"""Replay detection-only Kalman histories and score held-out box forecasts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from boxmot.engine.calibration.kalman_data import CalibrationTrack
from boxmot.engine.calibration.kalman_folds import matched_detections
from boxmot.engine.calibration.kalman_forecasts import FORECAST_BATCH_SIZE, iter_forecasts
from boxmot.engine.eval.motmetrics import _aabb_iou_matrix, _rotated_iou_batch
from boxmot.trackers.common.geometry.obb import xywha_to_corners
from boxmot.trackers.common.motion.kalman_filters.config import normalize_kalman_config
from boxmot.trackers.common.motion.models import MotionModelAdapter, MotionModelKind, create_motion_model

DEFAULT_HORIZONS = (1, 3, 5)


def box_metrics(predicted: np.ndarray, truth: np.ndarray, *, is_obb: bool) -> tuple[float, float]:
    """Score physical boxes in common units, using exact oriented overlap."""
    if not np.isfinite(predicted).all():
        raise ValueError("Kalman prediction produced non-finite geometry.")
    if is_obb:
        if np.any(predicted[2:4] <= 0):
            raise ValueError("Kalman prediction produced nonpositive box dimensions.")
        iou = _rotated_iou_batch(
            xywha_to_corners(truth[None]).reshape(1, 8), xywha_to_corners(predicted[None]).reshape(1, 8)
        )[0, 0]
        center_error = np.linalg.norm(predicted[:2] - truth[:2]) / np.linalg.norm(truth[2:4])
    else:
        expected, forecast = truth.copy(), predicted.copy()
        expected[2:] -= expected[:2]
        forecast[2:] -= forecast[:2]
        if np.any(forecast[2:] <= 0):
            raise ValueError("Kalman prediction produced nonpositive box dimensions.")
        iou = _aabb_iou_matrix(expected[None], forecast[None])[0, 0]
        center_error = np.linalg.norm((predicted[:2] + predicted[2:] - truth[:2] - truth[2:]) / 2)
        center_error /= np.linalg.norm(expected[2:])
    if not np.isfinite(iou) or not np.isfinite(center_error):
        raise ValueError("Kalman prediction produced non-finite evaluation metrics.")
    return float(np.clip(iou, 0.0, 1.0)), float(center_error)


def _validate_state(mean: np.ndarray, covariance: np.ndarray) -> None:
    """Reject divergence instead of selectively dropping difficult forecasts."""
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Kalman prediction produced a non-finite state or covariance.")


def batch_box_metrics(predicted: np.ndarray, truth: np.ndarray, *, is_obb: bool) -> np.ndarray:
    """Score corresponding boxes in linear space, retaining exact rotated IoU."""
    if is_obb:
        return np.asarray([box_metrics(box, target, is_obb=True) for box, target in zip(predicted, truth, strict=True)])
    if not np.isfinite(predicted).all():
        raise ValueError("Kalman prediction produced non-finite geometry.")
    expected_size = truth[:, 2:] - truth[:, :2]
    predicted_size = predicted[:, 2:] - predicted[:, :2]
    if np.any(predicted_size <= 0):
        raise ValueError("Kalman prediction produced nonpositive box dimensions.")
    # Match the evaluator's xywh arithmetic without constructing an N-by-N IoU matrix.
    intersection_size = np.maximum(
        np.minimum(truth[:, :2] + expected_size, predicted[:, :2] + predicted_size)
        - np.maximum(truth[:, :2], predicted[:, :2]),
        0.0,
    )
    intersection = np.prod(intersection_size, axis=1)
    union = np.prod(expected_size, axis=1) + np.prod(predicted_size, axis=1) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=intersection != 0.0)
    error = np.linalg.norm((predicted[:, :2] + predicted[:, 2:] - truth[:, :2] - truth[:, 2:]) / 2, axis=1)
    error /= np.linalg.norm(expected_size, axis=1)
    if not np.isfinite(iou).all() or not np.isfinite(error).all():
        raise ValueError("Kalman prediction produced non-finite evaluation metrics.")
    return np.column_stack((np.clip(iou, 0.0, 1.0), error))


def _score_forecasts(
    track: CalibrationTrack,
    model: MotionModelAdapter,
    kalman: Any,
    origins: Sequence[int],
    means: Sequence[np.ndarray],
    covariances: Sequence[np.ndarray],
    totals: dict[int, np.ndarray],
    camera_warps: np.ndarray | None = None,
) -> None:
    """Forecast a bounded set of posterior states and accumulate paired metrics."""
    boxes, targets, indices = [], [], []
    for horizon, source, target, predicted in iter_forecasts(
        model=model,
        kalman=kalman,
        means=np.asarray(means),
        covariances=np.asarray(covariances),
        origins=np.asarray(origins),
        frame_indices=track.frame_indices,
        horizons=tuple(totals),
        camera_warps=camera_warps,
    ):
        boxes.append(model.to_boxes(predicted))
        targets.append(target)
        indices.append(np.column_stack((source, np.full(len(source), horizon))))
    if not boxes:
        return
    indices = np.concatenate(indices)
    order = np.lexsort((indices[:, 1], indices[:, 0]))
    scores = batch_box_metrics(
        np.concatenate(boxes)[order], track.gt_boxes[np.concatenate(targets)[order]], is_obb=model.is_obb
    )
    horizons = indices[order, 1]
    for horizon, total in totals.items():
        selected = scores[horizons == horizon]
        total[:2] += selected.sum(axis=0)
        total[2] += len(selected)


def replay_track(
    track: CalibrationTrack,
    *,
    tracker_name: str,
    geometry: str,
    config: Mapping[str, Any],
    camera_warps: np.ndarray | None = None,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> dict[int, dict[str, Any]]:
    """Replay detection-only states and bounded open-loop forecasts from each observation.

    Forecasts use the same observed history and then withhold all intermediate
    observations. Fixed-step, nonadaptive forecasts are batched from bounded
    snapshots of posterior states. Adaptive noise and measured timestamps keep
    forecasts at their observation origin so future innovations cannot enter
    prediction. With CMC, each delivered frame applies its image-derived warp
    after prediction, including intermediate frames with detections withheld.
    These are detection-gap replays, not forecasts of unavailable future images.
    """
    settings = normalize_kalman_config(config, tracker_name=tracker_name)
    model = create_motion_model(
        settings.type,
        is_obb=geometry == "obb",
        cls_id=track.class_id,
        adaptive_kf=bool(settings.adaptive_kf),
    )
    noise = settings.noise.for_class(track.class_id)
    kalman = model.create_filter(noise_config=noise)
    mean = covariance = None
    matched = matched_detections(track)
    totals = {horizon: np.zeros(3) for horizon in horizons}
    batched = not settings.variable_dt and not (model.kind is MotionModelKind.XYHR and settings.adaptive_kf)
    origins, means, covariances = [], [], []
    for index, box in enumerate(track.detection_boxes):
        contiguous = index > 0 and track.frame_indices[index] == track.frame_indices[index - 1] + 1
        if not contiguous:
            if origins:
                _score_forecasts(track, model, kalman, origins, means, covariances, totals, camera_warps)
                origins.clear()
                means.clear()
                covariances.clear()
            mean = covariance = None
            kalman = model.create_filter(noise_config=noise)
        if mean is not None:
            dt = float(track.timestamps_s[index] - track.timestamps_s[index - 1]) if settings.variable_dt else None
            mean, covariance = model.predict(kalman, mean, covariance, dt=dt)
            if camera_warps is not None:
                warped_means, warped_covariances = model.transform_states(
                    mean[None], covariance[None], camera_warps[int(track.frame_indices[index])]
                )
                mean, covariance = warped_means[0], warped_covariances[0]
            _validate_state(mean, covariance)
        if not matched[index]:
            continue
        values = np.r_[box, track.scores[index]] if model.kind is MotionModelKind.XYSCR else box
        measurement = model.to_measurement(values, column=False)
        if mean is None:
            mean, covariance = model.initiate(kalman, measurement)
        else:
            confidence = float(track.scores[index]) if tracker_name == "strongsort" else 0.0
            mean, covariance = model.update(kalman, mean, covariance, measurement, confidence=confidence)
        _validate_state(mean, covariance)
        if batched:
            origins.append(index)
            means.append(mean.copy())
            covariances.append(covariance.copy())
            if len(origins) == FORECAST_BATCH_SIZE:
                _score_forecasts(track, model, kalman, origins, means, covariances, totals, camera_warps)
                origins.clear()
                means.clear()
                covariances.clear()
            continue
        forecast, uncertainty = mean.copy(), covariance.copy()
        for horizon in range(1, min(max(totals, default=0), len(track.gt_boxes) - index - 1) + 1):
            target = index + horizon
            if track.frame_indices[target] != track.frame_indices[target - 1] + 1:
                break
            dt = float(track.timestamps_s[target] - track.timestamps_s[target - 1]) if settings.variable_dt else None
            forecast, uncertainty = model.predict(kalman, forecast, uncertainty, dt=dt)
            if camera_warps is not None:
                warped_means, warped_covariances = model.transform_states(
                    forecast[None], uncertainty[None], camera_warps[int(track.frame_indices[target])]
                )
                forecast, uncertainty = warped_means[0], warped_covariances[0]
            _validate_state(forecast, uncertainty)
            if horizon in totals:
                iou, error = box_metrics(model.to_box(forecast)[0], track.gt_boxes[target], is_obb=model.is_obb)
                totals[horizon] += (iou, error, 1)
    if origins:
        _score_forecasts(track, model, kalman, origins, means, covariances, totals, camera_warps)
    return {
        horizon: {
            "mean_iou": float(iou / count),
            "normalized_center_error": float(error / count),
            "forecasts": int(count),
        }
        for horizon, (iou, error, count) in totals.items()
        if count
    }


def summarize_forecasts(
    records: Sequence[tuple[CalibrationTrack, dict]], *, primary_horizon: int | None = None
) -> dict[str, Any]:
    """Weight trajectories, sequences, and supported horizons equally in order."""
    horizons = {}
    for horizon in DEFAULT_HORIZONS:
        sequences = defaultdict(list)
        for track, metrics in records:
            if horizon in metrics:
                sequences[track.sequence_id].append(metrics[horizon])
        if not sequences:
            continue
        horizons[str(horizon)] = {
            name: float(np.mean([np.mean([row[name] for row in rows]) for rows in sequences.values()]))
            for name in ("mean_iou", "normalized_center_error")
        }
        horizons[str(horizon)].update(
            forecasts=sum(row["forecasts"] for rows in sequences.values() for row in rows),
            trajectories=sum(len(rows) for rows in sequences.values()),
            sequences=len(sequences),
        )
    if not horizons:
        raise ValueError("KF selection requires held-out predictions following matched detections.")
    objective_rows = list(horizons.values()) if primary_horizon is None else [horizons[str(primary_horizon)]]
    return {
        **{
            name: float(np.mean([row[name] for row in objective_rows]))
            for name in ("mean_iou", "normalized_center_error")
        },
        "forecasts": sum(row["forecasts"] for row in horizons.values()),
        "trajectories": sum(bool(metrics) for _, metrics in records),
        "sequences": len({track.sequence_id for track, metrics in records if metrics}),
        "horizons": horizons,
    }
