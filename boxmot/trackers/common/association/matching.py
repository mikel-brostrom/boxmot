# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from lapcpp import lapjv
from scipy.spatial.distance import cdist

from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.geometry.obb import align_obb_measurements, normalize_angle
from boxmot.trackers.common.motion.kalman_filters.base import BaseKalmanFilter

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


def solve_assignment(cost_matrix: np.ndarray, cost_limit: float | None = None) -> np.ndarray:
    """Return LAP pairs, allowing finite costs equal to an explicit limit."""
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    kwargs = {"extend_cost": True}
    eligible = np.isfinite(cost_matrix)
    if cost_limit is not None:
        # Keep the caller's cutoff in double precision even for float32 costs.
        eligible &= cost_matrix <= np.float64(cost_limit)
    if not eligible.any():
        return np.empty((0, 2), dtype=int)
    if not eligible.all():
        cost_matrix = np.where(eligible, cost_matrix, np.inf)
    if cost_limit is None and not eligible.all():
        # Permit unmatched rows rather than forcing forbidden edges, which can
        # steal a column from an otherwise eligible match.
        values = cost_matrix[eligible]
        cost_limit = float(values.max() + np.ptp(values) + max(1.0, np.abs(values).max()))
    if cost_limit is not None:
        # LAP can prefer rejection at equality. Raise its penalty in the
        # working precision; two steps survive cost_limit / 2 even at zero.
        # The original eligibility mask still forbids every above-limit edge.
        dtype = np.float32 if cost_matrix.dtype == np.float32 else np.float64
        limit = dtype(cost_limit)
        kwargs["cost_limit"] = np.nextafter(np.nextafter(limit, dtype(np.inf)), dtype(np.inf))

    matches = lapjv(cost_matrix, return_cost=False, output="pairs", **kwargs).astype(int, copy=False)
    return matches[eligible[matches[:, 0], matches[:, 1]]]


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve matches and return the remaining indices in their original order."""
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )
    matches = solve_assignment(cost_matrix, cost_limit=thresh)
    unmatched_a = np.ones(cost_matrix.shape[0], dtype=bool)
    unmatched_b = np.ones(cost_matrix.shape[1], dtype=bool)
    unmatched_a[matches[:, 0]] = False
    unmatched_b[matches[:, 1]] = False
    return matches, np.flatnonzero(unmatched_a), np.flatnonzero(unmatched_b)


def _iou_track_boxes(tracks: Sequence[Any], *, is_obb: bool) -> np.ndarray | list[np.ndarray]:
    """Use native batched geometry when it preserves the track's scalar hooks."""
    track_type = type(tracks[0])
    batch_geometry = getattr(track_type, "batch_geometry", None)
    if batch_geometry is not None and all(type(track) is track_type for track in tracks):
        boxes = batch_geometry(tracks, is_obb=is_obb)
        if boxes is not None:
            return boxes
    geometry_attr = "xywha" if is_obb else "xyxy"
    return [getattr(track, geometry_attr) for track in tracks]


def iou_distance(
    atracks: Sequence[Any] | np.ndarray,
    btracks: Sequence[Any] | np.ndarray,
    is_obb: bool = False,
) -> np.ndarray:
    """Return pairwise ``1 - IoU`` for box rows or native track geometry."""
    if not len(atracks) or not len(btracks):
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    if isinstance(atracks[0], np.ndarray) or isinstance(btracks[0], np.ndarray):
        atlbrs = atracks
        btlbrs = btracks
    else:
        is_obb = is_obb or any(getattr(track, "is_obb", False) for tracks in (atracks, btracks) for track in tracks)
        atlbrs = _iou_track_boxes(atracks, is_obb=is_obb)
        btlbrs = _iou_track_boxes(btracks, is_obb=is_obb)

    _ious = (
        AssociationFunction.iou_batch_obb(atlbrs, btlbrs) if is_obb else AssociationFunction.iou_batch(atlbrs, btlbrs)
    )

    cost_matrix = 1 - _ious

    return cost_matrix


def feature_distance(
    features_a,
    features_b,
    metric="cosine",
    *,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compute non-negative distances, leaving excluded candidates at infinity.

    Wide cosine features with a sparse geometric gate can avoid most pairwise
    dot products. Retain SciPy's per-pair arithmetic and use its dense operation
    for small workloads, dense gates, and metrics with batch-derived statistics.
    """
    features_a = np.asarray(features_a, dtype=np.float32)
    features_b = np.asarray(features_b, dtype=np.float32)
    shape = (len(features_a), len(features_b))
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask)
        if candidate_mask.dtype != np.bool_ or candidate_mask.shape != shape:
            raise ValueError("candidate_mask must be a boolean matrix matching the feature pair dimensions.")
    if len(features_a) == 0 or len(features_b) == 0:
        return np.zeros(shape, dtype=np.float32)
    if candidate_mask is not None:
        if features_a.ndim != 2 or features_b.ndim != 2 or features_a.shape[1] != features_b.shape[1]:
            raise ValueError("Feature arrays must be matrices with equal embedding dimensions.")
        candidates = np.count_nonzero(candidate_mask)
        if not candidates:
            return np.full(shape, np.inf)
        # Per-row dispatch costs more than a small dense cdist. Restrict it to
        # gates retaining at most 1/8 of pairs and skipping >=64K feature values
        # per track. These affect dispatch only, never numerical thresholds.
        skipped_values = (candidate_mask.size - candidates) * features_b.shape[1]
        if metric == "cosine" and skipped_values >= 65_536 * len(features_a) and candidates * 8 <= candidate_mask.size:
            distances = np.full(shape, np.inf)
            for row in np.flatnonzero(candidate_mask.any(axis=1)):
                columns = np.flatnonzero(candidate_mask[row])
                distances[row, columns] = np.maximum(
                    0.0, cdist(features_a[row:row + 1], features_b[columns], metric)[0],
                )
            return distances
    distances = np.maximum(0.0, cdist(features_a, features_b, metric))
    if candidate_mask is not None:
        distances[~candidate_mask] = np.inf
    return distances


def embedding_distance(
    tracks,
    detections,
    metric="cosine",
    *,
    candidate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Compare smoothed track and detection features within optional candidates."""

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    return feature_distance(track_features, det_features, metric, candidate_mask=candidate_mask)


def fuse_score(cost_matrix: np.ndarray, detections: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Multiply similarity by detection confidence, preserving forbidden edges.

    ``detections`` is a sequence of objects exposing ``conf`` or a one-dimensional
    NumPy array of confidences in cost-column order. Inputs are never modified.
    """
    costs = np.asarray(cost_matrix)
    if not costs.size:
        return costs.copy()
    confidences = (
        np.asarray(detections, dtype=float)
        if isinstance(detections, np.ndarray)
        else np.asarray([detection.conf for detection in detections], dtype=float)
    )
    if confidences.shape != (costs.shape[1],):
        raise ValueError("Expected one detection confidence per cost-matrix column.")
    result = np.full(costs.shape, np.inf, dtype=np.result_type(costs.dtype, confidences.dtype))
    rows, columns = np.nonzero(np.isfinite(costs))
    result[rows, columns] = 1.0 - (1.0 - costs[rows, columns]) * confidences[columns]
    return result


def fuse_iou(
    cost_matrix: np.ndarray,
    tracks: Sequence[Any] | np.ndarray,
    detections: Sequence[Any] | np.ndarray,
    *,
    is_obb: bool = False,
) -> np.ndarray:
    """Combine appearance similarity with IoU without changing input costs.

    The cost matrix must contain appearance distances in track-by-detection
    order. Each collection independently accepts track/detection objects or
    canonical box rows. Object ``is_obb`` flags also select oriented geometry.
    """
    costs = np.asarray(cost_matrix)
    if not costs.size:
        return costs.copy()
    is_obb = is_obb or any(getattr(item, "is_obb", False) for items in (tracks, detections) for item in items)
    geometry_attr = "xywha" if is_obb else "xyxy"
    track_boxes = np.asarray([getattr(track, geometry_attr, track) for track in tracks])
    detection_boxes = np.asarray([getattr(detection, geometry_attr, detection) for detection in detections])
    geometry_cost = iou_distance(track_boxes, detection_boxes, is_obb=is_obb)
    if geometry_cost.shape != costs.shape:
        raise ValueError("Box collections must match the appearance cost-matrix dimensions.")
    result = np.full(costs.shape, np.inf, dtype=np.result_type(costs.dtype, geometry_cost.dtype))
    finite = np.isfinite(costs)
    result[finite] = 1.0 - (1.0 - costs[finite]) * (1.0 - geometry_cost[finite] / 2.0)
    return result


def motion_gating_threshold(track: Any, *, only_position: bool = False) -> float:
    """Return the 95% chi-square gate for the selected geometric coordinates."""
    dimensions = 2 if only_position else len(track.motion_model.geometry_measurement_indices)
    return chi2inv95[dimensions]


def motion_gating_distance(track: Any, detection_boxes: np.ndarray, *, only_position: bool = False) -> np.ndarray:
    """Return squared Mahalanobis distances using a track's selected filter.

    Boxes are canonical ``xyxy`` or ``xywha`` rows, without metadata. The
    projection uses track-local covariance and measurement noise. Detector
    confidence is excluded from the geometric gate for XYSCR, and equivalent
    oriented rectangles are aligned before calculating angular residuals.
    """
    model = track.motion_model
    boxes = np.asarray(detection_boxes, dtype=float)
    dimensions = 5 if model.is_obb else 4
    if boxes.ndim != 2 or boxes.shape[1] != dimensions:
        raise ValueError(f"Expected detection boxes with shape (N, {dimensions}).")
    if not len(boxes):
        return np.empty(0, dtype=float)
    kalman = getattr(track, "kalman_filter", None)
    if kalman is None:
        kalman = track.kf
    mean = getattr(track, "mean", None)
    covariance = getattr(track, "covariance", None)
    if mean is None or covariance is None:
        mean, covariance = kalman.x, kalman.P
    mean = np.asarray(mean).reshape(-1)
    if model.is_obb:
        references = np.broadcast_to(model.to_box(mean)[0, :5], boxes.shape)
        boxes = align_obb_measurements(boxes, references)
    measurements = model.to_measurements(boxes)
    projected, uncertainty = model.project(kalman, mean, covariance)
    residuals = measurements - np.asarray(projected).reshape(-1)
    if model.is_obb:
        angle_index = model.angle_measurement_index
        residuals[:, angle_index] = normalize_angle(residuals[:, angle_index])
    indices = [0, 1] if only_position else list(model.geometry_measurement_indices)
    return BaseKalmanFilter._gating_from_residuals(residuals[:, indices], uncertainty[np.ix_(indices, indices)], "maha")


def fuse_motion(
    cost_matrix: np.ndarray,
    tracks: Sequence[Any],
    detection_boxes: np.ndarray,
    *,
    only_position: bool = False,
    lambda_: float = 0.98,
) -> np.ndarray:
    """Gate pairs and blend costs with raw squared Mahalanobis distance.

    ``lambda_`` weights the original cost; one performs gating only, and zero
    uses motion distance alone. Forbidden input edges remain forbidden at both
    endpoints. The 95% chi-square threshold is applied before blending.
    """
    if not np.isfinite(lambda_) or not 0.0 <= lambda_ <= 1.0:
        raise ValueError("Motion fusion lambda_ must be a finite value in [0, 1].")
    costs = np.asarray(cost_matrix)
    boxes = np.asarray(detection_boxes)
    if costs.ndim != 2 or costs.shape != (len(tracks), len(boxes)):
        raise ValueError("Tracks and detection boxes must match the cost-matrix dimensions.")
    result = np.array(costs, dtype=np.result_type(costs.dtype, float), copy=True)
    if not result.size:
        return result
    for row, track in enumerate(tracks):
        distances = motion_gating_distance(track, boxes, only_position=only_position)
        eligible = np.isfinite(costs[row]) & np.isfinite(distances)
        eligible &= distances <= motion_gating_threshold(track, only_position=only_position)
        result[row, ~eligible] = np.inf
        result[row, eligible] = lambda_ * costs[row, eligible] + (1.0 - lambda_) * distances[eligible]
    return result


def gate_cost_matrix(
    cost_matrix: np.ndarray,
    tracks: Sequence[Any],
    detection_boxes: np.ndarray,
    *,
    only_position: bool = False,
) -> np.ndarray:
    """Remove geometrically implausible pairs without rescaling eligible costs."""
    return fuse_motion(cost_matrix, tracks, detection_boxes, only_position=only_position, lambda_=1.0)


def apply_motion_fusion(
    cost_matrix: np.ndarray,
    tracks: Sequence[Any],
    detection_boxes: np.ndarray,
    *,
    mode: Literal["off", "gate", "blend"],
    only_position: bool = False,
    lambda_: float = 0.98,
) -> np.ndarray:
    """Apply the configured association motion policy without changing inputs."""
    if mode == "off":
        return cost_matrix
    if mode == "gate":
        return gate_cost_matrix(cost_matrix, tracks, detection_boxes, only_position=only_position)
    if mode == "blend":
        return fuse_motion(cost_matrix, tracks, detection_boxes, only_position=only_position, lambda_=lambda_)
    raise ValueError(f"Unknown motion fusion mode: {mode!r}.")
