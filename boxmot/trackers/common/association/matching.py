# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

import lap
import numpy as np
from scipy.spatial.distance import cdist

from boxmot.trackers.common.association.iou import AssociationFunction

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
    """Return ``(row, column)`` pairs from the shared LAP solver."""
    cost_matrix = np.asarray(cost_matrix)
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int)

    kwargs = {"extend_cost": True}
    if cost_limit is not None:
        kwargs["cost_limit"] = cost_limit
    _, row_to_col, _ = lap.lapjv(cost_matrix, **kwargs)
    assigned_rows = np.flatnonzero(row_to_col >= 0)
    return np.column_stack((assigned_rows, row_to_col[assigned_rows])).astype(int, copy=False)


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )
    matches = solve_assignment(cost_matrix, cost_limit=thresh)
    unmatched_a = np.setdiff1d(np.arange(cost_matrix.shape[0]), matches[:, 0])
    unmatched_b = np.setdiff1d(np.arange(cost_matrix.shape[1]), matches[:, 1])
    return matches, unmatched_a, unmatched_b


def iou_distance(atracks, btracks, is_obb: bool = False):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        is_obb = is_obb or any(getattr(track, "is_obb", False) for track in atracks + btracks)
        if is_obb:
            atlbrs = [track.xywha for track in atracks]
            btlbrs = [track.xywha for track in btracks]
        else:
            atlbrs = [track.xyxy for track in atracks]
            btlbrs = [track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    _ious = (
        AssociationFunction.iou_batch_obb(atlbrs, btlbrs) if is_obb else AssociationFunction.iou_batch(atlbrs, btlbrs)
    )

    cost_matrix = 1 - _ious

    return cost_matrix


def feature_distance(features_a, features_b, metric="cosine"):
    """Compute a non-negative pairwise distance matrix for feature arrays."""
    features_a = np.asarray(features_a, dtype=np.float32)
    features_b = np.asarray(features_b, dtype=np.float32)
    if len(features_a) == 0 or len(features_b) == 0:
        return np.zeros((len(features_a), len(features_b)), dtype=np.float32)
    return np.maximum(0.0, cdist(features_a, features_b, metric))


def embedding_distance(tracks, detections, metric="cosine"):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)
    return feature_distance(track_features, det_features, metric)


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    geometry_similarity = 1 - cost_matrix
    det_confs = np.array([det.conf for det in detections])
    det_confs = np.expand_dims(det_confs, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = geometry_similarity * det_confs
    fuse_cost = 1 - fuse_sim
    return fuse_cost
