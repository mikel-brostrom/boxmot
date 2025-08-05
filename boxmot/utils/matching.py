# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import lap
import numpy as np
from scipy.spatial.distance import cdist
from typing import Any, List, Tuple

from boxmot.utils.iou import AssociationFunction

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



def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve linear assignment using the LAPJV algorithm.

    Args:
        cost_matrix (np.ndarray): Square cost matrix of shape (N, M).
        thresh (float): Maximum allowed cost for assignments.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Matched indices, unmatched row indices, unmatched column indices.
    """

    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches: List[List[int]] = []
    unmatched_a: np.ndarray
    unmatched_b: np.ndarray
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches_array = np.asarray(matches)
    return matches_array, unmatched_a, unmatched_b


def iou_distance(atracks: List[Any], btracks: List[Any]) -> np.ndarray:
    """Compute cost matrix based on IoU.

    Args:
        atracks (List[STrack] | List[np.ndarray]): First set of tracks.
        btracks (List[STrack] | List[np.ndarray]): Second set of tracks.

    Returns:
        np.ndarray: Cost matrix where lower values indicate higher IoU.
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]

    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious
    _ious = AssociationFunction.iou_batch(atlbrs, btlbrs)

    cost_matrix = 1 - _ious

    return cost_matrix




def embedding_distance(
    tracks: List[Any],
    detections: List[Any],
    metric: str = "cosine",
) -> np.ndarray:
    """Compute embedding distance between tracks and detections.

    Args:
        tracks (List[STrack]): Existing tracks with embeddings.
        detections (List[BaseTrack]): Candidate detections with embeddings.
        metric (str, optional): Distance metric. Defaults to "cosine".

    Returns:
        np.ndarray: Cost matrix of embedding distances.
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray(
        [track.curr_feat for track in detections], dtype=np.float32
    )
    track_features = np.asarray(
        [track.smooth_feat for track in tracks], dtype=np.float32
    )
    cost_matrix = np.maximum(
        0.0, cdist(track_features, det_features, metric)
    )  # Normalized features
    return cost_matrix


def fuse_motion(
    kf: Any,
    cost_matrix: np.ndarray,
    tracks: List[Any],
    detections: List[Any],
    only_position: bool = False,
    lambda_: float = 0.98,
) -> np.ndarray:
    """Fuse motion information into the appearance cost matrix.

    Args:
        kf (Any): Kalman filter instance.
        cost_matrix (np.ndarray): Appearance-based cost matrix.
        tracks (List[STrack]): Existing tracks.
        detections (List[BaseTrack]): Candidate detections.
        only_position (bool, optional): Use only position for gating. Defaults to False.
        lambda_ (float, optional): Blending factor. Defaults to 0.98.

    Returns:
        np.ndarray: Updated cost matrix incorporating motion information.
    """

    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix: np.ndarray, tracks: List[Any], detections: List[Any]) -> np.ndarray:
    """Fuse IoU information into the appearance cost matrix.

    Args:
        cost_matrix (np.ndarray): Appearance-based cost matrix.
        tracks (List[STrack]): Existing tracks.
        detections (List[BaseTrack]): Candidate detections.

    Returns:
        np.ndarray: Updated cost matrix combining IoU and appearance.
    """

    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_confs = np.array([det.conf for det in detections])
    det_confs = np.expand_dims(det_confs, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_confs) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix: np.ndarray, detections: List[Any]) -> np.ndarray:
    """Fuse detection scores into the IoU cost matrix.

    Args:
        cost_matrix (np.ndarray): IoU cost matrix.
        detections (List[BaseTrack]): Candidate detections with confidence scores.

    Returns:
        np.ndarray: Updated cost matrix.
    """

    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_confs = np.array([det.conf for det in detections])
    det_confs = np.expand_dims(det_confs, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_confs
    fuse_cost = 1 - fuse_sim
    return fuse_cost
