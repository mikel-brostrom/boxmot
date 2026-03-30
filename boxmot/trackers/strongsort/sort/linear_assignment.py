# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import absolute_import

import lap
import numpy as np

from boxmot.utils.matching import chi2inv95

INFTY_COST = 1e5


def min_cost_matching(
    distance_metric,
    max_distance,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Solve the linear assignment problem for the given tracks and detections."""
    if track_indices is None:
        track_indices = np.arange(len(tracks), dtype=np.int32)
    else:
        track_indices = np.asarray(track_indices, dtype=np.int32)

    if detection_indices is None:
        detection_indices = np.arange(len(detections), dtype=np.int32)
    else:
        detection_indices = np.asarray(detection_indices, dtype=np.int32)

    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices.tolist(), detection_indices.tolist()

    cost_matrix = np.asarray(
        distance_metric(tracks, detections, track_indices, detection_indices),
        dtype=np.float32,
    )
    if cost_matrix.size == 0:
        return [], track_indices.tolist(), detection_indices.tolist()

    _, row_assignment, col_assignment = lap.lapjv(
        cost_matrix,
        extend_cost=True,
        cost_limit=max_distance,
    )

    matches = [
        (int(track_indices[row]), int(detection_indices[col]))
        for row, col in enumerate(row_assignment)
        if col >= 0
    ]
    unmatched_tracks = track_indices[row_assignment < 0].tolist()
    unmatched_detections = detection_indices[col_assignment < 0].tolist()
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric,
    max_distance,
    cascade_depth,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Run the StrongSORT matching cascade."""
    del cascade_depth

    if track_indices is None:
        track_indices = np.arange(len(tracks), dtype=np.int32)
    else:
        track_indices = np.asarray(track_indices, dtype=np.int32)

    if detection_indices is None:
        detection_indices = np.arange(len(detections), dtype=np.int32)
    else:
        detection_indices = np.asarray(detection_indices, dtype=np.int32)

    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices.tolist(), detection_indices.tolist()

    matches, _, unmatched_detections = min_cost_matching(
        distance_metric,
        max_distance,
        tracks,
        detections,
        track_indices,
        detection_indices,
    )
    matched_track_ids = {track_idx for track_idx, _ in matches}
    unmatched_tracks = [
        int(track_idx)
        for track_idx in track_indices
        if int(track_idx) not in matched_track_ids
    ]
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    cost_matrix,
    tracks,
    detections,
    track_indices,
    detection_indices,
    mc_lambda,
    gated_cost=INFTY_COST,
    only_position=False,
):
    """Apply Kalman gating to the cost matrix in place."""
    if cost_matrix.size == 0 or len(detection_indices) == 0 or len(track_indices) == 0:
        return cost_matrix

    gating_threshold = chi2inv95[4]
    measurements = np.asarray(
        [detections[i].to_xyah() for i in detection_indices],
        dtype=np.float32,
    )
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = track.kf.gating_distance(
            track.mean,
            track.covariance,
            measurements,
            only_position,
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
        cost_matrix[row] = (
            mc_lambda * cost_matrix[row] + (1.0 - mc_lambda) * gating_distance
        )
    return cost_matrix


def _normalize_rows(data) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        return data
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    return data / np.clip(norms, 1e-12, None)


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pairwise cosine distance between rows in `a` and `b`."""
    if not data_is_normalized:
        a = _normalize_rows(a)
        b = _normalize_rows(b)
    else:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
    return 1.0 - a @ b.T


def _pdist(a, b):
    """Compute pairwise squared euclidean distances between rows in `a` and `b`."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a2 = np.square(a).sum(axis=1)
    b2 = np.square(b).sum(axis=1)
    return np.clip(-2.0 * (a @ b.T) + a2[:, None] + b2[None, :], 0.0, np.inf)


def _nn_euclidean_distance(x, y):
    """Return the nearest squared euclidean distance for each row in `y`."""
    distances = _pdist(x, y)
    if distances.size == 0:
        return np.zeros((len(y),), dtype=np.float32)
    return distances.min(axis=0)


def _nn_cosine_distance(x, y):
    """Return the nearest cosine distance for each row in `y`."""
    distances = _cosine_distance(x, y)
    if distances.size == 0:
        return np.zeros((len(y),), dtype=np.float32)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest-neighbor distance metric that stores recent appearance embeddings per target.
    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """Update the target sample bank."""
        for feature, target in zip(features, targets):
            samples = self.samples.setdefault(int(target), [])
            samples.append(np.asarray(feature, dtype=np.float32))
            if self.budget is not None:
                self.samples[int(target)] = samples[-self.budget :]
        self.samples = {
            target: self.samples[target]
            for target in active_targets
            if target in self.samples
        }

    def distance(self, features, targets):
        """Compute distance between candidate features and stored target embeddings."""
        features = np.asarray(features, dtype=np.float32)
        if len(targets) == 0 or len(features) == 0:
            return np.zeros((len(targets), len(features)), dtype=np.float32)

        cost_matrix = np.zeros((len(targets), len(features)), dtype=np.float32)
        fallback_cost = self.matching_threshold + 1e-5
        for row, target in enumerate(targets):
            samples = self.samples.get(int(target))
            if not samples:
                cost_matrix[row, :] = fallback_cost
                continue
            cost_matrix[row, :] = self._metric(samples, features)
        return cost_matrix
