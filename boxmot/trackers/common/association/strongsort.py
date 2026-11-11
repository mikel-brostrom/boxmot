# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment

from boxmot.trackers.common.association.matching import (
    linear_assignment,
    motion_gating_distances,
    motion_gating_threshold,
)

INFTY_COST = 1e5


def min_cost_matching(
    cost_matrix: np.ndarray,
    max_distance: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve prepared costs and return local matched/unmatched indices as lists.

    Rows represent candidate tracks and columns represent candidate detections.
    Costs above ``max_distance`` are rejected; equality remains admissible.

    Finite costs retain the clipped Hungarian assignment and its partition
    order: unassigned indices precede rejected assignments. Non-finite costs
    use the shared solver so forbidden pairs can remain unmatched.
    The finite-cost path clips the input matrix in place.
    """
    track_indices = list(range(cost_matrix.shape[0]))
    detection_indices = list(range(cost_matrix.shape[1]))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    if not np.isfinite(cost_matrix).all():
        matches, unmatched_tracks, unmatched_detections = linear_assignment(cost_matrix, max_distance)
        return (
            [(track_indices[row], detection_indices[col]) for row, col in matches],
            [track_indices[row] for row in unmatched_tracks],
            [detection_indices[col] for col in unmatched_detections],
        )
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    matched_columns = np.zeros(len(detection_indices), dtype=bool)
    matched_rows = np.zeros(len(track_indices), dtype=bool)
    matched_columns[col_indices] = True
    matched_rows[row_indices] = True
    for col, detection_idx in enumerate(detection_indices):
        if not matched_columns[col]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if not matched_rows[row]:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    cost_matrix: np.ndarray,
    tracks: Sequence[Any],
    detections: Sequence[Any],
    mc_lambda: float,
    gated_cost: float = INFTY_COST,
    only_position: bool = False,
    *,
    gated_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend eligible costs with geometric Mahalanobis distance.

    Rejected pairs retain ``gated_cost`` independently of ``mc_lambda``.
    The gate uses selected-filter geometry coordinates, excluding confidence,
    and aligns equivalent oriented rectangles before measuring residuals.

    Rows and columns correspond directly to ``tracks`` and ``detections``.
    Eligible costs are weighted by ``mc_lambda`` and motion distances by its
    complement. ``only_position`` limits gating to the x and y coordinates.
    The cost matrix is modified in place and returned. If supplied,
    ``gated_mask`` receives the Boolean rejection mask before blending.
    """

    if not len(tracks):
        return cost_matrix
    rows_by_geometry: dict[bool, list[int]] = {}
    for row, track in enumerate(tracks):
        rows_by_geometry.setdefault(track.is_obb, []).append(row)
    distances = np.empty(cost_matrix.shape, dtype=float)
    for is_obb, rows in rows_by_geometry.items():
        geometry_attr = "xywha" if is_obb else "xyxy"
        boxes = np.asarray([getattr(detection, geometry_attr) for detection in detections]).reshape(
            -1, 5 if is_obb else 4
        )
        distances[rows] = motion_gating_distances([tracks[row] for row in rows], boxes, only_position=only_position)
    thresholds = np.asarray([motion_gating_threshold(track, only_position=only_position) for track in tracks])
    blocked = ~np.isfinite(cost_matrix) | ~np.isfinite(distances) | (distances > thresholds[:, None])
    if gated_mask is not None:
        gated_mask[:] = blocked
    eligible = ~blocked
    cost_matrix[eligible] = mc_lambda * cost_matrix[eligible] + (1 - mc_lambda) * distances[eligible]
    cost_matrix[blocked] = gated_cost
    return cost_matrix


def _cosine_distance(a: np.ndarray, b: np.ndarray, data_is_normalized: bool = False) -> np.ndarray:
    """Return pairwise cosine distances with shape ``(len(a), len(b))``.

    Rows are normalized to unit length unless ``data_is_normalized`` is true.
    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


def _pdist(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Return pairwise squared Euclidean distances with shape ``(len(a), len(b))``."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2.0 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0.0, float(np.inf))
    return r2


def _nn_euclidean_distance(x: ArrayLike, y: np.ndarray) -> np.ndarray:
    """Return each query row's smallest squared Euclidean distance to a sample in ``x``."""
    distances = _pdist(x, y)
    return distances.min(axis=0)


def _nn_cosine_distance(x: ArrayLike, y: np.ndarray) -> np.ndarray:
    """Return each query row's smallest cosine distance to a sample in ``x``."""
    distances = _cosine_distance(np.asarray(x), np.asarray(y))
    return distances.min(axis=0)


class NearestNeighborDistanceMetric:
    """Find the closest appearance sample in each target's gallery.

    ``metric`` selects cosine or squared Euclidean distance. Costs above
    ``matching_threshold`` are invalid matches. An optional ``budget`` limits
    each gallery to its most recent samples; ``None`` keeps all samples.
    ``samples`` maps target identities to lists of raw feature vectors.
    """

    def __init__(
        self,
        metric: Literal["euclidean", "cosine"],
        matching_threshold: float,
        budget: int | None = None,
    ) -> None:
        """Initialize an empty gallery using the selected distance metric."""
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples: dict[int, list[np.ndarray]] = {}

    def partial_fit(
        self,
        features: np.ndarray,
        targets: Sequence[int] | np.ndarray,
        active_targets: Sequence[int] | np.ndarray,
    ) -> None:
        """Append feature rows to their targets and discard inactive galleries.

        Each row in ``features`` corresponds to the same entry in ``targets``.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features: np.ndarray, targets: Sequence[int] | np.ndarray) -> np.ndarray:
        """Return gallery-to-query costs with shape ``(len(targets), len(features))``.

        Each entry is the minimum cosine or squared Euclidean distance between
        the corresponding target's stored samples and the query feature row.
        """
        cost_matrix = np.zeros((len(targets), len(features)))
        if self._metric is _nn_cosine_distance and len(targets):
            # Every gallery sees the same query matrix. Normalize it once,
            # retaining each gallery's own dot product and reduction order.
            # Galleries remain raw so direct sample edits and budget eviction
            # retain their existing behavior without a stale normalized cache.
            queries = np.asarray(features) / np.linalg.norm(features, axis=1, keepdims=True)
            for i, target in enumerate(targets):
                gallery = np.asarray(self.samples[target])
                gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
                # Reduce similarities before subtraction to avoid allocating
                # and writing a full gallery-by-query distance matrix.
                cost_matrix[i, :] = 1.0 - np.dot(gallery, queries.T).max(axis=0)
            return cost_matrix
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
