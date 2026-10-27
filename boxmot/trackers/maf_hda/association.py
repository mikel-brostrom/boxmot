"""GMPHD and mask affinity primitives translated from GMPHD_MAF.

Copyright (c) 2021, Young-min Song. See LICENSE in this package for the
upstream BSD 2-Clause license and attribution.
"""

from __future__ import annotations

import numpy as np

INITIAL_COVARIANCE = np.diag([25.0, 100.0, 25.0, 100.0])
PROCESS_NOISE = 0.5 * INITIAL_COVARIANCE
MEASUREMENT_NOISE = np.diag([25.0, 100.0])
TRANSITION = np.eye(4)
TRANSITION[:2, 2:] = np.eye(2)
MAX_COST = 10000.0


def predict_covariance(covariance: np.ndarray) -> np.ndarray:
    """Predict one frame, preserving the upstream diagonal covariance model."""
    predicted = TRANSITION @ covariance @ TRANSITION.T + PROCESS_NOISE
    return np.diag(np.diag(predicted))


def gaussian_affinity(
    boxes: np.ndarray,
    predicted_box: np.ndarray,
    covariance: np.ndarray,
    *,
    recovery: bool = False,
    gate_geometry: bool = True,
    max_area_ratio: float = 2.0,
    min_intersection_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gated center likelihoods and one posterior covariance per box.

    Overlap modes gate area changes and the fraction of the smaller box
    intersecting in S2TA; T2TA requires any overlap. Centroid mode supplies
    its own radius gate and keeps these cues from rejecting pairs.
    """
    innovation_covariance = covariance[:2, :2] + MEASUREMENT_NOISE
    gain = np.linalg.solve(innovation_covariance, covariance[:2, :]).T
    posterior = covariance - gain @ covariance[:2, :]
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    center = (predicted_box[:2] + predicted_box[2:]) * 0.5
    difference = centers - center
    distance = np.einsum("ij,ji->i", difference, np.linalg.solve(innovation_covariance, difference.T))
    likelihood = np.exp(-0.5 * distance) / (2.0 * np.pi * np.sqrt(np.linalg.det(innovation_covariance)))

    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    area = np.prod(predicted_box[2:] - predicted_box[:2])
    intersection = np.prod(
        np.maximum(0.0, np.minimum(boxes[:, 2:], predicted_box[2:]) - np.maximum(boxes[:, :2], predicted_box[:2])),
        axis=1,
    )
    valid = np.ones(len(boxes), dtype=bool)
    if gate_geometry:
        valid &= (areas < max_area_ratio * area) & (area < max_area_ratio * areas)
        valid &= intersection > 0 if recovery else intersection >= min_intersection_ratio * np.minimum(area, areas)
    # MAF can rescue a Gaussian-gated pair through appearance. Such a match
    # must retain the prediction's uncertainty: it supplied no motion update.
    posteriors = np.where(valid[:, None, None], posterior, covariance)
    if gate_geometry:
        likelihood[~valid | (likelihood < np.finfo(np.float32).tiny)] = 0.0
    else:
        # Large allowed radii can underflow the Gaussian likelihood. Retain a
        # finite ranking cost instead of introducing an implicit distance gate.
        likelihood = np.maximum(likelihood, np.finfo(float).tiny)
    return likelihood, posteriors


def minmax_affinity(values: np.ndarray) -> np.ndarray:
    """Normalize one affinity matrix, including constant and singleton cases."""
    if not values.size:
        return values.copy()
    minimum, maximum = float(values.min()), float(values.max())
    if minimum == maximum:
        if maximum == 0.0:
            return np.zeros_like(values)
        minimum = 0.0
    return (values - minimum) / (maximum - minimum)


def fusion_cost(
    motion: np.ndarray,
    appearance: np.ndarray,
    overlap: np.ndarray,
    *,
    mode: str,
    recovery: bool,
    appearance_lower: float,
    appearance_upper: float,
    overlap_lower: float,
    gate_geometry: bool = True,
    recovery_overlap_thresh: float = 0.9,
) -> np.ndarray:
    """Fuse motion and appearance; centroid uses only the lower appearance gate."""
    if mode == "motion":
        affinity = motion.copy()
    elif mode == "appearance":
        valid = appearance >= appearance_lower
        if gate_geometry:
            valid = (appearance >= appearance_upper) | (valid & (overlap >= overlap_lower))
        affinity = np.where(valid, appearance, 0.0)
    else:
        normalized_motion = minmax_affinity(motion)
        normalized_appearance = minmax_affinity(appearance)
        if recovery:
            normalized_appearance[appearance < appearance_lower] = 0.0
        affinity = normalized_motion * normalized_appearance
        if recovery:
            geometry_support = overlap >= (recovery_overlap_thresh if gate_geometry else 0.0)
            substitute = (normalized_motion == 0.0) & ((appearance >= appearance_upper) | geometry_support)
        else:
            substitute = (normalized_motion == 0.0) | (overlap > overlap_lower)
        # A zero centroid score lies exactly on the allowed radius. Keep it
        # rankable without overriding an absent or rejected appearance cue.
        geometry_affinity = overlap if gate_geometry else np.maximum(overlap, np.finfo(float).eps)
        affinity[substitute] = (geometry_affinity * normalized_appearance)[substitute]
    cost = np.full(affinity.shape, MAX_COST if gate_geometry else np.inf, dtype=np.float64)
    valid = affinity > (np.exp(-MAX_COST / 100.0) if gate_geometry else 0.0)
    cost[valid] = np.maximum(0.0, -100.0 * np.log(affinity[valid]))
    return cost


def mask_merge_groups(masks: list[np.ndarray], classes: list[int], threshold: float) -> list[list[int]]:
    """Find same-class connected components under the mask-IoU merge rule."""
    parents = list(range(len(masks)))
    areas = [int(mask.sum()) for mask in masks]
    supports = []
    for mask in masks:
        ys, xs = np.nonzero(mask)
        supports.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))

    def root(index: int) -> int:
        """Find a component representative with path compression."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    boxes = np.asarray(supports, dtype=int).reshape(-1, 4)
    labels = np.asarray(classes)
    candidates = labels[:, None] == labels[None, :]
    candidates &= (boxes[:, None, 0] < boxes[None, :, 2]) & (boxes[None, :, 0] < boxes[:, None, 2])
    candidates &= (boxes[:, None, 1] < boxes[None, :, 3]) & (boxes[None, :, 1] < boxes[:, None, 3])
    rows, columns = np.nonzero(np.triu(candidates, k=1))
    for i, j in zip(rows, columns, strict=True):
        a, b = supports[i], supports[j]
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
        intersection = int(np.count_nonzero(masks[i][y1:y2, x1:x2] & masks[j][y1:y2, x1:x2]))
        if intersection / (areas[i] + areas[j] - intersection) >= threshold:
            parents[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for index in range(len(masks)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())
