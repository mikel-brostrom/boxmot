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
    boxes: np.ndarray, predicted_box: np.ndarray, covariance: np.ndarray, *, recovery: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return gated center likelihoods and one posterior covariance per box.

    S2TA requires half of the smaller box to overlap; T2TA requires any
    overlap. Both reject changes of area by a factor of two or more.
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
    valid = (areas < 2.0 * area) & (area < 2.0 * areas)
    valid &= intersection > 0 if recovery else intersection >= 0.5 * np.minimum(area, areas)
    # MAF can rescue a Gaussian-gated pair through appearance. Such a match
    # must retain the prediction's uncertainty: it supplied no motion update.
    posteriors = np.where(valid[:, None, None], posterior, covariance)
    likelihood[~valid | (likelihood < np.finfo(np.float32).tiny)] = 0.0
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
) -> np.ndarray:
    """Fuse GMPHD likelihoods and KCF scores using the source min-max rules."""
    if mode == "motion":
        affinity = motion.copy()
    elif mode == "appearance":
        valid = (appearance >= appearance_upper) | ((appearance >= appearance_lower) & (overlap >= overlap_lower))
        affinity = np.where(valid, appearance, 0.0)
    else:
        normalized_motion = minmax_affinity(motion)
        normalized_appearance = minmax_affinity(appearance)
        if recovery:
            normalized_appearance[appearance < appearance_lower] = 0.0
        affinity = normalized_motion * normalized_appearance
        if recovery:
            substitute = (normalized_motion == 0.0) & ((appearance >= appearance_upper) | (overlap >= 0.9))
        else:
            substitute = (normalized_motion == 0.0) | (overlap > overlap_lower)
        affinity[substitute] = (overlap * normalized_appearance)[substitute]
    cost = np.full(affinity.shape, MAX_COST, dtype=np.float64)
    valid = affinity > np.exp(-MAX_COST / 100.0)
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

    for i, first in enumerate(masks):
        for j in range(i + 1, len(masks)):
            if classes[i] != classes[j]:
                continue
            a, b = supports[i], supports[j]
            x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
            if x2 <= x1 or y2 <= y1:
                continue
            intersection = int(np.count_nonzero(first[y1:y2, x1:x2] & masks[j][y1:y2, x1:x2]))
            if intersection / (areas[i] + areas[j] - intersection) >= threshold:
                parents[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for index in range(len(masks)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())
