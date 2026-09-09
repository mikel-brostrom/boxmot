"""Source EagerMOT 3D affinities and greedy sensor/track association.

Adapted from EagerMOT (MIT License), Copyright (c) 2021 Aleksandr Kim.
See LICENSE in this directory.
"""

from __future__ import annotations

import numpy as np

from boxmot.trackers.multimodal.eagermot.geometry import _boxes_array, iou3d_matrix, yaw_difference


def similarity_3d(detections: np.ndarray, predictions: np.ndarray, method: str = "dist_2d_full") -> np.ndarray:
    """Compute one of the source's four larger-is-better first-stage affinities.

    ``dist_2d`` uses ground-plane x/z centers. ``dist_2d_dims`` uses
    x/y/z/l/w/h; ``dist_2d_full`` multiplies that distance by
    ``2 - cos(pi-equivalent yaw difference)``. Distance affinities are
    negative, so their thresholds must also be nonpositive.
    """
    if method == "iou_3d":
        return iou3d_matrix(detections, predictions)
    columns = {"dist_2d": [0, 2], "dist_2d_dims": [0, 1, 2, 4, 5, 6], "dist_2d_full": [0, 1, 2, 4, 5, 6]}
    if method not in columns:
        raise ValueError(f"Unknown EagerMOT first matching method: {method!r}.")
    detections, predictions = _boxes_array(detections), _boxes_array(predictions)
    selected = columns[method]
    difference = detections[:, None, selected] - predictions[None, :, selected]
    distance = np.linalg.norm(difference, axis=2)
    if method == "dist_2d_full":
        angles = yaw_difference(detections[:, None, 3], predictions[None, :, 3])
        distance *= 2.0 - np.cos(angles)
    return -distance


def greedy_association(
    similarity: np.ndarray, threshold: float | np.ndarray, allowed: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily accept the largest available affinity, using stable row-order ties.

    Rows represent detections and columns tracks. ``threshold`` may be a
    scalar or broadcastable per-pair threshold. ``allowed`` can enforce
    class/camera membership. Invalid or below-threshold edges cannot occupy
    a row/column. Returns matched row/column pairs and the unmatched indices.
    """
    values = np.asarray(similarity, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("An association similarity matrix must be two-dimensional.")
    thresholds = np.broadcast_to(np.asarray(threshold, dtype=np.float64), values.shape)
    if not np.isfinite(thresholds).all():
        raise ValueError("Association thresholds must be finite.")
    valid = np.isfinite(values) & (values >= thresholds)
    if allowed is not None:
        membership = np.asarray(allowed, dtype=bool)
        if membership.shape != values.shape:
            raise ValueError("Association membership must match the similarity matrix shape.")
        valid &= membership
    flat = np.flatnonzero(valid)
    ranked = flat[np.argsort(-values.ravel()[flat], kind="stable")]
    used_rows, used_columns = np.zeros(values.shape[0], dtype=bool), np.zeros(values.shape[1], dtype=bool)
    matches = []
    for index in ranked:
        row, col = divmod(int(index), values.shape[1])
        if not used_rows[row] and not used_columns[col]:
            matches.append((row, col))
            used_rows[row], used_columns[col] = True, True
    return (
        np.asarray(matches, dtype=np.int64).reshape(-1, 2),
        np.flatnonzero(~used_rows),
        np.flatnonzero(~used_columns),
    )
