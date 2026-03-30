# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import absolute_import

import numpy as np

from boxmot.utils.iou import AssociationFunction

from . import linear_assignment


def _tlwh_to_tlbr(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    tlbr = np.asarray(boxes, dtype=np.float32).copy()
    tlbr[:, 2:] += tlbr[:, :2]
    return tlbr


def iou(bbox, candidates):
    """Compute intersection over union for a single tlwh bbox against tlwh candidates."""
    bbox_tlbr = _tlwh_to_tlbr(np.asarray([bbox], dtype=np.float32))
    candidates_tlbr = _tlwh_to_tlbr(np.asarray(candidates, dtype=np.float32))
    return AssociationFunction.iou_batch(bbox_tlbr, candidates_tlbr).reshape(-1)


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """Compute IoU cost using the shared vectorized IoU implementation."""
    if track_indices is None:
        track_indices = np.arange(len(tracks), dtype=np.int32)
    else:
        track_indices = np.asarray(track_indices, dtype=np.int32)

    if detection_indices is None:
        detection_indices = np.arange(len(detections), dtype=np.int32)
    else:
        detection_indices = np.asarray(detection_indices, dtype=np.int32)

    cost_matrix = np.full(
        (len(track_indices), len(detection_indices)),
        linear_assignment.INFTY_COST,
        dtype=np.float32,
    )
    if cost_matrix.size == 0:
        return cost_matrix

    valid_rows = [
        row
        for row, track_idx in enumerate(track_indices)
        if tracks[track_idx].time_since_update <= 1
    ]
    if not valid_rows:
        return cost_matrix

    valid_track_indices = track_indices[valid_rows]
    track_boxes = np.asarray(
        [tracks[track_idx].xyxy for track_idx in valid_track_indices],
        dtype=np.float32,
    )
    detection_boxes = _tlwh_to_tlbr(
        np.asarray([detections[idx].tlwh for idx in detection_indices], dtype=np.float32)
    )
    cost_matrix[np.asarray(valid_rows, dtype=np.int32)] = (
        1.0 - AssociationFunction.iou_batch(track_boxes, detection_boxes)
    )
    return cost_matrix
