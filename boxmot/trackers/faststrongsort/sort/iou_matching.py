# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

import numpy as np

from . import linear_assignment


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
    ]
    br = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
    ]
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)

def aiou(bbox, candidates):
    """
    IoU - Aspect Ratio

    """
    candidates = np.array(candidates)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)

    iou = area_intersection / (area_bbox + area_candidates - area_intersection)

    aspect_ratio = bbox[2] / bbox[3]
    candidates_aspect_ratio = candidates[:, 2] / candidates[:, 3]
    arctan = np.arctan(aspect_ratio) - np.arctan(candidates_aspect_ratio)
    v = 1 - ((4 / np.pi ** 2) * arctan ** 2)
    alpha = v / (1 - iou + v)
    
    return iou, alpha

def aiou_vectorized(bboxes1, bboxes2):
    """
    Vectorized implementation of AIOU (IoU with aspect ratio consideration)
    
    Args:
    bboxes1: numpy array of shape (N, 4) in format (x, y, w, h)
    bboxes2: numpy array of shape (M, 4) in format (x, y, w, h)
    
    Returns:
    ious: numpy array of shape (N, M) containing IoU values
    alphas: numpy array of shape (N, M) containing alpha values
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    bboxes1_x1y1 = bboxes1[:, :2]
    bboxes1_x2y2 = bboxes1[:, :2] + bboxes1[:, 2:]
    bboxes2_x1y1 = bboxes2[:, :2]
    bboxes2_x2y2 = bboxes2[:, :2] + bboxes2[:, 2:]
    
    # Compute intersection
    intersect_x1y1 = np.maximum(bboxes1_x1y1[:, None], bboxes2_x1y1[None, :])
    intersect_x2y2 = np.minimum(bboxes1_x2y2[:, None], bboxes2_x2y2[None, :])
    intersect_wh = np.maximum(0., intersect_x2y2 - intersect_x1y1)
    
    # Compute areas
    intersect_area = intersect_wh.prod(axis=2)
    bboxes1_area = bboxes1[:, 2:].prod(axis=1)
    bboxes2_area = bboxes2[:, 2:].prod(axis=1)
    union_area = bboxes1_area[:, None] + bboxes2_area[None, :] - intersect_area
    
    # Compute IoU
    ious = intersect_area / union_area
    
    # Compute aspect ratios
    bboxes1_aspect_ratio = bboxes1[:, 2] / bboxes1[:, 3]
    bboxes2_aspect_ratio = bboxes2[:, 2] / bboxes2[:, 3]
    
    # Compute alpha
    arctan_diff = np.arctan(bboxes1_aspect_ratio[:, None]) - np.arctan(bboxes2_aspect_ratio[None, :])
    v = 1 - ((4 / (np.pi ** 2)) * arctan_diff ** 2)
    alphas = v / (1 - ious + v)
    
    return ious, alphas


def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix
