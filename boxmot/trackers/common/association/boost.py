import warnings
from copy import deepcopy
from typing import Optional

import lap
import numpy as np

from boxmot.trackers.common.association.iou import AssociationFunction


def shape_similarity(
    detects: np.ndarray,
    tracks: np.ndarray,
    s_sim_corr: bool,
) -> np.ndarray:
    if not s_sim_corr:
        return shape_similarity_v1(detects, tracks)
    return shape_similarity_v2(detects, tracks)


def shape_similarity_v1(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw) / np.maximum(dw, tw) + np.abs(dh - th) / np.maximum(dw, tw)))


def shape_similarity_v2(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))

    dw = (detects[:, 2] - detects[:, 0]).reshape((-1, 1))
    dh = (detects[:, 3] - detects[:, 1]).reshape((-1, 1))
    tw = (tracks[:, 2] - tracks[:, 0]).reshape((1, -1))
    th = (tracks[:, 3] - tracks[:, 1]).reshape((1, -1))
    return np.exp(-(np.abs(dw - tw) / np.maximum(dw, tw) + np.abs(dh - th) / np.maximum(dh, th)))


def shape_similarity_obb(detects: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    """Return width/height similarity invariant to equivalent OBB forms."""
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((len(detects), len(tracks)), dtype=np.float32)

    dw = detects[:, 2].reshape((-1, 1))
    dh = detects[:, 3].reshape((-1, 1))
    tw = tracks[:, 2].reshape((1, -1))
    th = tracks[:, 3].reshape((1, -1))

    def relative_delta(lhs, rhs):
        return np.abs(lhs - rhs) / np.maximum(np.maximum(lhs, rhs), 1e-6)

    direct = relative_delta(dw, tw) + relative_delta(dh, th)
    swapped = relative_delta(dw, th) + relative_delta(dh, tw)
    return np.exp(-np.minimum(direct, swapped))


def soft_biou_batch_obb(detections: np.ndarray, trackers: np.ndarray) -> np.ndarray:
    """Compute confidence-buffered oriented IoU for DLO boosting."""
    if detections.size == 0 or trackers.size == 0:
        return np.zeros((len(detections), len(trackers)), dtype=np.float32)

    det_boxes = np.asarray(detections[:, :5], dtype=np.float32).copy()
    trk_boxes = np.asarray(trackers[:, :5], dtype=np.float32).copy()
    track_conf = np.clip(np.asarray(trackers[:, 5], dtype=np.float32), 0.0, 1.0)
    det_scale = 1.0 + (1.0 - float(track_conf.max())) * 0.5
    trk_scale = 1.0 + (1.0 - track_conf) * 1.0
    det_boxes[:, 2:4] *= det_scale
    trk_boxes[:, 2:4] *= trk_scale[:, None]
    return AssociationFunction.iou_batch_obb(det_boxes, trk_boxes)


def MhDist_similarity(
    mahalanobis_distance: np.ndarray,
    softmax_temp: float = 1.0,
) -> np.ndarray:
    limit = 13.2767  # 99% conf interval https://www.mathworks.com/help/stats/chi2inv.html
    mahalanobis_distance = deepcopy(mahalanobis_distance)
    mask = mahalanobis_distance > limit
    mahalanobis_distance[mask] = limit
    mahalanobis_distance = limit - mahalanobis_distance

    mahalanobis_distance = np.exp(mahalanobis_distance / softmax_temp) / np.exp(
        mahalanobis_distance / softmax_temp
    ).sum(0).reshape((1, -1))
    mahalanobis_distance = np.where(mask, 0, mahalanobis_distance)
    return mahalanobis_distance


def iou_batch(bboxes1, bboxes2):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)

    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    return wh / (
        (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        + (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        - wh
    )


def soft_biou_batch(bboxes1, bboxes2):
    """
    Computes soft BIoU between two bboxes in the form [x1,y1,x2,y2]
    BIoU is introduced in https://arxiv.org/pdf/2211.14317
    Soft BIoU is introduced as part of BoostTrack++
    # Author : Vukasin Stanojevic
    # Email  : vukasin.stanojevic@pmf.edu.rs
    """

    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    k1 = 0.25
    k2 = 0.5
    b2conf = bboxes2[..., 4]
    b1x1 = bboxes1[..., 0] - (bboxes1[..., 2] - bboxes1[..., 0]) * (1 - b2conf) * k1
    b2x1 = bboxes2[..., 0] - (bboxes2[..., 2] - bboxes2[..., 0]) * (1 - b2conf) * k2
    xx1 = np.maximum(b1x1, b2x1)

    b1y1 = bboxes1[..., 1] - (bboxes1[..., 3] - bboxes1[..., 1]) * (1 - b2conf) * k1
    b2y1 = bboxes2[..., 1] - (bboxes2[..., 3] - bboxes2[..., 1]) * (1 - b2conf) * k2
    yy1 = np.maximum(b1y1, b2y1)

    b1x2 = bboxes1[..., 2] + (bboxes1[..., 2] - bboxes1[..., 0]) * (1 - b2conf) * k1
    b2x2 = bboxes2[..., 2] + (bboxes2[..., 2] - bboxes2[..., 0]) * (1 - b2conf) * k2
    xx2 = np.minimum(b1x2, b2x2)

    b1y2 = bboxes1[..., 3] + (bboxes1[..., 3] - bboxes1[..., 1]) * (1 - b2conf) * k1
    b2y2 = bboxes2[..., 3] + (bboxes2[..., 3] - bboxes2[..., 1]) * (1 - b2conf) * k2
    yy2 = np.minimum(b1y2, b2y2)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h

    return wh / ((b1x2 - b1x1) * (b1y2 - b1y1) + (b2x2 - b2x1) * (b2y2 - b2y1) - wh)


def match(cost_matrix: np.ndarray, threshold: float) -> np.ndarray:
    if cost_matrix.size == 0:
        return np.empty(shape=(0, 2))

    a = (cost_matrix > threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        return np.stack(np.where(a), axis=1)
    _, x, y = lap.lapjv(-cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])


def linear_assignment(
    detections: np.ndarray,
    trackers: np.ndarray,
    iou_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    threshold: float,
    emb_cost: Optional[np.ndarray] = None,
):
    if iou_matrix is None and cost_matrix is None:
        raise Exception("Both iou_matrix and cost_matrix are None!")
    if iou_matrix is None:
        iou_matrix = deepcopy(cost_matrix)
    if cost_matrix is None:
        cost_matrix = deepcopy(iou_matrix)
    matched_indices = match(cost_matrix, threshold)
    unmatched_detections = []
    for d, _det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, _trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        valid_match = iou_matrix[m[0], m[1]] >= threshold or (
            False if emb_cost is None else (iou_matrix[m[0], m[1]] >= threshold / 2 and emb_cost[m[0], m[1]] >= 0.75)
        )
        if valid_match:
            matches.append(m.reshape(1, 2))
        else:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])

    matches = np.concatenate(matches, axis=0) if len(matches) else np.empty((0, 2), dtype=int)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers), cost_matrix


def associate(
    detections,
    trackers,
    iou_threshold,
    mahalanobis_distance: Optional[np.ndarray] = None,
    track_confidence: Optional[np.ndarray] = None,
    detection_confidence: Optional[np.ndarray] = None,
    emb_cost: Optional[np.ndarray] = None,
    lambda_iou: float = 0.5,
    lambda_mhd: float = 0.25,
    lambda_shape: float = 0.25,
    s_sim_corr: bool = False,
    lambda_emb_multiplier: float = 1.5,
    iou_matrix: Optional[np.ndarray] = None,
    shape_matrix: Optional[np.ndarray] = None,
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
            np.empty((0, 0)),
        )
    iou_matrix = iou_batch(detections, trackers) if iou_matrix is None else np.asarray(iou_matrix)

    cost_matrix = deepcopy(iou_matrix)

    if detection_confidence is not None and track_confidence is not None:
        conf = np.multiply(
            detection_confidence.reshape((-1, 1)),
            track_confidence.reshape((1, -1)),
        )
        conf[iou_matrix < iou_threshold] = 0

        cost_matrix += lambda_iou * conf * iou_matrix
    else:
        warnings.warn("Detections or tracklet confidence is None and detection-tracklet confidence cannot be computed!")
        conf = None

    if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
        mahalanobis_distance = MhDist_similarity(mahalanobis_distance)

        cost_matrix += lambda_mhd * mahalanobis_distance
        if conf is not None:
            if shape_matrix is None:
                shape_matrix = shape_similarity(detections, trackers, s_sim_corr)
            cost_matrix += lambda_shape * conf * shape_matrix

    if emb_cost is not None:
        lambda_emb = (1 + lambda_iou + lambda_shape + lambda_mhd) * lambda_emb_multiplier
        cost_matrix += lambda_emb * emb_cost

    return linear_assignment(
        detections,
        trackers,
        iou_matrix,
        cost_matrix,
        iou_threshold,
        emb_cost,
    )
