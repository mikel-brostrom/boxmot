import warnings
from copy import deepcopy
from typing import Optional
import numpy as np
import lap

__all__ = [
    'iou_batch', 'soft_biou_batch', 'shape_similarity', 'mahalanobis_similarity',
    'linear_assignment', 'associate'
]

# --------------------- Similarity Metrics ---------------------

def _shape_sim(detects: np.ndarray, tracks: np.ndarray, use_height_norm: bool = False) -> np.ndarray:
    if detects.size == 0 or tracks.size == 0:
        return np.zeros((len(detects), len(tracks)))
    dw = (detects[:, 2] - detects[:, 0])[:, None]
    dh = (detects[:, 3] - detects[:, 1])[:, None]
    tw = (tracks[:, 2] - tracks[:, 0])[None, :]
    th = (tracks[:, 3] - tracks[:, 1])[None, :]
    denom_w = np.maximum(dw, tw)
    denom_h = np.maximum(dh, th) if use_height_norm else denom_w
    return np.exp(-((np.abs(dw - tw) / denom_w) + (np.abs(dh - th) / denom_h)))


def shape_similarity(
    detects: np.ndarray,
    tracks: np.ndarray,
    height_normalize: bool = False
) -> np.ndarray:
    """
    Compute shape similarity between detections and tracks.
    If `height_normalize` is True, normalize height differences by max heights,
    otherwise by max widths.
    """
    return _shape_sim(detects, tracks, use_height_norm=height_normalize)


def mahalanobis_similarity(
    distances: np.ndarray,
    softmax_temp: float = 1.0,
    chi2_limit: float = 13.2767
) -> np.ndarray:
    """
    Convert Mahalanobis distances to similarity scores via clipped softmax.
    """
    d = deepcopy(distances)
    mask = d > chi2_limit
    d_clipped = chi2_limit - np.minimum(d, chi2_limit)
    exp_d = np.exp(d_clipped / softmax_temp)
    probs = exp_d / exp_d.sum(axis=0, keepdims=True)
    probs[mask] = 0
    return probs

# ------------------------- IOU Metrics -------------------------

def iou_batch(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two lists of boxes [x1,y1,x2,y2]."""
    if b1.size == 0 or b2.size == 0:
        return np.zeros((len(b1), len(b2)))
    b1 = b1[:, None, :]
    b2 = b2[None, :, :]
    xx1 = np.maximum(b1[..., 0], b2[..., 0])
    yy1 = np.maximum(b1[..., 1], b2[..., 1])
    xx2 = np.minimum(b1[..., 2], b2[..., 2])
    yy2 = np.minimum(b1[..., 3], b2[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    return inter / (area1 + area2 - inter)


def soft_biou_batch(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Soft BIoU as per BoostTrack++.
    Boxes b2 include confidence at index 4.
    """
    if b1.size == 0 or b2.size == 0:
        return np.zeros((len(b1), len(b2)))
    b2 = b2[None, :, :]
    b1 = b1[:, None, :]
    k1, k2 = 0.25, 0.5
    conf = b2[..., 4]
    # expand and contract corners
    b1_x1 = b1[..., 0] - (b1[..., 2] - b1[..., 0]) * (1 - conf) * k1
    b2_x1 = b2[..., 0] - (b2[..., 2] - b2[..., 0]) * (1 - conf) * k2
    b1_y1 = b1[..., 1] - (b1[..., 3] - b1[..., 1]) * (1 - conf) * k1
    b2_y1 = b2[..., 1] - (b2[..., 3] - b2[..., 1]) * (1 - conf) * k2
    b1_x2 = b1[..., 2] + (b1[..., 2] - b1[..., 0]) * (1 - conf) * k1
    b2_x2 = b2[..., 2] + (b2[..., 2] - b2[..., 0]) * (1 - conf) * k2
    b1_y2 = b1[..., 3] + (b1[..., 3] - b1[..., 1]) * (1 - conf) * k1
    b2_y2 = b2[..., 3] + (b2[..., 3] - b2[..., 1]) * (1 - conf) * k2
    xx1 = np.maximum(b1_x1, b2_x1)
    yy1 = np.maximum(b1_y1, b2_y1)
    xx2 = np.minimum(b1_x2, b2_x2)
    yy2 = np.minimum(b1_y2, b2_y2)
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter / (area1 + area2 - inter)

# ---------------------- Assignment Solver ----------------------

def match(cost: np.ndarray, thresh: float) -> np.ndarray:
    if cost.size == 0:
        return np.empty((0, 2), int)
    mask = (cost > thresh).astype(int)
    # trivial one-to-one
    if mask.sum(1).max() == 1 and mask.sum(0).max() == 1:
        return np.stack(np.where(mask), axis=1)
    _, x, y = lap.lapjv(-cost, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])


def linear_assignment(
    dets: np.ndarray,
    trks: np.ndarray,
    iou: np.ndarray,
    cost: np.ndarray,
    thresh: float,
    emb_cost: Optional[np.ndarray] = None
):
    if iou is None and cost is None:
        raise ValueError("Both iou and cost matrices are None")
    iou_mat = iou if iou is not None else deepcopy(cost)
    cost_mat = cost if cost is not None else deepcopy(iou)
    matched = match(cost_mat, thresh)

    all_d = set(range(len(dets)))
    all_t = set(range(len(trks)))
    m_d = set(m[:, 0] for m in matched)
    m_t = set(m[:, 1] for m in matched)
    unmatched_d = np.array(list(all_d - m_d), int)
    unmatched_t = np.array(list(all_t - m_t), int)

    # filter low-IoU
    final_matches, un_d, un_t = [], list(unmatched_d), list(unmatched_t)
    for d, t in matched:
        if iou_mat[d, t] >= thresh or \
           (emb_cost is not None and iou_mat[d, t] >= thresh/2 and emb_cost[d, t] >= 0.75):
            final_matches.append([d, t])
        else:
            un_d.append(d); un_t.append(t)
    if final_matches:
        final_matches = np.array(final_matches, int)
    else:
        final_matches = np.empty((0,2), int)
    return final_matches, np.array(un_d), np.array(un_t), cost_mat


def associate(
    dets: np.ndarray,
    trks: np.ndarray,
    iou_thresh: float,
    mahalanobis_distance: Optional[np.ndarray] = None,
    track_confidence: Optional[np.ndarray] = None,
    detection_confidence: Optional[np.ndarray] = None,
    emb_cost: Optional[np.ndarray] = None,
    lambda_iou: float = 0.5,
    lambda_mhd: float = 0.25,
    lambda_shape: float = 0.25,
    height_norm: bool = False
):
    """
    Multi-modal association combining IoU, detection confidence,
    Mahalanobis and shape similarity, and embedding cost.
    """
    if len(trks) == 0:
        return np.empty((0,2), int), np.arange(len(dets)), np.empty((0,), int), np.zeros((0,0))
    iou_mat = iou_batch(dets, trks)
    cost = deepcopy(iou_mat)

    # detection-track confidence
    if detection_confidence is not None and track_confidence is not None:
        conf = (detection_confidence[:,None] * track_confidence[None,:])
        conf[iou_mat < iou_thresh] = 0
        cost += lambda_iou * conf * iou_batch(dets, trks)
    else:
        warnings.warn("Cannot compute detection-track confidence; missing confidences.")

    # Mahalanobis & shape
    if mahalanobis_distance is not None and mahalanobis_distance.size:
        mhd_sim = mahalanobis_similarity(mahalanobis_distance)
        cost += lambda_mhd * mhd_sim
        if detection_confidence is not None and track_confidence is not None:
            cost += lambda_shape * conf * shape_similarity(dets, trks, height_norm)

    # embedding cost
    if emb_cost is not None:
        lam_emb = 1.5 * (1 + lambda_iou + lambda_shape + lambda_mhd)
        cost += lam_emb * emb_cost

    return linear_assignment(dets, trks, iou_mat, cost, iou_thresh, emb_cost)