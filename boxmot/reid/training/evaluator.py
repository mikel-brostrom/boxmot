"""ReID evaluation: CMC and mAP computation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Extracting",
    flip_tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from a dataset using the given model.

    Args:
        flip_tta: If True, average features from the original and
            horizontally-flipped image (standard ReID test-time augmentation).

    Returns:
        features: (N, D) array of L2-normalized feature vectors.
        pids: (N,) array of person IDs.
        camids: (N,) array of camera IDs.
    """
    model.eval()
    all_features, all_pids, all_camids = [], [], []

    for imgs, pids, camids in tqdm(dataloader, desc=f"    {desc}", leave=False, unit="batch"):
        imgs = imgs.to(device)
        feats = model(imgs)
        if flip_tta:
            feats_flip = model(torch.flip(imgs, dims=[3]))  # horizontal flip
            feats = (feats + feats_flip) / 2.0
        feats = F.normalize(feats, p=2, dim=1)
        all_features.append(feats.cpu().numpy())
        all_pids.append(np.asarray(pids))
        all_camids.append(np.asarray(camids))

    features = np.concatenate(all_features, axis=0)
    pids_arr = np.concatenate(all_pids, axis=0)
    camids_arr = np.concatenate(all_camids, axis=0)
    del all_features, all_pids, all_camids

    return features, pids_arr, camids_arr


def compute_distance_matrix(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
) -> np.ndarray:
    """Compute cosine distance matrix between query and gallery features."""
    # Features are already L2-normalized, so dot product = cosine similarity
    # Use float32 explicitly to avoid upcasting
    similarity = query_features.astype(np.float32) @ gallery_features.astype(np.float32).T
    return 1.0 - similarity


def evaluate_ranking(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank: int = 50,
) -> Tuple[np.ndarray, float]:
    """Compute CMC curve and mAP.

    Args:
        distmat: (num_query, num_gallery) distance matrix.
        q_pids: query person IDs.
        g_pids: gallery person IDs.
        q_camids: query camera IDs.
        g_camids: gallery camera IDs.
        max_rank: maximum rank for CMC.

    Returns:
        cmc: CMC curve array of shape (max_rank,).
        mAP: mean average precision.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g

    all_cmc = []
    all_AP = []
    num_valid_q = 0

    # Process row-by-row to avoid materializing full argsort/matches arrays
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = np.argsort(distmat[q_idx])
        # Remove gallery samples with same pid AND same camid
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        raw_cmc = (g_pids[order[keep]] == q_pid).astype(np.int32)
        if raw_cmc.sum() == 0:
            continue  # This query has no valid match in gallery

        num_valid_q += 1
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])

        # Compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        tmp_cmc_recall = tmp_cmc / num_rel
        # Use trapezoidal approximation
        recall_diff = np.zeros_like(tmp_cmc_recall)
        recall_diff[0] = tmp_cmc_recall[0]
        recall_diff[1:] = tmp_cmc_recall[1:] - tmp_cmc_recall[:-1]
        ap = (precision * recall_diff).sum()
        all_AP.append(ap)

    if num_valid_q == 0:
        return np.zeros(max_rank), 0.0

    all_cmc = np.asarray(all_cmc, dtype=np.float32)
    cmc = all_cmc.mean(axis=0)
    mAP = float(np.mean(all_AP))

    return cmc, mAP


def re_ranking(
    q_feats: np.ndarray,
    g_feats: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """k-reciprocal encoding re-ranking (Zhong et al., CVPR 2017).

    Returns a re-ranked distance matrix of shape (num_query, num_gallery).
    """
    feats = np.concatenate([q_feats, g_feats], axis=0)
    N = feats.shape[0]
    num_q = q_feats.shape[0]

    # Original cosine distance
    sim = feats @ feats.T
    original_dist = 1.0 - sim
    np.fill_diagonal(original_dist, 0.0)

    # Sorted index (ascending distance)
    indices = np.argsort(original_dist, axis=1)

    # k-reciprocal neighbors
    def _k_reciprocal(i: int, k: int):
        forward = set(indices[i, :k + 1].tolist())
        result = set()
        for candidate in forward:
            backward = set(indices[candidate, :k + 1].tolist())
            if i in backward:
                result.add(candidate)
        return result

    V = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        k_recip = _k_reciprocal(i, k1)
        # Expand with 1/2 * k1 reciprocal neighbors
        expanded = set(k_recip)
        for q in list(k_recip):
            q_recip = _k_reciprocal(q, int(np.round(k1 / 2)))
            if len(q_recip & k_recip) > 2 / 3 * len(q_recip):
                expanded |= q_recip
        expanded = sorted(expanded)
        weights = np.exp(-original_dist[i, expanded])
        V[i, expanded] = weights / weights.sum()

    # Local query expansion
    if k2 > 0:
        V_qe = np.zeros_like(V)
        for i in range(N):
            neighbors = indices[i, :k2 + 1]
            V_qe[i] = V[neighbors].mean(axis=0)
        V = V_qe

    # Jaccard distance
    jaccard = np.zeros((num_q, N), dtype=np.float32)
    for i in range(num_q):
        minimum = np.minimum(V[i], V)
        maximum = np.maximum(V[i], V)
        jaccard[i] = 1.0 - minimum.sum(axis=1) / (maximum.sum(axis=1) + 1e-12)

    final_dist = jaccard[:, num_q:] * (1 - lambda_value) + original_dist[:num_q, num_q:] * lambda_value
    return final_dist
