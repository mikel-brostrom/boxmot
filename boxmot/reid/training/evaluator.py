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
    normalize: bool = True,
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
        if normalize:
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
    *,
    metric: str = "cosine",
    part_dim: int | None = None,
    part_count: int | None = None,
    role_count: int | None = None,
    beta: float = 0.2,
    topk: int | None = None,
    sinkhorn_iters: int = 20,
    sinkhorn_temperature: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute cosine distance matrix between query and gallery features."""
    if metric == "visibility_weighted_parts":
        if part_dim is None or part_count is None:
            raise ValueError("visibility_weighted_parts distance requires part_dim and part_count")
        return compute_visibility_weighted_part_distance(
            query_features,
            gallery_features,
            part_dim=part_dim,
            part_count=part_count,
            beta=beta,
            eps=eps,
        )
    if metric == "evidence_sinkhorn":
        if part_dim is None or part_count is None:
            raise ValueError("evidence_sinkhorn distance requires part_dim and part_count")
        return compute_evidence_sinkhorn_distance(
            query_features,
            gallery_features,
            part_dim=part_dim,
            part_count=part_count,
            role_count=role_count,
            beta=beta,
            topk=topk,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_temperature=sinkhorn_temperature,
            eps=eps,
        )
    if metric != "cosine":
        raise ValueError(f"Unsupported distance metric: {metric}")
    # Features are already L2-normalized, so dot product = cosine similarity
    # Use float32 explicitly to avoid upcasting
    similarity = query_features.astype(np.float32) @ gallery_features.astype(np.float32).T
    return 1.0 - similarity


def compute_visibility_weighted_part_distance(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    *,
    part_dim: int,
    part_count: int,
    beta: float = 0.2,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute global distance plus mutually visible part distance.

    Expected feature layout:
      [global(D), part_0(D), ..., part_N(D), visibility_0, ..., visibility_N]
    """
    part_dim = int(part_dim)
    part_count = int(part_count)
    if part_dim <= 0 or part_count <= 0:
        raise ValueError("part_dim and part_count must be positive")

    expected_dim = part_dim * (1 + part_count) + part_count
    if query_features.shape[1] != expected_dim or gallery_features.shape[1] != expected_dim:
        raise ValueError(
            "visibility_weighted_parts feature dimension mismatch: "
            f"expected {expected_dim}, got query={query_features.shape[1]}, gallery={gallery_features.shape[1]}"
        )

    def _l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
        array = array.astype(np.float32, copy=False)
        denom = np.linalg.norm(array, axis=axis, keepdims=True).clip(min=eps)
        return array / denom

    q = query_features.astype(np.float32, copy=False)
    g = gallery_features.astype(np.float32, copy=False)
    q_global = _l2_normalize(q[:, :part_dim])
    g_global = _l2_normalize(g[:, :part_dim])

    parts_start = part_dim
    parts_end = part_dim * (1 + part_count)
    q_parts = _l2_normalize(q[:, parts_start:parts_end].reshape(-1, part_count, part_dim))
    g_parts = _l2_normalize(g[:, parts_start:parts_end].reshape(-1, part_count, part_dim))
    q_visibility = np.clip(q[:, parts_end:], 0.0, 1.0)
    g_visibility = np.clip(g[:, parts_end:], 0.0, 1.0)

    global_distance = 1.0 - (q_global @ g_global.T)
    part_similarity = np.einsum("qpd,gpd->qgp", q_parts, g_parts, optimize=True)
    part_distance = 1.0 - part_similarity
    visibility_weights = q_visibility[:, None, :] * g_visibility[None, :, :]
    weighted_part_distance = (visibility_weights * part_distance).sum(axis=2) / (
        eps + visibility_weights.sum(axis=2)
    )
    return global_distance + float(beta) * weighted_part_distance


def _parse_evidence_features(
    features: np.ndarray,
    *,
    part_dim: int,
    part_count: int,
    role_count: int | None,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse an IET evidence packet.

    Layout:
      [global(D), part_0(D), ..., part_N(D),
       visibility_N, rarity_N, role_probs_NxR, nullness_N]
    """
    part_dim = int(part_dim)
    part_count = int(part_count)
    if part_dim <= 0 or part_count <= 0:
        raise ValueError("part_dim and part_count must be positive")

    base_dim = part_dim * (1 + part_count)
    metadata_dim = features.shape[1] - base_dim
    if role_count is None:
        remainder = metadata_dim - (3 * part_count)
        if remainder < 0 or remainder % part_count != 0:
            raise ValueError(
                "Cannot infer evidence role count from feature dimension: "
                f"dim={features.shape[1]}, part_dim={part_dim}, part_count={part_count}"
            )
        role_count = remainder // part_count
    role_count = int(role_count)
    if role_count < 1:
        raise ValueError(f"evidence_sinkhorn role_count must be positive, got {role_count}")
    expected_dim = base_dim + (3 * part_count) + (part_count * role_count)
    if features.shape[1] != expected_dim:
        raise ValueError(
            "evidence_sinkhorn feature dimension mismatch: "
            f"expected {expected_dim}, got {features.shape[1]}"
        )

    def _l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
        array = array.astype(np.float32, copy=False)
        denom = np.linalg.norm(array, axis=axis, keepdims=True).clip(min=eps)
        return array / denom

    features = features.astype(np.float32, copy=False)
    global_features = _l2_normalize(features[:, :part_dim])
    parts_start = part_dim
    parts_end = part_dim * (1 + part_count)
    part_features = _l2_normalize(features[:, parts_start:parts_end].reshape(-1, part_count, part_dim))
    cursor = parts_end
    visibility = np.clip(features[:, cursor:cursor + part_count], 0.0, 1.0)
    cursor += part_count
    rarity = np.clip(features[:, cursor:cursor + part_count], 0.0, 1.0)
    cursor += part_count
    roles = np.clip(features[:, cursor:cursor + (part_count * role_count)], 0.0, 1.0)
    roles = roles.reshape(-1, part_count, role_count)
    role_norm = roles.sum(axis=2, keepdims=True).clip(min=eps)
    roles = roles / role_norm
    cursor += part_count * role_count
    nullness = np.clip(features[:, cursor:cursor + part_count], 0.0, 1.0)
    return global_features, part_features, visibility, rarity, roles, nullness


def _sinkhorn_plan_np(
    scores: np.ndarray,
    row_mass: np.ndarray,
    col_mass: np.ndarray,
    *,
    iters: int,
    temperature: float,
    eps: float,
) -> np.ndarray:
    """Return a small optimal-transport plan that maximizes ``scores``."""
    temperature = max(float(temperature), eps)
    kernel = np.exp((scores - scores.max()) / temperature).astype(np.float32)
    kernel = np.maximum(kernel, eps)
    row_mass = row_mass.astype(np.float32, copy=False)
    col_mass = col_mass.astype(np.float32, copy=False)
    u = np.ones_like(row_mass, dtype=np.float32)
    v = np.ones_like(col_mass, dtype=np.float32)
    for _ in range(max(int(iters), 1)):
        u = row_mass / np.maximum(kernel @ v, eps)
        v = col_mass / np.maximum(kernel.T @ u, eps)
    return u[:, None] * kernel * v[None, :]


def compute_evidence_sinkhorn_distance(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    *,
    part_dim: int,
    part_count: int,
    role_count: int | None = None,
    beta: float = 0.2,
    topk: int | None = None,
    sinkhorn_iters: int = 20,
    sinkhorn_temperature: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute global distance plus top-K Sinkhorn evidence alignment distance."""
    q_global, q_parts, q_visibility, q_rarity, q_roles, q_nullness = _parse_evidence_features(
        query_features,
        part_dim=part_dim,
        part_count=part_count,
        role_count=role_count,
        eps=eps,
    )
    g_global, g_parts, g_visibility, g_rarity, g_roles, g_nullness = _parse_evidence_features(
        gallery_features,
        part_dim=part_dim,
        part_count=part_count,
        role_count=role_count,
        eps=eps,
    )

    global_distance = 1.0 - (q_global @ g_global.T)
    distmat = global_distance.copy()
    num_gallery = gallery_features.shape[0]
    if num_gallery == 0:
        return distmat
    rerank_k = num_gallery if topk is None or topk <= 0 else min(int(topk), num_gallery)

    for query_index in range(query_features.shape[0]):
        gallery_indices = np.argpartition(global_distance[query_index], rerank_k - 1)[:rerank_k]
        q_mass = q_visibility[query_index] * q_rarity[query_index] * (1.0 - q_nullness[query_index])
        q_mass = q_mass / q_mass.sum().clip(min=eps) if q_mass.sum() > eps else np.full(part_count, 1.0 / part_count)
        for gallery_index in gallery_indices:
            g_mass = g_visibility[gallery_index] * g_rarity[gallery_index] * (1.0 - g_nullness[gallery_index])
            g_mass = (
                g_mass / g_mass.sum().clip(min=eps)
                if g_mass.sum() > eps
                else np.full(part_count, 1.0 / part_count)
            )
            part_similarity = q_parts[query_index] @ g_parts[gallery_index].T
            role_compatibility = q_roles[query_index] @ g_roles[gallery_index].T
            scores = part_similarity * role_compatibility
            plan = _sinkhorn_plan_np(
                scores,
                q_mass,
                g_mass,
                iters=sinkhorn_iters,
                temperature=sinkhorn_temperature,
                eps=eps,
            )
            alignment = float((plan * scores).sum())
            distmat[query_index, gallery_index] = global_distance[query_index, gallery_index] + float(beta) * (
                1.0 - alignment
            )
    return distmat


def visibility_part_count(head_parts: tuple[int, ...] | list[int] | str) -> int:
    """Return the number of non-global stripe parts represented by head_parts."""
    if isinstance(head_parts, str):
        values = [int(part.strip()) for part in head_parts.replace(";", ",").split(",") if part.strip()]
    else:
        values = [int(part) for part in head_parts]
    return sum(part for part in values if part > 1)


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
