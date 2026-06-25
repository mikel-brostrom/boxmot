from __future__ import annotations

import numpy as np

from boxmot.trackers.common.detections import DetectionBatch


def _float_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    dtype = array.dtype if np.issubdtype(array.dtype, np.floating) else np.float32
    return array.astype(dtype, copy=True)


def normalize_embedding(embedding: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a unit-length copy of an appearance embedding."""
    normalized = _float_array(embedding)
    norm = float(np.linalg.norm(normalized))
    if norm <= eps:
        return np.zeros_like(normalized)
    return normalized / norm


def blend_embeddings(
    previous: np.ndarray | None,
    current: np.ndarray,
    previous_weight: float,
    current_weight: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Blend two embeddings and normalize the result."""
    if previous is None:
        return normalize_embedding(current, eps=eps)
    blended = (float(previous_weight) * _float_array(previous)) + (float(current_weight) * _float_array(current))
    return normalize_embedding(blended, eps=eps)


def ema_update_embedding(
    previous: np.ndarray | None,
    current: np.ndarray,
    alpha: float = 0.9,
    eps: float = 1e-12,
) -> np.ndarray:
    """Apply an EMA appearance update and normalize the resulting embedding."""
    return blend_embeddings(previous, current, alpha, 1.0 - alpha, eps=eps)


def placeholder_embeddings(
    size: int,
    dim: int = 1,
    value: float = 1.0,
    dtype=np.float32,
) -> np.ndarray:
    """Return deterministic placeholder embeddings for disabled ReID paths."""
    return np.full((int(size), int(dim)), value, dtype=dtype)


def resolve_batch_embeddings(
    batch: DetectionBatch,
    img: np.ndarray,
    model=None,
    enabled: bool = True,
    boxes: np.ndarray | None = None,
    placeholder_dim: int = 1,
    placeholder_value: float = 1.0,
    dtype=np.float32,
) -> np.ndarray:
    """Return embeddings aligned with a detection batch.

    The order mirrors tracker update semantics: disabled appearance paths get
    deterministic placeholders, precomputed embeddings are reused as-is, empty
    batches avoid model calls, and only non-empty batches without precomputed
    embeddings are passed through the ReID model.
    """
    if not enabled:
        return placeholder_embeddings(
            len(batch),
            dim=placeholder_dim,
            value=placeholder_value,
            dtype=dtype,
        )

    if batch.embs is not None:
        return batch.embs

    if len(batch) == 0:
        return placeholder_embeddings(
            0,
            dim=placeholder_dim,
            value=placeholder_value,
            dtype=dtype,
        )

    if model is None:
        raise ValueError("A ReID model is required when embeddings are not provided")

    feature_boxes = batch.boxes if boxes is None else np.asarray(boxes)
    return model.get_features(feature_boxes, img)


def confidence_aware_alpha(
    confs: np.ndarray,
    det_thresh: float,
    base_alpha: float = 0.95,
    dtype=np.float32,
) -> np.ndarray:
    """Return confidence-aware EMA coefficients for appearance updates."""
    confs = np.asarray(confs)
    if confs.size == 0:
        return np.empty(0, dtype=dtype)

    trust = (confs.astype(dtype, copy=False) - float(det_thresh)) / max(
        1.0 - float(det_thresh),
        1e-12,
    )
    alpha = float(base_alpha) + (1.0 - float(base_alpha)) * (1.0 - trust)
    return alpha.astype(dtype, copy=False)
