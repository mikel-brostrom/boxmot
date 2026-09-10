from __future__ import annotations

from typing import Final

import numpy as np
import torch

_CPU: Final = "cpu"


def validate_tensor(
    value: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    ndim: int,
) -> None:
    """Validate a canonical tensor without casting, moving, or copying it."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}.")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}.")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {tuple(value.shape)}.")
    if value.device.type != _CPU:
        raise ValueError(f"{name} must be on CPU, got device {value.device}.")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use strided tensor layout, got {value.layout}.")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def validate_finite(value: torch.Tensor, *, name: str) -> None:
    """Reject non-finite values in an already validated canonical CPU tensor."""
    # NumPy scans the shared CPU storage without launching Torch's parallel
    # elementwise kernels for every small detection/embedding batch. force=True
    # also accepts autograd tensors and resolves lazy negative/conjugate views.
    if value.numel() and not np.isfinite(value.numpy(force=True)).all():
        raise ValueError(f"{name} must contain only finite values.")


def validate_scores_and_classes(scores: torch.Tensor, class_ids: torch.Tensor, *, owner: str) -> None:
    """Validate confidence scores and class IDs without implicit conversion."""
    validate_tensor(scores, name=f"{owner}.scores", dtype=torch.float32, ndim=1)
    validate_tensor(class_ids, name=f"{owner}.class_ids", dtype=torch.int64, ndim=1)
    validate_finite(scores, name=f"{owner}.scores")
    if scores.numel() and bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError(f"{owner}.scores must be in the inclusive range [0, 1].")
    if class_ids.numel() and bool((class_ids < 0).any()):
        raise ValueError(f"{owner}.class_ids must be non-negative.")


def validate_nonempty_string(value: str, *, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def normalize_row_indices(indices: torch.Tensor, *, count: int) -> torch.Tensor:
    """Validate a one-dimensional CPU row selector and return integer indices."""
    if not isinstance(indices, torch.Tensor):
        raise TypeError(f"indices must be a torch.Tensor, got {type(indices).__name__}.")
    if indices.device.type != _CPU:
        raise ValueError(f"indices must be on CPU, got device {indices.device}.")
    if indices.ndim != 1:
        raise ValueError(f"indices must have one dimension, got shape {tuple(indices.shape)}.")

    if indices.dtype == torch.bool:
        if len(indices) != count:
            raise ValueError(f"Boolean indices must have length {count}, got {len(indices)}.")
        return torch.nonzero(indices, as_tuple=True)[0]
    if indices.dtype != torch.int64:
        raise TypeError(f"indices must have dtype torch.int64 or torch.bool, got {indices.dtype}.")
    if indices.numel() and (bool((indices < 0).any()) or bool((indices >= count).any())):
        raise IndexError(f"indices must be between 0 and {count - 1}.")
    return indices
