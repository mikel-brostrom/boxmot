"""Model-agnostic primitives for human-centric encoder pretraining.

The utilities in this module deliberately operate on tensors and state-dict
key conventions rather than a specific ReID model.  Pose and parsing models
are privileged data producers: only the RGB encoder is exported for downstream
fine-tuning and inference.
"""

from __future__ import annotations

import math
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

MissingMaskPolicy = Literal["skip", "error"]
MissingWeightPolicy = Literal["uniform", "skip", "error"]
FeatureLossType = Literal["cosine", "smooth_l1", "mse"]

__all__ = (
    "WholePartMask",
    "export_tinyvit_backbone_checkpoint",
    "foreground_aware_patch_target_weights",
    "normalize_part_maps",
    "pose_parser_guided_whole_part_mask",
    "semantic_teacher_feature_reconstruction_loss",
    "two_view_masked_consistency_loss",
)


@dataclass(frozen=True)
class WholePartMask:
    """One batch of sampled semantic-part masks.

    Attributes:
        pixel_mask: Boolean tensor shaped ``(batch, 1, height, width)``.
        selected_parts: Boolean tensor shaped ``(batch, num_parts)``.
        valid_samples: Whether each sample contained at least one usable part.
    """

    pixel_mask: torch.Tensor
    selected_parts: torch.Tensor
    valid_samples: torch.Tensor

    def apply(self, images: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """Return ``images`` with every selected semantic part replaced."""
        if images.ndim != 4:
            raise ValueError(f"images must have shape (B, C, H, W), got {tuple(images.shape)}")
        if images.shape[0] != self.pixel_mask.shape[0]:
            raise ValueError(
                "images and pixel_mask must have the same batch size, got "
                f"{images.shape[0]} and {self.pixel_mask.shape[0]}"
            )
        mask = self.pixel_mask
        if mask.shape[-2:] != images.shape[-2:]:
            mask = F.interpolate(mask.float(), size=images.shape[-2:], mode="nearest") > 0.5
        return torch.where(
            mask.to(device=images.device),
            torch.as_tensor(fill_value, device=images.device, dtype=images.dtype),
            images,
        )


def _work_dtype(tensor: torch.Tensor) -> torch.dtype:
    """Use FP32 for numerically fragile reductions on low-precision inputs."""
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    if torch.is_floating_point(tensor):
        return tensor.dtype
    return torch.float32


def _positive_finite(tensor: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Return finite, non-negative floating weights without changing device."""
    return torch.nan_to_num(
        tensor.to(dtype=dtype),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)


def _foreground_tensor(
    foreground_mask: torch.Tensor,
    *,
    batch_size: int,
    size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate and resize a foreground mask to ``(B, 1, H, W)``."""
    if foreground_mask.ndim == 3:
        foreground_mask = foreground_mask[:, None]
    if foreground_mask.ndim != 4 or foreground_mask.shape[1] != 1:
        raise ValueError(
            f"foreground_mask must have shape (B, H, W) or (B, 1, H, W), got {tuple(foreground_mask.shape)}"
        )
    if foreground_mask.shape[0] != batch_size:
        raise ValueError(
            f"foreground_mask batch size does not match the target batch: {foreground_mask.shape[0]} != {batch_size}"
        )
    foreground = _positive_finite(
        foreground_mask.to(device=device),
        dtype=dtype,
    ).clamp_max(1.0)
    if foreground.shape[-2:] != size:
        foreground = F.interpolate(
            foreground,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
    return foreground


def normalize_part_maps(
    part_maps: torch.Tensor,
    foreground_mask: torch.Tensor | None = None,
    *,
    normalization: Literal["pixel", "spatial"] = "pixel",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize non-negative semantic part maps without producing NaNs.

    ``pixel`` normalization makes the part channels at each supported pixel
    sum to one.  ``spatial`` normalization makes each non-empty part map sum to
    one over the image.  Empty maps stay exactly zero in either mode.  Work is
    performed in FP32 for FP16/BF16 input and converted back on return.

    Args:
        part_maps: Non-negative parser masks/probabilities shaped ``(B, P, H, W)``.
        foreground_mask: Optional person mask used to suppress background evidence.
        normalization: Normalize across parts per pixel or across space per part.
        eps: Strictly positive denominator floor.
    """
    if part_maps.ndim != 4:
        raise ValueError(f"part_maps must have shape (B, P, H, W), got {tuple(part_maps.shape)}")
    if part_maps.shape[1] < 1:
        raise ValueError("part_maps must contain at least one semantic part")
    if normalization not in {"pixel", "spatial"}:
        raise ValueError(f"Unsupported part-map normalization: {normalization!r}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    output_dtype = part_maps.dtype if torch.is_floating_point(part_maps) else torch.float32
    work_dtype = _work_dtype(part_maps)
    maps = _positive_finite(part_maps, dtype=work_dtype)
    if foreground_mask is not None:
        foreground = _foreground_tensor(
            foreground_mask,
            batch_size=part_maps.shape[0],
            size=part_maps.shape[-2:],
            device=part_maps.device,
            dtype=work_dtype,
        )
        maps = maps * foreground

    reduce_dims = (1,) if normalization == "pixel" else (2, 3)
    denominator = maps.sum(dim=reduce_dims, keepdim=True)
    normalized = maps / denominator.clamp_min(eps)
    normalized = torch.where(denominator > eps, normalized, torch.zeros_like(normalized))
    return normalized.to(dtype=output_dtype)


def _randperm(count: int, generator: torch.Generator | None, device: torch.device) -> torch.Tensor:
    """Sample on the generator's device, then move indices to the data device."""
    if generator is None:
        return torch.randperm(count, device=device)
    generator_device = torch.device(generator.device)
    return torch.randperm(count, generator=generator, device=generator_device).to(device=device)


def pose_parser_guided_whole_part_mask(
    part_maps: torch.Tensor | None,
    *,
    mask_ratio: float,
    generator: torch.Generator | None = None,
    foreground_mask: torch.Tensor | None = None,
    missing_target: MissingMaskPolicy = "skip",
    eps: float = 1e-6,
) -> WholePartMask | None:
    """Select complete parser/pose parts and return their hard spatial union.

    A fixed number of available parts is sampled independently per image.  A
    pixel belongs to its highest-probability normalized part, so low-confidence
    probability tails cannot make a selected part mask the whole image.  All
    random choices use ``generator`` when provided.

    ``None`` targets return ``None`` under ``missing_target='skip'``.  Samples
    containing only empty maps remain in the returned batch with
    ``valid_samples=False`` and an all-false mask.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in [0, 1]")
    if missing_target not in {"skip", "error"}:
        raise ValueError(f"Unsupported missing_target policy: {missing_target!r}")
    if part_maps is None:
        if missing_target == "error":
            raise ValueError("pose/parser part maps are required")
        return None

    normalized = normalize_part_maps(
        part_maps,
        foreground_mask,
        normalization="pixel",
        eps=eps,
    )
    work = normalized.to(dtype=_work_dtype(normalized))
    part_available = work.sum(dim=(2, 3)) > eps
    valid_samples = part_available.any(dim=1)
    selected_parts = torch.zeros_like(part_available)

    for batch_index in range(part_maps.shape[0]):
        available_indices = torch.where(part_available[batch_index])[0]
        available_count = int(available_indices.numel())
        if available_count == 0 or mask_ratio == 0:
            continue
        selected_count = min(
            available_count,
            max(1, math.ceil(mask_ratio * available_count)),
        )
        order = _randperm(available_count, generator, part_maps.device)
        selected_parts[batch_index, available_indices[order[:selected_count]]] = True

    evidence = work.sum(dim=1, keepdim=True) > eps
    semantic_labels = work.argmax(dim=1, keepdim=True)
    chosen_at_pixel = selected_parts.gather(
        1,
        semantic_labels.flatten(1),
    ).view_as(semantic_labels)
    pixel_mask = evidence & chosen_at_pixel
    return WholePartMask(
        pixel_mask=pixel_mask,
        selected_parts=selected_parts,
        valid_samples=valid_samples,
    )


def foreground_aware_patch_target_weights(
    foreground_mask: torch.Tensor | None,
    patch_grid: tuple[int, int],
    *,
    batch_size: int | None = None,
    foreground_weight: float = 1.0,
    background_weight: float = 0.25,
    normalize: bool = True,
    missing_target: MissingWeightPolicy = "uniform",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert a person mask into foreground-aware patch reconstruction weights.

    The output has shape ``(B, grid_h, grid_w)``.  Patch coverage is computed
    with area averaging, then interpolated between ``background_weight`` and
    ``foreground_weight``.  Per-sample mean normalization keeps the global loss
    scale stable when foreground area changes.

    Missing masks can produce uniform weights, all-zero skip weights, or an
    exception.  ``batch_size`` is required only when ``foreground_mask`` is
    missing.
    """
    grid_h, grid_w = (int(value) for value in patch_grid)
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"patch_grid must be positive, got {patch_grid}")
    if foreground_weight < 0 or background_weight < 0:
        raise ValueError("foreground_weight and background_weight must be non-negative")
    if missing_target not in {"uniform", "skip", "error"}:
        raise ValueError(f"Unsupported missing_target policy: {missing_target!r}")
    if eps <= 0:
        raise ValueError("eps must be positive")

    if foreground_mask is None:
        if missing_target == "error":
            raise ValueError("foreground_mask is required")
        if batch_size is None or batch_size <= 0:
            raise ValueError("positive batch_size is required when foreground_mask is missing")
        fill = 1.0 if missing_target == "uniform" else 0.0
        return torch.full(
            (batch_size, grid_h, grid_w),
            fill,
            device=device,
            dtype=dtype,
        )

    if foreground_mask.ndim == 3:
        foreground_mask = foreground_mask[:, None]
    if foreground_mask.ndim != 4 or foreground_mask.shape[1] != 1:
        raise ValueError(
            f"foreground_mask must have shape (B, H, W) or (B, 1, H, W), got {tuple(foreground_mask.shape)}"
        )
    if batch_size is not None and foreground_mask.shape[0] != batch_size:
        raise ValueError(
            f"foreground_mask batch size {foreground_mask.shape[0]} does not match batch_size={batch_size}"
        )
    target_device = foreground_mask.device if device is None else torch.device(device)
    coverage = _positive_finite(
        foreground_mask.to(device=target_device),
        dtype=torch.float32,
    ).clamp_max(1.0)
    coverage = F.adaptive_avg_pool2d(coverage, (grid_h, grid_w)).squeeze(1)
    weights = background_weight + (foreground_weight - background_weight) * coverage
    if normalize:
        mean = weights.mean(dim=(1, 2), keepdim=True)
        weights = torch.where(mean > eps, weights / mean.clamp_min(eps), torch.zeros_like(weights))
    return weights.to(dtype=dtype)


def _feature_tokens(
    features: torch.Tensor,
    *,
    channel_dim: int,
) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """Return features as ``(B, tokens, channels)`` plus an optional 2D grid."""
    if features.ndim < 2:
        raise ValueError(f"features must include batch and channel dimensions, got {tuple(features.shape)}")
    resolved_channel_dim = channel_dim if channel_dim >= 0 else features.ndim + channel_dim
    if resolved_channel_dim <= 0 or resolved_channel_dim >= features.ndim:
        raise ValueError(
            f"channel_dim must identify a non-batch dimension for shape {tuple(features.shape)}, got {channel_dim}"
        )
    moved = features.movedim(resolved_channel_dim, -1)
    grid = tuple(int(value) for value in moved.shape[1:-1]) if moved.ndim == 4 else None
    return moved.reshape(moved.shape[0], -1, moved.shape[-1]), grid


def _token_weights(
    weights: torch.Tensor | None,
    *,
    batch_size: int,
    token_count: int,
    token_grid: tuple[int, int] | None,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Broadcast or spatially pool weights to ``(B, tokens)``."""
    if weights is None:
        return torch.ones((batch_size, token_count), device=device, dtype=dtype)
    if not torch.is_tensor(weights):
        raise TypeError(f"{name} must be a tensor or None")
    value = _positive_finite(weights.to(device=device), dtype=dtype)
    if value.ndim > 0 and value.shape[0] not in {1, batch_size}:
        raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {value.shape[0]}")
    if value.ndim == 0:
        return value.expand(batch_size, token_count)
    if value.shape[0] == 1 and batch_size > 1:
        value = value.expand(batch_size, *value.shape[1:])
    if value.numel() == batch_size:
        return value.reshape(batch_size, 1).expand(batch_size, token_count)
    if value.numel() == batch_size * token_count:
        return value.reshape(batch_size, token_count)
    if token_grid is not None and value.ndim in {3, 4}:
        if value.ndim == 3:
            value = value[:, None]
        if value.shape[1] != 1:
            raise ValueError(f"spatial {name} must have one channel, got {tuple(value.shape)}")
        return F.adaptive_avg_pool2d(value, token_grid).reshape(batch_size, token_count)
    raise ValueError(
        f"{name} with shape {tuple(weights.shape)} cannot be aligned to (batch={batch_size}, tokens={token_count})"
    )


def _weighted_token_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Return a weighted mean or a differentiable zero when no token is valid."""
    denominator = weights.sum()
    numerator = (values * weights).sum()
    zero = values.sum() * 0.0
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12), zero)


def two_view_masked_consistency_loss(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    token_mask: torch.Tensor | None = None,
    *,
    valid_samples: torch.Tensor | None = None,
    channel_dim: int = -1,
    token_grid: tuple[int, int] | None = None,
    detach_view_b: bool = False,
) -> torch.Tensor:
    """Match aligned features from two augmented views on selected tokens.

    The loss is cosine distance.  ``token_mask`` may be per-token, per-sample,
    or a spatial mask.  An all-zero mask returns a differentiable zero.  Set
    ``detach_view_b`` for an online/target-teacher arrangement.
    """
    tokens_a, inferred_grid_a = _feature_tokens(view_a, channel_dim=channel_dim)
    tokens_b, inferred_grid_b = _feature_tokens(view_b, channel_dim=channel_dim)
    if tokens_a.shape != tokens_b.shape:
        raise ValueError(
            "two-view features must have identical canonical shape, got "
            f"{tuple(tokens_a.shape)} and {tuple(tokens_b.shape)}"
        )
    inferred_grid = inferred_grid_a or inferred_grid_b
    if inferred_grid_a is not None and inferred_grid_b is not None and inferred_grid_a != inferred_grid_b:
        raise ValueError(f"two-view spatial grids differ: {inferred_grid_a} != {inferred_grid_b}")
    grid = token_grid or inferred_grid
    batch_size, token_count, _ = tokens_a.shape
    weights = _token_weights(
        token_mask,
        batch_size=batch_size,
        token_count=token_count,
        token_grid=grid,
        device=tokens_a.device,
        dtype=_work_dtype(tokens_a),
        name="token_mask",
    )
    if valid_samples is not None:
        weights = weights * _token_weights(
            valid_samples,
            batch_size=batch_size,
            token_count=token_count,
            token_grid=None,
            device=tokens_a.device,
            dtype=weights.dtype,
            name="valid_samples",
        )
    target_b = tokens_b.detach() if detach_view_b else tokens_b
    per_token = 1.0 - F.cosine_similarity(
        tokens_a.to(dtype=weights.dtype),
        target_b.to(device=tokens_a.device, dtype=weights.dtype),
        dim=-1,
        eps=1e-6,
    )
    return _weighted_token_mean(per_token, weights)


def semantic_teacher_feature_reconstruction_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor | None,
    token_mask: torch.Tensor | None = None,
    *,
    target_weights: torch.Tensor | None = None,
    valid_samples: torch.Tensor | None = None,
    channel_dim: int = -1,
    token_grid: tuple[int, int] | None = None,
    loss_type: FeatureLossType = "cosine",
    normalize_features: bool = True,
    detach_teacher: bool = True,
    missing_target: MissingMaskPolicy = "skip",
) -> torch.Tensor:
    """Reconstruct privileged semantic-teacher features on selected patches.

    Teacher gradients are stopped by default.  Missing teacher targets either
    produce a differentiable zero attached to the student or raise explicitly.
    Token masks, foreground-aware target weights, and per-sample validity are
    multiplied before reduction.
    """
    if missing_target not in {"skip", "error"}:
        raise ValueError(f"Unsupported missing_target policy: {missing_target!r}")
    if teacher_features is None:
        if missing_target == "error":
            raise ValueError("semantic teacher features are required")
        return student_features.sum() * 0.0
    if loss_type not in {"cosine", "smooth_l1", "mse"}:
        raise ValueError(f"Unsupported semantic feature loss: {loss_type!r}")

    student_tokens, student_grid = _feature_tokens(student_features, channel_dim=channel_dim)
    teacher_tokens, teacher_grid = _feature_tokens(teacher_features, channel_dim=channel_dim)
    if student_tokens.shape != teacher_tokens.shape:
        raise ValueError(
            "student and teacher features must have identical canonical shape, got "
            f"{tuple(student_tokens.shape)} and {tuple(teacher_tokens.shape)}"
        )
    inferred_grid = student_grid or teacher_grid
    if student_grid is not None and teacher_grid is not None and student_grid != teacher_grid:
        raise ValueError(f"student and teacher spatial grids differ: {student_grid} != {teacher_grid}")
    grid = token_grid or inferred_grid
    batch_size, token_count, _ = student_tokens.shape
    work_dtype = _work_dtype(student_tokens)
    weights = _token_weights(
        token_mask,
        batch_size=batch_size,
        token_count=token_count,
        token_grid=grid,
        device=student_tokens.device,
        dtype=work_dtype,
        name="token_mask",
    )
    weights = weights * _token_weights(
        target_weights,
        batch_size=batch_size,
        token_count=token_count,
        token_grid=grid,
        device=student_tokens.device,
        dtype=work_dtype,
        name="target_weights",
    )
    if valid_samples is not None:
        weights = weights * _token_weights(
            valid_samples,
            batch_size=batch_size,
            token_count=token_count,
            token_grid=None,
            device=student_tokens.device,
            dtype=work_dtype,
            name="valid_samples",
        )

    student = student_tokens.to(dtype=work_dtype)
    teacher = teacher_tokens.to(device=student.device, dtype=work_dtype)
    if detach_teacher:
        teacher = teacher.detach()
    if normalize_features:
        student = F.normalize(student, p=2, dim=-1, eps=1e-6)
        teacher = F.normalize(teacher, p=2, dim=-1, eps=1e-6)

    if loss_type == "cosine":
        per_token = 1.0 - F.cosine_similarity(student, teacher, dim=-1, eps=1e-6)
    elif loss_type == "smooth_l1":
        per_token = F.smooth_l1_loss(student, teacher, reduction="none").mean(dim=-1)
    else:
        per_token = F.mse_loss(student, teacher, reduction="none").mean(dim=-1)
    return _weighted_token_mean(per_token, weights)


def _unwrap_state_dict(source: nn.Module | Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a module state dict or unwrap one common checkpoint container."""
    if isinstance(source, nn.Module):
        return source.state_dict()
    if not isinstance(source, Mapping):
        raise TypeError(f"source must be an nn.Module or mapping, got {type(source).__name__}")
    for key in ("state_dict", "model", "model_state_dict"):
        nested = source.get(key)
        if isinstance(nested, Mapping):
            return nested
    return source


def export_tinyvit_backbone_checkpoint(
    source: nn.Module | Mapping[str, Any],
    output_path: str | Path,
    *,
    source_prefix: str = "",
    backbone_prefixes: Sequence[str] = ("patch_embed.", "layers."),
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Export CPU backbone weights accepted by ``load_pretrained_tinyvit``.

    The saved wrapper uses the existing ``state_dict`` extraction convention,
    and exported keys retain the native ``patch_embed.*`` / ``layers.*`` names
    required by the TinyViT loader.  Pass ``source_prefix='encoder.'`` when the
    exact encoder is nested inside a pretraining wrapper.
    """
    prefixes = tuple(str(prefix) for prefix in backbone_prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("backbone_prefixes must contain at least one non-empty prefix")
    state_dict = _unwrap_state_dict(source)
    exported: OrderedDict[str, torch.Tensor] = OrderedDict()
    for original_key, value in state_dict.items():
        if not isinstance(original_key, str):
            continue
        if source_prefix:
            if not original_key.startswith(source_prefix):
                continue
            key = original_key[len(source_prefix) :]
        else:
            key = original_key
        if not key.startswith(prefixes):
            continue
        if not torch.is_tensor(value):
            raise TypeError(f"Backbone state value for {original_key!r} is not a tensor")
        if key in exported:
            raise ValueError(f"Duplicate exported backbone key: {key}")
        exported[key] = value.detach().to(device="cpu").contiguous().clone()

    missing_prefixes = [prefix for prefix in prefixes if not any(key.startswith(prefix) for key in exported)]
    if missing_prefixes:
        raise ValueError(
            "Source does not contain every requested TinyViT backbone prefix after stripping "
            f"source_prefix={source_prefix!r}: missing {missing_prefixes}"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": "boxmot-tinyvit-backbone-v1",
        "state_dict": exported,
        "metadata": dict(metadata or {}),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination
