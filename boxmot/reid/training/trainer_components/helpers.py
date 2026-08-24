"""Stateless data-worker and anatomical geometry helpers."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


def _seed_data_worker(worker_id: int) -> None:
    """Seed every worker-local RNG from the PyTorch DataLoader worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _bilinear_sample_2d(
    feature_map: torch.Tensor,
    grid: torch.Tensor,
) -> torch.Tensor:
    """Sample a 2D feature map with an MPS-differentiable bilinear path."""
    if feature_map.ndim != 4:
        raise ValueError("feature map must have shape [N,C,H,W]")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError("sampling grid must have shape [N,H,W,2]")
    if feature_map.shape[0] != grid.shape[0]:
        raise ValueError("feature map and sampling grid batches must match")

    batch_size, channels, height, width = feature_map.shape
    output_height, output_width = grid.shape[1:3]
    x = (grid[..., 0] + 1.0) * width * 0.5 - 0.5
    y = (grid[..., 1] + 1.0) * height * 0.5 - 0.5
    x0_float = torch.floor(x)
    y0_float = torch.floor(y)
    x1_float = x0_float + 1.0
    y1_float = y0_float + 1.0
    x0 = x0_float.to(torch.long)
    y0 = y0_float.to(torch.long)
    x1 = x1_float.to(torch.long)
    y1 = y1_float.to(torch.long)
    flattened = feature_map.flatten(2)

    def gather_neighbor(
        neighbor_x: torch.Tensor,
        neighbor_y: torch.Tensor,
    ) -> torch.Tensor:
        valid = (neighbor_x >= 0) & (neighbor_x < width) & (neighbor_y >= 0) & (neighbor_y < height)
        linear_index = neighbor_y.clamp(0, height - 1) * width + neighbor_x.clamp(0, width - 1)
        gathered = flattened.gather(
            2,
            linear_index.reshape(batch_size, 1, -1).expand(
                -1,
                channels,
                -1,
            ),
        ).reshape(
            batch_size,
            channels,
            output_height,
            output_width,
        )
        return gathered * valid[:, None].to(gathered.dtype)

    top_left = gather_neighbor(x0, y0)
    top_right = gather_neighbor(x1, y0)
    bottom_left = gather_neighbor(x0, y1)
    bottom_right = gather_neighbor(x1, y1)
    top_left_weight = (x1_float - x) * (y1_float - y)
    top_right_weight = (x - x0_float) * (y1_float - y)
    bottom_left_weight = (x1_float - x) * (y - y0_float)
    bottom_right_weight = (x - x0_float) * (y - y0_float)
    return (
        top_left * top_left_weight[:, None]
        + top_right * top_right_weight[:, None]
        + bottom_left * bottom_left_weight[:, None]
        + bottom_right * bottom_right_weight[:, None]
    )


def _scale_aware_anatomical_targets(
    source: torch.Tensor,
    masks: torch.Tensor,
    canonical_grid: torch.Tensor,
    grid_valid: torch.Tensor,
    mask_valid: torch.Tensor,
    *,
    fine_scale: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Build fixed pose-mask routing and same-scale RGB cell targets.

    Local routing intentionally covers a broader fraction of the person,
    matching its coarser semantic receptive field. Fine routing is sharper and
    therefore uses the additional spatial resolution instead of imitating the
    local map. RGB values are detached only after geometry has fixed where
    pooling occurs.
    """
    if source.ndim != 4:
        raise ValueError("anatomical source must have shape [B,C,H,W]")
    if masks.ndim != 4:
        raise ValueError("anatomical masks must have shape [B,P,H,W]")
    if canonical_grid.ndim != 4 or canonical_grid.shape[-1] != 2:
        raise ValueError("flattened canonical grid must have shape [B,P,K,2]")
    if grid_valid.shape != canonical_grid.shape[:-1]:
        raise ValueError("canonical grid validity must match the grid")
    if mask_valid.shape != (source.shape[0],):
        raise ValueError("mask validity must have shape [B]")
    if source.shape[0] != masks.shape[0] or masks.shape[:2] != canonical_grid.shape[:2]:
        raise ValueError("anatomical target batches and part counts must match")

    # Target construction is intentionally FP32 even when the model runs
    # under CUDA AMP. In FP16, 1e-8 rounds to zero, so an invalid/empty cell
    # would otherwise perform 0 / 0 and contaminate the complete loss.
    geometry_dtype = torch.float64 if source.dtype == torch.float64 else torch.float32
    masks = masks.to(
        device=source.device,
        dtype=geometry_dtype,
    )
    canonical_grid = canonical_grid.to(
        device=source.device,
        dtype=geometry_dtype,
    )
    grid_valid = grid_valid.to(
        device=source.device,
        dtype=torch.bool,
    )
    mask_valid = mask_valid.to(
        device=source.device,
        dtype=torch.bool,
    )
    source_values = source.detach().to(dtype=geometry_dtype)
    eps = torch.finfo(geometry_dtype).eps
    target_height, target_width = source.shape[-2:]
    resized_masks = F.interpolate(
        masks,
        size=(target_height, target_width),
        mode="area",
    ).clamp_min(0)
    grid_x = (canonical_grid[..., 0] + 1.0) * target_width * 0.5 - 0.5
    grid_y = (canonical_grid[..., 1] + 1.0) * target_height * 0.5 - 0.5
    spatial_x = torch.arange(
        target_width,
        device=source.device,
        dtype=geometry_dtype,
    )
    spatial_y = torch.arange(
        target_height,
        device=source.device,
        dtype=geometry_dtype,
    )
    height_fraction = 0.04 if fine_scale else 0.06
    width_fraction = 0.06 if fine_scale else 0.09
    sigma_y = max(target_height * height_fraction, 1.0)
    sigma_x = max(target_width * width_fraction, 0.75)
    distance = ((spatial_y[None, None, None, :, None] - grid_y[..., None, None]) / sigma_y).square() + (
        (spatial_x[None, None, None, None, :] - grid_x[..., None, None]) / sigma_x
    ).square()
    cell_routing = torch.exp(-0.5 * distance)
    mask_routing = cell_routing * resized_masks[:, :, None, :, :]
    cell_routing = torch.where(
        mask_valid[:, None, None, None, None],
        mask_routing,
        cell_routing,
    )
    cell_routing = cell_routing * grid_valid[..., None, None].to(cell_routing.dtype)
    routing_mass = cell_routing.sum(
        dim=(-1, -2),
        keepdim=True,
    )
    routing_valid = grid_valid & (routing_mass.squeeze(-1).squeeze(-1) > eps)
    cell_routing = torch.where(
        routing_valid[..., None, None],
        cell_routing / routing_mass.clamp_min(eps),
        torch.zeros_like(cell_routing),
    )

    dense_mask_mass = resized_masks.sum(
        dim=(-1, -2),
        keepdim=True,
    )
    dense_mask_target = resized_masks / dense_mask_mass.clamp_min(eps)
    pose_target = cell_routing.sum(dim=2)
    pose_target = pose_target / pose_target.sum(
        dim=(-1, -2),
        keepdim=True,
    ).clamp_min(eps)
    dense_target = torch.where(
        mask_valid[:, None, None, None],
        dense_mask_target,
        pose_target,
    )
    teacher_cell_tokens = torch.einsum(
        "bpkhw,bchw->bpkc",
        cell_routing,
        source_values,
    )
    return (
        cell_routing,
        dense_target,
        routing_valid,
        teacher_cell_tokens,
    )


def _cross_scale_role_relation_loss(
    local_tokens: torch.Tensor,
    fine_tokens: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Align anatomical role structure without equating raw scale features."""
    if local_tokens.shape != fine_tokens.shape:
        raise ValueError("local and fine anatomical tokens must match")
    if local_tokens.ndim != 3:
        raise ValueError("anatomical tokens must have shape [B,P,C]")
    if reliability.shape != local_tokens.shape[:2]:
        raise ValueError("anatomical reliability must have shape [B,P]")
    relation_dtype = torch.float64 if local_tokens.dtype == torch.float64 else torch.float32
    local_tokens = local_tokens.to(dtype=relation_dtype)
    fine_tokens = fine_tokens.to(dtype=relation_dtype)
    reliability = reliability.to(
        device=local_tokens.device,
        dtype=relation_dtype,
    )
    local_relations = torch.einsum(
        "bpc,bqc->bpq",
        F.normalize(local_tokens, p=2, dim=-1),
        F.normalize(local_tokens, p=2, dim=-1),
    )
    fine_relations = torch.einsum(
        "bpc,bqc->bpq",
        F.normalize(fine_tokens, p=2, dim=-1),
        F.normalize(fine_tokens, p=2, dim=-1),
    )
    relation_weights = (reliability[:, :, None] * reliability[:, None, :]).sqrt()
    relation_weights = relation_weights * (
        ~torch.eye(
            local_tokens.shape[1],
            device=relation_weights.device,
            dtype=torch.bool,
        )
    )[None].to(relation_weights.dtype)
    return ((local_relations - fine_relations).square() * relation_weights).sum() / relation_weights.sum().clamp_min(
        1e-6
    )
