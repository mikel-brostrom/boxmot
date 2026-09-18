"""Equivalent EdgeTAM point embeddings without MPS boolean-index synchronization."""

from __future__ import annotations

from typing import Any

import torch


def _embed_points(encoder: Any, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
    """Preserve the official prompt math using fixed-shape selection on MPS.

    Upstream uses boolean indexed reads and writes for each label, forcing MPS
    to resolve variable-size indices repeatedly for every tracked object. This
    instance-bound implementation selects the same additions with ``where``;
    it reuses the original positional encoder and all learned parameters.
    """
    points = points + 0.5
    if pad:
        padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
        padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        points = torch.cat([points, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)
    embedding = encoder.pe_layer.forward_with_coords(points, encoder.input_image_size)
    embedding = torch.where((labels == -1).unsqueeze(-1), encoder.not_a_point_embed.weight, embedding)
    for index, weights in enumerate(encoder.point_embeddings):
        embedding = torch.where((labels == index).unsqueeze(-1), embedding + weights.weight, embedding)
    return embedding
