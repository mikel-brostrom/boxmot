"""Preserve EdgeTAM's spatial Perceiver math for multiple object memories."""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Adapted from EdgeTAM's sam2/modeling/perceiver.py under EDGETAM_LICENSE.
# BoxMOT modification: reshape expanded tensors without assuming contiguous strides.

from __future__ import annotations

import math
from typing import Any

import torch


def _forward_2d(perceiver: Any, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the original windowed layers with a batch-safe learned-latent layout.

    Expanding the learned latents creates a stride-zero batch dimension. The
    upstream ``view`` cannot merge that dimension with the latent dimension
    when more than one object is present. ``reshape`` preserves the same object
    and window ordering while materializing that expansion when necessary.
    Bind this method to one predictor instance; all original modules, weights,
    positional encodings, normalization and training behavior remain in use.
    """
    batch, channels, height, width = features.shape
    latents = perceiver.latents_2d.unsqueeze(0).expand(batch, -1, -1).reshape(-1, 1, channels)

    windows_per_side = int(math.sqrt(perceiver.num_latents_2d))
    window_size = height // windows_per_side
    windows = features.permute(0, 2, 3, 1).reshape(
        batch, height // window_size, window_size, width // window_size, window_size, channels
    )
    windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, channels)
    for layer in perceiver.layers:
        latents = layer(latents, windows)

    latents = latents.reshape(batch, windows_per_side, windows_per_side, channels).permute(0, 3, 1, 2)
    positions = perceiver.position_encoding(latents).permute(0, 2, 3, 1).flatten(1, 2)
    latents = perceiver.norm(latents.permute(0, 2, 3, 1).flatten(1, 2))
    return latents, positions
