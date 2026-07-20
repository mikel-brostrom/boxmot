# BoxMOT AGPL-3.0 license

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from boxmot.reid.backbones.common.pretrained import (
    load_hub_checkpoint,
    load_partial_state_dict,
    log_pretrained_result,
)
from boxmot.reid.backbones.families.csl_tinyvit.attention import Attention
from boxmot.reid.backbones.families.csl_tinyvit.model import CSLTinyViT
from boxmot.utils import logger as LOGGER

__all__ = ["load_pretrained_tinyvit"]

# TinyViT-5M (ImageNet-1k, distilled from 22k): embed_dims=[64,128,160,320]
_TINYVIT_5M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth"
)

# TinyViT-11M (ImageNet-1k, distilled from 22k): embed_dims=[64,128,256,448]
_TINYVIT_11M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth"
)

# TinyViT-21M (ImageNet-1k, distilled from 22k): embed_dims=[96,192,384,576]
_TINYVIT_21M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth"
)


def _resize_absolute_attention_bias(
    bias: torch.Tensor,
    target_resolution: tuple[int, int],
) -> torch.Tensor:
    """Resize one square absolute-offset bias table to a target window."""
    if bias.ndim != 2:
        raise ValueError(f"Expected a 2D attention-bias table, got shape {tuple(bias.shape)}")
    source_side = math.isqrt(bias.shape[1])
    if source_side * source_side != bias.shape[1]:
        raise ValueError(f"Expected a square 2D attention-bias table, got shape {tuple(bias.shape)}")

    target_height, target_width = target_resolution
    resized = F.interpolate(
        bias.float().reshape(1, bias.shape[0], source_side, source_side),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=True,
    )
    return resized.reshape(bias.shape[0], target_height * target_width).to(dtype=bias.dtype)


def load_pretrained_tinyvit(model: CSLTinyViT, url: str) -> None:
    """Load TinyViT pretrained weights with partial key matching.

    Loads backbone layers (patch_embed, layers, neck) from the ImageNet
    checkpoint. Skips head/classifier and any keys with shape mismatches.
    Rectangular models can opt into resizing compatible absolute attention-bias
    tables instead of initializing those resolution-dependent tensors at zero.
    """
    state_dict = load_hub_checkpoint(url, logger=LOGGER, weights_only=False)
    head_skipped = [key for key in state_dict if "head" in key]
    backbone_state = {key: value for key, value in state_dict.items() if "head" not in key}
    interpolate_bias = bool(getattr(model, "interpolate_pretrained_attention_bias", False))
    target_bias_resolutions = {
        f"{name}.attention_biases": module.resolution
        for name, module in model.named_modules()
        if isinstance(module, Attention) and module.bias_mode == "absolute"
    }
    interpolated_biases: list[str] = []
    sliced_stage3_mlp: list[str] = []

    def transform_tensor(key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        target = model.state_dict().get(key)
        if target is not None and value.shape != target.shape and ".layers.3.blocks." in f".{key}":
            if key.endswith(".mlp.fc1.weight") and value.shape[1:] == target.shape[1:]:
                value = value[: target.shape[0]]
            elif key.endswith(".mlp.fc1.bias") and value.ndim == target.ndim == 1:
                value = value[: target.shape[0]]
            elif key.endswith(".mlp.fc2.weight") and value.shape[0] == target.shape[0]:
                value = value[:, : target.shape[1]]
            if value.shape == target.shape:
                sliced_stage3_mlp.append(key)
        target_resolution = target_bias_resolutions.get(key)
        if not interpolate_bias or target_resolution is None:
            return key, value
        target_shape = model.state_dict()[key].shape
        if value.shape == target_shape:
            return key, value
        try:
            value = _resize_absolute_attention_bias(value, target_resolution)
        except ValueError:
            return key, value
        if value.shape == target_shape:
            interpolated_biases.append(key)
        return key, value

    matched, skipped = load_partial_state_dict(
        model,
        backbone_state,
        strip_prefix=None,
        tensor_transform=transform_tensor,
    )
    skipped = [*head_skipped, *skipped]

    total = len(matched) + len(skipped)
    model.pretrained_match_count = len(matched)
    model.pretrained_total_count = total
    model.pretrained_url = url
    model.pretrained_interpolated_attention_biases = tuple(interpolated_biases)
    log_pretrained_result(f"TinyViT ({url})", matched, skipped, logger=LOGGER)
    if matched:
        LOGGER.info(f"Loaded {len(matched)}/{total} pretrained tensors from TinyViT ({url})")
    if skipped:
        LOGGER.info(f"Skipped {len(skipped)}/{total} layers (resolution-dependent / head)")
    if interpolated_biases:
        LOGGER.info(f"Interpolated {len(interpolated_biases)} pretrained absolute attention-bias tables")
    if sliced_stage3_mlp:
        LOGGER.info(f"Sliced {len(sliced_stage3_mlp)} pretrained Stage-3 MLP tensors to the reduced ratio")


_load_pretrained_tinyvit = load_pretrained_tinyvit
