# BoxMOT AGPL-3.0 license

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn.functional as F

from boxmot.reid.backbones.common.pretrained import (
    load_hub_checkpoint,
    load_partial_state_dict,
    log_pretrained_result,
)
from boxmot.reid.backbones.families.csl_tinyvit.attention import Attention
from boxmot.reid.backbones.families.csl_tinyvit.model import CSLTinyViT
from boxmot.reid.core.artifacts import file_sha256
from boxmot.utils import logger as LOGGER

__all__ = ["load_pretrained_tinyvit", "load_pretrained_tinyvit_checkpoint"]

# TinyViT-5M (ImageNet-1k, distilled from 22k): embed_dims=[64,128,160,320]
_TINYVIT_5M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_5m_22kto1k_distill.pth"
)
_TINYVIT_5M_SHA256 = "e4894bab91be2d8f1b5e9a2147b57f35ae84be46e195d6586308864fb06fad3d"

# TinyViT-11M (ImageNet-1k, distilled from 22k): embed_dims=[64,128,256,448]
_TINYVIT_11M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_11m_22kto1k_distill.pth"
)
_TINYVIT_11M_SHA256 = "98d4dde231bb9b8d98df178393e725ae8258115e939a6fb50210970f5f0d3192"

# TinyViT-21M (ImageNet-1k, distilled from 22k): embed_dims=[96,192,384,576]
_TINYVIT_21M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/tiny_vit_21m_22kto1k_distill.pth"
)
_TINYVIT_21M_SHA256 = "eb20032633663fc3e1e49c1cb21332c11e6804c9f920b77da5f832dad248b897"


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


def _required_pretrained_keys(model_state: dict[str, torch.Tensor]) -> set[str]:
    """Return target tensors that an official TinyViT backbone must provide."""
    required = {key for key in model_state if key.startswith(("patch_embed.", "layers."))}
    # Factorized signed biases are a BoxMOT treatment with no equivalent in
    # the official absolute-offset checkpoint.  All other backbone tensors,
    # including adapted absolute biases, are mandatory.
    return {
        key
        for key in required
        if ".reid_adapters." not in key and not key.endswith((".attention_bias_h", ".attention_bias_w"))
    }


def load_pretrained_tinyvit(model: CSLTinyViT, url: str, *, sha256: str | None = None) -> None:
    """Load TinyViT pretrained weights with partial key matching.

    Loads the official ``patch_embed`` and ``layers`` tensors from the
    ImageNet checkpoint.  The ReID neck and head are task-specific and remain
    freshly initialized.  Every compatible target-backbone tensor is required;
    a silently partial initialization raises instead of starting training with
    random backbone blocks.
    Rectangular models can opt into resizing compatible absolute attention-bias
    tables instead of initializing those resolution-dependent tensors at zero.
    """
    state_dict = load_hub_checkpoint(
        url,
        logger=LOGGER,
        weights_only=True,
        sha256=sha256,
    )
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
    target_state = model.state_dict()

    def transform_tensor(key: str, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        target = target_state.get(key)
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
        target_shape = target_state[key].shape
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

    required_keys = _required_pretrained_keys(target_state)
    matched_required = required_keys.intersection(matched)
    missing_required = sorted(required_keys.difference(matched_required))
    required_numel = sum(target_state[key].numel() for key in required_keys)
    matched_numel = sum(target_state[key].numel() for key in matched_required)
    tensor_coverage = len(matched_required) / max(len(required_keys), 1)
    numel_coverage = matched_numel / max(required_numel, 1)
    model.pretrained_backbone_tensor_coverage = tensor_coverage
    model.pretrained_backbone_numel_coverage = numel_coverage
    model.pretrained_backbone_required_tensor_count = len(required_keys)
    model.pretrained_backbone_matched_tensor_count = len(matched_required)
    model.pretrained_backbone_required_numel = required_numel
    model.pretrained_backbone_matched_numel = matched_numel
    model.pretrained_missing_backbone_keys = tuple(missing_required)
    if missing_required:
        preview = ", ".join(missing_required[:12])
        remainder = len(missing_required) - min(len(missing_required), 12)
        suffix = f" (+{remainder} more)" if remainder else ""
        raise RuntimeError(
            "Incomplete TinyViT pretrained backbone load: "
            f"{len(matched_required)}/{len(required_keys)} tensors and "
            f"{matched_numel}/{required_numel} elements matched; "
            f"missing {preview}{suffix}"
        )

    total = len(matched) + len(skipped)
    model.pretrained_match_count = len(matched)
    model.pretrained_total_count = total
    model.pretrained_url = url
    model.pretrained_sha256 = sha256
    model.pretrained_interpolated_attention_biases = tuple(interpolated_biases)
    log_pretrained_result(f"TinyViT ({url})", matched, skipped, logger=LOGGER)
    if matched:
        LOGGER.info(
            f"Loaded {len(matched)}/{total} pretrained tensors from TinyViT ({url}); "
            f"required backbone coverage={tensor_coverage:.2%} tensors/{numel_coverage:.2%} elements"
        )
    if skipped:
        LOGGER.info(f"Skipped {len(skipped)}/{total} layers (resolution-dependent / head)")
    if interpolated_biases:
        LOGGER.info(f"Interpolated {len(interpolated_biases)} pretrained absolute attention-bias tables")
    if sliced_stage3_mlp:
        LOGGER.info(f"Sliced {len(sliced_stage3_mlp)} pretrained Stage-3 MLP tensors to the reduced ratio")


_load_pretrained_tinyvit = load_pretrained_tinyvit


def load_pretrained_tinyvit_checkpoint(
    model: CSLTinyViT,
    checkpoint_path: str | Path,
) -> None:
    """Load an exact, local TinyViT backbone export into ``model``.

    Human-centric pretraining intentionally exports only ``patch_embed`` and
    ``layers``.  Unlike generic transfer loading, this handoff is strict about
    every deployable TinyViT backbone tensor: a wrong variant or incomplete
    cache fails before ReID fine-tuning can silently start from mixed random
    and pretrained weights.
    """
    source = Path(checkpoint_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"TinyViT pretrained checkpoint does not exist: {source}")
    source_sha256 = file_sha256(source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"TinyViT pretrained checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    state_dict: Mapping = checkpoint
    for key in ("state_dict", "model", "model_state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            state_dict = nested
            break

    backbone_state: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for original_key, value in state_dict.items():
        if not isinstance(original_key, str):
            skipped.append(str(original_key))
            continue
        key = original_key
        for prefix in ("module.encoder.", "encoder.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.startswith(("patch_embed.", "layers.")) and torch.is_tensor(value):
            backbone_state[key] = value
        else:
            skipped.append(original_key)

    matched, shape_skipped = load_partial_state_dict(
        model,
        backbone_state,
        strip_prefix=None,
    )
    skipped.extend(shape_skipped)
    target_state = model.state_dict()
    required_keys = _required_pretrained_keys(target_state)
    matched_required = required_keys.intersection(matched)
    missing_required = sorted(required_keys.difference(matched_required))
    required_numel = sum(target_state[key].numel() for key in required_keys)
    matched_numel = sum(target_state[key].numel() for key in matched_required)
    tensor_coverage = len(matched_required) / max(len(required_keys), 1)
    numel_coverage = matched_numel / max(required_numel, 1)
    if missing_required:
        preview = ", ".join(missing_required[:12])
        remainder = len(missing_required) - min(len(missing_required), 12)
        suffix = f" (+{remainder} more)" if remainder else ""
        raise RuntimeError(
            f"Incomplete local TinyViT backbone checkpoint {source}: "
            f"{len(matched_required)}/{len(required_keys)} required tensors and "
            f"{matched_numel}/{required_numel} elements matched; missing {preview}{suffix}"
        )

    model.pretrained_match_count = len(matched)
    model.pretrained_total_count = len(matched) + len(skipped)
    model.pretrained_url = str(source.resolve())
    model.pretrained_sha256 = source_sha256
    model.pretrained_backbone_tensor_coverage = tensor_coverage
    model.pretrained_backbone_numel_coverage = numel_coverage
    model.pretrained_backbone_required_tensor_count = len(required_keys)
    model.pretrained_backbone_matched_tensor_count = len(matched_required)
    model.pretrained_backbone_required_numel = required_numel
    model.pretrained_backbone_matched_numel = matched_numel
    model.pretrained_missing_backbone_keys = ()
    LOGGER.info(
        f"Loaded exact local TinyViT backbone from {source}: "
        f"{len(matched_required)}/{len(required_keys)} tensors, "
        f"{numel_coverage:.2%} elements"
    )
