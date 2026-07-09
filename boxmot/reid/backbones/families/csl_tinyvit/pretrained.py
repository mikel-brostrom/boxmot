# BoxMOT AGPL-3.0 license

from __future__ import annotations

from boxmot.reid.backbones.common.pretrained import (
    load_hub_checkpoint,
    load_partial_state_dict,
    log_pretrained_result,
)
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


def load_pretrained_tinyvit(model: CSLTinyViT, url: str) -> None:
    """Load TinyViT pretrained weights with partial key matching.

    Loads backbone layers (patch_embed, layers, neck) from the ImageNet
    checkpoint. Skips head/classifier and any keys with shape mismatches
    (e.g. attention biases that depend on input resolution).
    """
    state_dict = load_hub_checkpoint(url, logger=LOGGER, weights_only=False)
    head_skipped = [key for key in state_dict if "head" in key]
    backbone_state = {key: value for key, value in state_dict.items() if "head" not in key}
    matched, skipped = load_partial_state_dict(
        model,
        backbone_state,
        strip_prefix=None,
    )
    skipped = [*head_skipped, *skipped]

    total = len(matched) + len(skipped)
    model.pretrained_match_count = len(matched)
    model.pretrained_total_count = total
    model.pretrained_url = url
    log_pretrained_result(f"TinyViT ({url})", matched, skipped, logger=LOGGER)
    if matched:
        LOGGER.info(f"Loaded {len(matched)}/{total} pretrained tensors from TinyViT ({url})")
    if skipped:
        LOGGER.info(f"Skipped {len(skipped)}/{total} layers (resolution-dependent / head)")


_load_pretrained_tinyvit = load_pretrained_tinyvit
