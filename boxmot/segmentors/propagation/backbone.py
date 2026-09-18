"""Use official EdgeTAM backbone functionality without unused initialization."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from sam2.modeling.backbones.timm import TimmBackbone
from timm import create_model


class InferenceTimmBackbone(TimmBackbone):
    """Preserve upstream forward and checkpoint keys, loading only the full checkpoint."""

    def __init__(self, name: str, features: Sequence[str]) -> None:
        """Construct the same timm feature extractor with pretrained loading disabled."""
        torch.nn.Module.__init__(self)
        self.body = create_model(
            name,
            pretrained=False,
            in_chans=3,
            features_only=True,
            out_indices=tuple(int(feature.removeprefix("layer")) for feature in features),
        )
        self.channel_list = self.body.feature_info.channels()[::-1]
