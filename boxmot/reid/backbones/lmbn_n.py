# BoxMOT AGPL-3.0 license

from __future__ import annotations

from typing import Any

from boxmot.reid.backbones.families.osnet import OSBlock, osnet_x1_0
from boxmot.reid.backbones.lmbn import LMBNBackbone
from boxmot.reid.backbones.registry import register_backbone


@register_backbone(
    "lmbn_n",
    family="cnn",
    default_recipe="lmbn_reid",
    default_img_size=(384, 128),
)
class LMBN_n(LMBNBackbone):
    """LightMBN with an OSNet-x1.0 backbone."""

    def __init__(
        self,
        num_classes: int,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: Any = None,
    ) -> None:
        del use_gpu
        super().__init__(
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            osnet_builder=osnet_x1_0,
            drop_bottleneck_type=OSBlock,
        )
