# BoxMOT AGPL-3.0 license

from __future__ import annotations

from typing import Any

from boxmot.reid.backbones.families.osnet import OSBlockAIN, osnet_ain_x1_0
from boxmot.reid.backbones.lmbn import LMBNBackbone
from boxmot.reid.backbones.registry import register_backbone


@register_backbone(
    "lmbn_ain_n",
    family="cnn",
    default_recipe="lmbn_reid",
    default_img_size=(384, 128),
)
class LMBN_ain_n(LMBNBackbone):
    """LightMBN with an OSNet-AIN-x1.0 backbone."""

    def __init__(
        self,
        args: Any = None,
        test_only: bool = True,
        num_classes: int | None = None,
        loss: str | None = "softmax",
        pretrained: bool | None = None,
        use_gpu: Any = None,
    ) -> None:
        del use_gpu
        if args is not None:
            num_classes = args.num_classes if num_classes is None else num_classes
            test_only = getattr(args, "test_only", test_only)
            feat_dim = getattr(args, "feats", 512)
            activation_map = getattr(args, "activation_map", False)
        else:
            feat_dim = 512
            activation_map = False

        if num_classes is None:
            raise ValueError("num_classes is required to build LMBN_ain_n")

        if args is not None:
            backbone_pretrained = not test_only
        elif pretrained is not None:
            backbone_pretrained = bool(pretrained)
        else:
            backbone_pretrained = not test_only

        super().__init__(
            num_classes=num_classes,
            loss=loss or "softmax",
            pretrained=backbone_pretrained,
            osnet_builder=osnet_ain_x1_0,
            drop_bottleneck_type=OSBlockAIN,
            use_ain_pools=True,
            feat_dim=feat_dim,
            activation_map=activation_map,
        )
