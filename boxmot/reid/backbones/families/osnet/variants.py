# BoxMOT AGPL-3.0 license

from __future__ import annotations

from dataclasses import dataclass

from boxmot.reid.backbones.families.osnet.blocks import OSBlock, OSBlockAIN, OSBlockINin
from boxmot.reid.backbones.families.osnet.model import OSNet
from boxmot.reid.backbones.families.osnet.pretrained import load_osnet_pretrained
from boxmot.reid.backbones.registry import BackboneVariant, register_variant

_OSNET_PUBLIC_NAMES = (
    "osnet_ain_x0_25",
    "osnet_ain_x0_5",
    "osnet_ain_x0_75",
    "osnet_ain_x1_0",
    "osnet_ibn_x1_0",
    "osnet_x0_25",
    "osnet_x0_5",
    "osnet_x0_75",
    "osnet_x1_0",
)

__all__ = list(_OSNET_PUBLIC_NAMES)


@dataclass(frozen=True)
class OSNetVariant:
    name: str
    channels: tuple[int, int, int, int]
    blocks: tuple
    conv1_in: bool = False
    use_intermediate_pools: bool = False
    conv2_in: bool = False


STANDARD_BLOCKS = (OSBlock, OSBlock, OSBlock)
AIN_BLOCKS = (
    (OSBlockINin, OSBlockINin),
    (OSBlockAIN, OSBlockINin),
    (OSBlockINin, OSBlockAIN),
)


def _build_osnet_variant(
    *,
    key: str,
    spec: OSNetVariant,
    num_classes: int = 1000,
    pretrained: bool = True,
    loss: str = "softmax",
    use_gpu=None,
    **kwargs,
) -> OSNet:
    del use_gpu
    model = OSNet(
        num_classes,
        blocks=spec.blocks,
        layers=(2, 2, 2),
        channels=spec.channels,
        loss=loss,
        IN=spec.conv2_in,
        conv1_IN=spec.conv1_in,
        use_intermediate_pools=spec.use_intermediate_pools,
        **kwargs,
    )
    if pretrained:
        load_osnet_pretrained(model, key=key)
    return model


def make_osnet_builder(spec: OSNetVariant):
    def builder(num_classes=1000, pretrained=True, loss="softmax", use_gpu=None, **kwargs):
        return _build_osnet_variant(
            key=spec.name,
            spec=spec,
            num_classes=num_classes,
            pretrained=pretrained,
            loss=loss,
            use_gpu=use_gpu,
            **kwargs,
        )

    builder.__name__ = spec.name
    builder.__qualname__ = spec.name
    builder.__module__ = __name__
    return builder


_OSNET_VARIANTS = (
    OSNetVariant("osnet_x1_0", channels=(64, 256, 384, 512), blocks=STANDARD_BLOCKS),
    OSNetVariant("osnet_x0_75", channels=(48, 192, 288, 384), blocks=STANDARD_BLOCKS),
    OSNetVariant("osnet_x0_5", channels=(32, 128, 192, 256), blocks=STANDARD_BLOCKS),
    OSNetVariant("osnet_x0_25", channels=(16, 64, 96, 128), blocks=STANDARD_BLOCKS),
    OSNetVariant("osnet_ibn_x1_0", channels=(64, 256, 384, 512), blocks=STANDARD_BLOCKS, conv2_in=True),
    OSNetVariant(
        "osnet_ain_x1_0",
        channels=(64, 256, 384, 512),
        blocks=AIN_BLOCKS,
        conv1_in=True,
        use_intermediate_pools=True,
    ),
    OSNetVariant(
        "osnet_ain_x0_75",
        channels=(48, 192, 288, 384),
        blocks=AIN_BLOCKS,
        conv1_in=True,
        use_intermediate_pools=True,
    ),
    OSNetVariant(
        "osnet_ain_x0_5",
        channels=(32, 128, 192, 256),
        blocks=AIN_BLOCKS,
        conv1_in=True,
        use_intermediate_pools=True,
    ),
    OSNetVariant(
        "osnet_ain_x0_25",
        channels=(16, 64, 96, 128),
        blocks=AIN_BLOCKS,
        conv1_in=True,
        use_intermediate_pools=True,
    ),
)


for _variant in _OSNET_VARIANTS:
    globals()[_variant.name] = register_variant(
        BackboneVariant(
            name=_variant.name,
            family="cnn",
            default_recipe="cnn_reid",
            pretrained_source="imagenet",
        )
    )(make_osnet_builder(_variant))

del _variant
