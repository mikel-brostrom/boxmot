# BoxMOT AGPL-3.0 license

from boxmot.reid.backbones.families.osnet.blocks import ChannelGate, OSBlock, OSBlockAIN, OSBlockINin
from boxmot.reid.backbones.families.osnet.layers import (
    Conv1x1,
    Conv1x1Linear,
    Conv3x3,
    ConvLayer,
    LightConv3x3,
    LightConvStream,
)
from boxmot.reid.backbones.families.osnet.model import OSNet
from boxmot.reid.backbones.families.osnet.pretrained import load_osnet_pretrained, pretrained_urls
from boxmot.reid.backbones.families.osnet.variants import (
    osnet_ain_x0_5,
    osnet_ain_x0_25,
    osnet_ain_x0_75,
    osnet_ain_x1_0,
    osnet_ibn_x1_0,
    osnet_x0_5,
    osnet_x0_25,
    osnet_x0_75,
    osnet_x1_0,
)

__all__ = [
    "ChannelGate",
    "Conv1x1",
    "Conv1x1Linear",
    "Conv3x3",
    "ConvLayer",
    "LightConv3x3",
    "LightConvStream",
    "OSBlock",
    "OSBlockAIN",
    "OSBlockINin",
    "OSNet",
    "load_osnet_pretrained",
    "osnet_ain_x0_25",
    "osnet_ain_x0_5",
    "osnet_ain_x0_75",
    "osnet_ain_x1_0",
    "osnet_ibn_x1_0",
    "osnet_x0_25",
    "osnet_x0_5",
    "osnet_x0_75",
    "osnet_x1_0",
    "pretrained_urls",
]
