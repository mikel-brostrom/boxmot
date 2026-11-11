# BoxMOT AGPL-3.0 license

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from boxmot.reid.backbones.base import ReIDBackbone, format_reid_output
from boxmot.reid.backbones.common import init_kaiming_reid
from boxmot.reid.backbones.families.osnet.layers import Conv1x1, ConvLayer

__all__ = ["OSNet"]


class OSNet(ReIDBackbone):
    """Omni-Scale Network shared by standard, IBN, and AIN variants."""

    def __init__(
        self,
        num_classes: int,
        blocks,
        layers: Sequence[int],
        channels: Sequence[int],
        feature_dim: int = 512,
        loss: str = "softmax",
        IN: bool = False,
        conv1_IN: bool = False,
        use_intermediate_pools: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        num_blocks = len(blocks)
        if num_blocks != len(layers) or num_blocks != len(channels) - 1:
            raise ValueError("OSNet blocks/layers/channels lengths are inconsistent")
        self.loss = loss
        self.feature_dim = feature_dim
        self.use_intermediate_pools = bool(use_intermediate_pools)

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=conv1_IN or IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        if self.use_intermediate_pools:
            self.conv2 = self._make_layer_from_block_list(blocks[0], channels[0], channels[1])
            self.pool2 = nn.Sequential(Conv1x1(channels[1], channels[1]), nn.AvgPool2d(2, stride=2))
            self.conv3 = self._make_layer_from_block_list(blocks[1], channels[1], channels[2])
            self.pool3 = nn.Sequential(Conv1x1(channels[2], channels[2]), nn.AvgPool2d(2, stride=2))
            self.conv4 = self._make_layer_from_block_list(blocks[2], channels[2], channels[3])
        else:
            self.conv2 = self._make_layer(
                blocks[0],
                layers[0],
                channels[0],
                channels[1],
                reduce_spatial_size=True,
                IN=IN,
            )
            self.conv3 = self._make_layer(
                blocks[1],
                layers[1],
                channels[1],
                channels[2],
                reduce_spatial_size=True,
            )
            self.conv4 = self._make_layer(
                blocks[2],
                layers[2],
                channels[2],
                channels[3],
                reduce_spatial_size=False,
            )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(self.feature_dim, channels[3], dropout_p=None)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    def _make_layer(self, block, layer: int, in_channels: int, out_channels: int, reduce_spatial_size: bool, IN=False):
        layers = [block(in_channels, out_channels, IN=IN)]
        layers.extend(block(out_channels, out_channels, IN=IN) for _ in range(1, layer))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_channels, out_channels), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    @staticmethod
    def _make_layer_from_block_list(blocks, in_channels: int, out_channels: int):
        layers = [blocks[0](in_channels, out_channels)]
        layers.extend(block(out_channels, out_channels) for block in blocks[1:])
        return nn.Sequential(*layers)

    def _construct_fc_layer(self, fc_dims, input_dim: int, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*layers)

    def _init_params(self) -> None:
        init_kaiming_reid(self)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        if self.use_intermediate_pools:
            x = self.pool2(x)
        x = self.conv3(x)
        if self.use_intermediate_pools:
            x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward_head(self, features):
        v = self.global_avgpool(features)
        v = v.flatten(1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        return format_reid_output(self.loss, y, v)

    def forward(self, x, return_featuremaps: bool = False):
        x = self.forward_features(x)
        if return_featuremaps:
            return x
        return self.forward_head(x)
