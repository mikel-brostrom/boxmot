# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import absolute_import, division

import torch
from torch import nn
from torch.nn import functional as F

from boxmot.reid.backbones.base import ReIDBackbone, format_reid_output
from boxmot.reid.backbones.common import init_kaiming_reid, warn_manual_pretrained_download
from boxmot.reid.backbones.registry import register_backbone

__all__ = ["mlfn"]

model_urls = {
    # training epoch = 5, top1 = 51.6
    "imagenet": "https://mega.nz/#!YHxAhaxC!yu9E6zWl0x5zscSouTdbZu8gdFFytDdl-RAdD2DEfpk",
}


class MLFNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, fsm_channels, groups=32):
        super(MLFNBlock, self).__init__()
        self.groups = groups
        mid_channels = out_channels // 2

        # Factor Modules
        self.fm_conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.fm_bn1 = nn.BatchNorm2d(mid_channels)
        self.fm_conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            3,
            stride=stride,
            padding=1,
            bias=False,
            groups=self.groups,
        )
        self.fm_bn2 = nn.BatchNorm2d(mid_channels)
        self.fm_conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.fm_bn3 = nn.BatchNorm2d(out_channels)

        # Factor Selection Module
        self.fsm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, fsm_channels[0], 1),
            nn.BatchNorm2d(fsm_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(fsm_channels[0], fsm_channels[1], 1),
            nn.BatchNorm2d(fsm_channels[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(fsm_channels[1], self.groups, 1),
            nn.BatchNorm2d(self.groups),
            nn.Sigmoid(),
        )

        self.downsample = None
        if in_channels != out_channels or stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        s = self.fsm(x)

        # reduce dimension
        x = self.fm_conv1(x)
        x = self.fm_bn1(x)
        x = F.relu(x, inplace=True)

        # group convolution
        x = self.fm_conv2(x)
        x = self.fm_bn2(x)
        x = F.relu(x, inplace=True)

        # factor selection
        b, c = x.size(0), x.size(1)
        n = c // self.groups
        ss = s.repeat(1, n, 1, 1)  # from (b, g, 1, 1) to (b, g*n=c, 1, 1)
        ss = ss.view(b, n, self.groups, 1, 1)
        ss = ss.permute(0, 2, 1, 3, 4).contiguous()
        ss = ss.view(b, c, 1, 1)
        x = ss * x

        # recover dimension
        x = self.fm_conv3(x)
        x = self.fm_bn3(x)
        x = F.relu(x, inplace=True)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return F.relu(residual + x, inplace=True), s


class MLFN(ReIDBackbone):
    """Multi-Level Factorisation Net.

    Reference:
        Chang et al. Multi-Level Factorisation Net for
        Person Re-Identification. CVPR 2018.

    Public keys:
        - ``mlfn``: MLFN (Multi-Level Factorisation Net).
    """

    def __init__(
        self,
        num_classes,
        loss="softmax",
        groups=32,
        channels=[64, 256, 512, 1024, 2048],
        embed_dim=1024,
        **kwargs,
    ):
        super(MLFN, self).__init__()
        self.loss = loss
        self.groups = groups

        # first convolutional layer
        self.conv1 = nn.Conv2d(3, channels[0], 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # main body
        self.feature = nn.ModuleList(
            [
                # layer 1-3
                MLFNBlock(channels[0], channels[1], 1, [128, 64], self.groups),
                MLFNBlock(channels[1], channels[1], 1, [128, 64], self.groups),
                MLFNBlock(channels[1], channels[1], 1, [128, 64], self.groups),
                # layer 4-7
                MLFNBlock(channels[1], channels[2], 2, [256, 128], self.groups),
                MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups),
                MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups),
                MLFNBlock(channels[2], channels[2], 1, [256, 128], self.groups),
                # layer 8-13
                MLFNBlock(channels[2], channels[3], 2, [512, 128], self.groups),
                MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups),
                MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups),
                MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups),
                MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups),
                MLFNBlock(channels[3], channels[3], 1, [512, 128], self.groups),
                # layer 14-16
                MLFNBlock(channels[3], channels[4], 2, [512, 128], self.groups),
                MLFNBlock(channels[4], channels[4], 1, [512, 128], self.groups),
                MLFNBlock(channels[4], channels[4], 1, [512, 128], self.groups),
            ]
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # projection functions
        self.fc_x = nn.Sequential(
            nn.Conv2d(channels[4], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.fc_s = nn.Sequential(
            nn.Conv2d(self.groups * 16, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

        self.init_params()

    def init_params(self):
        init_kaiming_reid(self)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        s_hat = []
        for block in self.feature:
            x, s = block(x)
            s_hat.append(s)
        s_hat = torch.cat(s_hat, 1)

        x = self.global_avgpool(x)
        x = self.fc_x(x)
        s_hat = self.fc_s(s_hat)

        v = (x + s_hat) * 0.5
        return v.flatten(1)

    def forward_head(self, features):
        if not self.training:
            return features

        y = self.classifier(features)

        return format_reid_output(self.loss, y, features)


@register_backbone(
    "mlfn",
    family="legacy",
    default_recipe="legacy_reid",
    default_img_size=(256, 128),
    pretrained_source="imagenet",
)
def mlfn(num_classes, loss="softmax", pretrained=True, use_gpu=None, **kwargs):
    del use_gpu
    model = MLFN(num_classes, loss, **kwargs)
    if pretrained:
        warn_manual_pretrained_download(model_urls["imagenet"])
    return model
