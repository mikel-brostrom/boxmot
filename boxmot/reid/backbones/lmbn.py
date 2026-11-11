# BoxMOT AGPL-3.0 license

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import nn

from boxmot.reid.backbones.base import ReIDBackbone
from boxmot.reid.backbones.common.attention import BatchFeatureErase_Top
from boxmot.reid.backbones.heads.bnneck import BNNeck, BNNeck3
from boxmot.utils import logger as LOGGER


class LMBNFeatureMaps(NamedTuple):
    global_map: torch.Tensor
    partial_map: torch.Tensor
    channel_map: torch.Tensor


def _weights_init_kaiming(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Linear" in classname:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif "Conv" in classname:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif "BatchNorm" in classname and module.affine:
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class LMBNBackbone(ReIDBackbone):
    """Shared LightMBN branch/head implementation for OSNet-based variants."""

    def __init__(
        self,
        *,
        num_classes: int,
        loss: str,
        pretrained: bool,
        osnet_builder: Callable[..., nn.Module],
        drop_bottleneck_type: type[nn.Module],
        use_ain_pools: bool = False,
        feat_dim: int = 512,
        activation_map: bool = False,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.num_classes = int(num_classes)
        self.n_ch = 2
        self.chs = 512 // self.n_ch
        self.feature_dim = feat_dim * 7
        self.activation_map = activation_map

        osnet = osnet_builder(pretrained=bool(pretrained))

        stem_layers = [osnet.conv1, osnet.maxpool, osnet.conv2]
        if use_ain_pools:
            stem_layers.append(osnet.pool2)
        stem_layers.append(osnet.conv3[0])
        self.backbone = nn.Sequential(*stem_layers)

        branch_layers = [copy.deepcopy(osnet.conv3[1:])]
        if use_ain_pools:
            branch_layers.append(copy.deepcopy(osnet.pool3))
        branch_layers.extend([copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5)])

        self.global_branch = nn.Sequential(*copy.deepcopy(branch_layers))
        self.partial_branch = nn.Sequential(*copy.deepcopy(branch_layers))
        self.channel_branch = nn.Sequential(*copy.deepcopy(branch_layers))

        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 1))

        reduction = BNNeck3(512, self.num_classes, feat_dim, return_f=True)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)

        self.shared = nn.Sequential(
            nn.Conv2d(self.chs, feat_dim, 1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(True),
        )
        self.shared.apply(_weights_init_kaiming)

        self.reduction_ch_0 = BNNeck(feat_dim, self.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(feat_dim, self.num_classes, return_f=True)
        self.batch_drop_block = BatchFeatureErase_Top(512, drop_bottleneck_type)

    def forward_features(self, x: torch.Tensor) -> LMBNFeatureMaps:
        x = self.backbone(x)
        return LMBNFeatureMaps(
            global_map=self.global_branch(x),
            partial_map=self.partial_branch(x),
            channel_map=self.channel_branch(x),
        )

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x).global_map

    def forward_head(self, features: LMBNFeatureMaps):
        glo = features.global_map
        par = features.partial_map
        cha = features.channel_map

        glo_for_activation = glo
        glo_drop, glo = self.batch_drop_block(glo)

        if self.activation_map:
            _, _, h_par, _ = par.size()
            fmap_p0 = par[:, :, : h_par // 2, :]
            fmap_p1 = par[:, :, h_par // 2 :, :]
            fmap_c0 = cha[:, : self.chs, :, :]
            fmap_c1 = cha[:, self.chs :, :, :]
            LOGGER.debug("Generating activation maps...")
            return glo, glo_for_activation, fmap_c0, fmap_c1, fmap_p0, fmap_p1

        glo_drop = self.global_pooling(glo_drop)
        glo = self.channel_pooling(glo)
        g_par = self.global_pooling(par)
        p_par = self.partial_pooling(par)
        cha = self.channel_pooling(cha)

        p0 = p_par[:, :, 0:1, :]
        p1 = p_par[:, :, 1:2, :]

        f_glo = self.reduction_0(glo)
        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p0)
        f_p2 = self.reduction_3(p1)
        f_glo_drop = self.reduction_4(glo_drop)

        c0 = self.shared(cha[:, : self.chs, :, :])
        c1 = self.shared(cha[:, self.chs :, :, :])
        f_c0 = self.reduction_ch_0(c0)
        f_c1 = self.reduction_ch_1(c1)

        metric_features = [f_glo[-1], f_glo_drop[-1], f_p0[-1]]

        if not self.training:
            embeddings = torch.stack(
                [f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0]],
                dim=2,
            )
            return embeddings.flatten(1, 2)

        logits = [
            f_glo[1],
            f_glo_drop[1],
            f_p0[1],
            f_p1[1],
            f_p2[1],
            f_c0[1],
            f_c1[1],
        ]
        return logits, metric_features

    def forward(self, x: torch.Tensor, return_featuremaps: bool = False):
        features = self.forward_features(x)
        if return_featuremaps:
            return features.global_map
        return self.forward_head(features)
