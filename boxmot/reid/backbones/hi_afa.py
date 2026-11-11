# BoxMOT AGPL-3.0 license

"""Hierarchical Attentive Feature Aggregation (Hi-AFA) ReID backbone.

The implementation follows Dong and Lu, *Hierarchical Attentive Feature
Aggregation for Person Re-Identification*, IEEE Access 2024.  The paper's
stage-index equation is ambiguous at the branch diagonals, so the forward
graph below follows Figure 1: one C2 node, two C3 nodes, three C4 nodes and
four C5 nodes.  Every lower branch starts from the suppressed attentive
feature above it and later receives both its own feature and the suppressed
feature from the adjacent upper branch.

The v2 BoxMOT contract makes two explicit stability choices for details that
the paper does not make reproducible: LDAM residual gates are trainable and
zero-initialized so ImageNet features survive construction, and retrieval
normalizes each unique stream before concatenation instead of allowing raw
global-feature magnitudes to overwhelm the local descriptors.
"""

from __future__ import annotations

import copy
import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from boxmot.reid.backbones.base import ReIDBackbone
from boxmot.reid.backbones.families.osnet import osnet_x1_0
from boxmot.reid.backbones.heads.bnneck import BNNeck
from boxmot.reid.backbones.registry import register_backbone

__all__ = [
    "ChannelAttention",
    "DropBlock2d",
    "FeatureSuppression",
    "HiAFA",
    "HiAFAFeatureMaps",
    "HorizontalStripePool",
    "LightweightDualAttention",
    "SpatialAttention",
]


def _effective_groups(channels: int, hidden_channels: int, requested: int) -> int:
    """Use the paper's group count when possible and remain shape-safe otherwise."""
    return max(1, math.gcd(int(requested), math.gcd(int(channels), int(hidden_channels))))


class _GroupedPointwiseBottleneck(nn.Module):
    """Two grouped 1x1 convolutions with the paper's C -> C/r -> C shape."""

    def __init__(self, channels: int, reduction: int, groups: int) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0 or groups <= 0:
            raise ValueError("channels, reduction, and groups must be positive")
        hidden_channels = max(1, channels // reduction)
        effective_groups = _effective_groups(channels, hidden_channels, groups)
        self.groups = effective_groups
        self.reduce = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=1,
            groups=effective_groups,
            bias=True,
        )
        self.expand = nn.Conv2d(
            hidden_channels,
            channels,
            kernel_size=1,
            groups=effective_groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The paper specifies no activation between the grouped convolutions.
        return self.expand(self.reduce(x))


class FeatureSuppression(nn.Module):
    """Suppress spatial positions whose channel-mean response exceeds ``tau``."""

    def __init__(self, tau: float = 0.7, eps: float = 1e-12) -> None:
        super().__init__()
        if not 0.0 < tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {tau}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.tau = float(tau)
        self.eps = float(eps)

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FeatureSuppression expects BCHW input, got shape {tuple(x.shape)}")
        response = x.mean(dim=1, keepdim=True)
        flat = response.flatten(2)
        minimum = flat.amin(dim=2, keepdim=True).unsqueeze(-1)
        maximum = flat.amax(dim=2, keepdim=True).unsqueeze(-1)
        normalized = (response - minimum) / (maximum - minimum).clamp_min(self.eps)
        return (normalized <= self.tau).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask(x)


class SpatialAttention(nn.Module):
    """Resolution-specific grouped spatial attention used by LDAM."""

    def __init__(
        self,
        spatial_size: tuple[int, int],
        reduction: int = 8,
        groups: int = 8,
        gamma: float = 0.0,
    ) -> None:
        super().__init__()
        height, width = (int(value) for value in spatial_size)
        if height <= 0 or width <= 0:
            raise ValueError(f"spatial_size must be positive, got {spatial_size}")
        self.spatial_size = (height, width)
        self.spatial_channels = height * width
        self.transform = _GroupedPointwiseBottleneck(
            self.spatial_channels,
            reduction,
            groups,
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.spatial_size:
            raise ValueError(
                "SpatialAttention was built for "
                f"{self.spatial_size}, got {tuple(x.shape[-2:])}"
            )
        batch_size = x.shape[0]
        spatial_descriptor = x.sum(dim=1).reshape(batch_size, self.spatial_channels, 1, 1)
        logits = self.transform(spatial_descriptor).reshape(batch_size, 1, *self.spatial_size)
        return torch.softmax(logits.flatten(2), dim=2).reshape_as(logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * x * self.attention(x)


class ChannelAttention(nn.Module):
    """Grouped channel attention used by LDAM."""

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        groups: int = 8,
        gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.transform = _GroupedPointwiseBottleneck(self.channels, reduction, groups)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        descriptor = F.adaptive_avg_pool2d(x, 1) + F.adaptive_max_pool2d(x, 1)
        return torch.softmax(self.transform(descriptor), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * x * self.attention(x)


class LightweightDualAttention(nn.Module):
    """Paper-order SAM followed by CAM."""

    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        reduction: int = 8,
        groups: int = 8,
        gamma: float = 0.0,
    ) -> None:
        super().__init__()
        self.spatial = SpatialAttention(spatial_size, reduction, groups, gamma)
        self.channel = ChannelAttention(channels, reduction, groups, gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel(self.spatial(x))


class HorizontalStripePool(nn.Module):
    """Adaptive horizontal average pooling implemented with slice reductions.

    PyTorch's MPS adaptive-pooling kernel does not support non-divisible sizes
    such as the paper's 24 -> 5 partition.  These start/end rules are exactly
    those used by adaptive average pooling and work on every backend.
    """

    def __init__(self, parts: int) -> None:
        super().__init__()
        if parts <= 0:
            raise ValueError(f"parts must be positive, got {parts}")
        self.parts = int(parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"HorizontalStripePool expects BCHW input, got {tuple(x.shape)}")
        height = x.shape[2]
        if height < self.parts:
            raise ValueError(f"Cannot pool height {height} into {self.parts} non-empty stripes")
        stripes = []
        for index in range(self.parts):
            start = math.floor(index * height / self.parts)
            end = math.ceil((index + 1) * height / self.parts)
            stripes.append(x[:, :, start:end, :].mean(dim=(2, 3), keepdim=True))
        return torch.cat(stripes, dim=2)


class DropBlock2d(nn.Module):
    """Ghiasi et al. contiguous DropBlock for 2-D feature maps.

    Hi-AFA names DropBlock but does not report its probability or block size.
    The defaults below record the common ImageNet choices of drop probability
    0.1 and a 7x7 block.  Masks are sampled independently per sample/channel,
    expanded with max pooling, and rescaled by their realized keep rate.
    """

    def __init__(self, drop_prob: float = 0.1, block_size: int = 7) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        self.drop_prob = float(drop_prob)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        if x.ndim != 4:
            raise ValueError(f"DropBlock2d expects BCHW input, got {tuple(x.shape)}")
        height, width = x.shape[-2:]
        block_size = min(self.block_size, height, width)
        valid_height = height - block_size + 1
        valid_width = width - block_size + 1
        gamma = self.drop_prob * height * width
        gamma /= block_size**2 * valid_height * valid_width
        gamma = min(max(gamma, 0.0), 1.0)

        rows = torch.arange(height, device=x.device).view(1, 1, height, 1)
        cols = torch.arange(width, device=x.device).view(1, 1, 1, width)
        left = (block_size - 1) // 2
        right = block_size // 2
        valid_centers = (
            (rows >= left)
            & (rows < height - right)
            & (cols >= left)
            & (cols < width - right)
        )
        seeds = (torch.rand_like(x) < gamma) & valid_centers
        dropped = F.max_pool2d(
            seeds.to(dtype=x.dtype),
            kernel_size=block_size,
            stride=1,
            padding=block_size // 2,
        )
        dropped = dropped[:, :, :height, :width]
        keep = 1.0 - dropped
        scale = keep.numel() / keep.sum().clamp_min(1.0)
        return x * keep * scale


class HiAFAFeatureMaps(NamedTuple):
    """Terminal maps from the four upper-triangular branches."""

    branch1: torch.Tensor
    branch2: torch.Tensor
    branch3: torch.Tensor
    branch4: torch.Tensor


@register_backbone(
    "hi_afa",
    family="cnn",
    default_recipe="cnn_reid",
    default_img_size=(384, 128),
    accepts_model_kwargs=True,
    pretrained_source="imagenet",
)
class HiAFA(ReIDBackbone):
    """OSNet-x1.0 Hi-AFA with stabilized transfer and retrieval contracts."""

    num_identity_streams = 17
    num_inference_streams = 16
    num_ranking_streams = 5
    num_center_streams = 22
    stream_dim = 512

    def __init__(
        self,
        num_classes: int,
        loss: str = "ms",
        pretrained: bool = False,
        use_gpu=None,
        img_size: tuple[int, int] = (384, 128),
        feat_dim: int = 512,
        attention_reduction: int = 8,
        attention_groups: int = 8,
        attention_gamma: float = 0.0,
        suppression_tau: float = 0.7,
        dropblock_prob: float = 0.1,
        dropblock_size: int = 7,
        **kwargs,
    ) -> None:
        super().__init__()
        del use_gpu, kwargs
        if int(feat_dim) != self.stream_dim:
            raise ValueError(f"Hi-AFA uses fixed 512-D streams, got feat_dim={feat_dim}")
        if len(img_size) != 2:
            raise ValueError(f"img_size must be (height, width), got {img_size}")
        self.img_size = tuple(int(value) for value in img_size)
        if any(value <= 0 or value % 16 for value in self.img_size):
            raise ValueError(f"Hi-AFA img_size values must be positive multiples of 16, got {self.img_size}")
        self.num_classes = int(num_classes)
        self.loss = str(loss).lower()
        self.feature_dim = self.num_inference_streams * self.stream_dim

        # These attributes let the generic trainer reproduce Eq. (10) without
        # changing the output contract of any other backbone.
        self.metric_loss_kwargs = {
            "ms": {
                "alpha": 2.0,
                "beta": 40.0,
                "thresh": 0.5,
                "mining_margin": 0.1,
            },
        }
        self.allow_center_with_ms = True

        osnet = osnet_x1_0(pretrained=bool(pretrained))
        self.stem = nn.Sequential(osnet.conv1, osnet.maxpool)
        self.stage2 = osnet.conv2
        self.stage3 = nn.ModuleList(copy.deepcopy(osnet.conv3) for _ in range(2))
        self.stage4 = nn.ModuleList(copy.deepcopy(osnet.conv4) for _ in range(3))
        self.stage5 = nn.ModuleList(copy.deepcopy(osnet.conv5) for _ in range(4))

        height, width = self.img_size
        stage2_size = (height // 8, width // 8)
        stage3_size = (height // 16, width // 16)
        self.attention2 = LightweightDualAttention(
            256,
            stage2_size,
            attention_reduction,
            attention_groups,
            attention_gamma,
        )
        self.attention3 = nn.ModuleList(
            LightweightDualAttention(
                384,
                stage3_size,
                attention_reduction,
                attention_groups,
                attention_gamma,
            )
            for _ in range(2)
        )
        self.attention4 = nn.ModuleList(
            LightweightDualAttention(
                512,
                stage3_size,
                attention_reduction,
                attention_groups,
                attention_gamma,
            )
            for _ in range(3)
        )
        self.suppression = FeatureSuppression(suppression_tau)

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.part_pools = nn.ModuleList(HorizontalStripePool(parts) for parts in (5, 3, 2))
        self.dropblock = DropBlock2d(dropblock_prob, dropblock_size)
        self.channel_projections = nn.ModuleList(
            nn.Conv2d(256, self.stream_dim, kernel_size=1, bias=False) for _ in range(2)
        )
        for projection in self.channel_projections:
            nn.init.kaiming_normal_(projection.weight, mode="fan_out")
        self.bn_necks = nn.ModuleList(
            BNNeck(self.stream_dim, self.num_classes, return_f=True)
            for _ in range(self.num_identity_streams)
        )
        self.reproduction_contract = {
            "implementation_version": 2,
            "paper": "Dong and Lu, IEEE Access 2024",
            "model_kwargs": {
                "img_size": list(self.img_size),
                "feat_dim": self.stream_dim,
                "attention_reduction": int(attention_reduction),
                "attention_groups": int(attention_groups),
                "attention_gamma": float(attention_gamma),
                "suppression_tau": float(suppression_tau),
                "dropblock_prob": float(dropblock_prob),
                "dropblock_size": int(dropblock_size),
            },
            "architecture": {
                "backbone": "osnet_x1_0",
                "img_size": list(self.img_size),
                "branch_stage_counts": [1, 2, 3, 4],
                "attention": {
                    "reduction": int(attention_reduction),
                    "groups": int(attention_groups),
                    "gamma_init": float(attention_gamma),
                    "gamma_trainable": True,
                    "spatial_residual": "elementwise_feature_attention",
                },
                "suppression_tau": float(suppression_tau),
                "dropblock": {
                    "probability": float(dropblock_prob),
                    "block_size": int(dropblock_size),
                },
            },
            "objective": {
                "identity_streams": self.num_identity_streams,
                "ranking_streams": self.num_ranking_streams,
                "center_streams": self.num_center_streams,
                "branch_reduction": "sum",
                "multi_similarity": dict(self.metric_loss_kwargs["ms"]),
                "center_loss_internal_scale": self.num_center_streams / 2.0,
            },
            "inference": {
                "descriptor": "balanced_unique_raw_pooled_streams",
                "streams": self.num_inference_streams,
                "dimension": self.feature_dim,
                "normalization": "per_stream_l2_then_concat_final_l2",
                "training_only_streams": ["dropped_global"],
            },
            "unreported_assumptions": {
                "attention_gamma_init": float(attention_gamma),
                "dropblock_probability": float(dropblock_prob),
                "dropblock_size": int(dropblock_size),
                "multi_similarity_mining_margin": 0.1,
                "center_table": "shared_across_streams",
                "center_optimizer": {"name": "sgd", "lr": 0.5},
            },
        }

    def forward_features(self, x: torch.Tensor) -> HiAFAFeatureMaps:
        if x.shape[-2:] != self.img_size:
            raise ValueError(f"Hi-AFA expects input size {self.img_size}, got {tuple(x.shape[-2:])}")

        stem = self.stem(x)
        branch1_c2 = self.attention2(self.stage2(stem))

        branch1_c3 = self.attention3[0](self.stage3[0](branch1_c2))
        branch2_c3 = self.attention3[1](self.stage3[1](self.suppression(branch1_c2)))

        branch1_c4 = self.attention4[0](self.stage4[0](branch1_c3))
        branch2_c4 = self.attention4[1](
            self.stage4[1](branch2_c3 + self.suppression(branch1_c3))
        )
        branch3_c4 = self.attention4[2](self.stage4[2](self.suppression(branch2_c3)))

        branch1_c5 = self.stage5[0](branch1_c4)
        branch2_c5 = self.stage5[1](branch2_c4 + self.suppression(branch1_c4))
        branch3_c5 = self.stage5[2](branch3_c4 + self.suppression(branch2_c4))
        branch4_c5 = self.stage5[3](self.suppression(branch3_c4))
        return HiAFAFeatureMaps(branch1_c5, branch2_c5, branch3_c5, branch4_c5)

    @staticmethod
    def _flatten_pooled(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)

    def _raw_pooled_streams(
        self,
        features: HiAFAFeatureMaps,
    ) -> list[torch.Tensor]:
        branches = list(features)
        globals_ = [self._flatten_pooled(self.global_max_pool(branch)) for branch in branches]

        parts: list[torch.Tensor] = []
        for branch, pool in zip(branches[:3], self.part_pools, strict=True):
            pooled = pool(branch).squeeze(-1)
            parts.extend(pooled[:, :, index] for index in range(pooled.shape[2]))

        channel_pool = self.global_avg_pool(features.branch4)
        channel_halves = channel_pool.chunk(2, dim=1)
        channels = [
            self._flatten_pooled(projection(half))
            for projection, half in zip(self.channel_projections, channel_halves, strict=True)
        ]

        raw_streams = [*globals_, *parts, *channels]
        if len(raw_streams) != self.num_inference_streams:
            raise RuntimeError(f"Expected 16 unique raw streams, built {len(raw_streams)}")
        return raw_streams

    def _pooled_streams(
        self,
        features: HiAFAFeatureMaps,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        raw_streams = self._raw_pooled_streams(features)
        globals_ = raw_streams[:4]
        dropped_global = self._flatten_pooled(self.global_max_pool(self.dropblock(features.branch4)))

        ranking_streams = [dropped_global, *globals_]
        identity_streams = [dropped_global, *raw_streams]
        if len(identity_streams) != self.num_identity_streams:
            raise RuntimeError(f"Expected 17 identity streams, built {len(identity_streams)}")
        return identity_streams, ranking_streams

    def forward_head(self, features: HiAFAFeatureMaps):
        if not self.training:
            raw_streams = self._raw_pooled_streams(features)
            balanced = [F.normalize(stream, p=2, dim=1) for stream in raw_streams]
            return F.normalize(torch.cat(balanced, dim=1), p=2, dim=1)

        identity_streams, ranking_streams = self._pooled_streams(features)

        neck_outputs = [neck(stream) for neck, stream in zip(self.bn_necks, identity_streams, strict=True)]
        embeddings = [output[0] for output in neck_outputs]
        logits = [output[1] for output in neck_outputs]
        if self.loss == "softmax":
            return logits

        packet = {
            "global": ranking_streams[0],
            "raw_mean": torch.stack(ranking_streams, dim=0).mean(dim=0),
            "_metric_features": tuple(ranking_streams),
            "_center_features": tuple([*embeddings, *ranking_streams]),
            "_classification_loss_aggregation": "sum",
            "_metric_loss_aggregation": "sum",
            # Existing CenterLoss omits the paper's 1/2 factor.  Averaging the
            # concatenated streams and multiplying by 22/2 reproduces Eq. (10).
            "_center_loss_scale": self.num_center_streams / 2.0,
        }
        return logits, packet

    def forward(self, x: torch.Tensor, return_featuremaps: bool = False):
        features = self.forward_features(x)
        if return_featuremaps:
            return features.branch1
        return self.forward_head(features)
