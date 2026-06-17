# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
CSL-TinyViT: Cross-Scale Learning TinyViT for Person Re-Identification.

A hybrid CNN-Transformer architecture inspired by:
- TinyViT (Wu et al.): efficient multi-stage hybrid backbone with
  MBConv (early stages) and windowed self-attention (later stages).
- CSL (Cross-Scale Learning): self-supervised pretraining that learns
  scale-invariant and instance-discriminative features.

Architecture overview:
  PatchEmbed (stride-4 conv stem)
  → Stage 0: ConvLayer (MBConv blocks)          → 64ch
  → Stage 1: BasicLayer (windowed attention)     → 128ch
  → Stage 2: BasicLayer (windowed attention) ×6  → variant channels
  → Stage 3: BasicLayer (windowed attention)     → variant channels
  → Neck (1×1 + LN + 3×3 + LN → 512ch)
  → Multi-Branch Head (global + 2× partial)      → 3×512 = 1536-d

The model family follows TinyViT-5M/11M/21M width scaling with fixed
depths=[2, 2, 6, 2]. Input: 384×128 (H×W, same as LMBN/CSPReID).

Follows the BoxMOT ReID backbone contract:
  * training + softmax  → class logits list
  * training + triplet  → (logits list, embedding)
  * inference           → concatenated feature embedding
"""

from __future__ import annotations

import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from boxmot.reid.backbones.lmbn.bnneck import BNNeck3
from boxmot.utils import logger as LOGGER

# ---------------------------------------------------------------------------
# Utilities (no timm dependency)
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """Stochastic depth (per-sample drop of entire residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep)
        return x.div(keep) * mask


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Conv2d_BN(nn.Sequential):
    """Conv2d + BatchNorm2d (fused at deployment)."""

    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module("c", nn.Conv2d(
            in_ch, out_ch, ks, stride, pad, dilation, groups, bias=False))
        bn = nn.BatchNorm2d(out_ch)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)


class PatchEmbed(nn.Module):
    """Stride-4 convolutional patch embedding."""

    def __init__(self, in_chans, embed_dim, img_size, activation):
        super().__init__()
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def forward(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv block."""

    def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()
        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()
        self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    """Downsampling layer between stages."""

    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        # No spatial downsample for last two stages (320, 448, 576 dims)
        stride_c = 1 if out_dim in (320, 448, 576) else 2
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def forward(self, x, hw_size):
        if x.ndim == 3:
            H, W = hw_size
            B = x.shape[0]
            x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        out_size = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


class ConvLayer(nn.Module):
    """Convolutional stage (MBConv blocks)."""

    def __init__(self, dim, input_resolution, depth, activation,
                 drop_path=0.0, downsample=None, use_checkpoint=False,
                 out_dim=None, conv_expand_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x, out_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        if self.downsample is not None:
            x, out_size = self.downsample(x, out_size)
        return x, out_size


# ---------------------------------------------------------------------------
# Windowed Self-Attention blocks
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head attention with learned attention biases."""

    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=(14, 14)):
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer("attention_bias_idxs",
                             torch.LongTensor(idxs).view(N, N), persistent=False)

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode:
            if hasattr(self, "ab"):
                del self.ab
        else:
            self.register_buffer("ab",
                                 self.attention_biases[:, self.attention_bias_idxs],
                                 persistent=False)

    def forward(self, x):
        B, N, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn + bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTMlp(nn.Module):
    """MLP with pre-norm for TinyViT blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TinyViTBlock(nn.Module):
    """TinyViT block: windowed attention + local depthwise conv + MLP."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 local_conv_size=3, activation=nn.GELU):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        head_dim = dim // num_heads
        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TinyViTMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x, hw_size):
        B, L, C = x.shape
        H, W = hw_size
        assert L == H * W

        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # Window partition
            x = x.view(B, nH, self.window_size, nW, self.window_size, C)
            x = x.transpose(2, 3).reshape(B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # Window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size, C)
            x = x.transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()
            x = x.view(B, L, C)

        x = res_x + self.drop_path(x)

        # Local depthwise convolution
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.view(B, C, L).transpose(1, 2)

        # MLP
        x = x + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):
    """A stage of TinyViT blocks (windowed attention)."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0, downsample=None,
                 use_checkpoint=False, local_conv_size=3, activation=nn.GELU,
                 out_dim=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            TinyViTBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                local_conv_size=local_conv_size, activation=activation)
            for i in range(depth)
        ])

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x, out_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, out_size, use_reentrant=False)
            else:
                x = blk(x, out_size)
        if self.downsample is not None:
            x, out_size = self.downsample(x, out_size)
        return x, out_size


class LayerNorm2d(nn.Module):
    """LayerNorm for channel-first (B, C, H, W) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ---------------------------------------------------------------------------
# Multi-Branch ReID Head
# ---------------------------------------------------------------------------


class GeM(nn.Module):
    """Generalized mean pooling with optional spatial output size."""

    def __init__(self, output_size: tuple[int, int], p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.output_size = output_size
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=self.eps)
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, self.output_size)
        return x.pow(1.0 / p)


class SpatialTopDrop(nn.Module):
    """Drop top-activation rows in a feature map during training."""

    def __init__(self, h_ratio: float = 0.33):
        super().__init__()
        self.h_ratio = h_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        b, c, h, w = x.size()
        rh = max(1, min(h, round(self.h_ratio * h)))
        act = (x**2).sum(1)
        max_act, _ = act.max(2)
        top_rows = torch.argsort(max_act, dim=1)[:, -rh:]
        mask = x.new_ones((b, h))
        for i in range(b):
            mask[i, top_rows[i]] = 0
        mask = mask.unsqueeze(1).unsqueeze(-1).expand(-1, c, -1, w)
        return x * mask


class MultiBranchHead(nn.Module):
    """Multi-granularity feature head with configurable horizontal stripes.

    Produces:
      - Training: (cls_scores_list, features_tensor)
      - Inference: (B, feat_dim × num_branches) concatenated features
    """

    def __init__(
        self,
        in_ch,
        feat_dim,
        num_classes,
        metric_feature: str = "raw_mean",
        inference_feature: str = "concat_bn",
        head_pool: str = "avg",
        branch_metric: bool = False,
        head_parts: tuple[int, ...] = (1, 2),
    ):
        super().__init__()
        self.metric_feature = metric_feature
        self.inference_feature = inference_feature
        self.branch_metric = branch_metric
        self.head_parts = self._normalize_head_parts(head_parts)
        self.branch_specs = self._build_branch_specs(self.head_parts)
        self.part_keys = [key for key, granularity, _ in self.branch_specs if granularity > 1]
        self.set_pooling(head_pool)

        for key, _, _ in self.branch_specs:
            setattr(self, self._bn_attr(key), BNNeck3(in_ch, num_classes, feat_dim, return_f=True))

    @staticmethod
    def _normalize_head_parts(head_parts) -> tuple[int, ...]:
        if isinstance(head_parts, str):
            values = [part for part in head_parts.replace(";", ",").split(",") if part.strip()]
        elif isinstance(head_parts, int):
            values = [head_parts]
        else:
            values = list(head_parts or (1, 2))
        normalized = tuple(dict.fromkeys(int(part) for part in values))
        if not normalized:
            raise ValueError("CSL-TinyViT head_parts must not be empty")
        if any(part < 1 for part in normalized):
            raise ValueError(f"CSL-TinyViT head_parts must be positive, got {normalized}")
        if 1 not in normalized:
            raise ValueError(f"CSL-TinyViT head_parts must include 1 for the global branch, got {normalized}")
        return normalized

    @staticmethod
    def _build_branch_specs(head_parts: tuple[int, ...]) -> list[tuple[str, int, int]]:
        specs = [("global", 1, 0)]
        part_index = 0
        for granularity in head_parts:
            if granularity == 1:
                continue
            for stripe_index in range(granularity):
                specs.append((f"part{part_index}", granularity, stripe_index))
                part_index += 1
        return specs

    @staticmethod
    def _bn_attr(key: str) -> str:
        return "bn_global" if key == "global" else f"bn_{key}"

    @staticmethod
    def _pool_attr(granularity: int) -> str:
        if granularity == 1:
            return "global_pool"
        if granularity == 2:
            return "partial_pool"
        return f"part_pool_{granularity}"

    @staticmethod
    def _make_pool(head_pool: str, output_size: tuple[int, int]) -> nn.Module:
        if head_pool == "avg":
            return nn.AdaptiveAvgPool2d(output_size)
        if head_pool == "gem":
            return GeM(output_size)
        raise ValueError(f"Unsupported CSL-TinyViT head_pool: {head_pool}")

    def set_pooling(self, head_pool: str) -> None:
        head_pool = str(head_pool).lower()
        for granularity in self.head_parts:
            setattr(
                self,
                self._pool_attr(granularity),
                self._make_pool(head_pool, (granularity, 1)),
            )
        self.head_pool = head_pool

    def set_branch_metric(self, branch_metric: bool) -> None:
        self.branch_metric = bool(branch_metric)

    def forward(self, x):
        # x: (B, C, H, W)
        pooled_by_granularity = {
            granularity: getattr(self, self._pool_attr(granularity))(x)
            for granularity in self.head_parts
        }

        branch_outputs = {}
        bn_features_list = []
        raw_features_list = []
        cls_scores = []
        raw_features = {}
        for key, granularity, stripe_index in self.branch_specs:
            pooled = pooled_by_granularity[granularity]
            if granularity > 1:
                pooled = pooled[:, :, stripe_index:stripe_index + 1, :]
            branch_output = getattr(self, self._bn_attr(key))(pooled)
            branch_outputs[key] = branch_output
            bn_features_list.append(branch_output[0])
            cls_scores.append(branch_output[1])
            raw_features_list.append(branch_output[2])
            raw_features[key] = branch_output[2]

        bn_features = torch.stack(bn_features_list, dim=2).flatten(1, 2)
        raw_features["raw_mean"] = torch.stack(raw_features_list, dim=0).mean(dim=0)
        raw_features["concat_bn"] = bn_features

        if not self.training:
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            raise ValueError(f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}")

        if self.branch_metric:
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        else:
            feats = raw_features["raw_mean"]
        return cls_scores, feats


class LMBNStyleMultiBranchHead(MultiBranchHead):
    """LMBN-style head with drop-global and channel split branches."""

    def __init__(
        self,
        in_ch,
        feat_dim,
        num_classes,
        metric_feature: str = "raw_mean",
        inference_feature: str = "concat_bn",
        head_pool: str = "avg",
        branch_metric: bool = False,
        head_parts: tuple[int, ...] = (1, 2),
        drop_h_ratio: float = 0.33,
    ):
        super().__init__(
            in_ch=in_ch,
            feat_dim=feat_dim,
            num_classes=num_classes,
            metric_feature=metric_feature,
            inference_feature=inference_feature,
            head_pool=head_pool,
            branch_metric=branch_metric,
            head_parts=head_parts,
        )
        if in_ch % 2 != 0:
            raise ValueError(f"LMBN-style channel split requires even channels, got {in_ch}")
        self.drop_global = SpatialTopDrop(h_ratio=drop_h_ratio)
        self.channel_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_shared = nn.Sequential(
            nn.Conv2d(in_ch // 2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.bn_drop_global = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_part_global = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_ch0 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)
        self.bn_ch1 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)

    def forward(self, x):
        pooled_by_granularity = {
            granularity: getattr(self, self._pool_attr(granularity))(x)
            for granularity in self.head_parts
        }
        branch_outputs = {"global": getattr(self, self._bn_attr("global"))(pooled_by_granularity[1])}
        dropped = self.drop_global(x)
        branch_outputs["drop_global"] = self.bn_drop_global(
            getattr(self, self._pool_attr(1))(dropped)
        )
        branch_outputs["part_global"] = self.bn_part_global(pooled_by_granularity[1])

        for key, granularity, stripe_index in self.branch_specs:
            if key == "global" or granularity <= 1:
                continue
            pooled = pooled_by_granularity[granularity][:, :, stripe_index : stripe_index + 1, :]
            branch_outputs[key] = getattr(self, self._bn_attr(key))(pooled)

        pooled_channel = self.channel_pool(x)
        channel_0, channel_1 = torch.chunk(pooled_channel, chunks=2, dim=1)
        channel_0 = self.channel_shared(channel_0)
        channel_1 = self.channel_shared(channel_1)
        branch_outputs["ch0"] = self.bn_ch0(channel_0)
        branch_outputs["ch1"] = self.bn_ch1(channel_1)

        ordered_keys = ["global", "drop_global", "part_global", *self.part_keys, "ch0", "ch1"]
        bn_features_list = [branch_outputs[key][0] for key in ordered_keys]
        cls_scores = [branch_outputs[key][1] for key in ordered_keys]
        raw_features_list = [branch_outputs[key][2] for key in ordered_keys]

        bn_features = torch.stack(bn_features_list, dim=2).flatten(1, 2)
        raw_features = {
            key: branch_outputs[key][2]
            for key in ordered_keys
        }
        raw_features["raw_mean"] = torch.stack(raw_features_list, dim=0).mean(dim=0)
        raw_features["concat_bn"] = bn_features

        if not self.training:
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            raise ValueError(f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}")

        if self.branch_metric:
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        else:
            feats = [
                raw_features["global"],
                raw_features["drop_global"],
                raw_features["part_global"],
            ]
        return cls_scores, feats


class CSLTinyViTFeatureFusion(nn.Module):
    """Swappable spatial feature fusion module for CSL-TinyViT stage outputs."""

    _VALID_MODES = {"final", "last2", "last3", "weighted_last2", "weighted_last3"}
    _VALID_FUSION_TYPES = {"final", "residual", "weighted"}

    def __init__(
        self,
        fusion_type: str,
        stage_indices: tuple[int, ...],
        path_channels: dict[int, int],
        out_channels: int,
    ):
        super().__init__()
        self.fusion_type = str(fusion_type).lower()
        if self.fusion_type not in self._VALID_FUSION_TYPES:
            raise ValueError(f"Unsupported CSL-TinyViT feature fusion type: {fusion_type}")
        self.mode = self.fusion_type
        self.stage_indices = tuple(stage_indices)
        if self.fusion_type == "final" and self.stage_indices:
            raise ValueError("CSL-TinyViT final feature fusion must not define path stages")
        self.weighted = self.fusion_type == "weighted"

        missing = [index for index in self.stage_indices if index not in path_channels]
        if missing:
            raise ValueError(f"Missing CSL-TinyViT fusion path channels for stages: {missing}")

        self.projections = nn.ModuleDict(
            {
                str(index): nn.Sequential(
                    nn.Conv2d(path_channels[index], out_channels, kernel_size=1, bias=False),
                    LayerNorm2d(out_channels),
                )
                for index in self.stage_indices
            }
        )
        self.residual_scales = nn.ParameterDict(
            {
                str(index): nn.Parameter(torch.zeros(()))
                for index in (() if self.weighted else self.stage_indices)
            }
        )
        if self.weighted:
            self.fusion_weights = nn.Parameter(torch.tensor([1.0, *([1e-3] * len(self.stage_indices))]))
        else:
            self.register_parameter("fusion_weights", None)

    @classmethod
    def from_mode(
        cls,
        mode: str,
        path_channels: dict[int, int],
        out_channels: int,
    ) -> CSLTinyViTFeatureFusion:
        normalized_mode = cls.normalize_mode(mode)
        module = cls(
            fusion_type=cls.fusion_type_for_mode(normalized_mode),
            stage_indices=cls.stage_indices_for_mode(normalized_mode),
            path_channels=path_channels,
            out_channels=out_channels,
        )
        module.mode = normalized_mode
        return module

    @classmethod
    def normalize_mode(cls, mode: str) -> str:
        mode = str(mode).lower()
        if mode not in cls._VALID_MODES:
            raise ValueError(f"Unsupported CSL-TinyViT feature_fusion: {mode}")
        return mode

    @staticmethod
    def fusion_type_for_mode(mode: str) -> str:
        if mode == "final":
            return "final"
        if mode.startswith("weighted_"):
            return "weighted"
        return "residual"

    @staticmethod
    def stage_indices_for_mode(mode: str) -> tuple[int, ...]:
        if mode in {"last2", "weighted_last2"}:
            return (2,)
        if mode in {"last3", "weighted_last3"}:
            return (1, 2)
        return ()

    def normalized_weights(self) -> torch.Tensor:
        if self.fusion_weights is None:
            raise RuntimeError("Normalized fusion weights are only available for weighted feature fusion")
        weights = F.relu(self.fusion_weights)
        return weights / (weights.sum() + 1e-4)

    def _project_path(self, stage_index: int, feature: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        feature = self.projections[str(stage_index)](feature)
        if feature.shape[-2:] != output_size:
            feature = F.interpolate(feature, size=output_size, mode="bilinear", align_corners=False)
        return feature

    def forward(self, final_feature: torch.Tensor, path_features: dict[int, torch.Tensor]) -> torch.Tensor:
        if not self.stage_indices:
            return final_feature

        output_size = final_feature.shape[-2:]
        fused = final_feature
        weighted_features = [final_feature]
        for stage_index in self.stage_indices:
            path_feature = self._project_path(stage_index, path_features[stage_index], output_size)
            if self.weighted:
                weighted_features.append(path_feature)
            else:
                fused = fused + self.residual_scales[str(stage_index)] * path_feature

        if not self.weighted:
            return fused

        fusion_weights = self.normalized_weights()
        fused = fusion_weights[0] * weighted_features[0]
        for weight, feature in zip(fusion_weights[1:], weighted_features[1:], strict=True):
            fused = fused + weight * feature
        return fused


# ---------------------------------------------------------------------------
# CSL-TinyViT Backbone
# ---------------------------------------------------------------------------


class CSLTinyViT(nn.Module):
    """CSL-TinyViT: hybrid CNN-Transformer ReID backbone.

    Combines efficient MBConv early stages with windowed self-attention
    later stages, producing multi-granularity features via a multi-branch head.

    Input: 3×384×128 (H×W)
    Output:
      - Inference: num_branches × feat_dim feature vector
      - Training: (cls_scores_per_branch, features)
    """

    def __init__(
        self,
        num_classes: int,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: bool = True,
        *,
        img_size: tuple[int, int] = (384, 128),
        in_chans: int = 3,
        embed_dims: list[int] = None,
        depths: list[int] = None,
        num_heads: list[int] = None,
        window_sizes: list[int] = None,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        feat_dim: int = 512,
        neck_dim: int = 512,
        inference_feature: str = "concat_bn",
        feature_fusion: str = "final",
        head_pool: str = "avg",
        head_parts: tuple[int, ...] = (1, 2),
        branch_metric: bool = False,
        lmbn_style_head: bool = False,
        drop_h_ratio: float = 0.33,
    ):
        super().__init__()
        if embed_dims is None:
            embed_dims = [64, 128, 160, 320]
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [2, 4, 5, 10]
        if window_sizes is None:
            window_sizes = [7, 7, 14, 7]

        self.loss = loss
        self.img_size = img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.feature_fusion = CSLTinyViTFeatureFusion.normalize_mode(feature_fusion)

        activation = nn.GELU

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0],
            img_size=img_size, activation=activation)
        patches_resolution = self.patch_embed.patches_resolution

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build stages
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=(
                    patches_resolution[0] // (2 ** (i_layer if i_layer < 2 else 2)),
                    patches_resolution[1] // (2 ** (i_layer if i_layer < 2 else 2)),
                ),
                depth=depths[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=False,
                out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                activation=activation,
            )
            if i_layer == 0:
                layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs,
                )
            self.layers.append(layer)

        # Alias for trainer ViT detection (checks for model.blocks + model.patch_embed)
        self.blocks = self.layers

        # Feature neck: project to consistent dim
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dims[-1], neck_dim, kernel_size=1, bias=False),
            LayerNorm2d(neck_dim),
            nn.Conv2d(neck_dim, neck_dim, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(neck_dim),
        )
        fusion_stage_indices = CSLTinyViTFeatureFusion.stage_indices_for_mode(self.feature_fusion)
        fusion_path_channels = {
            index: embed_dims[min(index + 1, len(embed_dims) - 1)]
            for index in fusion_stage_indices
        }
        self.feature_fusion_module = CSLTinyViTFeatureFusion.from_mode(
            mode=self.feature_fusion,
            path_channels=fusion_path_channels,
            out_channels=neck_dim,
        )
        self._fusion_stage_indices = self.feature_fusion_module.stage_indices

        # Multi-branch ReID head.
        # For standard CSL-TinyViT, MS loss trains on the same concatenated BN
        # embedding used at inference. For LMBN-style heads, keep LightMBN-like
        # metric supervision on the three raw branch features
        # (global/drop-global/part-global) regardless of loss type.
        metric_feature = "concat_bn" if loss == "ms" else "raw_mean"
        if lmbn_style_head:
            metric_feature = "raw_mean"
        if lmbn_style_head:
            self.head = LMBNStyleMultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                branch_metric=branch_metric,
                drop_h_ratio=drop_h_ratio,
            )
        else:
            self.head = MultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                branch_metric=branch_metric,
            )

        # Initialize weights
        self.apply(self._init_weights)

    @property
    def fusion_scales(self) -> nn.ParameterDict:
        return self.feature_fusion_module.residual_scales

    @property
    def fusion_weights(self) -> nn.Parameter | None:
        return self.feature_fusion_module.fusion_weights

    def _normalized_fusion_weights(self) -> torch.Tensor:
        return self.feature_fusion_module.normalized_weights()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        old_to_new_prefixes = {
            "fusion_projections.": "feature_fusion_module.projections.",
            "fusion_scales.": "feature_fusion_module.residual_scales.",
        }
        for old_prefix, new_prefix in old_to_new_prefixes.items():
            old_full_prefix = f"{prefix}{old_prefix}"
            for key in list(state_dict.keys()):
                if key.startswith(old_full_prefix):
                    new_key = f"{prefix}{new_prefix}{key[len(old_full_prefix):]}"
                    state_dict.setdefault(new_key, state_dict[key])
                    del state_dict[key]

        old_weight_key = f"{prefix}fusion_weights"
        new_weight_key = f"{prefix}feature_fusion_module.fusion_weights"
        if old_weight_key in state_dict:
            state_dict.setdefault(new_weight_key, state_dict[old_weight_key])
            del state_dict[old_weight_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """Extract spatial feature map from backbone."""
        x = self.patch_embed(x)
        out_size = (x.shape[2], x.shape[3])
        fusion_features: dict[int, tuple[torch.Tensor, tuple[int, int]]] = {}

        # Stage 0 (conv layer operates on 4D tensor)
        x, out_size = self.layers[0](x, out_size)

        # Stages 1+ (attention layers operate on 3D tokens)
        for i in range(1, len(self.layers)):
            x, out_size = self.layers[i](x, out_size)
            if i in self._fusion_stage_indices:
                fusion_features[i] = (x, out_size)

        # Reshape back to spatial for neck
        B, _, C = x.size()
        x = x.view(B, out_size[0], out_size[1], C).permute(0, 3, 1, 2)
        x = self.neck(x)
        path_features: dict[int, torch.Tensor] = {}
        for index in self._fusion_stage_indices:
            stage_tokens, stage_size = fusion_features[index]
            stage = stage_tokens.view(B, stage_size[0], stage_size[1], -1)
            path_features[index] = stage.permute(0, 3, 1, 2)
        return self.feature_fusion_module(x, path_features)

    def forward(self, x):
        x = self.forward_features(x)  # (B, neck_dim, H, W)
        return self.head(x)


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

# TinyViT-5M (ImageNet-1k, distilled from 22k): embed_dims=[64,128,160,320]
_TINYVIT_5M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/"
    "tiny_vit_5m_22kto1k_distill.pth"
)

# TinyViT-11M (ImageNet-1k, distilled from 22k): embed_dims=[64,128,256,448]
_TINYVIT_11M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/"
    "tiny_vit_11m_22kto1k_distill.pth"
)

# TinyViT-21M (ImageNet-1k, distilled from 22k): embed_dims=[96,192,384,576]
_TINYVIT_21M_URL = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/"
    "tiny_vit_21m_22kto1k_distill.pth"
)


def _download_pretrained(url: str) -> dict:
    """Download pretrained weights to the torch hub cache."""
    cache_dir = os.path.join(
        os.path.expanduser(os.getenv("TORCH_HOME", os.path.join(
            os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))),
        "checkpoints",
    )
    os.makedirs(cache_dir, exist_ok=True)
    filename = url.rsplit("/", 1)[-1]
    cached = os.path.join(cache_dir, filename)
    if not os.path.exists(cached):
        LOGGER.info(f"Downloading pretrained weights from {url}")
        torch.hub.download_url_to_file(url, cached, progress=True)
    return torch.load(cached, map_location="cpu", weights_only=False)


def _load_pretrained_tinyvit(model: CSLTinyViT, url: str) -> None:
    """Load TinyViT pretrained weights with partial key matching.

    Loads backbone layers (patch_embed, layers, neck) from the ImageNet
    checkpoint. Skips head/classifier and any keys with shape mismatches
    (e.g. attention biases that depend on input resolution).
    """
    ckpt = _download_pretrained(url)
    state_dict = ckpt.get("model", ckpt)
    model_dict = model.state_dict()

    matched, skipped = [], []

    for ckpt_key, ckpt_val in state_dict.items():
        # Skip classification head from pretrained model
        if "head" in ckpt_key:
            skipped.append(ckpt_key)
            continue

        if ckpt_key not in model_dict:
            skipped.append(ckpt_key)
            continue

        if model_dict[ckpt_key].shape == ckpt_val.shape:
            model_dict[ckpt_key] = ckpt_val
            matched.append(ckpt_key)
        else:
            skipped.append(
                f"{ckpt_key} ({ckpt_val.shape} vs {model_dict[ckpt_key].shape})"
            )

    model.load_state_dict(model_dict)

    total = len(matched) + len(skipped)
    if matched:
        LOGGER.info(f"Loaded {len(matched)}/{total} pretrained layers from TinyViT")
    if skipped:
        LOGGER.debug(f"Skipped pretrained layers: {skipped}")
        LOGGER.info(f"Skipped {len(skipped)}/{total} layers (resolution-dependent / head)")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def _build_csl_tinyvit_variant(
    *,
    num_classes: int,
    loss: str,
    pretrained: bool,
    use_gpu: bool,
    embed_dims: list[int],
    num_heads: list[int],
    drop_path_rate: float,
    pretrained_url: str,
    **kwargs,
) -> CSLTinyViT:
    """Build one CSL-TinyViT size variant with shared ReID head defaults."""
    model = CSLTinyViT(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        img_size=kwargs.pop("img_size", (384, 128)),
        embed_dims=embed_dims,
        depths=[2, 2, 6, 2],
        num_heads=num_heads,
        window_sizes=[7, 7, 14, 7],
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        feat_dim=kwargs.pop("feat_dim", 512),
        neck_dim=kwargs.pop("neck_dim", 512),
        **kwargs,
    )
    if pretrained:
        _load_pretrained_tinyvit(model, pretrained_url)
    return model


def csl_tinyvit_7m(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """CSL-TinyViT 7M/small: lightweight hybrid ReID backbone.

    TinyViT-5M-style backbone | embed_dims=[64, 128, 160, 320] | depths=[2, 2, 6, 2]
    Input: 384×128 (H×W) | Output: 1536-d (3×512)
    """
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[64, 128, 160, 320],
        num_heads=[2, 4, 5, 10],
        drop_path_rate=0.0,
        pretrained_url=_TINYVIT_5M_URL,
        **kwargs,
    )


def csl_tinyvit_11m(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """CSL-TinyViT 11M/normal: standard mid-size hybrid ReID backbone.

    TinyViT-11M-style backbone | embed_dims=[64, 128, 256, 448] | depths=[2, 2, 6, 2]
    Input: 384×128 (H×W) | Output: 1536-d (3×512)
    """
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[64, 128, 256, 448],
        num_heads=[2, 4, 8, 14],
        drop_path_rate=0.1,
        pretrained_url=_TINYVIT_11M_URL,
        **kwargs,
    )


def csl_tinyvit_23m(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """CSL-TinyViT 23M/large: high-capacity hybrid ReID backbone.

    TinyViT-21M-style backbone | embed_dims=[96, 192, 384, 576] | depths=[2, 2, 6, 2]
    Input: 384×128 (H×W) | Output: 1536-d (3×512)
    """
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[96, 192, 384, 576],
        num_heads=[3, 6, 12, 18],
        drop_path_rate=0.2,
        pretrained_url=_TINYVIT_21M_URL,
        **kwargs,
    )


def csl_tinyvit_small(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """Alias for the small CSL-TinyViT 7M variant."""
    return csl_tinyvit_7m(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs,
    )


def csl_tinyvit_normal(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """Alias for the normal CSL-TinyViT 11M variant."""
    return csl_tinyvit_11m(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs,
    )


def csl_tinyvit_large(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """Alias for the large CSL-TinyViT 23M variant."""
    return csl_tinyvit_23m(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs,
    )


def csl_tinyvit_7m_lmbn(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """LMBN-style CSL-TinyViT 7M variant with drop-global and channel branches."""
    kwargs["lmbn_style_head"] = True
    return csl_tinyvit_7m(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs,
    )


def csl_tinyvit_11m_lmbn(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """LMBN-style CSL-TinyViT 11M variant with drop-global and channel branches."""
    kwargs["lmbn_style_head"] = True
    return csl_tinyvit_11m(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs,
    )


def csl_tinyvit_23m_lmbn(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """LMBN-style CSL-TinyViT 23M variant with drop-global and channel branches."""
    kwargs["lmbn_style_head"] = True
    return csl_tinyvit_23m(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs,
    )


def csl_tinyvit_lmbn(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """Backward-compatible alias for LMBN-style CSL-TinyViT 11M."""
    return csl_tinyvit_11m_lmbn(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        **kwargs,
    )


if __name__ == "__main__":
    model = CSLTinyViT(num_classes=751, loss="softmax")

    # Inference mode
    model.eval()
    x = torch.randn(2, 3, 384, 128)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Feature dim:  {out.shape[1]}")

    # Training mode
    model.train()
    cls_scores, feats = model(x)
    print(f"Training: {len(cls_scores)} branches, feats={feats.shape}")

    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
