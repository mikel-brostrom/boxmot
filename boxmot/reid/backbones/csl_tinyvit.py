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
import math
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

    def __init__(
        self,
        dim,
        key_dim,
        num_heads=8,
        attn_ratio=4,
        resolution=(14, 14),
        bias_mode: str = "absolute",
    ):
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.resolution = resolution
        self.bias_mode = str(bias_mode).lower()
        if self.bias_mode not in {"absolute", "signed_factorized"}:
            raise ValueError(f"Unsupported CSL-TinyViT attention bias mode: {bias_mode}")
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        if self.bias_mode == "absolute":
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
        else:
            idxs_h = []
            idxs_w = []
            for p1 in points:
                for p2 in points:
                    idxs_h.append(p1[0] - p2[0] + resolution[0] - 1)
                    idxs_w.append(p1[1] - p2[1] + resolution[1] - 1)
            self.attention_bias_h = nn.Parameter(torch.zeros(num_heads, 2 * resolution[0] - 1))
            self.attention_bias_w = nn.Parameter(torch.zeros(num_heads, 2 * resolution[1] - 1))
            self.register_buffer("attention_bias_h_idxs",
                                 torch.LongTensor(idxs_h).view(N, N), persistent=False)
            self.register_buffer("attention_bias_w_idxs",
                                 torch.LongTensor(idxs_w).view(N, N), persistent=False)

    def _attention_bias(self) -> torch.Tensor:
        if self.bias_mode == "absolute":
            return self.attention_biases[:, self.attention_bias_idxs]
        return (
            self.attention_bias_h[:, self.attention_bias_h_idxs]
            + self.attention_bias_w[:, self.attention_bias_w_idxs]
        )

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode:
            if hasattr(self, "ab"):
                del self.ab
        else:
            if hasattr(self, "ab"):
                del self.ab
            self.register_buffer("ab", self._attention_bias(), persistent=False)

    def forward(self, x, attn_mask: torch.Tensor | None = None):
        B, N, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self._attention_bias() if self.training else self.ab
        attn = attn + bias
        if attn_mask is not None:
            attn = attn.masked_fill(~attn_mask[:, None, :, :], torch.finfo(attn.dtype).min)
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


class ReIDResidualAdapter(nn.Module):
    """Zero-gated ReID adapter for TinyViT token features."""

    def __init__(self, dim: int, reduction_ratio: int = 4):
        super().__init__()
        if reduction_ratio < 1:
            raise ValueError(f"reduction_ratio must be positive, got {reduction_ratio}")
        hidden_dim = max(dim // int(reduction_ratio), 1)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.adapter = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=(1, 3),
                padding=(0, 1),
                groups=hidden_dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor, hw_size: tuple[int, int]) -> torch.Tensor:
        B, L, C = x.shape
        H, W = hw_size
        if L != H * W:
            raise ValueError(f"Adapter token count {L} does not match spatial size {hw_size}")
        spatial = x.transpose(1, 2).reshape(B, C, H, W)
        adapted = self.adapter(spatial).flatten(2).transpose(1, 2)
        return x + self.gamma * adapted


def _is_window_size(value) -> bool:
    return isinstance(value, int) or (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(part, int) for part in value)
    )


def _to_2tuple(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an int or (height, width) tuple, got {value!r}")


def _expand_block_values(value, depth: int) -> list:
    if _is_window_size(value):
        return [value for _ in range(depth)]
    values = list(value)
    if len(values) != depth:
        raise ValueError(f"Expected {depth} block values, got {len(values)}: {value!r}")
    return values


def _shift_for_window(window_size) -> tuple[int, int]:
    window_h, window_w = _to_2tuple(window_size)
    return window_h // 2, window_w // 2


class TinyViTBlock(nn.Module):
    """TinyViT block: windowed attention + local depthwise conv + MLP."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4.0, drop=0.0, drop_path=0.0,
                 local_conv_size=3, activation=nn.GELU,
                 shift_size=0, attention_bias: str = "absolute",
                 attention_mask: bool = False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = _to_2tuple(window_size)
        self.shift_size = _to_2tuple(shift_size)
        if any(shift < 0 for shift in self.shift_size):
            raise ValueError(f"CSL-TinyViT shift_size must be non-negative, got {shift_size}")
        if any(shift >= window for shift, window in zip(self.shift_size, self.window_size, strict=True)):
            raise ValueError(
                f"CSL-TinyViT shift_size {self.shift_size} must be smaller than window_size {self.window_size}"
            )
        self.attention_mask = bool(attention_mask)
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        head_dim = dim // num_heads
        window_resolution = self.window_size
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution,
                              bias_mode=attention_bias)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TinyViTMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                              act_layer=activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    @staticmethod
    def _partition_windows(x: torch.Tensor, window_size: tuple[int, int]) -> torch.Tensor:
        window_h, window_w = window_size
        B, H, W, C = x.shape
        nH = H // window_h
        nW = W // window_w
        x = x.view(B, nH, window_h, nW, window_w, C)
        return x.transpose(2, 3).reshape(B * nH * nW, window_h * window_w, C)

    @staticmethod
    def _mask_slices(size: int, window: int, shift: int) -> tuple[slice, ...]:
        if shift == 0:
            return (slice(0, size),)
        return (slice(0, -window), slice(-window, -shift), slice(-shift, None))

    def _window_attention_mask(
        self,
        *,
        batch_size: int,
        original_size: tuple[int, int],
        padded_size: tuple[int, int],
        device: torch.device,
        shift_size: tuple[int, int],
    ) -> torch.Tensor | None:
        window_h, window_w = self.window_size
        shift_h, shift_w = shift_size
        H, W = original_size
        pH, pW = padded_size
        nH = pH // window_h
        nW = pW // window_w
        num_windows = nH * nW
        num_tokens = window_h * window_w
        allowed: torch.Tensor | None = None

        if shift_h > 0 or shift_w > 0:
            region_mask = torch.zeros((1, pH, pW, 1), device=device, dtype=torch.long)
            counter = 0
            for h_slice in self._mask_slices(pH, window_h, shift_h):
                for w_slice in self._mask_slices(pW, window_w, shift_w):
                    region_mask[:, h_slice, w_slice, :] = counter
                    counter += 1
            mask_windows = self._partition_windows(region_mask, self.window_size).view(num_windows, num_tokens)
            allowed = mask_windows[:, :, None] == mask_windows[:, None, :]

        if self.attention_mask and (H != pH or W != pW):
            valid = torch.ones((1, H, W, 1), device=device, dtype=torch.bool)
            valid = F.pad(valid, (0, 0, 0, pW - W, 0, pH - H), value=False)
            if shift_h > 0 or shift_w > 0:
                valid = torch.roll(valid, shifts=(-shift_h, -shift_w), dims=(1, 2))
            valid_windows = self._partition_windows(valid, self.window_size).view(num_windows, num_tokens)
            valid_allowed = valid_windows[:, None, :].expand(num_windows, num_tokens, num_tokens)
            allowed = valid_allowed if allowed is None else allowed & valid_allowed

        if allowed is None:
            return None
        return allowed.repeat(batch_size, 1, 1)

    def forward(self, x, hw_size):
        B, L, C = x.shape
        H, W = hw_size
        assert L == H * W

        res_x = x
        window_h, window_w = self.window_size
        shift_h, shift_w = self.shift_size
        if H <= window_h:
            shift_h = 0
        if W <= window_w:
            shift_w = 0
        active_shift = (shift_h, shift_w)

        if H == window_h and W == window_w and active_shift == (0, 0):
            x = self.attn(x)
        else:
            x = x.view(B, H, W, C)
            pad_b = (window_h - H % window_h) % window_h
            pad_r = (window_w - W % window_w) % window_w
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            if active_shift != (0, 0):
                x = torch.roll(x, shifts=(-active_shift[0], -active_shift[1]), dims=(1, 2))

            nH = pH // window_h
            nW = pW // window_w
            attn_mask = self._window_attention_mask(
                batch_size=B,
                original_size=(H, W),
                padded_size=(pH, pW),
                device=x.device,
                shift_size=active_shift,
            )
            # Window partition
            x = x.view(B, nH, window_h, nW, window_w, C)
            x = x.transpose(2, 3).reshape(B * nH * nW, window_h * window_w, C)
            x = self.attn(x, attn_mask=attn_mask)
            # Window reverse
            x = x.view(B, nH, nW, window_h, window_w, C)
            x = x.transpose(2, 3).reshape(B, pH, pW, C)

            if active_shift != (0, 0):
                x = torch.roll(x, shifts=active_shift, dims=(1, 2))
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
                 out_dim=None, shift_size=0, attention_bias: str = "absolute",
                 attention_mask: bool = False, adapter_reduction_ratio: int | None = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        block_window_sizes = _expand_block_values(window_size, depth)
        block_shift_sizes = _expand_block_values(shift_size, depth)

        self.blocks = nn.ModuleList([
            TinyViTBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=block_window_sizes[i],
                shift_size=block_shift_sizes[i],
                mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                local_conv_size=local_conv_size, activation=activation,
                attention_bias=attention_bias,
                attention_mask=attention_mask)
            for i in range(depth)
        ])
        self.reid_adapters = nn.ModuleList(
            [
                ReIDResidualAdapter(dim, adapter_reduction_ratio)
                for _ in range(depth)
            ]
            if adapter_reduction_ratio is not None
            else []
        )

        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def forward(self, x, out_size):
        for index, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, out_size, use_reentrant=False)
            else:
                x = blk(x, out_size)
            if self.reid_adapters:
                x = self.reid_adapters[index](x, out_size)
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
        initial_p = max(float(p), 1.0 + eps)
        self.raw_p = nn.Parameter(torch.tensor([math.log(math.expm1(initial_p - 1.0))]))
        self.eps = eps

    def effective_p(self) -> torch.Tensor:
        return (1.0 + F.softplus(self.raw_p)).clamp(max=8.0)

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
        old_key = f"{prefix}p"
        new_key = f"{prefix}raw_p"
        if old_key in state_dict and new_key not in state_dict:
            p = state_dict[old_key].clamp(min=1.0 + self.eps, max=8.0)
            state_dict[new_key] = torch.log(torch.expm1(p - 1.0))
            del state_dict[old_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.effective_p()
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, self.output_size)
        return x.pow(1.0 / p)


class ActivatedGeM(nn.Sequential):
    """Apply an activation before GeM pooling."""

    def __init__(self, activation: nn.Module, output_size: tuple[int, int]):
        super().__init__(activation, GeM(output_size))


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


class PatternAdapter(nn.Module):
    """Zero-initialized residual adapter for pattern-specific feature maps."""

    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(f"pattern_adapter_dim must be positive, got {hidden_dim}")
        self.projection = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.projection[-1].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.projection(x)


class LearnedPartTokenPool(nn.Module):
    """Pool spatial features with learned queries and a trainable band prior."""

    def __init__(self, channels: int, num_parts: int):
        super().__init__()
        if num_parts < 1:
            raise ValueError(f"num_part_tokens must be positive, got {num_parts}")
        self.channels = channels
        self.num_parts = num_parts
        self.queries = nn.Parameter(torch.empty(num_parts, channels))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(channels)
        self.token_norm = nn.LayerNorm(channels)

        centers = (torch.arange(num_parts, dtype=torch.float32) + 0.5) / num_parts
        initial_width = 0.5 / num_parts
        self.band_centers = nn.Parameter(centers)
        self.band_log_widths = nn.Parameter(
            torch.full((num_parts,), math.log(math.expm1(initial_width)))
        )
        self.band_log_strength = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))

    def _band_bias(self, height: int, width: int) -> torch.Tensor:
        rows = (torch.arange(height, device=self.queries.device, dtype=self.queries.dtype) + 0.5) / height
        widths = F.softplus(self.band_log_widths).clamp_min(1e-3)
        strength = F.softplus(self.band_log_strength)
        bias = -0.5 * ((rows[None, :] - self.band_centers[:, None]) / widths[:, None]).square()
        return (strength * bias)[:, :, None].expand(-1, -1, width).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        keys = self.token_norm(tokens)
        queries = self.query_norm(self.queries)
        logits = torch.einsum("kc,bnc->bkn", queries, keys) / math.sqrt(channels)
        logits = logits + self._band_bias(height, width)[None, :, :]
        attention = logits.softmax(dim=-1)
        pooled = torch.einsum("bkn,bnc->bkc", attention, tokens)
        return pooled.reshape(batch_size, self.num_parts, channels, 1, 1)


class DSELitePool(nn.Module):
    """DSE-lite weighted spatial pooling without token pruning or merging."""

    def __init__(self, output_size: tuple[int, int], eps: float = 1e-6):
        super().__init__()
        if len(output_size) != 2 or output_size[1] != 1:
            raise ValueError(f"DSE-lite pooling expects output_size=(parts, 1), got {output_size}")
        self.output_size = tuple(int(value) for value in output_size)
        self.eps = float(eps)

    def _center_gaussian_prior(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        parts = self.output_size[0]
        rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        centers = (torch.arange(parts, device=device, dtype=dtype) + 0.5) / parts
        sigma = max(0.5 / parts, self.eps)
        prior = torch.exp(-0.5 * ((rows[None, :] - centers[:, None]) / sigma) ** 2)
        return prior[:, :, None].expand(parts, height, width)

    def _entropy_inverse_attention_score(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels <= 1:
            return x.new_ones((x.shape[0], 1, x.shape[2], x.shape[3]))
        probabilities = torch.softmax(x.float().square(), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(self.eps).log()).sum(dim=1, keepdim=True)
        inverse = 1.0 - entropy / math.log(channels)
        return inverse.clamp_min(self.eps).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        prior = self._center_gaussian_prior(height, width, x.device, x.dtype)
        token_score = self._entropy_inverse_attention_score(x)
        weights = token_score[:, None, :, :, :] * prior[None, :, None, :, :]
        denominator = weights.sum(dim=(-1, -2), keepdim=True).clamp_min(self.eps)
        weights = (weights / denominator).squeeze(2)
        pooled = torch.einsum("bghw,bchw->bcg", weights, x)
        return pooled.reshape(batch_size, x.shape[1], self.output_size[0], 1)


class StripeVisibilityGate(nn.Module):
    """Predict a confidence for each pooled stripe from its local feature."""

    def __init__(self, channels: int, num_stripes: int):
        super().__init__()
        if num_stripes < 1:
            raise ValueError(f"num_stripes must be positive, got {num_stripes}")
        self.num_stripes = int(num_stripes)
        self.norm = nn.LayerNorm(channels)
        self.predictor = nn.Linear(channels, 1)
        nn.init.zeros_(self.predictor.weight)
        nn.init.constant_(self.predictor.bias, math.log(9.0))

    def forward(self, pooled_stripes: torch.Tensor) -> torch.Tensor:
        """Return sigmoid confidences with shape ``(batch, num_stripes)``."""
        if pooled_stripes.ndim != 4 or pooled_stripes.shape[2] != self.num_stripes:
            raise ValueError(
                "Expected pooled stripes shaped "
                f"(B, C, {self.num_stripes}, 1), got {tuple(pooled_stripes.shape)}"
            )
        stripe_tokens = pooled_stripes.squeeze(-1).transpose(1, 2)
        return torch.sigmoid(self.predictor(self.norm(stripe_tokens))).squeeze(-1)


class MultiBranchHead(nn.Module):
    """Multi-granularity head with fixed stripes or learned part tokens.

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
        part_pooling: str = "stripes",
        num_part_tokens: int = 4,
        decouple_patterns: bool = False,
        pattern_adapter_dim: int = 128,
        stripe_visibility: bool = False,
    ):
        super().__init__()
        self.metric_feature = metric_feature
        self.inference_feature = inference_feature
        self.branch_metric = branch_metric
        self.part_pooling = str(part_pooling).lower()
        if self.part_pooling not in {"stripes", "tokens"}:
            raise ValueError(f"Unsupported CSL-TinyViT part_pooling: {part_pooling}")
        self.num_part_tokens = int(num_part_tokens)
        self.head_parts = self._normalize_head_parts(head_parts)
        if self.part_pooling == "tokens":
            if self.num_part_tokens < 1:
                raise ValueError(f"num_part_tokens must be positive, got {num_part_tokens}")
            self.branch_specs = [
                ("global", 1, 0),
                *[(f"part{index}", 0, index) for index in range(self.num_part_tokens)],
            ]
            self.part_token_pool = LearnedPartTokenPool(in_ch, self.num_part_tokens)
        else:
            self.branch_specs = self._build_branch_specs(self.head_parts)
            self.part_token_pool = None
        self.part_keys = [key for key, granularity, _ in self.branch_specs if granularity > 1]
        if self.part_pooling == "tokens":
            self.part_keys = [key for key, _, _ in self.branch_specs if key != "global"]
        self.decouple_patterns = bool(decouple_patterns)
        self.pattern_adapter_dim = int(pattern_adapter_dim)
        if self.decouple_patterns:
            self.global_adapter = PatternAdapter(in_ch, self.pattern_adapter_dim)
            self.local_adapter = PatternAdapter(in_ch, self.pattern_adapter_dim)
        else:
            self.global_adapter = nn.Identity()
            self.local_adapter = nn.Identity()
        self.stripe_visibility = bool(stripe_visibility)
        if self.stripe_visibility:
            if self.part_pooling != "stripes":
                raise ValueError("stripe_visibility requires fixed stripe pooling")
            local_specs = [spec for spec in self.branch_specs if spec[0] != "global"]
            granularities = {granularity for _, granularity, _ in local_specs}
            if len(granularities) != 1:
                raise ValueError(
                    "stripe_visibility requires exactly one local stripe granularity, "
                    f"got head_parts={self.head_parts}"
                )
            self.visibility_granularity = granularities.pop()
            self.visibility_gate = StripeVisibilityGate(in_ch, len(local_specs))
        else:
            self.visibility_granularity = None
            self.visibility_gate = None
        self.dse_descriptor_pool = DSELitePool((1, 1))
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
        if head_pool == "dse":
            return DSELitePool(output_size)
        if head_pool == "gelu_gem":
            return ActivatedGeM(nn.GELU(), output_size)
        if head_pool == "relu_gem":
            return ActivatedGeM(nn.ReLU(inplace=False), output_size)
        if head_pool == "softplus_gem":
            return ActivatedGeM(nn.Softplus(), output_size)
        raise ValueError(f"Unsupported CSL-TinyViT head_pool: {head_pool}")

    def set_pooling(self, head_pool: str) -> None:
        head_pool = str(head_pool).lower()
        granularities = (1,) if self.part_pooling == "tokens" else self.head_parts
        for granularity in granularities:
            setattr(
                self,
                self._pool_attr(granularity),
                self._make_pool(head_pool, (granularity, 1)),
            )
        self.head_pool = head_pool

    def set_branch_metric(self, branch_metric: bool) -> None:
        self.branch_metric = bool(branch_metric)

    def _needs_dse_descriptor(self) -> bool:
        return self.metric_feature in {"dse_weighted", "dse_mix"} or self.inference_feature in {
            "dse_weighted",
            "dse_mix",
        }

    def _add_dse_descriptors(self, raw_features: dict[str, torch.Tensor], source: torch.Tensor) -> None:
        if not self._needs_dse_descriptor():
            return
        dse_weighted = self.dse_descriptor_pool(source).flatten(1)
        raw_features["dse_weighted"] = dse_weighted
        raw_features["dse_mix"] = torch.cat(
            (
                F.normalize(raw_features["raw_mean"], p=2, dim=1),
                F.normalize(dse_weighted, p=2, dim=1),
                F.normalize(raw_features["raw_concat"], p=2, dim=1),
            ),
            dim=1,
        )

    def forward(self, x):
        # x: (B, C, H, W)
        global_feature = self.global_adapter(x)
        local_feature = self.local_adapter(x)
        pooled_by_granularity = {1: self.global_pool(global_feature)}
        token_parts = None
        if self.part_pooling == "tokens":
            token_parts = self.part_token_pool(local_feature)
        else:
            pooled_by_granularity.update(
                {
                    granularity: getattr(self, self._pool_attr(granularity))(local_feature)
                    for granularity in self.head_parts
                    if granularity > 1
                }
            )
        visibility_by_key = {"global": None}
        if self.visibility_gate is not None:
            visibility = self.visibility_gate(
                pooled_by_granularity[self.visibility_granularity]
            )
            visibility_by_key.update(
                {
                    key: visibility[:, index : index + 1]
                    for index, (key, _, _) in enumerate(
                        spec for spec in self.branch_specs if spec[0] != "global"
                    )
                }
            )

        branch_outputs = {}
        bn_features_list = []
        raw_features_list = []
        normalized_bn_features_list = []
        normalized_raw_features_list = []
        cls_scores = []
        raw_features = {}
        for key, granularity, stripe_index in self.branch_specs:
            if key == "global":
                pooled = pooled_by_granularity[1]
            elif self.part_pooling == "tokens":
                pooled = token_parts[:, stripe_index]
            else:
                pooled = pooled_by_granularity[granularity]
                pooled = pooled[:, :, stripe_index:stripe_index + 1, :]
            branch_output = getattr(self, self._bn_attr(key))(pooled)
            branch_outputs[key] = branch_output
            confidence = visibility_by_key.get(key)
            base_bn_feature = branch_output[0]
            base_raw_feature = branch_output[2]
            bn_feature = base_bn_feature
            raw_feature = base_raw_feature
            normalized_bn_feature = F.normalize(base_bn_feature, p=2, dim=1)
            normalized_raw_feature = F.normalize(base_raw_feature, p=2, dim=1)
            if confidence is not None:
                bn_feature = bn_feature * confidence
                raw_feature = raw_feature * confidence
                normalized_bn_feature = normalized_bn_feature * confidence
                normalized_raw_feature = normalized_raw_feature * confidence
            bn_features_list.append(bn_feature)
            normalized_bn_features_list.append(normalized_bn_feature)
            cls_scores.append(branch_output[1])
            raw_features_list.append(raw_feature)
            normalized_raw_features_list.append(normalized_raw_feature)
            raw_features[key] = raw_feature

        bn_features = torch.stack(bn_features_list, dim=2).flatten(1, 2)
        raw_features["raw_mean"] = torch.stack(raw_features_list, dim=0).mean(dim=0)
        raw_features["raw_concat"] = torch.cat(normalized_raw_features_list, dim=1)
        raw_features["concat_bn"] = bn_features
        raw_features["norm_concat_bn"] = F.normalize(
            torch.cat(normalized_bn_features_list, dim=1),
            p=2,
            dim=1,
        )
        self._add_dse_descriptors(raw_features, local_feature)

        if not self.training:
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "norm_concat_bn":
                return raw_features["norm_concat_bn"]
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            if self.inference_feature == "raw_concat":
                return raw_features["raw_concat"]
            if self.inference_feature in {"dse_weighted", "dse_mix"}:
                return raw_features[self.inference_feature]
            raise ValueError(f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}")

        if self.branch_metric:
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        elif self.metric_feature == "raw_concat":
            feats = raw_features["raw_concat"]
        elif self.metric_feature in {"global", "dse_weighted", "dse_mix"}:
            feats = raw_features[self.metric_feature]
        else:
            feats = raw_features["raw_mean"]
        return cls_scores, feats


class GPCLiteMultiBranchHead(MultiBranchHead):
    """Global/part/channel head with CE on every branch and global metric supervision."""

    def __init__(
        self,
        in_ch,
        feat_dim,
        num_classes,
        metric_feature: str = "raw_mean",
        inference_feature: str = "norm_concat_bn",
        head_pool: str = "avg",
        branch_metric: bool = False,
        head_parts: tuple[int, ...] = (1, 3),
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
            part_pooling="stripes",
            decouple_patterns=False,
            stripe_visibility=False,
        )
        if in_ch % 2 != 0:
            raise ValueError(f"GPC-lite channel split requires even channels, got {in_ch}")
        self.channel_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.channel_shared = nn.Sequential(
            nn.Conv2d(in_ch // 2, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.bn_ch0 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)
        self.bn_ch1 = BNNeck3(feat_dim, num_classes, feat_dim, return_f=True)

    def forward(self, x):
        pooled_by_granularity = {
            granularity: getattr(self, self._pool_attr(granularity))(x)
            for granularity in self.head_parts
        }
        branch_outputs = {
            "global": getattr(self, self._bn_attr("global"))(pooled_by_granularity[1])
        }
        for key, granularity, stripe_index in self.branch_specs:
            if key == "global":
                continue
            pooled = pooled_by_granularity[granularity][
                :, :, stripe_index : stripe_index + 1, :
            ]
            branch_outputs[key] = getattr(self, self._bn_attr(key))(pooled)

        channel_0, channel_1 = torch.chunk(self.channel_pool(x), chunks=2, dim=1)
        branch_outputs["ch0"] = self.bn_ch0(self.channel_shared(channel_0))
        branch_outputs["ch1"] = self.bn_ch1(self.channel_shared(channel_1))

        ordered_keys = ["global", *self.part_keys, "ch0", "ch1"]
        bn_features_list = [branch_outputs[key][0] for key in ordered_keys]
        raw_features_list = [branch_outputs[key][2] for key in ordered_keys]
        cls_scores = [branch_outputs[key][1] for key in ordered_keys]
        bn_features = torch.cat(bn_features_list, dim=1)
        raw_features = {key: branch_outputs[key][2] for key in ordered_keys}
        # GPC-lite deliberately applies metric and center losses only to the
        # global raw descriptor while every branch retains CE supervision.
        raw_features["raw_mean"] = raw_features["global"]
        raw_features["raw_concat"] = torch.cat(
            [F.normalize(feature, p=2, dim=1) for feature in raw_features_list],
            dim=1,
        )
        raw_features["concat_bn"] = bn_features
        raw_features["norm_concat_bn"] = F.normalize(
            torch.cat(
                [F.normalize(feature, p=2, dim=1) for feature in bn_features_list],
                dim=1,
            ),
            p=2,
            dim=1,
        )

        if not self.training:
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "norm_concat_bn":
                return raw_features["norm_concat_bn"]
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            if self.inference_feature == "raw_concat":
                return raw_features["raw_concat"]
            if self.inference_feature in raw_features:
                return raw_features[self.inference_feature]
            raise ValueError(
                f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}"
            )

        if self.branch_metric:
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        elif self.metric_feature == "raw_concat":
            feats = raw_features["raw_concat"]
        elif self.metric_feature in raw_features:
            feats = raw_features[self.metric_feature]
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
        raw_features["raw_concat"] = torch.cat(
            [F.normalize(feature, p=2, dim=1) for feature in raw_features_list],
            dim=1,
        )
        raw_features["concat_bn"] = bn_features
        raw_features["norm_concat_bn"] = F.normalize(
            torch.cat(
                [F.normalize(feature, p=2, dim=1) for feature in bn_features_list],
                dim=1,
            ),
            p=2,
            dim=1,
        )

        if not self.training:
            if self.inference_feature == "concat_bn":
                return bn_features
            if self.inference_feature == "norm_concat_bn":
                return raw_features["norm_concat_bn"]
            if self.inference_feature == "global":
                return branch_outputs["global"][0]
            if self.inference_feature == "raw_mean":
                return raw_features["raw_mean"]
            if self.inference_feature == "raw_concat":
                return raw_features["raw_concat"]
            if self.inference_feature in raw_features:
                return raw_features[self.inference_feature]
            raise ValueError(f"Unsupported CSL-TinyViT inference_feature: {self.inference_feature}")

        if self.branch_metric:
            feats = raw_features
        elif self.metric_feature == "concat_bn":
            feats = bn_features
        elif self.metric_feature == "raw_concat":
            feats = raw_features["raw_concat"]
        elif self.metric_feature != "raw_mean" and self.metric_feature in raw_features:
            feats = raw_features[self.metric_feature]
        else:
            feats = [
                raw_features["global"],
                raw_features["drop_global"],
                raw_features["part_global"],
            ]
        return cls_scores, feats


class CSLTinyViTFeatureFusion(nn.Module):
    """Swappable spatial feature fusion module for CSL-TinyViT stage outputs."""

    _VALID_MODES = {
        "final",
        "last2",
        "last3",
        "weighted_last2",
        "weighted_last3",
        "normpres_last2",
        "normpres_last3",
        "dynamic_last3",
        "dynamic_last3_scale_token",
    }
    _VALID_FUSION_TYPES = {"final", "residual", "weighted", "norm_preserved", "dynamic", "dynamic_scale_token"}

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
        self.norm_preserved = self.fusion_type == "norm_preserved"
        self.dynamic = self.fusion_type in {"dynamic", "dynamic_scale_token"}
        self.use_scale_token = self.fusion_type == "dynamic_scale_token"

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
                for index in (self.stage_indices if self.fusion_type == "residual" else ())
            }
        )
        if self.weighted:
            self.fusion_weights = nn.Parameter(torch.tensor([1.0, *([1e-3] * len(self.stage_indices))]))
        else:
            self.register_parameter("fusion_weights", None)

        if self.dynamic:
            num_paths = 1 + len(self.stage_indices)
            gate_hidden_dim = max(out_channels // 4, 64)
            scale_token_dim = max(min(out_channels // 16, 64), 16)
            if self.use_scale_token:
                self.scale_token_projection = nn.Sequential(
                    nn.LayerNorm(out_channels),
                    nn.Linear(out_channels, scale_token_dim),
                    nn.GELU(),
                )
                self.scale_tokens = nn.Parameter(torch.empty(num_paths, scale_token_dim))
                nn.init.trunc_normal_(self.scale_tokens, std=0.02)
                self.scale_token_norm = nn.LayerNorm(scale_token_dim)
                gate_input_dim = out_channels + num_paths * scale_token_dim
            else:
                self.scale_token_projection = None
                self.register_parameter("scale_tokens", None)
                self.scale_token_norm = None
                gate_input_dim = out_channels
            self.dynamic_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, num_paths),
            )
            self.initialize_dynamic_gate()
        else:
            self.scale_token_projection = None
            self.register_parameter("scale_tokens", None)
            self.scale_token_norm = None
            self.dynamic_gate = None

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
        if mode.startswith("normpres_"):
            return "norm_preserved"
        if mode == "dynamic_last3":
            return "dynamic"
        if mode == "dynamic_last3_scale_token":
            return "dynamic_scale_token"
        return "residual"

    @staticmethod
    def stage_indices_for_mode(mode: str) -> tuple[int, ...]:
        if mode in {"last2", "weighted_last2"}:
            return (2,)
        if mode in {"last3", "weighted_last3"}:
            return (1, 2)
        if mode == "normpres_last2":
            return (2,)
        if mode == "normpres_last3":
            return (1, 2)
        if mode in {"dynamic_last3", "dynamic_last3_scale_token"}:
            # Dynamic fusion follows the semantic order final, stage 2, stage 1.
            return (2, 1)
        return ()

    def initialize_dynamic_gate(self) -> None:
        """Initialize dynamic fusion with a stable 80/10/10 path mixture."""
        if self.dynamic_gate is None:
            return
        output = self.dynamic_gate[-1]
        nn.init.trunc_normal_(output.weight, std=1e-3)
        with torch.no_grad():
            initial_weights = output.bias.new_tensor([0.8, *([0.1] * len(self.stage_indices))])
            output.bias.copy_(initial_weights.log())

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

    @staticmethod
    def _pooled_descriptor(feature: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)

    def _ordered_features(
        self,
        final_feature: torch.Tensor,
        path_features: dict[int, torch.Tensor],
    ) -> list[torch.Tensor]:
        output_size = final_feature.shape[-2:]
        return [
            final_feature,
            *[
                self._project_path(stage_index, path_features[stage_index], output_size)
                for stage_index in self.stage_indices
            ],
        ]

    def dynamic_weights(
        self,
        final_feature: torch.Tensor,
        path_features: dict[int, torch.Tensor],
    ) -> torch.Tensor:
        """Return per-image softmax weights in fusion path order."""
        if self.dynamic_gate is None:
            raise RuntimeError("Dynamic weights are only available for dynamic feature fusion")
        features = self._ordered_features(final_feature, path_features)
        return self._dynamic_weights_from_features(features)

    def _dynamic_weights_from_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Compute dynamic weights from already projected, ordered features."""
        descriptors = [self._pooled_descriptor(feature) for feature in features]
        final_descriptor = descriptors[0]
        gate_inputs = [final_descriptor]
        if self.scale_token_projection is not None:
            scale_descriptors = torch.stack(
                [self.scale_token_projection(descriptor) for descriptor in descriptors],
                dim=1,
            )
            scale_queries = self.scale_tokens.unsqueeze(0).expand(scale_descriptors.shape[0], -1, -1)
            scale_key_values = scale_descriptors + scale_queries
            attention = (
                scale_queries @ scale_key_values.transpose(1, 2)
                / math.sqrt(scale_queries.shape[-1])
            ).softmax(dim=-1)
            scale_context = self.scale_token_norm(scale_queries + attention @ scale_key_values)
            gate_inputs.append(scale_context.flatten(1))
        return self.dynamic_gate(torch.cat(gate_inputs, dim=1)).softmax(dim=1)

    def forward(self, final_feature: torch.Tensor, path_features: dict[int, torch.Tensor]) -> torch.Tensor:
        if not self.stage_indices:
            return final_feature

        if self.dynamic:
            features = self._ordered_features(final_feature, path_features)
            weights = self._dynamic_weights_from_features(features)
            return sum(
                weight[:, None, None, None] * feature
                for weight, feature in zip(weights.unbind(dim=1), features, strict=True)
            )

        if self.norm_preserved:
            features = self._ordered_features(final_feature, path_features)
            mean_feature = torch.stack(features, dim=0).mean(dim=0)
            max_norm = torch.stack(
                [feature.norm(p=2, dim=1, keepdim=True) for feature in features],
                dim=0,
            ).max(dim=0).values
            return F.normalize(mean_feature, p=2, dim=1) * max_norm

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
        window_sizes: list[int | tuple[int, int]] = None,
        attention_window_layout: str = "legacy",
        attention_bias: str = "absolute",
        attention_mask: bool = False,
        attention_shift: bool = False,
        stage3_global: bool = False,
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
        part_pooling: str = "stripes",
        num_part_tokens: int = 4,
        decouple_patterns: bool = False,
        pattern_adapter_dim: int = 128,
        head_type: str = "standard",
        stripe_visibility: bool = False,
        reid_adapter_stages: tuple[int, ...] = (),
        reid_adapter_reduction: int = 4,
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
            attention_window_layout = str(attention_window_layout).lower()
            if attention_window_layout == "legacy":
                window_sizes = [7, 7, 14, 7]
            elif attention_window_layout == "rect":
                window_sizes = [7, (12, 4), (12, 8), (12, 8)]
            else:
                raise ValueError(f"Unsupported CSL-TinyViT attention_window_layout: {attention_window_layout}")
        else:
            attention_window_layout = str(attention_window_layout).lower()

        self.loss = loss
        self.img_size = img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = float(drop_path_rate)
        self.window_sizes = tuple(_to_2tuple(size) for size in window_sizes)
        self.attention_window_layout = attention_window_layout
        self.attention_bias = str(attention_bias).lower()
        self.attention_mask = bool(attention_mask)
        self.attention_shift = bool(attention_shift)
        self.stage3_global = bool(stage3_global)
        self.feature_fusion = CSLTinyViTFeatureFusion.normalize_mode(feature_fusion)
        self.head_type = "lmbn" if lmbn_style_head else str(head_type).lower()
        if self.head_type not in {"standard", "gpc_lite", "lmbn"}:
            raise ValueError(f"Unsupported CSL-TinyViT head_type: {head_type}")
        self.reid_adapter_stages = self._normalize_adapter_stages(reid_adapter_stages)
        self.reid_adapter_reduction = int(reid_adapter_reduction)
        if self.reid_adapter_reduction < 1:
            raise ValueError("reid_adapter_reduction must be positive")
        self.pretrained_match_count: int | None = None
        self.pretrained_total_count: int | None = None
        self.pretrained_url: str | None = None

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
            input_resolution = (
                patches_resolution[0] // (2 ** (i_layer if i_layer < 2 else 2)),
                patches_resolution[1] // (2 ** (i_layer if i_layer < 2 else 2)),
            )
            kwargs = dict(
                dim=embed_dims[i_layer],
                input_resolution=input_resolution,
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
                layer_window_size = window_sizes[i_layer]
                layer_shift_size = 0
                if self.attention_shift and i_layer in (1, 2):
                    shift_size = _shift_for_window(layer_window_size)
                    layer_shift_size = [
                        (0, 0) if block_index % 2 == 0 else shift_size
                        for block_index in range(depths[i_layer])
                    ]
                if self.stage3_global and i_layer == self.num_layers - 1:
                    layer_window_size = [
                        layer_window_size if block_index < depths[i_layer] - 1 else input_resolution
                        for block_index in range(depths[i_layer])
                    ]
                    layer_shift_size = [(0, 0) for _ in range(depths[i_layer])]
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=layer_window_size,
                    shift_size=layer_shift_size,
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    attention_bias=self.attention_bias,
                    attention_mask=self.attention_mask,
                    adapter_reduction_ratio=(
                        self.reid_adapter_reduction
                        if i_layer in self.reid_adapter_stages
                        else None
                    ),
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
        if self.head_type == "lmbn":
            metric_feature = "raw_mean"
        if self.head_type == "lmbn":
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
        elif self.head_type == "gpc_lite":
            self.head = GPCLiteMultiBranchHead(
                neck_dim,
                feat_dim=feat_dim,
                num_classes=num_classes,
                metric_feature=metric_feature,
                inference_feature=inference_feature,
                head_pool=head_pool,
                head_parts=head_parts,
                branch_metric=branch_metric,
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
                part_pooling=part_pooling,
                num_part_tokens=num_part_tokens,
                decouple_patterns=decouple_patterns,
                pattern_adapter_dim=pattern_adapter_dim,
                stripe_visibility=stripe_visibility,
                branch_metric=branch_metric,
            )

        # Initialize weights
        self.apply(self._init_weights)
        self.feature_fusion_module.initialize_dynamic_gate()

    @staticmethod
    def _normalize_adapter_stages(stages) -> tuple[int, ...]:
        if stages is None:
            return ()
        if isinstance(stages, str):
            if stages.lower() in {"", "none", "off"}:
                return ()
            values = [part for part in stages.replace(";", ",").split(",") if part.strip()]
        elif isinstance(stages, int):
            values = [stages]
        else:
            values = list(stages)
        normalized = tuple(dict.fromkeys(int(stage) for stage in values))
        invalid = [stage for stage in normalized if stage not in {1, 2, 3}]
        if invalid:
            raise ValueError(f"CSL-TinyViT ReID adapters only support attention stages 1, 2, 3; got {invalid}")
        return normalized

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
    model.pretrained_match_count = len(matched)
    model.pretrained_total_count = total
    model.pretrained_url = url
    if matched:
        LOGGER.info(f"Loaded {len(matched)}/{total} pretrained tensors from TinyViT ({url})")
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
    drop_path_rate = float(kwargs.pop("drop_path_rate", drop_path_rate))
    model = CSLTinyViT(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        img_size=kwargs.pop("img_size", (384, 128)),
        embed_dims=embed_dims,
        depths=[2, 2, 6, 2],
        num_heads=num_heads,
        window_sizes=kwargs.pop("window_sizes", None),
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
    drop_path_rate = kwargs.pop("drop_path_rate", 0.0)
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[64, 128, 160, 320],
        num_heads=[2, 4, 5, 10],
        drop_path_rate=drop_path_rate,
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
    drop_path_rate = kwargs.pop("drop_path_rate", 0.1)
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[64, 128, 256, 448],
        num_heads=[2, 4, 8, 14],
        drop_path_rate=drop_path_rate,
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
    drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
    return _build_csl_tinyvit_variant(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        embed_dims=[96, 192, 384, 576],
        num_heads=[3, 6, 12, 18],
        drop_path_rate=drop_path_rate,
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
