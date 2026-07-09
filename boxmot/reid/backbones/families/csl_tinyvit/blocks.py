# BoxMOT AGPL-3.0 license

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn

from boxmot.reid.backbones.families.csl_tinyvit.attention import Attention

__all__ = [
    "BasicLayer",
    "Conv2d_BN",
    "ConvLayer",
    "DropPath",
    "LayerNorm2d",
    "MBConv",
    "PatchEmbed",
    "PatchMerging",
    "ReIDResidualAdapter",
    "TinyViTBlock",
    "TinyViTMlp",
]


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

    def __init__(self, in_ch, out_ch, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module("c", nn.Conv2d(in_ch, out_ch, ks, stride, pad, dilation, groups, bias=False))
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
        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1, groups=self.hidden_chans)
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

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        activation,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        out_dim=None,
        conv_expand_ratio=4.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                MBConv(
                    dim, dim, conv_expand_ratio, activation, drop_path[i] if isinstance(drop_path, list) else drop_path
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
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


class TinyViTMlp(nn.Module):
    """MLP with pre-norm for TinyViT blocks."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
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
        isinstance(value, tuple) and len(value) == 2 and all(isinstance(part, int) for part in value)
    )


def _to_2tuple(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
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

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        local_conv_size=3,
        activation=nn.GELU,
        shift_size=0,
        attention_bias: str = "absolute",
        attention_mask: bool = False,
    ):
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
        self.attn = Attention(
            dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution, bias_mode=attention_bias
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TinyViTMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

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

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        downsample=None,
        use_checkpoint=False,
        local_conv_size=3,
        activation=nn.GELU,
        out_dim=None,
        shift_size=0,
        attention_bias: str = "absolute",
        attention_mask: bool = False,
        adapter_reduction_ratio: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        block_window_sizes = _expand_block_values(window_size, depth)
        block_shift_sizes = _expand_block_values(shift_size, depth)

        self.blocks = nn.ModuleList(
            [
                TinyViTBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=block_window_sizes[i],
                    shift_size=block_shift_sizes[i],
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    local_conv_size=local_conv_size,
                    activation=activation,
                    attention_bias=attention_bias,
                    attention_mask=attention_mask,
                )
                for i in range(depth)
            ]
        )
        self.reid_adapters = nn.ModuleList(
            [ReIDResidualAdapter(dim, adapter_reduction_ratio) for _ in range(depth)]
            if adapter_reduction_ratio is not None
            else []
        )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
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
