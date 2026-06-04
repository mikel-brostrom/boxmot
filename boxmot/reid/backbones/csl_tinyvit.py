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
  → Stage 2: BasicLayer (windowed attention) ×6  → 160ch
  → Stage 3: BasicLayer (windowed attention)     → 320ch
  → Neck (1×1 + LN + 3×3 + LN → 256ch)
  → Multi-Branch Head (global + 2× partial)      → 3×512 = 1536-d

Parameters: ~5.4M  |  Input: 384×128 (H×W, same as LMBN/CSPReID)

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
            B = len(x)
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


class MultiBranchHead(nn.Module):
    """Multi-granularity feature head: global + 2× horizontal parts.

    Produces:
      - Training: (cls_scores_list, features_tensor)
      - Inference: (B, feat_dim × 3) concatenated features
    """

    def __init__(self, in_ch, feat_dim, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.partial_pool = nn.AdaptiveAvgPool2d((2, 1))

        self.bn_global = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_part0 = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_part1 = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)

    def forward(self, x):
        # x: (B, C, H, W)
        g = self.global_pool(x)
        f_glo = self.bn_global(g)

        p = self.partial_pool(x)
        p0 = p[:, :, 0:1, :]
        p1 = p[:, :, 1:2, :]
        f_p0 = self.bn_part0(p0)
        f_p1 = self.bn_part1(p1)

        if not self.training:
            features = torch.stack([f_glo[0], f_p0[0], f_p1[0]], dim=2)
            return features.flatten(1, 2)

        cls_scores = [f_glo[1], f_p0[1], f_p1[1]]
        feats = torch.stack([f_glo[2], f_p0[2], f_p1[2]], dim=0).mean(dim=0)
        return cls_scores, feats


# ---------------------------------------------------------------------------
# CSL-TinyViT Backbone
# ---------------------------------------------------------------------------


class CSLTinyViT(nn.Module):
    """CSL-TinyViT: hybrid CNN-Transformer ReID backbone.

    Combines efficient MBConv early stages with windowed self-attention
    later stages, producing multi-granularity features via a multi-branch head.

    Input: 3×384×128 (H×W)
    Output:
      - Inference: 1536-d feature vector (3 × 512)
      - Training: ([cls_score_global, cls_score_p0, cls_score_p1], features)
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
        neck_dim: int = 256,
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

        # Multi-branch ReID head
        self.head = MultiBranchHead(neck_dim, feat_dim=feat_dim, num_classes=num_classes)

        # Initialize weights
        self.apply(self._init_weights)

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

        # Stage 0 (conv layer operates on 4D tensor)
        x, out_size = self.layers[0](x, out_size)

        # Stages 1+ (attention layers operate on 3D tokens)
        for i in range(1, len(self.layers)):
            x, out_size = self.layers[i](x, out_size)

        # Reshape back to spatial for neck
        B, _, C = x.size()
        x = x.view(B, out_size[0], out_size[1], C).permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

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


def csl_tinyvit_5m(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """CSL-TinyViT 5M: lightweight hybrid ReID backbone.

    ~5.4M params | embed_dims=[64, 128, 160, 320] | depths=[2, 2, 6, 2]
    Input: 384×128 (H×W) | Output: 1536-d (3×512)
    """
    model = CSLTinyViT(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        img_size=kwargs.pop("img_size", (384, 128)),
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_rate=0.0,
        drop_path_rate=0.0,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        feat_dim=kwargs.pop("feat_dim", 512),
        neck_dim=kwargs.pop("neck_dim", 256),
        **kwargs,
    )
    if pretrained:
        _load_pretrained_tinyvit(model, _TINYVIT_5M_URL)
    return model


def csl_tinyvit_11m(
    num_classes: int = 1000,
    loss: str = "softmax",
    pretrained: bool = False,
    use_gpu: bool = True,
    **kwargs,
) -> CSLTinyViT:
    """CSL-TinyViT 11M: medium hybrid ReID backbone.

    ~11M params | embed_dims=[96, 192, 384, 576] | depths=[2, 2, 6, 2]
    Input: 384×128 (H×W) | Output: 1536-d (3×512)
    """
    model = CSLTinyViT(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        img_size=kwargs.pop("img_size", (384, 128)),
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_rate=0.0,
        drop_path_rate=0.2,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        feat_dim=kwargs.pop("feat_dim", 512),
        neck_dim=kwargs.pop("neck_dim", 256),
        **kwargs,
    )
    if pretrained:
        _load_pretrained_tinyvit(model, _TINYVIT_21M_URL)
    return model


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
