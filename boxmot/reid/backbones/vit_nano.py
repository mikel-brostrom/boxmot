"""ViT-Nano: a lightweight Vision Transformer backbone for person ReID.

Architecture (defaults):
    embed_dim   = 192
    depth       = 6
    num_heads   = 3
    mlp_ratio   = 4.0
    patch_size  = 16

Same width as DeiT-Tiny but half the depth — ~3.0 M parameters.
Loads the first 6 of 12 DeiT-Tiny pretrained blocks exactly.

OSNet-inspired enhancements (Zhou et al. ICCV'19):
    * **Omni-Scale Aggregation** — multi-scale spatial pooling (1/2/4/8
      horizontal strips) with a unified aggregation gate that dynamically
      fuses features with input-dependent channel-wise weights.
    * **AIN (Adaptive Instance-LayerNorm)** — early blocks blend
      InstanceNorm (strips camera/domain style) with LayerNorm
      (preserves content) via a learned per-channel gate.  Improves
      cross-domain generalization.

Follows the standard BoxMOT backbone ``forward()`` contract:
    * training + softmax  → class logits
    * training + triplet  → (logits, embedding)
    * inference           → embedding vector

When ``pretrained=True``:
    * **vit_tiny** loads DeiT-Tiny (ImageNet-1k) weights from Facebook.
    * **vit_nano** loads DeiT-Tiny weights with partial matching (layers
      whose dimensions differ are skipped and randomly initialized).
"""

from __future__ import annotations

import math
import os
import warnings
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from boxmot.utils import logger as LOGGER

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image to non-overlapping patch embedding."""

    def __init__(self, img_size=(256, 128), patch_size=16, in_chans=3, embed_dim=128, stride=None):
        super().__init__()
        stride = stride or patch_size
        self.grid_size = (
            (img_size[0] - patch_size) // stride + 1,
            (img_size[1] - patch_size) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) → (B, N, D)
        x = self.proj(x)  # (B, D, Gh, Gw)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class DropPath(nn.Module):
    """Stochastic depth (per-sample drop of entire residual branch).

    Critical ViT regularizer — DeiT uses 0.1 for Tiny.  Linearly
    increases per block (deeper blocks drop more often).
    """

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
# Adaptive Instance-LayerNorm (AIN, inspired by OSNet-AIN, Zhou et al. 2019)
# ---------------------------------------------------------------------------

class AdaptiveINLN(nn.Module):
    """Adaptive Instance-LayerNorm for cross-domain robustness.

    Computes both InstanceNorm (strips camera/domain style) and LayerNorm
    (preserves discriminative content) then blends them per-channel via a
    learned sigmoid gate::

        output = gate * IN(x) + (1 - gate) * LN(x)

    Applied to early transformer blocks so the network learns which
    channels need style-invariance (IN) vs. content-preservation (LN).
    This is the ViT equivalent of IBN/AIN from OSNet.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        # IN along the token (spatial) dimension — per-channel, per-instance
        self.in_norm = nn.InstanceNorm1d(dim, affine=True)
        # Learnable gate per channel, initialised near 0.5
        self.gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        ln_out = self.ln(x)
        # InstanceNorm1d expects (B, D, N)
        in_out = self.in_norm(x.transpose(1, 2)).transpose(1, 2)
        g = torch.sigmoid(self.gate)  # (D,) broadcast to (B, N, D)
        return g * in_out + (1.0 - g) * ln_out


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, use_ain: bool = False):
        super().__init__()
        self.norm1 = AdaptiveINLN(dim) if use_ain else nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Omni-Scale Aggregation (inspired by OSNet, Zhou et al. ICCV'19)
# ---------------------------------------------------------------------------

class UnifiedAggregationGate(nn.Module):
    """Unified AG: shared mini-network that produces channel-wise fusion weights.

    One gate is shared across all spatial-scale streams (Eq. 3-4 in the
    OSNet paper).  Sharing means the gradients from *all* streams jointly
    guide the gate, which is exactly the property that makes AG effective.

    Architecture: GAP → Linear(D, D//r) → ReLU → Linear(D//r, D) → Sigmoid
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        mid = max(dim // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(dim, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, D) — a single stream's pooled feature.  Returns (B, D) weights."""
        return self.fc(x)


class OmniScaleAggregation(nn.Module):
    """Multi-scale spatial pooling with unified aggregation gate.

    Given (B, N, D) spatial patch tokens from the transformer, pool at
    T=4 vertical granularities (person images are upright):
        - global average pool          (scale 1 — whole body)
        - 2 horizontal strips           (scale 2 — upper/lower body)
        - 4 horizontal strips           (scale 3 — head/torso/legs/feet)
        - 8 horizontal strips           (scale 4 — fine-grained parts)

    Each scale produces a (B, D) vector via strip-pool → average.
    A single **unified AG** (shared across all scales) generates dynamic
    channel-wise weights g_t ∈ (0,1)^D, and the final feature is:

        f = Σ_t  g_t(x_t) ⊙ x_t          (Eq. 3, OSNet paper)

    This adds ~0.1 M extra parameters on top of the transformer.
    """

    def __init__(self, dim: int, num_scales: int = 4, reduction: int = 16):
        super().__init__()
        # Strip counts per scale: 1, 2, 4, 8
        self.strip_counts = [2 ** i for i in range(num_scales)]
        self.gate = UnifiedAggregationGate(dim, reduction=reduction)
        # Per-strip BN for each scale — stabilises multi-scale features
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_scales)
        ])

    def forward(self, patch_tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, N, D) spatial tokens (no CLS).
            grid_h, grid_w: spatial grid dimensions so we can reshape.
        Returns:
            (B, D) omni-scale fused feature.
        """
        B, N, D = patch_tokens.shape
        # Reshape to spatial grid: (B, D, H, W)
        spatial = patch_tokens.transpose(1, 2).reshape(B, D, grid_h, grid_w)

        fused = torch.zeros(B, D, device=patch_tokens.device, dtype=patch_tokens.dtype)
        for i, num_strips in enumerate(self.strip_counts):
            # Adaptive pool to (num_strips, 1) then average across strips → (B, D)
            pooled = F.adaptive_avg_pool2d(spatial, (num_strips, 1))  # (B, D, S, 1)
            pooled = pooled.squeeze(-1).mean(dim=-1)                  # (B, D)
            pooled = self.scale_norms[i](pooled)
            # Unified AG: same gate for every scale
            g = self.gate(pooled)  # (B, D) channel-wise weights
            fused = fused + g * pooled

        return fused


# ---------------------------------------------------------------------------
# ViT-Nano ReID model
# ---------------------------------------------------------------------------

class ViTNano(nn.Module):
    """Lightweight Vision Transformer for person re-identification.

    Parameters match the standard BoxMOT backbone constructor contract:
    ``__init__(num_classes, loss, pretrained, use_gpu, **kwargs)``.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: bool = True,
        *,
        img_size: tuple[int, int] = (256, 128),
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        omni_scale: bool = True,
        ain: bool = True,
        pool: str = "cls",
        patch_stride: int | None = None,
        feat_dim: int | None = None,
    ):
        super().__init__()
        self.loss = loss
        feat_dim = feat_dim or embed_dim
        self.feature_dim = feat_dim
        self.omni_scale = omni_scale
        self.pool = pool
        self.depth = depth
        self.img_size = img_size

        # ---------- patch embedding + positional ----------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim, stride=patch_stride)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # ---------- transformer blocks ----------
        # DropPath: linearly increasing per block (DeiT recipe)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # AIN (Adaptive Instance-LayerNorm) applied to the first half of
        # blocks — early layers strip camera/domain style via InstanceNorm
        # while later layers keep standard LayerNorm for content features.
        ain_depth = depth // 2 if ain else 0
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], use_ain=(i < ain_depth))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ---------- omni-scale aggregation (OSNet-inspired) ----------
        if omni_scale:
            self.os_agg = OmniScaleAggregation(embed_dim, num_scales=4, reduction=16)

        # ---------- projection head (optional) ----------
        if feat_dim != embed_dim:
            self.proj = nn.Linear(embed_dim, feat_dim, bias=False)
        else:
            self.proj = None

        # ---------- ReID head ----------
        self.bottleneck = nn.BatchNorm1d(feat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.classifier:
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        if self.omni_scale:
            # Use omni-scale aggregation over spatial patch tokens
            patch_tokens = x[:, 1:]  # drop CLS token
            gh, gw = self.patch_embed.grid_size
            return self.os_agg(patch_tokens, gh, gw)
        elif self.pool == "gap":
            # Global average pooling over patch tokens (better for ReID
            # fine-tuning — preserves spatial part-level information)
            return x[:, 1:].mean(dim=1)
        else:
            return x[:, 0]  # CLS token only

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        v = self.forward_features(x)  # raw embedding (embed_dim)
        if self.proj is not None:
            v = self.proj(v)           # project to feat_dim
        feat = self.bottleneck(v)      # BN-normalized for classifier

        if not self.training:
            # Return BNNeck output for cosine-distance retrieval (BoT paper)
            return feat

        y = self.classifier(feat)
        if self.loss == "softmax":
            return y
        elif self.loss == "triplet":
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


# ---------------------------------------------------------------------------
# Pretrained weight loading
# ---------------------------------------------------------------------------

# DeiT-Tiny (ImageNet-1k): embed_dim=192, depth=12, heads=3, mlp_ratio=4.0
_DEIT_TINY_URL = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"


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


def _interpolate_pos_embed(pos_embed_ckpt: torch.Tensor, model: ViTNano) -> torch.Tensor:
    """Bicubic-interpolate positional embeddings from checkpoint to model grid."""
    num_patches_model = model.patch_embed.num_patches
    num_patches_ckpt = pos_embed_ckpt.shape[1] - 1  # minus CLS token
    if num_patches_ckpt == num_patches_model:
        return pos_embed_ckpt

    cls_token = pos_embed_ckpt[:, :1, :]
    patch_tokens = pos_embed_ckpt[:, 1:, :]

    # Checkpoint was 224x224 / 16 = 14x14
    gs_old = int(math.sqrt(num_patches_ckpt))
    gs_new_h, gs_new_w = model.patch_embed.grid_size

    patch_tokens = patch_tokens.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(
        patch_tokens, size=(gs_new_h, gs_new_w), mode="bicubic", align_corners=False
    )
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, pos_embed_ckpt.shape[-1])
    return torch.cat([cls_token, patch_tokens], dim=1)


def _load_pretrained_deit_tiny(model: ViTNano) -> None:
    """Load DeiT-Tiny pretrained weights with partial matching.

    Layers with matching names and shapes are loaded; others (e.g.
    classifier, dimension mismatches) are skipped.
    """
    ckpt = _download_pretrained(_DEIT_TINY_URL)
    state_dict = ckpt.get("model", ckpt)
    model_dict = model.state_dict()

    # Map DeiT key names to our naming convention
    key_map = {
        "cls_token": "cls_token",
        "pos_embed": "pos_embed",
        "patch_embed.proj.weight": "patch_embed.proj.weight",
        "patch_embed.proj.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    # Map transformer block keys — handle AIN blocks where norm1 is
    # AdaptiveINLN (norm1.ln.weight instead of norm1.weight)
    for k in list(state_dict.keys()):
        if k.startswith("blocks."):
            model_key = k
            # If this block uses AIN, norm1.weight → norm1.ln.weight
            if model_key not in model_dict:
                ain_key = k.replace(".norm1.weight", ".norm1.ln.weight") \
                           .replace(".norm1.bias", ".norm1.ln.bias")
                if ain_key in model_dict:
                    model_key = ain_key
            key_map[k] = model_key

    new_state_dict = OrderedDict()
    matched, skipped = [], []

    for ckpt_key, ckpt_val in state_dict.items():
        model_key = key_map.get(ckpt_key, ckpt_key)

        # Skip head/classifier weights from pretrained model
        if "head" in ckpt_key:
            skipped.append(ckpt_key)
            continue

        if model_key not in model_dict:
            skipped.append(ckpt_key)
            continue

        # Handle positional embedding interpolation
        if model_key == "pos_embed":
            if ckpt_val.shape[-1] != model_dict[model_key].shape[-1]:
                skipped.append(ckpt_key + " (embed_dim mismatch)")
                continue
            ckpt_val = _interpolate_pos_embed(ckpt_val, model)

        if model_dict[model_key].shape == ckpt_val.shape:
            new_state_dict[model_key] = ckpt_val
            matched.append(model_key)
        else:
            skipped.append(f"{ckpt_key} ({ckpt_val.shape} vs {model_dict[model_key].shape})")

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    total = len(matched) + len(skipped)
    if matched:
        LOGGER.info(f"Loaded {len(matched)}/{total} pretrained layers from DeiT-Tiny")
    if skipped:
        LOGGER.debug(f"Skipped pretrained layers: {skipped}")
        LOGGER.info(f"Skipped {len(skipped)}/{total} layers (deeper blocks / classifier head)")


# ---------------------------------------------------------------------------
# Factory functions (match osnet_x0_25 convention)
# ---------------------------------------------------------------------------

def vit_nano(num_classes: int = 1000, pretrained: bool = False, loss: str = "softmax", use_gpu: bool = True, **kwargs) -> ViTNano:
    """ViT-Nano (plain): embed_dim=192, depth=6, heads=3. No AIN, no OmniScale.

    Baseline variant for ablation comparison.
    """
    model = ViTNano(num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu,
                    embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                    ain=False, omni_scale=False, **kwargs)
    if pretrained:
        _load_pretrained_deit_tiny(model)
    return model


def vit_nano_ain(num_classes: int = 1000, pretrained: bool = False, loss: str = "softmax", use_gpu: bool = True, **kwargs) -> ViTNano:
    """ViT-Nano + AIN: embed_dim=192, depth=6, heads=3. AIN in first half of blocks.

    Adds Adaptive Instance-LayerNorm for camera/domain style normalization.
    """
    model = ViTNano(num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu,
                    embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                    ain=True, omni_scale=False, **kwargs)
    if pretrained:
        _load_pretrained_deit_tiny(model)
    return model


def vit_nano_ain_os(num_classes: int = 1000, pretrained: bool = False, loss: str = "softmax", use_gpu: bool = True, **kwargs) -> ViTNano:
    """ViT-Nano + AIN + OmniScale: embed_dim=192, depth=6, heads=3.

    Full variant with Adaptive Instance-LayerNorm and OmniScale Aggregation
    (multi-scale spatial pooling with unified aggregation gate).
    """
    model = ViTNano(num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu,
                    embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                    ain=True, omni_scale=True, **kwargs)
    if pretrained:
        _load_pretrained_deit_tiny(model)
    return model


def vit_tiny(num_classes: int = 1000, pretrained: bool = False, loss: str = "softmax", use_gpu: bool = True, **kwargs) -> ViTNano:
    """ViT-Tiny: ~5.5 M params, embed_dim=192, depth=12, heads=3.

    Architecture matches DeiT-Tiny (standard LayerNorm, CLS-token pooling).
    Uses overlapping patches (stride=12, kernel=16) for finer spatial
    resolution (TransReID-style, +3-4% mAP).
    When ``pretrained=True``, loads DeiT-Tiny (ImageNet-1k) weights.
    """
    img_size = kwargs.pop("img_size", (384, 128))
    feat_dim = kwargs.pop("feat_dim", 512)
    model = ViTNano(num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu,
                    img_size=img_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                    ain=False, omni_scale=False, patch_stride=12, feat_dim=feat_dim, **kwargs)
    if pretrained:
        _load_pretrained_deit_tiny(model)
    return model
