# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

"""
CSPReID-n: Cross-Stage Partial ReID Nano

A lightweight ReID backbone inspired by:
- YOLO26: C3k2 blocks, SPPF, SiLU activations, efficient downsampling
- LMBN_n: Multi-branch heads (global + partial + channel) for multi-granularity features

Architecture overview:
  Stem → [C3k2 + Downsample] × 4 → SPPF → LightAttention
  ├── Global branch  → BNNeck → 512-d
  └── Partial branch → 2× BNNeck → 2×512-d
  Inference: concat 3 heads → 1536-d feature vector

Target: ~2M parameters, optimized for person/vehicle re-identification.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from boxmot.reid.backbones.lmbn.bnneck import BNNeck3
from boxmot.utils import logger as LOGGER


# ---------------------------------------------------------------------------
# Building blocks (YOLO26-inspired)
# ---------------------------------------------------------------------------


class ConvBNSiLU(nn.Module):
    """Conv2d + BatchNorm + SiLU (standard YOLO conv block)."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise separable convolution: DW 3×3 + PW 1×1."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBNSiLU(in_ch, in_ch, 3, stride=stride, groups=in_ch)
        self.pw = ConvBNSiLU(in_ch, out_ch, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class Bottleneck(nn.Module):
    """Standard bottleneck block: 1×1 reduce → 3×3 → residual."""

    def __init__(self, in_ch, out_ch, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        mid_ch = int(out_ch * expansion)
        self.cv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cv2 = ConvBNSiLU(mid_ch, out_ch, 3, groups=groups)
        self.shortcut = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """CSP Bottleneck with 2 convolutions (YOLO26-style).

    Splits input channels, applies n bottlenecks to one part,
    concatenates all partial outputs, then fuses with a 1×1 conv.
    """

    def __init__(self, in_ch, out_ch, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        mid_ch = int(out_ch * expansion)
        self.cv1 = ConvBNSiLU(in_ch, 2 * mid_ch, 1)
        self.cv2 = ConvBNSiLU((2 + n) * mid_ch, out_ch, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(mid_ch, mid_ch, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = self.cv1(x)
        # Split into two chunks along channel dim
        chunks = y.chunk(2, dim=1)
        outputs = list(chunks)
        # Each bottleneck takes the last output
        current = outputs[-1]
        for bn in self.bottlenecks:
            current = bn(current)
            outputs.append(current)
        return self.cv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (YOLO-style).

    Applies MaxPool2d with kernel k sequentially to produce multi-scale features.
    """

    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        mid_ch = in_ch // 2
        self.cv1 = ConvBNSiLU(in_ch, mid_ch, 1)
        self.cv2 = ConvBNSiLU(mid_ch * 4, out_ch, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class LightSelfAttention(nn.Module):
    """Lightweight channel + spatial attention for ReID feature maps.

    Combines squeeze-excitation (channel) with a depthwise spatial gate.
    Much cheaper than full self-attention but effective on small feature maps.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 16)
        # Channel attention (SE-like)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        # Spatial attention (lightweight depthwise)
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel gate
        ca = self.fc(self.gap(x))
        x = x * ca
        # Spatial gate
        sa = self.spatial(x)
        x = x * sa
        return x


# ---------------------------------------------------------------------------
# Multi-branch head (LMBN-inspired)
# ---------------------------------------------------------------------------


class MultiBranchHead(nn.Module):
    """Multi-granularity feature aggregation head.

    Branches (informed by LMBN ablation — channel splits add <0.06 HOTA):
      - Global: captures holistic appearance via avg pooling
      - Partial (×2): horizontal upper/lower body splits — most discriminative for tracking
    """

    def __init__(self, in_ch, feat_dim, num_classes):
        super().__init__()

        # Pooling strategies
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.partial_pool = nn.AdaptiveAvgPool2d((2, 1))

        # BNNeck heads (input_dim, num_classes, feat_dim)
        self.bn_global = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_part0 = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)
        self.bn_part1 = BNNeck3(in_ch, num_classes, feat_dim, return_f=True)

    def forward(self, x):
        """
        Args:
            x: feature map (B, C, H, W) from backbone

        Returns:
            - Training: ([cls_scores...], features_tensor)
            - Inference: (B, feat_dim * 3) concatenated features
        """
        # Global branch
        g = self.global_pool(x)  # (B, C, 1, 1)
        f_glo = self.bn_global(g)

        # Partial branch (horizontal 2-split: upper/lower body)
        p = self.partial_pool(x)  # (B, C, 2, 1)
        p0 = p[:, :, 0:1, :]
        p1 = p[:, :, 1:2, :]
        f_p0 = self.bn_part0(p0)
        f_p1 = self.bn_part1(p1)

        if not self.training:
            # Concatenate all feature embeddings: 3 × feat_dim
            features = torch.stack(
                [f_glo[0], f_p0[0], f_p1[0]], dim=2
            )
            return features.flatten(1, 2)

        # Training: return (logits_list, features_tensor)
        cls_scores = [f_glo[1], f_p0[1], f_p1[1]]
        feats = torch.stack(
            [f_glo[2], f_p0[2], f_p1[2]], dim=0
        ).mean(dim=0)  # (B, feat_dim)
        return cls_scores, feats


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class CSPReID_n(nn.Module):
    """CSPReID Nano: lightweight multi-granularity ReID model.

    Combines YOLO26's C3k2/SPPF efficiency with LMBN's multi-branch heads.

    Architecture:
        Input 3×384×128 (H×W, same as LMBN)
        Stem: 2× ConvBNSiLU with stride 2       → 64ch, 96×32
        Stage1: C3k2 + Downsample               → 128ch, 48×16
        Stage2: C3k2 + Downsample               → 256ch, 24×8
        Stage3: C3k2×2 + Downsample             → 384ch, 12×4
        Stage4: C3k2 + SPPF + LightAttention    → 512ch, 12×4
        Head: MultiBranchHead                   → 3×512 = 1536-d features

    Parameters: ~5.5M  |  GFLOPs: ~1.8 (at 384×128 input)
    """

    def __init__(self, num_classes, loss="softmax", pretrained=False, use_gpu=True):
        super().__init__()
        self.loss = loss
        self.training = False

        # Stem
        self.stem = nn.Sequential(
            ConvBNSiLU(3, 32, 3, stride=2),    # 128×64
            ConvBNSiLU(32, 64, 3, stride=2),   # 64×32
        )

        # Stage 1: 64 → 128
        self.stage1 = nn.Sequential(
            C3k2(64, 128, n=1, shortcut=True),
            ConvBNSiLU(128, 128, 3, stride=2),  # 32×16
        )

        # Stage 2: 128 → 256
        self.stage2 = nn.Sequential(
            C3k2(128, 256, n=1, shortcut=True),
            ConvBNSiLU(256, 256, 3, stride=2),  # 16×8
        )

        # Stage 3: 256 → 384
        self.stage3 = nn.Sequential(
            C3k2(256, 384, n=2, shortcut=True),
            ConvBNSiLU(384, 384, 3, stride=2),  # 8×4
        )

        # Stage 4: 384 → 512 with SPPF + attention
        self.stage4 = nn.Sequential(
            C3k2(384, 512, n=1, shortcut=True),
            SPPF(512, 512, k=3),
            LightSelfAttention(512, reduction=8),
        )

        # Multi-branch head
        self.head = MultiBranchHead(512, feat_dim=512, num_classes=num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following ReID best practices (Luo et al. 'Bag of Tricks').

        Strategy:
          - Conv2d: Kaiming normal (fan_out) — preserves gradient variance in backward.
          - BatchNorm: gamma=1, beta=0 (standard).
          - Linear (classifiers): normal(0, 0.001) — small logits at start prevent
            early over-confident predictions with label smoothing.
          - Zero-init residual: set gamma=0 in the last BN of each Bottleneck so
            residual blocks start as identity — dramatically smooths early training
            when no pretrained weights are available.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init residual: last BN gamma in each Bottleneck → block = identity at init
        for m in self.modules():
            if isinstance(m, Bottleneck) and m.shortcut:
                nn.init.zeros_(m.cv2.bn.weight)

    def featuremaps(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x

    def forward(self, x):
        x = self.featuremaps(x)
        return self.head(x)


def cspreid_n(num_classes, loss="softmax", pretrained=False, use_gpu=True, **kwargs):
    """Construct CSPReID-n model."""
    model = CSPReID_n(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
    )
    return model


if __name__ == "__main__":
    from torchinfo import summary

    model = CSPReID_n(num_classes=751, loss="softmax")
    model.eval()

    x = torch.randn(2, 3, 384, 128)  # Same input as LMBN
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Feature dim:  {out.shape[1]}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    try:
        summary(model, input_size=(2, 3, 384, 128), col_names=["output_size", "num_params"])
    except ImportError:
        pass
