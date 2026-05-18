"""ViT-Tiny with part-based pooling for person ReID.

Extends the base ViTNano backbone with LMBN-inspired multi-part horizontal
pooling.  Patch tokens are split into ``num_parts`` horizontal strips, each
processed through an independent projection → BNNeck → classifier head.

At inference the global CLS-based feature and all part features are
concatenated, yielding a ``(1 + num_parts) * feat_dim`` embedding that
captures both holistic and local appearance cues (improves retrieval under
partial occlusion).

Training returns multi-branch logits so that every head receives ID
supervision, while triplet / center loss still uses the global embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from boxmot.reid.backbones.vit_nano import (
    ViTNano,
    _load_pretrained_deit_tiny,
)


class ViTTinyParts(ViTNano):
    """ViT-Tiny with part-based horizontal pooling heads."""

    def __init__(
        self,
        num_classes: int = 1000,
        loss: str = "softmax",
        pretrained: bool = False,
        use_gpu: bool = True,
        *,
        num_parts: int = 2,
        **kwargs,
    ):
        # feat_dim / embed_dim needed before super().__init__ for part heads
        feat_dim = kwargs.get("feat_dim") or kwargs.get("embed_dim", 192)
        embed_dim = kwargs.get("embed_dim", 192)

        super().__init__(
            num_classes=num_classes,
            loss=loss,
            pretrained=pretrained,
            use_gpu=use_gpu,
            **kwargs,
        )
        self.num_parts = num_parts

        # Per-part: projection → BNNeck → classifier
        self.part_projs = nn.ModuleList([
            nn.Linear(embed_dim, feat_dim, bias=False) if feat_dim != embed_dim else nn.Identity()
            for _ in range(num_parts)
        ])
        self.part_bns = nn.ModuleList([
            nn.BatchNorm1d(feat_dim) for _ in range(num_parts)
        ])
        self.part_classifiers = nn.ModuleList([
            nn.Linear(feat_dim, num_classes, bias=False) for _ in range(num_parts)
        ])

        # Init part heads
        for bn in self.part_bns:
            bn.bias.requires_grad_(False)
            nn.init.constant_(bn.weight, 1.0)
            nn.init.constant_(bn.bias, 0.0)
        for clf in self.part_classifiers:
            nn.init.normal_(clf.weight, std=0.01)
        for proj in self.part_projs:
            if isinstance(proj, nn.Linear):
                nn.init.trunc_normal_(proj.weight, std=0.02)

    # ------------------------------------------------------------------
    # Part pooling helpers
    # ------------------------------------------------------------------

    def _pool_parts(self, patch_tokens: torch.Tensor) -> list[torch.Tensor]:
        """Split patch tokens into horizontal strips and average-pool each."""
        B, _N, D = patch_tokens.shape
        gh, gw = self.patch_embed.grid_size
        spatial = patch_tokens.transpose(1, 2).reshape(B, D, gh, gw)
        parts = []
        strip_h = gh // self.num_parts
        for i in range(self.num_parts):
            h_start = i * strip_h
            h_end = h_start + strip_h if i < self.num_parts - 1 else gh
            parts.append(spatial[:, :, h_start:h_end, :].mean(dim=[2, 3]))
        return parts

    def _part_features(self, parts: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Project + BN each part; return (bn_feats, raw_projs)."""
        bn_feats, raw_projs = [], []
        for i, p in enumerate(parts):
            pv = self.part_projs[i](p)
            raw_projs.append(pv)
            bn_feats.append(self.part_bns[i](pv))
        return bn_feats, raw_projs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        # Global feature (CLS token)
        global_raw = x[:, 0]
        if self.proj is not None:
            v = self.proj(global_raw)
        else:
            v = global_raw
        feat = self.bottleneck(v)

        # Part features from patch tokens
        patch_tokens = x[:, 1:]
        parts = self._pool_parts(patch_tokens)
        part_bn_feats, _part_raw = self._part_features(parts)

        if not self.training:
            return torch.cat([feat] + part_bn_feats, dim=1)

        y = self.classifier(feat)
        part_logits = [self.part_classifiers[i](part_bn_feats[i]) for i in range(self.num_parts)]

        if self.loss == "softmax":
            return [y] + part_logits
        elif self.loss == "triplet":
            return [y] + part_logits, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def vit_tiny_parts(
    num_classes: int = 1000,
    pretrained: bool = False,
    loss: str = "softmax",
    use_gpu: bool = True,
    **kwargs,
) -> ViTTinyParts:
    """ViT-Tiny + part pooling: ~5.7 M params, embed_dim=192, depth=12, heads=3.

    Overlapping patches (stride=12, kernel=16) for finer spatial resolution.
    Part-based pooling (LMBN-inspired) splits patch tokens into horizontal
    strips with independent BNNeck heads; concatenated at inference for a
    ``(1 + num_parts) * feat_dim`` embedding.
    When ``pretrained=True``, loads DeiT-Tiny (ImageNet-1k) weights.
    """
    img_size = kwargs.pop("img_size", (384, 128))
    feat_dim = kwargs.pop("feat_dim", 512)
    num_parts = kwargs.pop("num_parts", 2)
    model = ViTTinyParts(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu,
        img_size=img_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        ain=False,
        omni_scale=False,
        patch_stride=12,
        feat_dim=feat_dim,
        num_parts=num_parts,
        **kwargs,
    )
    if pretrained:
        _load_pretrained_deit_tiny(model)
    return model
