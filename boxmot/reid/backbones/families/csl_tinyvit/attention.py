# BoxMOT AGPL-3.0 license

from __future__ import annotations

import itertools

import torch
from torch import nn

__all__ = ["Attention"]


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
        self.scale = key_dim**-0.5
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
            self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
            self.register_buffer("attention_bias_idxs", torch.LongTensor(idxs).view(N, N), persistent=False)
        else:
            idxs_h = []
            idxs_w = []
            for p1 in points:
                for p2 in points:
                    idxs_h.append(p1[0] - p2[0] + resolution[0] - 1)
                    idxs_w.append(p1[1] - p2[1] + resolution[1] - 1)
            self.attention_bias_h = nn.Parameter(torch.zeros(num_heads, 2 * resolution[0] - 1))
            self.attention_bias_w = nn.Parameter(torch.zeros(num_heads, 2 * resolution[1] - 1))
            self.register_buffer("attention_bias_h_idxs", torch.LongTensor(idxs_h).view(N, N), persistent=False)
            self.register_buffer("attention_bias_w_idxs", torch.LongTensor(idxs_w).view(N, N), persistent=False)

    def _attention_bias(self) -> torch.Tensor:
        if self.bias_mode == "absolute":
            return self.attention_biases[:, self.attention_bias_idxs]
        return (
            self.attention_bias_h[:, self.attention_bias_h_idxs] + self.attention_bias_w[:, self.attention_bias_w_idxs]
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
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
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
