"""Exportable attention with real rotary arithmetic and explicit memory masks."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def attend(q: Tensor, k: Tensor, v: Tensor, valid: Tensor | None = None) -> Tensor:
    """Evaluate attention in query blocks to bound the eager attention workspace."""
    outputs = []
    # Image/token dimensions are fixed; only the independent object batch varies.
    for query in q.split(256, dim=-2):
        logits = (query * (q.shape[-1] ** -0.5)) @ k.transpose(-2, -1)
        if valid is not None:
            logits = torch.where(valid[:, None, None, :] > 0, logits, -float("inf"))
        outputs.append(logits.softmax(dim=-1) @ v)
    return torch.cat(outputs, dim=-2)


class ExportAttention(nn.Module):
    """Reuse trained SAM projections without CUDA context managers during export."""

    def __init__(self, source: Any) -> None:
        """Share projection parameters with the source evaluation attention."""
        super().__init__()
        self.q_proj = source.q_proj
        self.k_proj = source.k_proj
        self.v_proj = source.v_proj
        self.out_proj = source.out_proj
        self.num_heads = source.num_heads

    def heads(self, x: Tensor) -> Tensor:
        """Convert batch/token/channel tensors into attention heads."""
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, -1).transpose(1, 2)

    def combine(self, x: Tensor) -> Tensor:
        """Restore projected channels after attention."""
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Apply scaled dot product attention with unchanged learned projections."""
        output = attend(self.heads(self.q_proj(q)), self.heads(self.k_proj(k)), self.heads(self.v_proj(v)))
        return self.out_proj(self.combine(output))


def real_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Apply the upstream interleaved complex rotation using real operations."""
    real, imaginary = x[..., 0::2], x[..., 1::2]
    return torch.stack((real * cosine - imaginary * sine, real * sine + imaginary * cosine), -1).flatten(-2)


class RotaryAttention(ExportAttention):
    """Real-valued RoPE attention for fixed image and padded memory token grids."""

    def __init__(
        self,
        source: Any,
        *,
        image_size: int,
        memory_slots: int = 0,
        tokens_per_memory: int = 0,
    ) -> None:
        """Precompute exactly the upstream rotary frequencies outside the graph."""
        super().__init__(source)
        query_frequencies = source.compute_cis(end_x=image_size, end_y=image_size)
        self.register_buffer("query_cosine", query_frequencies.real.clone())
        self.register_buffer("query_sine", query_frequencies.imag.clone())
        self.memory_slots = memory_slots
        self.tokens_per_memory = tokens_per_memory
        if memory_slots:
            spatial_frequencies = source.freqs_cis_k
            global_tokens = tokens_per_memory - spatial_frequencies.shape[0]
            cosine = torch.cat((torch.ones(global_tokens, spatial_frequencies.shape[1]), spatial_frequencies.real))
            sine = torch.cat((torch.zeros(global_tokens, spatial_frequencies.shape[1]), spatial_frequencies.imag))
            self.register_buffer("key_cosine", cosine.repeat(memory_slots, 1))
            self.register_buffer("key_sine", sine.repeat(memory_slots, 1))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, valid: Tensor | None = None) -> Tensor:
        """Exclude absent slots and leave global memories and object pointers unrotated."""
        q = real_rotary(self.heads(self.q_proj(q)), self.query_cosine, self.query_sine)
        k = self.heads(self.k_proj(k))
        v = self.heads(self.v_proj(v))
        if self.memory_slots:
            spatial_end = self.memory_slots * self.tokens_per_memory
            k = torch.cat(
                (real_rotary(k[..., :spatial_end, :], self.key_cosine, self.key_sine), k[..., spatial_end:, :]),
                dim=-2,
            )
        else:
            k = real_rotary(k, self.query_cosine, self.query_sine)
        return self.out_proj(self.combine(attend(q, k, v, valid)))


class MemoryAttentionLayer(nn.Module):
    """Preserve an official memory layer while masking absent history entries."""

    def __init__(self, source: Any, *, image_size: int, memory_slots: int, tokens_per_memory: int) -> None:
        """Copy the reference layer's evaluation computation and parameters."""
        super().__init__()
        self.source = source
        self.self_attention = RotaryAttention(source.self_attn, image_size=image_size)
        self.cross_attention = RotaryAttention(
            source.cross_attn_image,
            image_size=image_size,
            memory_slots=memory_slots,
            tokens_per_memory=tokens_per_memory,
        )

    def forward(self, x: Tensor, memory: Tensor, image_pos: Tensor, memory_pos: Tensor, valid: Tensor) -> Tensor:
        """Attend to valid histories without changing their softmax normalization."""
        layer = self.source
        normalized = layer.norm1(x)
        query = normalized + image_pos if layer.pos_enc_at_attn else normalized
        x = x + self.self_attention(query, query, normalized)
        normalized = layer.norm2(x)
        query = normalized + image_pos if layer.pos_enc_at_cross_attn_queries else normalized
        keys = memory + memory_pos if layer.pos_enc_at_cross_attn_keys else memory
        x = x + self.cross_attention(query, keys, memory, valid)
        return x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))


class MemoryAttention(nn.Module):
    """Batch-first memory attention accepting masked, fixed-capacity histories."""

    def __init__(self, source: Any, *, image_size: int, memory_slots: int, tokens_per_memory: int) -> None:
        """Build tensor-only layers from the official trained memory transformer."""
        super().__init__()
        self.layers = nn.ModuleList(
            MemoryAttentionLayer(
                layer, image_size=image_size, memory_slots=memory_slots, tokens_per_memory=tokens_per_memory
            )
            for layer in source.layers
        )
        self.norm = source.norm
        self.pos_enc_at_input = source.pos_enc_at_input

    def forward(self, image: Tensor, memory: Tensor, image_pos: Tensor, memory_pos: Tensor, valid: Tensor) -> Tensor:
        """Fuse one independently masked history with each object's current features."""
        output = image + 0.1 * image_pos if self.pos_enc_at_input else image
        for layer in self.layers:
            output = layer(output, memory, image_pos, memory_pos, valid)
        return self.norm(output)
