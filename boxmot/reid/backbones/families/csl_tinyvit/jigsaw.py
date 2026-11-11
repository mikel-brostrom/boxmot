# BoxMOT AGPL-3.0 license

"""Training-only Jigsaw Patch Module for CSL-TinyViT ReID models."""

from __future__ import annotations

import torch
from torch import nn

from boxmot.reid.backbones.heads.bnneck import BNNeck

__all__ = ["JigsawPatchAuxiliary"]


class JigsawPatchAuxiliary(nn.Module):
    """Apply TransReID-style shift/shuffle supervision to spatial tokens.

    The module is deliberately auxiliary: it returns four local training
    features and classifiers, while the retrieval head remains unchanged.
    CSL-TinyViT has no class token, so the per-image global pooled feature is
    used as the shared token prepended to every shuffled group.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        num_groups: int = 4,
        shift: int = 5,
        token_dim: int = 96,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("JPM input_dim must be positive")
        if num_classes < 1:
            raise ValueError("JPM num_classes must be positive")
        if num_groups < 2:
            raise ValueError("JPM num_groups must be at least two")
        if shift < 0:
            raise ValueError("JPM shift must be non-negative")
        if token_dim < 1:
            raise ValueError("JPM token_dim must be positive")
        if num_heads < 1 or token_dim % num_heads:
            raise ValueError(
                "JPM token_dim must be divisible by a positive num_heads"
            )
        if mlp_ratio <= 0:
            raise ValueError("JPM mlp_ratio must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("JPM dropout must be in [0, 1)")

        self.input_dim = int(input_dim)
        self.num_groups = int(num_groups)
        self.shift = int(shift)
        self.token_dim = int(token_dim)

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_projection = nn.Linear(
            self.input_dim,
            self.token_dim,
            bias=False,
        )
        self.shared_block = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(round(self.token_dim * mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(self.token_dim)
        self.necks = nn.ModuleList(
            BNNeck(self.token_dim, num_classes, return_f=True)
            for _ in range(self.num_groups)
        )

    @staticmethod
    def rearrange_patches(
        patches: torch.Tensor,
        *,
        num_groups: int,
        shift: int,
    ) -> torch.Tensor:
        """Cyclically shift then TransReID-shuffle a patch-token sequence."""
        if patches.ndim != 3:
            raise ValueError(
                "JPM patches must have shape (batch, tokens, channels)"
            )
        token_count = int(patches.shape[1])
        if token_count < 1:
            raise ValueError("JPM requires at least one spatial token")
        if token_count % int(num_groups):
            raise ValueError(
                f"JPM token count {token_count} must be divisible by "
                f"num_groups={num_groups}"
            )
        if shift:
            patches = torch.roll(
                patches,
                shifts=-(int(shift) % token_count),
                dims=1,
            )
        batch_size, _, channels = patches.shape
        return (
            patches.reshape(
                batch_size,
                int(num_groups),
                token_count // int(num_groups),
                channels,
            )
            .transpose(1, 2)
            .contiguous()
            .reshape(batch_size, token_count, channels)
        )

    def forward(
        self,
        spatial_map: torch.Tensor,
        global_map: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return per-group classifier logits and pre-BN metric features."""
        if spatial_map.ndim != 4 or global_map.ndim != 4:
            raise ValueError("JPM expects BCHW spatial and global feature maps")
        if spatial_map.shape[0] != global_map.shape[0]:
            raise ValueError("JPM spatial and global batch sizes must match")
        if spatial_map.shape[1] != self.input_dim:
            raise ValueError(
                f"JPM expected {self.input_dim} input channels, got "
                f"{spatial_map.shape[1]}"
            )
        if global_map.shape[1] != self.input_dim:
            raise ValueError(
                f"JPM expected {self.input_dim} global channels, got "
                f"{global_map.shape[1]}"
            )

        patches = spatial_map.flatten(2).transpose(1, 2)
        patches = self.input_projection(self.input_norm(patches))
        patches = self.rearrange_patches(
            patches,
            num_groups=self.num_groups,
            shift=self.shift,
        )
        batch_size, token_count, _ = patches.shape
        tokens_per_group = token_count // self.num_groups
        groups = patches.reshape(
            batch_size,
            self.num_groups,
            tokens_per_group,
            self.token_dim,
        )

        shared_token = global_map.mean(dim=(2, 3))
        shared_token = self.input_projection(self.input_norm(shared_token))
        shared_token = shared_token[:, None, None, :].expand(
            -1,
            self.num_groups,
            1,
            -1,
        )
        grouped_sequences = torch.cat((shared_token, groups), dim=2)
        grouped_sequences = grouped_sequences.reshape(
            batch_size * self.num_groups,
            tokens_per_group + 1,
            self.token_dim,
        )
        encoded = self.shared_block(grouped_sequences)
        local_features = self.output_norm(encoded[:, 0]).reshape(
            batch_size,
            self.num_groups,
            self.token_dim,
        )

        outputs = tuple(
            neck(local_features[:, group_index])
            for group_index, neck in enumerate(self.necks)
        )
        logits = tuple(output[1] for output in outputs)
        metric_features = tuple(output[2] for output in outputs)
        return logits, metric_features
