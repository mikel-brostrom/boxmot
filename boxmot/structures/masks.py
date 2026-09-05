from __future__ import annotations

from dataclasses import dataclass

import torch

from ._validation import normalize_row_indices, validate_tensor


@dataclass(frozen=True, slots=True, eq=False)
class MaskBatch:
    """Full-frame boolean masks stored as ``bool[N, H, W]``."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def validate(self) -> None:
        validate_tensor(self.values, name="MaskBatch.values", dtype=torch.bool, ndim=3)
        if self.values.shape[1] <= 0 or self.values.shape[2] <= 0:
            raise ValueError(f"MaskBatch.values height and width must be positive, got {tuple(self.values.shape[1:])}.")

    @property
    def image_size(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return int(self.values.shape[1]), int(self.values.shape[2])

    def select(self, indices: torch.Tensor) -> MaskBatch:
        selected = normalize_row_indices(indices, count=len(self))
        return MaskBatch(self.values.index_select(0, selected))


__all__ = ("MaskBatch",)
