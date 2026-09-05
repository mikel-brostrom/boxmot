from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch

from ._validation import normalize_row_indices, validate_finite, validate_tensor
from .kinds import GeometryKind


@dataclass(frozen=True, slots=True, eq=False)
class Boxes:
    """Axis-aligned ``xyxy`` boxes stored as ``float32[N, 4]``."""

    values: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def is_obb(self) -> bool:
        return False

    def validate(self) -> None:
        validate_tensor(self.values, name="Boxes.values", dtype=torch.float32, ndim=2)
        if self.values.shape[1] != 4:
            raise ValueError(f"Boxes.values must have shape [N, 4], got {tuple(self.values.shape)}.")
        validate_finite(self.values, name="Boxes.values")
        if len(self) and (
            bool((self.values[:, 2] <= self.values[:, 0]).any()) or bool((self.values[:, 3] <= self.values[:, 1]).any())
        ):
            raise ValueError("Boxes.values must satisfy x2 > x1 and y2 > y1.")

    def select(self, indices: torch.Tensor) -> Boxes:
        selected = normalize_row_indices(indices, count=len(self))
        return Boxes(self.values.index_select(0, selected))


@dataclass(frozen=True, slots=True, eq=False)
class OrientedBoxes:
    """Oriented ``cxcywha`` boxes stored as ``float32[N, 5]``.

    Angles are expressed in radians and intentionally remain unwrapped.
    """

    values: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def is_obb(self) -> bool:
        return True

    def validate(self) -> None:
        validate_tensor(self.values, name="OrientedBoxes.values", dtype=torch.float32, ndim=2)
        if self.values.shape[1] != 5:
            raise ValueError(f"OrientedBoxes.values must have shape [N, 5], got {tuple(self.values.shape)}.")
        validate_finite(self.values, name="OrientedBoxes.values")
        if len(self) and bool((self.values[:, 2:4] <= 0).any()):
            raise ValueError("OrientedBoxes.values width and height must be positive.")

    def select(self, indices: torch.Tensor) -> OrientedBoxes:
        selected = normalize_row_indices(indices, count=len(self))
        return OrientedBoxes(self.values.index_select(0, selected))


Geometry: TypeAlias = Boxes | OrientedBoxes


__all__ = ("Boxes", "Geometry", "GeometryKind", "OrientedBoxes")
