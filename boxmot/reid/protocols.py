"""Runtime protocol for appearance embedding encoders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from boxmot.structures import Detections, Frame


@dataclass(frozen=True, slots=True)
class EncoderRequirements:
    """Detection enrichments needed before appearance encoding."""

    masks: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.masks, bool):
            raise TypeError("EncoderRequirements.masks must be bool.")


@runtime_checkable
class AppearanceEncoder(Protocol):
    """Encode detection-aligned appearances for one or more frames."""

    @property
    def embedding_dim(self) -> int:
        """Width of every descriptor returned by :meth:`encode`."""
        ...

    @property
    def requirements(self) -> EncoderRequirements:
        """Declare the payloads required to extract appearance crops."""
        ...

    def encode(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
    ) -> list[torch.Tensor]:
        """Return one float32 ``[N, D]`` CPU tensor per input frame."""
        ...


__all__ = ("AppearanceEncoder", "EncoderRequirements")
