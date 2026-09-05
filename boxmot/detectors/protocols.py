"""Runtime protocol for independently usable object detectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from boxmot.structures import Detections, Frame


@dataclass(frozen=True, slots=True)
class DetectorCapabilities:
    """Geometry and enrichment outputs provided directly by a detector."""

    provides_masks: bool = False
    provides_embeddings: bool = False
    supports_aabb: bool = True
    supports_obb: bool = False

    def __post_init__(self) -> None:
        for name in ("provides_masks", "provides_embeddings", "supports_aabb", "supports_obb"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"DetectorCapabilities.{name} must be bool.")
        if not self.supports_aabb and not self.supports_obb:
            raise ValueError("A detector must support at least one geometry type.")


@runtime_checkable
class Detector(Protocol):
    """Detect objects in a batch of canonical frames."""

    @property
    def capabilities(self) -> DetectorCapabilities:
        """Declare detector geometry and enrichment support."""
        ...

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        """Return one detection collection for every input frame, in order."""
        ...


__all__ = ("Detector", "DetectorCapabilities")
