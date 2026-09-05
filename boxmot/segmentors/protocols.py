"""Runtime protocol for detection-aligned instance segmentors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from boxmot.structures import Detections, Frame, MaskBatch


@runtime_checkable
class Segmentor(Protocol):
    """Produce one full-frame mask for every supplied detection."""

    def segment(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
    ) -> list[MaskBatch]:
        """Return one detection-aligned mask collection per frame."""
        ...


__all__ = ("Segmentor",)
