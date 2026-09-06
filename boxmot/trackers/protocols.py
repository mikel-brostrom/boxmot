"""Public tracker protocol independent of detector and engine implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers.specs import TrackerCapabilities


@dataclass(frozen=True, slots=True)
class TrackerRequirements:
    """Inputs required by one resolved tracker configuration."""

    embeddings: bool = False
    masks: bool = False
    frame: bool = False
    frame_dimensions_only: bool = False

    def __post_init__(self) -> None:
        for name in ("embeddings", "masks", "frame", "frame_dimensions_only"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"TrackerRequirements.{name} must be bool.")
        if self.frame_dimensions_only and not self.frame:
            raise ValueError("TrackerRequirements.frame_dimensions_only requires frame=True.")

    @property
    def frame_pixels(self) -> bool:
        """Return whether the tracker needs decoded source-image pixels."""

        return self.frame and not self.frame_dimensions_only


@runtime_checkable
class Tracker(Protocol):
    """Stateful, single-sequence tracker contract."""

    name: str

    @property
    def capabilities(self) -> TrackerCapabilities:
        """Return static capabilities of the tracker algorithm."""
        ...

    @property
    def requirements(self) -> TrackerRequirements:
        """Return the inputs required by this resolved configuration."""
        ...

    def update(self, detections: Detections | np.ndarray, frame: Frame | None = None) -> Tracks:
        """Advance one frame from canonical detections or packed NumPy rows."""
        ...

    def reset(self) -> None:
        """Clear sequence-local state while preserving configuration."""
        ...


__all__ = ("Tracker", "TrackerRequirements")
