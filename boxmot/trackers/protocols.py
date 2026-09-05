"""Public tracker protocol independent of detector and engine implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers.specs import TrackerCapabilities


@dataclass(frozen=True, slots=True)
class TrackerRequirements:
    """Inputs required by one resolved tracker configuration."""

    embeddings: bool = False
    masks: bool = False
    frame: bool = False

    def __post_init__(self) -> None:
        for name in ("embeddings", "masks", "frame"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"TrackerRequirements.{name} must be bool.")


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

    def update(self, detections: Detections, frame: Frame | None = None) -> Tracks:
        """Advance the tracker by exactly one frame."""
        ...

    def reset(self) -> None:
        """Clear sequence-local state while preserving configuration."""
        ...


__all__ = ("Tracker", "TrackerRequirements")
