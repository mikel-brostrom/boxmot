"""Public tracker protocol independent of detector and engine implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, overload, runtime_checkable

import numpy as np

from boxmot.reid.specs import ReIDEncoderSpec
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
    supports_variable_dt: bool
    variable_dt: bool

    @property
    def capabilities(self) -> TrackerCapabilities:
        """Return static capabilities of the tracker algorithm."""
        ...

    @property
    def requirements(self) -> TrackerRequirements:
        """Return the inputs required by this resolved configuration."""
        ...

    @property
    def generates_embeddings(self) -> bool:
        """Return whether missing embeddings can be generated from a frame."""
        ...

    def validate_timing(
        self, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None
    ) -> float | None:
        """Validate capture timestamps without advancing sequence state."""
        ...

    @overload
    def update(
        self, detections: Detections, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None
    ) -> Tracks:
        """Advance one frame and return canonical tracks."""
        ...

    @overload
    def update(
        self, detections: np.ndarray, frame: Frame | np.ndarray | None = None, *, timestamp_s: float | None = None
    ) -> np.ndarray:
        """Advance one frame and return packed NumPy track rows."""
        ...

    def update(
        self,
        detections: Detections | np.ndarray,
        frame: Frame | np.ndarray | None = None,
        *,
        timestamp_s: float | None = None,
    ) -> Tracks | np.ndarray:
        """Advance with a Frame or uint8 HWC BGR image; preserve the detection representation."""
        ...

    def reset(self) -> None:
        """Clear sequence-local state while preserving configuration."""
        ...


@runtime_checkable
class ReIDConfigurableTracker(Tracker, Protocol):
    """Optional tracker interface for lazy, tracker-owned ReID inference."""

    def configure_reid(self, spec: ReIDEncoderSpec) -> None:
        """Configure tracker-owned ReID before the first update of a sequence."""
        ...


__all__ = ("ReIDConfigurableTracker", "Tracker", "TrackerRequirements")
