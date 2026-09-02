from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TrackerProtocol(Protocol):
    """Public contract implemented by tracker classes."""

    name: str
    supports_obb: bool
    uses_img: bool
    uses_embs: bool
    supports_masks: bool

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray | None = None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update the tracker and return public track rows."""
        ...

    def reset(self) -> None:
        """Clear all sequence-local tracker state."""
        ...

    def requires_image(
        self,
        dets: np.ndarray,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ) -> bool:
        """Return whether the current update needs an image."""
        ...
