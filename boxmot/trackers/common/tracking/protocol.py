from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TrackerProtocol(Protocol):
    """Public contract implemented by tracker classes."""

    name: str
    supports_obb: bool

    def update(
        self,
        dets: np.ndarray,
        img: np.ndarray | None = None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
        *,
        image: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> np.ndarray:
        """Update the tracker and return public track rows."""
        ...

    def reset(self) -> None:
        """Clear all sequence-local tracker state."""
        ...
