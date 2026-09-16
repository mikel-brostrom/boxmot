"""Select a temporal mask backend independently of the box tracker."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from boxmot.segmentors.propagation.weights import is_edgetam_tflite_bundle

if TYPE_CHECKING:
    import torch


class MaskPropagator(Protocol):
    """Temporal state owned by one tracker, with masks keyed by its identities."""

    device: str | torch.device
    max_objects: int
    prompt_overlap: float

    def propagate(
        self,
        frame_index: int,
        frame: np.ndarray,
        active_boxes: Mapping[int, np.ndarray],
        new_boxes: Mapping[int, np.ndarray],
    ) -> dict[int, np.ndarray | torch.Tensor]:
        """Predict masks from prior observations, retaining the backend's device."""
        ...

    def retain_tracks(self, track_ids: set[int]) -> None:
        """Release identities no longer retained by the tracker."""
        ...

    def reset(self) -> None:
        """Release sequence state while retaining the loaded inference models."""
        ...


def mask_propagation_device(checkpoint: str | Path, device: str) -> str:
    """Place LiteRT propagation on CPU independently of other workflow models."""
    return "cpu" if is_edgetam_tflite_bundle(checkpoint) else device


def create_mask_propagator(
    checkpoint: str | Path, *, device: str, max_objects: int, prompt_overlap: float
) -> MaskPropagator:
    """Load the selected EdgeTAM backend lazily on the first tracker update."""
    if is_edgetam_tflite_bundle(checkpoint):
        from boxmot.segmentors.propagation.tflite import EdgeTAMTFLiteMaskPropagator

        return EdgeTAMTFLiteMaskPropagator(
            checkpoint, device=device, max_objects=max_objects, prompt_overlap=prompt_overlap
        )

    from boxmot.segmentors.propagation.edgetam import EdgeTAMMaskPropagator

    return EdgeTAMMaskPropagator(
        checkpoint, device=device, max_objects=max_objects, prompt_overlap=prompt_overlap
    )
