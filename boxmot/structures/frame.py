from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ._validation import validate_nonempty_string, validate_tensor


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One canonical RGB frame.

    ``image`` is an unbatched, CPU-contiguous ``torch.uint8`` tensor in RGB
    channel-first layout: ``[3, height, width]``. Construction never casts,
    moves, or copies the supplied tensor.
    """

    image: torch.Tensor
    sample_id: str
    sequence_id: str | None = None
    frame_index: int | None = None
    timestamp_s: float | None = None
    source_uri: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_tensor(self.image, name="Frame.image", dtype=torch.uint8, ndim=3)
        if self.image.shape[0] != 3:
            raise ValueError(f"Frame.image must have shape [3, H, W], got {tuple(self.image.shape)}.")
        if self.image.shape[1] <= 0 or self.image.shape[2] <= 0:
            raise ValueError(f"Frame.image height and width must be positive, got {tuple(self.image.shape[1:])}.")

        validate_nonempty_string(self.sample_id, name="Frame.sample_id")
        if self.sequence_id is not None:
            validate_nonempty_string(self.sequence_id, name="Frame.sequence_id")
        if self.frame_index is not None:
            if isinstance(self.frame_index, bool) or not isinstance(self.frame_index, int):
                raise TypeError(f"Frame.frame_index must be an int, got {type(self.frame_index).__name__}.")
            if self.frame_index < 0:
                raise ValueError("Frame.frame_index must be non-negative.")
        if self.timestamp_s is not None:
            if not isinstance(self.timestamp_s, float):
                raise TypeError(f"Frame.timestamp_s must be a float, got {type(self.timestamp_s).__name__}.")
            if not math.isfinite(self.timestamp_s):
                raise ValueError("Frame.timestamp_s must be finite.")
        if self.source_uri is not None:
            validate_nonempty_string(self.source_uri, name="Frame.source_uri")

    @property
    def height(self) -> int:
        return int(self.image.shape[1])

    @property
    def width(self) -> int:
        return int(self.image.shape[2])

    @property
    def image_size(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return self.height, self.width
