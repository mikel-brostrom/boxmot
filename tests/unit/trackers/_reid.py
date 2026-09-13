"""Canonical appearance encoder fixture shared by Python and native tracker tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from boxmot.reid.protocols import EncoderRequirements
from boxmot.structures import Detections, Frame


class RecordingEncoder:
    """Record complete canonical inputs while returning configured descriptors."""

    requirements = EncoderRequirements()

    def __init__(self, features: np.ndarray) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.embedding_dim = self.features.shape[1]
        self.calls: list[tuple[tuple[Frame, ...], tuple[Detections, ...]]] = []

    def encode(self, frames: Sequence[Frame], detections: Sequence[Detections]) -> list[torch.Tensor]:
        self.calls.append((tuple(frames), tuple(detections)))
        return [torch.from_numpy(self.features[: len(value)].copy()) for value in detections]


class OutputEncoder:
    """Expose a controlled output and a dimension that is read only after encoding."""

    requirements = EncoderRequirements()

    def __init__(self, output: torch.Tensor, dimension: object = 3, *, defer_dimension: bool = False) -> None:
        self.output = output
        self.dimension = dimension
        self.defer_dimension = defer_dimension
        self.calls = 0
        self.dimension_reads = 0

    @property
    def embedding_dim(self) -> int:
        self.dimension_reads += 1
        if self.defer_dimension and not self.calls:
            raise RuntimeError("The dimension is unavailable before the first encode call.")
        return self.dimension

    def encode(self, frames: Sequence[Frame], detections: Sequence[Detections]) -> list[torch.Tensor]:
        self.calls += 1
        return [self.output]


INVALID_ENCODER_OUTPUTS = (
    "declared-zero",
    "declared-bool",
    "declared-text",
    "declared-float",
    "declared-none",
    "width",
    "rows",
    "rank",
    "dtype",
    "device",
    "contiguous",
)


def invalid_output_encoder(case: str) -> OutputEncoder:
    """Build malformed canonical outputs without optional devices or model weights."""
    declarations = {
        "declared-zero": 0,
        "declared-bool": True,
        "declared-text": "3",
        "declared-float": 3.5,
        "declared-none": None,
    }
    if case in declarations:
        return OutputEncoder(torch.ones((2, 3)), declarations[case])
    outputs = {
        "width": lambda: torch.ones((2, 4)),
        "rows": lambda: torch.ones((1, 3)),
        "rank": lambda: torch.ones(6),
        "dtype": lambda: torch.ones((2, 3), dtype=torch.float64),
        "device": lambda: torch.empty((2, 3), device="meta"),
        "contiguous": lambda: torch.ones((3, 2)).t(),
    }
    return OutputEncoder(outputs[case]())
