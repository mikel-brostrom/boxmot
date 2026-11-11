"""Independent runtime model and algorithm settings for EdgeTAM tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeTAMConfig(TrackerConfig):
    """Bound prompted object memory and the size of propagation batches."""

    max_objects: int = field(default=32, metadata={"ge": 1})
    batch_size: int = field(default=8, metadata={"ge": 1})


@dataclass(frozen=True, slots=True)
class EdgeTAMModelConfig:
    """Select EdgeTAM weights and inference placement without loading a model."""

    checkpoint: str | Path = "edgetam.pt"
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate the model selection without checking files or hardware."""
        if not isinstance(self.checkpoint, (str, Path)):
            raise TypeError("checkpoint must be a path string or Path.")
        if not str(self.checkpoint).strip():
            raise ValueError("checkpoint must not be empty.")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string.")
        from boxmot.utils.devices import normalize_device

        object.__setattr__(self, "device", normalize_device(self.device))


__all__ = ("EdgeTAMConfig", "EdgeTAMModelConfig")
