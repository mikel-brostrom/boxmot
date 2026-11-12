"""JSON-ready attribute results with application-owned track provenance."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AttributeSample:
    """One source frame included in a same-track analysis batch."""

    sample_id: str
    frame_index: int | None
    timestamp_s: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return provenance using only JSON-compatible values."""
        return {"sample_id": self.sample_id, "frame_index": self.frame_index, "timestamp_s": self.timestamp_s}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AttributeSample:
        """Restore a sample from a persisted record."""
        return cls(payload["sample_id"], payload.get("frame_index"), payload.get("timestamp_s"))


@dataclass(frozen=True, slots=True)
class AttributeResult:
    """A detached result envelope; model output cannot choose its owner."""

    run_id: str
    sequence_id: str | None
    generation: int
    track_id: int
    class_id: int
    model: str
    samples: tuple[AttributeSample, ...]
    status: str
    values: dict[str, Any] | None = None
    error: str | None = None
    display_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize ownership, provenance, and values without sharing mutable data."""
        return {
            "run_id": self.run_id,
            "sequence_id": self.sequence_id,
            "generation": self.generation,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "model": self.model,
            "samples": [sample.to_dict() for sample in self.samples],
            "status": self.status,
            "values": deepcopy(self.values),
            "error": self.error,
            "display_fields": list(self.display_fields),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AttributeResult:
        """Restore one result snapshot without aliasing persisted values."""
        return cls(
            run_id=payload["run_id"],
            sequence_id=payload.get("sequence_id"),
            generation=payload["generation"],
            track_id=payload["track_id"],
            class_id=payload["class_id"],
            model=payload["model"],
            samples=tuple(AttributeSample.from_dict(sample) for sample in payload["samples"]),
            status=payload["status"],
            values=deepcopy(payload.get("values")),
            error=payload.get("error"),
            display_fields=tuple(payload.get("display_fields", ())),
        )
