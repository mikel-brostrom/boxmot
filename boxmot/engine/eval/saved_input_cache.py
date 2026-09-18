"""Cache generated appearances for evaluation of declared saved detections."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from boxmot import __version__
from boxmot.datasets.annotation_cache import load_cached_annotation
from boxmot.datasets.readers.images import _local_path
from boxmot.reid.protocols import AppearanceEncoder, EncoderRequirements
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Detections, Frame

_SCHEMA = "boxmot.saved-appearance-cache/v1"


def _tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    """Hash canonical tensors without converting integer identifiers to floats."""
    array = value.detach().numpy()
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


class CachedAppearanceEncoder:
    """Reuse frame-aligned embeddings with complete crop and encoder provenance.

    Source file mutation, decoded pixel changes, ordered detection rows and the
    complete resolved encoder spec independently invalidate cached appearances.
    The numeric annotation cache owns atomic publication and corruption recovery.
    Results are private writable tensors; tracker trials never share memory.
    """

    def __init__(self, encoder: AppearanceEncoder, spec: ReIDEncoderSpec, cache_root: str | Path) -> None:
        if spec.artifact is not None and spec.artifact_sha256 is None:
            raise ValueError("Cached appearance encoding requires a resolved artifact SHA-256.")
        if type(encoder.embedding_dim) is not int or encoder.embedding_dim <= 0:
            raise ValueError("Cached appearance encoding requires a positive embedding dimension.")
        self._encoder = encoder
        self._cache_root = Path(cache_root)
        self._identity = {
            "schema": _SCHEMA,
            "boxmot_version": __version__,
            "torch_version": torch.__version__,
            "encoder": asdict(spec),
            "implementation": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
            "embedding_dim": encoder.embedding_dim,
            "requirements": asdict(encoder.requirements),
        }

    @property
    def embedding_dim(self) -> int:
        """Return the wrapped encoder's descriptor width."""
        return self._encoder.embedding_dim

    @property
    def requirements(self) -> EncoderRequirements:
        """Keep crop requirements visible to the tracking pipeline."""
        return self._encoder.requirements

    def _format(self, frame: Frame, detections: Detections) -> str:
        identity = {
            **self._identity,
            "frame": {
                "sample_id": frame.sample_id,
                "sequence_id": frame.sequence_id,
                "frame_index": frame.frame_index,
                "timestamp_s": frame.timestamp_s,
                "image": _tensor_identity(frame.image),
            },
            "detections": {
                "geometry": "obb" if detections.is_obb else "aabb",
                "boxes": _tensor_identity(detections.geometry.values),
                "scores": _tensor_identity(detections.scores),
                "class_ids": _tensor_identity(detections.class_ids),
                "instance_ids": detections.instance_ids,
                "masks": _tensor_identity(detections.masks.values) if self.requirements.masks else None,
            },
        }
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return f"{_SCHEMA}:{hashlib.sha256(payload).hexdigest()}"

    def _validate_embeddings(self, values: torch.Tensor, detections: Detections) -> torch.Tensor:
        detections.with_embeddings(values)
        if values.shape[1] != self.embedding_dim:
            raise ValueError(f"Appearance encoder must return {self.embedding_dim} columns, got {values.shape[1]}.")
        return values

    def encode(self, frames: Sequence[Frame], detections: Sequence[Detections]) -> list[torch.Tensor]:
        """Encode cache misses and return independently owned descriptors in order."""
        if len(frames) != len(detections):
            raise ValueError("Appearance frames and detection batches must be aligned.")
        results = []
        for frame, detection in zip(frames, detections, strict=True):
            frame.validate()
            detection.validate()
            if frame.sample_id != detection.sample_id:
                raise ValueError("Appearance frame and detections must have the same sample_id.")
            if not len(detection):
                results.append(torch.empty((0, self.embedding_dim), dtype=torch.float32))
                continue
            if self.requirements.masks and detection.masks is None:
                raise ValueError("The appearance encoder requires detection masks.")
            if frame.source_uri is None:
                raise ValueError("Cached appearance encoding requires a local image source_uri.")
            source, fragment = _local_path(frame.source_uri, description="appearance source")
            if fragment:
                raise ValueError("Cached saved-detection appearances require individual image files.")

            def read_embeddings(_: Path) -> np.ndarray:
                values = self._encoder.encode((frame,), (detection,))
                if len(values) != 1:
                    raise ValueError("Appearance encoder must return one descriptor batch per frame.")
                return self._validate_embeddings(values[0], detection).detach().numpy()

            cached = load_cached_annotation(
                source,
                reader=read_embeddings,
                format=self._format(frame, detection),
                cache_root=self._cache_root,
            )
            results.append(self._validate_embeddings(torch.from_numpy(cached), detection))
        return results
