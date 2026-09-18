"""Factory for canonical instance segmentor components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from boxmot.components.artifacts import require_resolved_artifact
from boxmot.components.registry import LazyComponentRegistry
from boxmot.segmentors.protocols import Segmentor
from boxmot.segmentors.specs import SegmentorSpec

SegmentorFactory = Callable[[SegmentorSpec], Segmentor]

_SEGMENTOR_FACTORIES: LazyComponentRegistry[SegmentorFactory] = LazyComponentRegistry(
    "segmentor",
    {
        "sam": "boxmot.segmentors.backends.sam:create_sam_segmentor",
        "maskrcnn": "boxmot.segmentors.backends.maskrcnn:create_maskrcnn_segmentor",
        "edgetam": "boxmot.segmentors.backends.edgetam:create_edgetam_segmentor",
    },
)


def create_segmentor(spec: SegmentorSpec, *, model: Any | None = None) -> Segmentor:
    """Construct a segmentor, optionally sharing an official EdgeTAM model."""
    if not isinstance(spec, SegmentorSpec):
        raise TypeError(f"spec must be a SegmentorSpec, not {type(spec).__name__}.")
    if model is not None and spec.backend != "edgetam":
        raise ValueError("Model injection through create_segmentor() is supported only for backend='edgetam'.")
    require_resolved_artifact(
        spec.artifact,
        spec.artifact_sha256,
        component=f"Segmentor backend {spec.backend!r}",
    )
    factory = _SEGMENTOR_FACTORIES.resolve(spec.backend)
    if model is None:
        segmentor = factory(spec)
    else:
        from boxmot.segmentors.backends.edgetam import create_edgetam_segmentor

        segmentor = create_edgetam_segmentor(spec, model=model)
    if not isinstance(segmentor, Segmentor):
        raise TypeError(f"Segmentor backend {spec.backend!r} does not implement segment().")
    return segmentor


__all__ = ("create_segmentor",)
