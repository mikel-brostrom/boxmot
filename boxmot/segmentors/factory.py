"""Factory for canonical instance segmentor components."""

from __future__ import annotations

from collections.abc import Callable

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
    },
)


def create_segmentor(spec: SegmentorSpec) -> Segmentor:
    """Construct the segmentor described by ``spec`` through the lazy registry."""
    if not isinstance(spec, SegmentorSpec):
        raise TypeError(f"spec must be a SegmentorSpec, not {type(spec).__name__}.")
    require_resolved_artifact(
        spec.artifact,
        spec.artifact_sha256,
        component=f"Segmentor backend {spec.backend!r}",
    )
    factory = _SEGMENTOR_FACTORIES.resolve(spec.backend)
    segmentor = factory(spec)
    if not isinstance(segmentor, Segmentor):
        raise TypeError(f"Segmentor backend {spec.backend!r} does not implement segment().")
    return segmentor


__all__ = ("create_segmentor",)
