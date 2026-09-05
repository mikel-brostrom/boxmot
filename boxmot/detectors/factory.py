"""Factory for canonical detector components."""

from __future__ import annotations

from collections.abc import Callable

from boxmot.components.artifacts import require_resolved_artifact
from boxmot.components.registry import LazyComponentRegistry
from boxmot.detectors._capabilities import capabilities_from_spec
from boxmot.detectors.protocols import Detector, DetectorCapabilities
from boxmot.detectors.specs import DetectorSpec

DetectorFactory = Callable[[DetectorSpec], Detector]

_DETECTOR_FACTORIES: LazyComponentRegistry[DetectorFactory] = LazyComponentRegistry(
    "detector",
    {
        "ultralytics": "boxmot.detectors.backends.ultralytics:UltralyticsDetector",
        "yolox": "boxmot.detectors.backends.yolox:YoloXDetector",
        "rtdetr": "boxmot.detectors.backends.rtdetr:RTDetrDetector",
    },
)


def detector_capabilities(spec: DetectorSpec) -> DetectorCapabilities:
    """Resolve detector capabilities without constructing or importing a model."""

    if not isinstance(spec, DetectorSpec):
        raise TypeError(f"spec must be a DetectorSpec, not {type(spec).__name__}.")
    if spec.backend not in _DETECTOR_FACTORIES.entries:
        available = ", ".join(_DETECTOR_FACTORIES.entries) or "(none)"
        raise ValueError(f"Unknown detector backend {spec.backend!r}. Available backends: {available}.")
    return capabilities_from_spec(spec)


def create_detector(spec: DetectorSpec) -> Detector:
    """Construct the detector described by ``spec`` through the lazy registry."""
    if not isinstance(spec, DetectorSpec):
        raise TypeError(f"spec must be a DetectorSpec, not {type(spec).__name__}.")
    require_resolved_artifact(
        spec.artifact,
        spec.artifact_sha256,
        component=f"Detector backend {spec.backend!r}",
    )
    factory = _DETECTOR_FACTORIES.resolve(spec.backend)
    detector = factory(spec)
    if not isinstance(detector, Detector):
        raise TypeError(f"Detector backend {spec.backend!r} does not implement predict().")
    return detector


__all__ = ("create_detector",)
