"""Factory for canonical detector components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, overload

from boxmot.components.artifacts import require_resolved_artifact
from boxmot.components.registry import LazyComponentRegistry
from boxmot.detectors._capabilities import capabilities_from_spec
from boxmot.detectors._model_names import DetectorName
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


@overload
def create_detector(
    spec: DetectorName,
    *,
    device: str | None = None,
    precision: str | None = None,
    preprocessing: str | None = None,
    geometry: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_download: bool = True,
) -> Detector: ...


@overload
def create_detector(
    spec: DetectorSpec | str | Path | Mapping[str, Any],
    *,
    device: str | None = None,
    precision: str | None = None,
    preprocessing: str | None = None,
    geometry: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_download: bool = True,
) -> Detector: ...


def create_detector(
    spec: DetectorSpec | str | Path | Mapping[str, Any],
    *,
    device: str | None = None,
    precision: str | None = None,
    preprocessing: str | None = None,
    geometry: str | None = None,
    options: Mapping[str, Any] | None = None,
    allow_download: bool = True,
) -> Detector:
    """Construct a detector from a spec, profile name, YAML, or model artifact.

    Runtime keywords override the resolved settings; ``options`` merges backend
    settings over authored options. References resolve and hash their artifacts
    before construction, downloading missing models when ``allow_download`` is
    true. An existing ``DetectorSpec`` keeps its resolved artifact identity.
    """
    if not isinstance(spec, (DetectorSpec, str, Path, Mapping)):
        raise TypeError(f"spec must be a DetectorSpec, name, Path, or mapping, not {type(spec).__name__}.")
    if isinstance(spec, str) and (not spec or spec != spec.strip()):
        raise ValueError("Detector reference must be a non-empty canonical string.")
    if not isinstance(allow_download, bool):
        raise TypeError("allow_download must be bool.")
    if geometry is not None and geometry not in ("auto", "aabb", "obb"):
        raise ValueError("geometry must be one of: auto, aabb, obb.")

    option_overrides = None
    if options is not None:
        from boxmot.components.resolution import component_options

        option_overrides = component_options(options)
    if not isinstance(spec, DetectorSpec):
        from boxmot.detectors.config import resolve_detector_spec

        spec, _ = resolve_detector_spec(spec, geometry=geometry, allow_download=allow_download)

    overrides: dict[str, Any] = {
        key: value
        for key, value in (("device", device), ("precision", precision), ("preprocessing", preprocessing))
        if value is not None
    }
    if geometry not in (None, "auto"):
        if spec.geometry_mode not in ("auto", geometry):
            raise ValueError(
                f"Detector geometry_mode {spec.geometry_mode!r} does not match requested geometry {geometry!r}."
            )
        overrides["geometry_mode"] = geometry
    if option_overrides is not None:
        overrides["options"] = tuple(sorted({**spec.option_values(), **dict(option_overrides)}.items()))
    if overrides:
        spec = replace(spec, **overrides)
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
