"""Lightweight detector capability resolution shared by factories and backends."""

from __future__ import annotations

from boxmot.detectors.protocols import DetectorCapabilities
from boxmot.detectors.specs import DetectorSpec


def capabilities_from_spec(spec: DetectorSpec) -> DetectorCapabilities:
    """Resolve static capabilities without importing a model implementation."""

    if not isinstance(spec, DetectorSpec):
        raise TypeError(f"spec must be a DetectorSpec, not {type(spec).__name__}.")
    ultralytics = spec.backend == "ultralytics"
    axis_aligned_only = spec.backend in {"rtdetr", "yolox"}
    artifact = spec.artifact or ""
    provides_masks = ultralytics and any(marker in artifact.lower() for marker in ("-seg", "_seg"))
    supports_aabb = spec.geometry_mode in {"auto", "aabb"}
    supports_obb = spec.geometry_mode == "obb"
    if ultralytics:
        supports_obb = spec.geometry_mode in {"auto", "obb"}
    elif axis_aligned_only:
        supports_obb = False
    return DetectorCapabilities(
        provides_masks=provides_masks,
        provides_embeddings=False,
        supports_aabb=supports_aabb,
        supports_obb=supports_obb,
    )


__all__: tuple[str, ...] = ()
