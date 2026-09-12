"""Canonical immutable dataset contracts, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cached import CachedVisionDataset, DatasetSample
    from .manifest import ArtifactRecord, DatasetManifest, ManifestError, PublishedContent, ShardRecord, StageProvenance
    from .sequence import ImageDataset, ImageSample, MultimodalSequence, SensorFrame

__all__ = (
    "ArtifactRecord",
    "CachedVisionDataset",
    "DatasetSample",
    "DatasetManifest",
    "ImageDataset",
    "ImageSample",
    "ManifestError",
    "MultimodalSequence",
    "PublishedContent",
    "SensorFrame",
    "ShardRecord",
    "StageProvenance",
)

_EXPORTS = {
    "ArtifactRecord": ("boxmot.datasets.manifest", "ArtifactRecord"),
    "CachedVisionDataset": ("boxmot.datasets.cached", "CachedVisionDataset"),
    "DatasetSample": ("boxmot.datasets.cached", "DatasetSample"),
    "DatasetManifest": ("boxmot.datasets.manifest", "DatasetManifest"),
    "ImageDataset": ("boxmot.datasets.sequence", "ImageDataset"),
    "ImageSample": ("boxmot.datasets.sequence", "ImageSample"),
    "ManifestError": ("boxmot.datasets.manifest", "ManifestError"),
    "MultimodalSequence": ("boxmot.datasets.sequence", "MultimodalSequence"),
    "PublishedContent": ("boxmot.datasets.manifest", "PublishedContent"),
    "SensorFrame": ("boxmot.datasets.sequence", "SensorFrame"),
    "ShardRecord": ("boxmot.datasets.manifest", "ShardRecord"),
    "StageProvenance": ("boxmot.datasets.manifest", "StageProvenance"),
}


def __getattr__(name: str) -> Any:
    """Resolve dataset exports without loading readers for configuration imports."""

    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose public names without importing their implementations."""

    return sorted((*globals(), *__all__))
