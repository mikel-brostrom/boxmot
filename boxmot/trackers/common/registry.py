# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from boxmot.structures import GeometryKind
from boxmot.trackers.common.config import get_tracker_config_path
from boxmot.trackers.common.manifest import _TRACKER_MANIFEST
from boxmot.trackers.common.specs import TrackerCapabilities, TrackerFamily, TrackerSpec

_BOX_GEOMETRIES = frozenset({GeometryKind.AABB, GeometryKind.OBB})

_TRACKER_CAPABILITIES: dict[str, TrackerCapabilities] = {
    "boosttrack": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_embeddings=True,
        accepts_frame=True,
    ),
    "botsort": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_embeddings=True,
        accepts_frame=True,
    ),
    "bytetrack": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_frame=True,
    ),
    "deepocsort": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_embeddings=True,
        accepts_frame=True,
    ),
    "eagermot": TrackerCapabilities(
        family=TrackerFamily.MULTIMODAL,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        accepts_masks=True,
        accepts_frame=True,
        requires_detections_3d=True,
        accepts_detections_3d=True,
        requires_camera=True,
        accepts_camera=True,
        accepts_ego_motion=True,
    ),
    "hybridsort": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_embeddings=True,
        accepts_frame=True,
    ),
    "maf_hda": TrackerCapabilities(
        family=TrackerFamily.MULTIMODAL,
        geometry_kinds=frozenset({GeometryKind.AABB}),
        requires_masks=True,
        accepts_masks=True,
        accepts_frame=True,
    ),
    "occluboost": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_embeddings=True,
        accepts_frame=True,
    ),
    "ocsort": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_frame=True,
    ),
    "sfsort": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        accepts_frame=True,
    ),
    "strongsort": TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=_BOX_GEOMETRIES,
        requires_embeddings=True,
        accepts_embeddings=True,
        requires_frame=True,
        accepts_frame=True,
    ),
}

if _TRACKER_CAPABILITIES.keys() != _TRACKER_MANIFEST.keys():
    missing = sorted(_TRACKER_MANIFEST.keys() - _TRACKER_CAPABILITIES.keys())
    extra = sorted(_TRACKER_CAPABILITIES.keys() - _TRACKER_MANIFEST.keys())
    raise RuntimeError(f"Tracker capability registry mismatch: missing={missing}, extra={extra}.")


@dataclass(frozen=True, slots=True)
class TrackerDefinition:
    """Registered tracker metadata used by factories and workflow adapters."""

    name: str
    class_path: str
    capabilities: TrackerCapabilities
    config_name: str | None = None
    accepts_per_class: bool = True
    native_class_path: str | None = None
    native_geometry_kinds: frozenset[GeometryKind] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, TrackerCapabilities):
            raise TypeError("TrackerDefinition.capabilities must be TrackerCapabilities.")
        if not isinstance(self.accepts_per_class, bool):
            raise TypeError("TrackerDefinition.accepts_per_class must be bool.")
        if not isinstance(self.native_geometry_kinds, frozenset) or any(
            not isinstance(kind, GeometryKind) for kind in self.native_geometry_kinds
        ):
            raise TypeError("TrackerDefinition.native_geometry_kinds must be a frozenset of GeometryKind values.")
        if self.native_class_path is None and self.native_geometry_kinds:
            raise ValueError("TrackerDefinition without a native class cannot declare native geometry kinds.")
        unsupported = self.native_geometry_kinds - self.capabilities.geometry_kinds
        if unsupported:
            values = ", ".join(sorted(kind.value for kind in unsupported))
            raise ValueError(f"Native geometry kinds must be supported by the tracker algorithm: {values}.")

    @property
    def config_path(self) -> Path:
        return get_tracker_config_path(self.config_name or self.name)


TRACKER_DEFINITIONS = {
    name: TrackerDefinition(
        name=name,
        class_path=entry.class_path,
        capabilities=_TRACKER_CAPABILITIES[name],
        native_class_path=entry.native_class_path,
        native_geometry_kinds=(
            frozenset(GeometryKind(mode) for mode in entry.native_geometry_modes)
            if entry.native_class_path is not None
            else frozenset()
        ),
    )
    for name, entry in _TRACKER_MANIFEST.items()
}

TRACKER_MAPPING = {name: definition.class_path for name, definition in TRACKER_DEFINITIONS.items()}
TRACKER_CLASS_SPECS = {
    class_path: TrackerSpec(name=name, backend=backend)
    for name, entry in _TRACKER_MANIFEST.items()
    for class_path, backend in ((entry.class_path, "python"), (entry.native_class_path, "cpp"))
    if class_path is not None
}


def get_tracker_definition(tracker_type: str) -> TrackerDefinition:
    """Return registered metadata for a tracker type."""
    try:
        return TRACKER_DEFINITIONS[tracker_type]
    except KeyError as exc:
        available = ", ".join(TRACKER_MAPPING)
        raise ValueError(f"Unknown tracker type: {tracker_type!r}. Available trackers are: {available}") from exc


def get_tracker_config(tracker_type: str) -> Path:
    """Return the built-in configuration path for a registered tracker."""
    return get_tracker_definition(tracker_type).config_path


def _load_tracker_class(definition: TrackerDefinition):
    module_path, class_name = definition.class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def supported_native_trackers() -> tuple[str, ...]:
    """Return tracker names with registered C++ domain adapters."""

    return tuple(sorted(name for name, definition in TRACKER_DEFINITIONS.items() if definition.native_class_path))


def get_tracker_class(tracker_type: str):
    """Return the lazily imported tracker class for a registered type."""
    return _load_tracker_class(get_tracker_definition(tracker_type))


__all__ = (
    "TRACKER_CLASS_SPECS",
    "TRACKER_DEFINITIONS",
    "TRACKER_MAPPING",
    "TrackerDefinition",
    "get_tracker_class",
    "get_tracker_config",
    "get_tracker_definition",
    "supported_native_trackers",
)
