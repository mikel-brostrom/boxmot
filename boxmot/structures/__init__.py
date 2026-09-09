"""Canonical values shared by BoxMOT components, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detections import Detections
    from .frame import Frame
    from .geometry import Boxes, Geometry, OrientedBoxes
    from .kinds import GeometryKind
    from .masks import MaskBatch
    from .spatial import Boxes3D, CameraModel, Detections3D, MultimodalTracks, Tracks3D
    from .tracks import Tracks

__all__ = (
    "Boxes",
    "Boxes3D",
    "CameraModel",
    "Detections",
    "Detections3D",
    "Frame",
    "Geometry",
    "GeometryKind",
    "MaskBatch",
    "MultimodalTracks",
    "OrientedBoxes",
    "Tracks",
    "Tracks3D",
)

_EXPORTS = {
    "Boxes": ("boxmot.structures.geometry", "Boxes"),
    "Boxes3D": ("boxmot.structures.spatial", "Boxes3D"),
    "CameraModel": ("boxmot.structures.spatial", "CameraModel"),
    "Detections": ("boxmot.structures.detections", "Detections"),
    "Detections3D": ("boxmot.structures.spatial", "Detections3D"),
    "Frame": ("boxmot.structures.frame", "Frame"),
    "Geometry": ("boxmot.structures.geometry", "Geometry"),
    "GeometryKind": ("boxmot.structures.kinds", "GeometryKind"),
    "MaskBatch": ("boxmot.structures.masks", "MaskBatch"),
    "MultimodalTracks": ("boxmot.structures.spatial", "MultimodalTracks"),
    "OrientedBoxes": ("boxmot.structures.geometry", "OrientedBoxes"),
    "Tracks": ("boxmot.structures.tracks", "Tracks"),
    "Tracks3D": ("boxmot.structures.spatial", "Tracks3D"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
