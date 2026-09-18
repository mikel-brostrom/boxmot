"""Canonical values shared by BoxMOT components, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .camera import CameraModel
    from .detections import Detections, Detections3D
    from .frame import Frame
    from .geometry import Boxes, Boxes3D, Geometry, OrientedBoxes
    from .kinds import GeometryKind
    from .masks import MaskBatch
    from .tracks import MultimodalTracks, Tracks, Tracks3D

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
    "Boxes3D": ("boxmot.structures.geometry", "Boxes3D"),
    "CameraModel": ("boxmot.structures.camera", "CameraModel"),
    "Detections": ("boxmot.structures.detections", "Detections"),
    "Detections3D": ("boxmot.structures.detections", "Detections3D"),
    "Frame": ("boxmot.structures.frame", "Frame"),
    "Geometry": ("boxmot.structures.geometry", "Geometry"),
    "GeometryKind": ("boxmot.structures.kinds", "GeometryKind"),
    "MaskBatch": ("boxmot.structures.masks", "MaskBatch"),
    "MultimodalTracks": ("boxmot.structures.tracks", "MultimodalTracks"),
    "OrientedBoxes": ("boxmot.structures.geometry", "OrientedBoxes"),
    "Tracks": ("boxmot.structures.tracks", "Tracks"),
    "Tracks3D": ("boxmot.structures.tracks", "Tracks3D"),
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
