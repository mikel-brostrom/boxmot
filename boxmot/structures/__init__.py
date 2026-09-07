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
    from .tracks import Tracks

__all__ = (
    "Boxes",
    "Detections",
    "Frame",
    "Geometry",
    "GeometryKind",
    "MaskBatch",
    "OrientedBoxes",
    "Tracks",
)

_EXPORTS = {
    "Boxes": ("boxmot.structures.geometry", "Boxes"),
    "Detections": ("boxmot.structures.detections", "Detections"),
    "Frame": ("boxmot.structures.frame", "Frame"),
    "Geometry": ("boxmot.structures.geometry", "Geometry"),
    "GeometryKind": ("boxmot.structures.kinds", "GeometryKind"),
    "MaskBatch": ("boxmot.structures.masks", "MaskBatch"),
    "OrientedBoxes": ("boxmot.structures.geometry", "OrientedBoxes"),
    "Tracks": ("boxmot.structures.tracks", "Tracks"),
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
