"""Public tracker contracts and factory, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.trackers.protocols import Tracker, TrackerRequirements
    from boxmot.trackers.specs import GeometryKind, TrackerCapabilities, TrackerFamily, TrackerSpec

__all__ = (
    "GeometryKind",
    "Tracker",
    "TrackerCapabilities",
    "TrackerFamily",
    "TrackerRequirements",
    "TrackerSpec",
    "create_tracker",
)

_EXPORTS = {
    "GeometryKind": ("boxmot.trackers.specs", "GeometryKind"),
    "Tracker": ("boxmot.trackers.protocols", "Tracker"),
    "TrackerCapabilities": ("boxmot.trackers.specs", "TrackerCapabilities"),
    "TrackerFamily": ("boxmot.trackers.specs", "TrackerFamily"),
    "TrackerRequirements": ("boxmot.trackers.protocols", "TrackerRequirements"),
    "TrackerSpec": ("boxmot.trackers.specs", "TrackerSpec"),
    "create_tracker": ("boxmot.trackers.factory", "create_tracker"),
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
