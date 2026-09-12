"""Public tracker contracts and factory, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.trackers.common.protocols import ReIDConfigurableTracker, Tracker, TrackerRequirements
    from boxmot.trackers.common.specs import GeometryKind, TrackerCapabilities, TrackerFamily, TrackerSpec

__all__ = (
    "GeometryKind",
    "ReIDConfigurableTracker",
    "Tracker",
    "TrackerCapabilities",
    "TrackerFamily",
    "TrackerRequirements",
    "TrackerSpec",
    "create_tracker",
)

_EXPORTS = {
    "GeometryKind": ("boxmot.trackers.common.specs", "GeometryKind"),
    "ReIDConfigurableTracker": ("boxmot.trackers.common.protocols", "ReIDConfigurableTracker"),
    "Tracker": ("boxmot.trackers.common.protocols", "Tracker"),
    "TrackerCapabilities": ("boxmot.trackers.common.specs", "TrackerCapabilities"),
    "TrackerFamily": ("boxmot.trackers.common.specs", "TrackerFamily"),
    "TrackerRequirements": ("boxmot.trackers.common.protocols", "TrackerRequirements"),
    "TrackerSpec": ("boxmot.trackers.common.specs", "TrackerSpec"),
    "create_tracker": ("boxmot.trackers.common.factory", "create_tracker"),
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
