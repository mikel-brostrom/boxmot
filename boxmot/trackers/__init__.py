"""Public tracker contracts and factory, resolved lazily."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from boxmot._tracker_exports import _TRACKER_MANIFEST

if TYPE_CHECKING:
    from boxmot.trackers.common.algorithm_config import TrackerConfig as TrackerConfig
    from boxmot.trackers.common.factory import create_tracker as create_tracker
    from boxmot.trackers.common.mask_guidance import MaskGuidance, MaskGuidanceConfig
    from boxmot.trackers.common.protocols import ReIDConfigurableTracker, Tracker, TrackerRequirements
    from boxmot.trackers.common.specs import GeometryKind, TrackerCapabilities, TrackerFamily, TrackerSpec

__all__ = (
    "GeometryKind",
    "MaskGuidance",
    "MaskGuidanceConfig",
    "ReIDConfigurableTracker",
    "Tracker",
    "TrackerCapabilities",
    "TrackerConfig",
    "TrackerFamily",
    "TrackerRequirements",
    "TrackerSpec",
    "create_tracker",
)

_EXPORTS = {
    "TrackerConfig": ("boxmot.trackers.common.algorithm_config", "TrackerConfig"),
    "GeometryKind": ("boxmot.trackers.common.specs", "GeometryKind"),
    "MaskGuidance": ("boxmot.trackers.common.mask_guidance", "MaskGuidance"),
    "MaskGuidanceConfig": ("boxmot.trackers.common.mask_guidance", "MaskGuidanceConfig"),
    "ReIDConfigurableTracker": ("boxmot.trackers.common.protocols", "ReIDConfigurableTracker"),
    "Tracker": ("boxmot.trackers.common.protocols", "Tracker"),
    "TrackerCapabilities": ("boxmot.trackers.common.specs", "TrackerCapabilities"),
    "TrackerFamily": ("boxmot.trackers.common.specs", "TrackerFamily"),
    "TrackerRequirements": ("boxmot.trackers.common.protocols", "TrackerRequirements"),
    "TrackerSpec": ("boxmot.trackers.common.specs", "TrackerSpec"),
    "create_tracker": ("boxmot.trackers.common.factory", "create_tracker"),
}


def _tracker_name_hints() -> dict[str, str]:
    """Guidance for concrete-tracker lookups; never used for resolution.

    Concrete tracker implementations are deliberately absent from this
    namespace (public API is contracts + factory -- see the v24 package
    contract test and issue #2353). These hints only fire on *failed*
    attribute lookups to tell the user where the tracker actually lives, so
    ``hasattr`` stays False and ``__all__`` is untouched.
    """
    hints: dict[str, str] = {}
    for key, entry in _TRACKER_MANIFEST.items():
        module_path, _, class_name = entry.class_path.rpartition(".")
        message = (
            "concrete tracker implementations are not exported from the "
            "'boxmot.trackers' namespace (public API is contracts + factory). "
            f"Use create_tracker(TrackerSpec(name={key!r})), "
            f"from boxmot import {class_name}, "
            f"or from {module_path} import {class_name}."
        )
        hints[class_name] = message
        # Also catch the lowercase manifest key (e.g. ``botsort``).
        hints.setdefault(key, message)
    return hints


_TRACKER_NAME_HINTS = _tracker_name_hints()
del _tracker_name_hints


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        pass
    else:
        value = getattr(import_module(module_name), attribute)
        globals()[name] = value
        return value
    hint = _TRACKER_NAME_HINTS.get(name)
    if hint is not None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}: {hint}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
