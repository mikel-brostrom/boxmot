"""Bounding-box tracker public API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.trackers.bbox.boosttrack import BoostTrack
    from boxmot.trackers.bbox.botsort import BotSort
    from boxmot.trackers.bbox.bytetrack import ByteTrack
    from boxmot.trackers.bbox.deepocsort import DeepOcSort
    from boxmot.trackers.bbox.hybridsort import HybridSort
    from boxmot.trackers.bbox.occluboost import OccluBoost
    from boxmot.trackers.bbox.ocsort import OcSort
    from boxmot.trackers.bbox.sfsort import SFSORT
    from boxmot.trackers.bbox.strongsort import StrongSort

_EXPORTS = {
    "BoostTrack": ("boxmot.trackers.bbox.boosttrack", "BoostTrack"),
    "BotSort": ("boxmot.trackers.bbox.botsort", "BotSort"),
    "ByteTrack": ("boxmot.trackers.bbox.bytetrack", "ByteTrack"),
    "DeepOcSort": ("boxmot.trackers.bbox.deepocsort", "DeepOcSort"),
    "HybridSort": ("boxmot.trackers.bbox.hybridsort", "HybridSort"),
    "OccluBoost": ("boxmot.trackers.bbox.occluboost", "OccluBoost"),
    "OcSort": ("boxmot.trackers.bbox.ocsort", "OcSort"),
    "SFSORT": ("boxmot.trackers.bbox.sfsort", "SFSORT"),
    "StrongSort": ("boxmot.trackers.bbox.strongsort", "StrongSort"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
