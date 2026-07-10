"""Tracker package public API."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxmot.trackers.bbox.occluboost import OccluBoost as OccluBoost

_EXPORTS = {
    "OccluBoost": ("boxmot.trackers.occluboost", "OccluBoost"),
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
