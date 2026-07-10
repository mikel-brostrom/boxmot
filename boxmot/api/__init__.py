# Mikel Brostrom - BoxMOT - AGPL-3.0 license

"""Public BoxMOT Python API."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BoxMOT": ("boxmot.pipeline", "BoxMOT"),
    "Detector": ("boxmot.models.detector", "Detector"),
    "ReIDModel": ("boxmot.models.reid", "ReIDModel"),
}

__all__ = tuple(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
