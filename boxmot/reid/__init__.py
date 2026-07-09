"""Lazy public ReID package exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ("ReID",)

if TYPE_CHECKING:
    from boxmot.reid.core.reid import ReID


def __getattr__(name: str):
    if name == "ReID":
        return import_module("boxmot.reid.core.reid").ReID
    raise AttributeError(f"module 'boxmot.reid' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
