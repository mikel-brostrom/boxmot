"""Lightweight canonical structure kind declarations."""

from __future__ import annotations

from enum import Enum


class GeometryKind(str, Enum):
    """Canonical geometry representations shared across components."""

    AABB = "aabb"
    OBB = "obb"

    def __str__(self) -> str:
        return self.value


__all__ = ("GeometryKind",)
