# BoxMOT AGPL-3.0 license

"""Lazy public ReID core exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ("ReID", "export_formats")

if TYPE_CHECKING:
    from boxmot.reid.core.runtime import ReID


def export_formats():
    """Return supported ReID export formats as the public pandas table."""
    import pandas as pd

    from boxmot.reid.core.formats import REID_EXPORT_FORMAT_COLUMNS, REID_EXPORT_FORMAT_ROWS

    return pd.DataFrame(REID_EXPORT_FORMAT_ROWS, columns=REID_EXPORT_FORMAT_COLUMNS)


def __getattr__(name: str):
    if name == "ReID":
        return import_module("boxmot.reid.core.runtime").ReID
    raise AttributeError(f"module 'boxmot.reid.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
