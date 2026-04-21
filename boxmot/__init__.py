"""BoxMOT package metadata and lazy Python API re-exports.

The package root stays intentionally small so CLI/docs tooling can import
``boxmot`` without immediately pulling in the runtime stack. Public Python API
symbols are re-exported lazily from ``boxmot.api`` so both ``import boxmot.api``
and ``from boxmot import Boxmot`` remain supported.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "17.0.0"

_API_EXPORTS = (
    "Boxmot",
    "ExportResult",
    "GenerateResult",
    "ResearchResult",
    "TrackRunResult",
    "TuneResult",
    "TuneTrialResult",
    "ValidationResult",
    "evaluate",
    "track",
)

__all__ = ("__version__", *_API_EXPORTS)


if TYPE_CHECKING:
    from boxmot.api import (
        Boxmot,
        ExportResult,
        GenerateResult,
        ResearchResult,
        TrackRunResult,
        TuneResult,
        TuneTrialResult,
        ValidationResult,
        evaluate,
        track,
    )


def __getattr__(name: str):
    if name in _API_EXPORTS:
        api_module = import_module("boxmot.api")
        return getattr(api_module, name)
    raise AttributeError(f"module 'boxmot' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
