from __future__ import annotations

from typing import Any

from boxmot.postprocessing.base import Postprocessor

_POSTPROCESSORS = ("gsi", "gbrc", "gta")


def supported_postprocessors() -> tuple[str, ...]:
    """Return supported postprocessing step names."""
    return _POSTPROCESSORS


def create_postprocessor(name: str, **kwargs: Any) -> Postprocessor:
    """Create a postprocessor by name."""
    normalized = name.strip().lower()
    if normalized == "gsi":
        from boxmot.postprocessing.gsi import GSIPostprocessor

        return GSIPostprocessor(**kwargs)
    if normalized == "gbrc":
        from boxmot.postprocessing.gbrc import GBRCPostprocessor

        return GBRCPostprocessor(**kwargs)
    if normalized == "gta":
        from boxmot.postprocessing.gta import GTAPostprocessor

        return GTAPostprocessor(**kwargs)
    raise ValueError(
        f"Unknown postprocessing step '{name}'. Valid options: {sorted(_POSTPROCESSORS)}"
    )
