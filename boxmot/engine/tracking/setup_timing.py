"""Stable startup-timing schema shared by tracking runtime and reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SETUP_TIMING_COMPONENTS = (
    "detector_load",
    "tracker_reid_load",
    "reid_adapter",
    "output_prepare",
    "source_first_frame",
)


def normalize_setup_timings_ms(values: Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return startup timings with stable keys and a derived total."""

    source = values or {}
    normalized = {
        key: max(float(source.get(key, 0.0) or 0.0), 0.0)
        for key in SETUP_TIMING_COMPONENTS
    }
    normalized["total"] = sum(normalized.values())
    return normalized


__all__ = ("SETUP_TIMING_COMPONENTS", "normalize_setup_timings_ms")
