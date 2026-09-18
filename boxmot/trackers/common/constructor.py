"""Runtime keyword contracts shared by concrete tracker constructors.

Algorithm settings belong in the tracker's immutable config. These optional
keywords describe input selection, class metadata, and guidance components.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from typing_extensions import TypedDict

from boxmot.trackers.common.mask_guidance import MaskGuidance, MaskGuidanceConfig


class TrackerMetadataOptions(TypedDict, total=False):
    """Detector class metadata and class-separated execution."""

    class_ids: Iterable[int] | None
    class_names: Mapping[int, str] | None
    per_class: bool


class BoxTrackerOptions(TrackerMetadataOptions, total=False):
    """Box geometry selection and optional temporal mask guidance."""

    is_obb: bool
    mask_guidance: MaskGuidanceConfig | MaskGuidance | None
    edgetam: Mapping[str, object] | None


def validate_runtime_options(options: Mapping[str, object], *, box: bool = True) -> None:
    """Reject algorithm keywords before they can reach a shared tracker base."""
    contract = BoxTrackerOptions if box else TrackerMetadataOptions
    unexpected = options.keys() - contract.__optional_keys__
    if unexpected:
        name = sorted(unexpected)[0]
        if name in {"kalman", "reid"}:
            component = "ReID" if name == "reid" else name
            raise TypeError(f"This tracker does not accept {component} configuration.")
        raise TypeError(f"unexpected keyword argument {name!r}; pass algorithm settings in the tracker config.")


__all__ = (
    "BoxTrackerOptions",
    "TrackerMetadataOptions",
    "validate_runtime_options",
)
