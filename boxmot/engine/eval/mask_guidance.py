"""Validate the supported temporal mask guidance evaluation path."""

from __future__ import annotations

from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.mask_guidance import validate_mask_guidance_spec


def validate_mask_guidance_tracker(spec: TrackerSpec, *, output_format: str = "mot") -> None:
    """Keep evaluation on the same supported association path as live tracking."""
    validate_mask_guidance_spec(spec)
    if output_format != "mot":
        raise ValueError("Mask guidance evaluates bounding-box tracks, not segmentation outputs; use MOT metrics.")
