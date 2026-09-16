"""Canonical algorithm settings for SFSORT."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class SFSORTConfig(TrackerConfig):
    """Configure confidence stages, density adjustments, and image regions.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        high_th: Confidence threshold for first-pass detections.
        match_th_first: Maximum geometric cost for the first association
            pass; smaller values require closer matches.
        new_track_th: Minimum confidence for creating a new track.
        low_th: Minimum confidence for second-pass detections.
        match_th_second: Maximum geometric cost for the second association
            pass.
        dynamic_tuning: Adjust thresholds using the current detection count.
        cth: Confidence cutoff for detections counted by dynamic tuning.
        high_th_m: Scale for decreasing ``high_th`` during dynamic tuning.
        new_track_th_m: Scale for increasing ``new_track_th`` during dynamic
            tuning.
        match_th_first_m: Scale for decreasing ``match_th_first`` during
            dynamic tuning.
        obb_theta_damping: Previous angular-update weight in OBB smoothing;
            larger values reduce the influence of the latest angle change.
        marginal_timeout: Frames to retain tracks lost near an image edge.
        central_timeout: Frames to retain tracks lost inside the central
            region.
        frame_width: Optional width in pixels; configure together with
            ``frame_height``, or supply frame dimensions during updates.
        frame_height: Optional height in pixels, paired with ``frame_width``.
        horizontal_margin: Left and right margin size in pixels.
        vertical_margin: Top and bottom margin size in pixels.
    """

    high_th: float | None = field(default=0.6, metadata={"ge": 0.0, "le": 1.0, "none_uses_default": True})
    match_th_first: float | None = field(default=0.67, metadata={"none_uses_default": True})
    new_track_th: float | None = field(default=0.7, metadata={"ge": 0.0, "le": 1.0, "none_uses_default": True})
    low_th: float | None = field(default=0.1, metadata={"ge": 0.0, "le": 1.0, "none_uses_default": True})
    match_th_second: float | None = field(default=0.3, metadata={"none_uses_default": True})
    dynamic_tuning: bool = False
    cth: float | None = field(default=0.5, metadata={"ge": 0.0, "le": 1.0, "none_uses_default": True})
    high_th_m: float | None = field(default=0.0, metadata={"ge": 0, "none_uses_default": True})
    new_track_th_m: float | None = field(default=0.0, metadata={"ge": 0, "none_uses_default": True})
    match_th_first_m: float | None = field(default=0.0, metadata={"ge": 0, "none_uses_default": True})
    obb_theta_damping: float = field(default=0.8, metadata={"ge": 0.0, "le": 1.0})
    marginal_timeout: int | None = field(default=0, metadata={"ge": 0, "none_uses_default": True})
    central_timeout: int | None = field(default=0, metadata={"ge": 0, "none_uses_default": True})
    frame_width: int | None = field(default=None, metadata={"gt": 0})
    frame_height: int | None = field(default=None, metadata={"gt": 0})
    horizontal_margin: int | None = field(default=0, metadata={"ge": 0})
    vertical_margin: int | None = field(default=0, metadata={"ge": 0})

    def __post_init__(self) -> None:
        """Normalize default-valued inputs and validate paired dimensions."""
        TrackerConfig.__post_init__(self)
        if (self.frame_width is None) != (self.frame_height is None):
            raise ValueError("frame_width and frame_height must be configured together.")


__all__ = ("SFSORTConfig",)
