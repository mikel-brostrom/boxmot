"""Canonical algorithm settings for ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ByteTrackConfig(TrackerConfig):
    """Configure ByteTrack's confidence stages and lost-track buffer.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        min_conf: Minimum confidence for the low-score association stage.
        track_thresh: Confidence threshold for the first association pass
            and for creating new tracks.
        match_thresh: Maximum score-fused geometric cost for ordinary
            ByteTrack's first association pass, including when mask
            guidance is enabled.
        track_buffer: Lost-track retention in frames at 30 FPS, scaled by
            ``frame_rate``. This controls tracking expiry.
        frame_rate: Frame rate used to scale ``track_buffer``.
    """

    min_conf: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    track_thresh: float = field(default=0.6, metadata={"ge": 0.0, "le": 1.0})
    match_thresh: float = field(default=0.9, metadata={"ge": 0})
    track_buffer: int = field(default=30, metadata={"ge": 0})
    frame_rate: int = field(default=30, metadata={"gt": 0})


__all__ = ("ByteTrackConfig",)
