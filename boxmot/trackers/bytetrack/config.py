"""Canonical algorithm settings for ByteTrack."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class ByteTrackConfig(TrackerConfig):
    """Configure ByteTrack's confidence stages and lost-track lifetime.

    Shared history and association settings are inherited from ``TrackerConfig``.
    Match cost thresholds apply to overlap metrics. Centroid matching uses
    ``centroid_max_distance`` for every stage while confidence still ranks pairs.
    Optional first-pass motion fusion adds an independent uncertainty gate.

    Args:
        min_conf: Minimum confidence for the low-score association stage.
        track_thresh: Confidence threshold for the first association pass
            and for creating new tracks.
        match_thresh: Maximum score-fused geometric cost for ordinary
            ByteTrack's first association pass, including when mask
            guidance is enabled.
        second_match_thresh: Maximum geometric cost for low-confidence matches.
        unconfirmed_match_thresh: Maximum score-fused cost for tentative tracks.
        max_lost_frames: Maximum missed tracker updates before a lost track is
            marked removed. Measured directly in frames, without FPS scaling.
        fuse_first_associate: Fuse detection confidence into first-pass matching costs.
        fuse_unconfirmed_associate: Fuse detection confidence into tentative-track costs.
        motion_fusion: First-pass Kalman gating: ``off``, ``gate``, or ``blend``.
        motion_fusion_lambda: Existing-cost weight when blending squared motion distance.
        motion_fusion_only_position: Gate and blend centers instead of full box geometry.
        duplicate_distance_thresh: Maximum IoU distance for suppressing duplicates
            between active and lost tracks. Zero disables duplicate suppression.
    """

    min_conf: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    track_thresh: float = field(default=0.6, metadata={"ge": 0.0, "le": 1.0})
    match_thresh: float = field(default=0.9, metadata={"ge": 0})
    second_match_thresh: float = field(default=0.5, metadata={"ge": 0})
    unconfirmed_match_thresh: float = field(default=0.7, metadata={"ge": 0})
    max_lost_frames: int = field(default=30, metadata={"ge": 0})
    fuse_first_associate: bool = True
    fuse_unconfirmed_associate: bool = True
    motion_fusion: str = field(default="off", metadata={"choices": ("off", "gate", "blend")})
    motion_fusion_lambda: float = field(default=0.98, metadata={"ge": 0.0, "le": 1.0})
    motion_fusion_only_position: bool = False
    duplicate_distance_thresh: float = field(default=0.15, metadata={"ge": 0.0, "le": 1.0})

    use_cmc: bool = False
    cmc_method: str = field(default="sof", metadata={"choices": ("ecc", "orb", "sof", "sift")})
    cmc_min_conf: float | None = field(default=None, metadata={"ge": 0.0, "le": 1.0})

__all__ = ("ByteTrackConfig",)
