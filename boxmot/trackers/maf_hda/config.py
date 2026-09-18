"""Canonical algorithm settings for MafHda."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class MafHdaConfig(TrackerConfig):
    """Configure mask association, KCF appearance, and tracklet recovery.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        max_age: Maximum frame gap for reconnecting a lost tracklet.
        min_hits: Observations required for confirmation; only current-frame
            observations are emitted.
        iou_threshold: Lower geometry-affinity bound for appearance gating
            and fusion.
        det_thresh: Minimum detection confidence.
        velocity_alpha: Previous-velocity weight in the motion update.
        merge_iou_thresh: Mask IoU threshold for merging duplicate instances.
        appearance_lower: Minimum KCF affinity for track-to-track recovery.
        appearance_upper: Strong KCF affinity allowing overlap-based recovery.
        appearance_gate: Gate KCF evaluation by geometry affinity.
        s2ta_mode: Segment-to-track affinity: ``motion``, ``appearance``, or
            ``maf``. Appearance modes require image pixels.
        t2ta_mode: Track-to-track affinity, using the same modes as
            ``s2ta_mode``.
        template_size: Maximum spatial dimension of the KCF feature template.
    """

    max_age: int = field(default=30, metadata={"ge": 1})
    min_hits: int = field(default=1, metadata={"ge": 1})
    iou_threshold: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    asso_func: str = field(default="iou", metadata={"choices": ("iou", "giou", "diou", "ciou", "hmiou", "centroid")})
    det_thresh: float = field(default=0.7, metadata={"ge": 0.0, "le": 1.0})
    velocity_alpha: float = field(default=0.4, metadata={"ge": 0.0, "le": 1.0})
    merge_iou_thresh: float = field(default=0.4, metadata={"gt": 0.0, "le": 1.0})
    appearance_lower: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    appearance_upper: float = field(default=0.7, metadata={"ge": 0.0, "le": 1.0})
    appearance_gate: bool = True
    s2ta_mode: str = field(default="maf", metadata={"choices": ("motion", "appearance", "maf")})
    t2ta_mode: str = field(default="maf", metadata={"choices": ("motion", "appearance", "maf")})
    template_size: int = field(default=96, metadata={"ge": 16, "le": 512})

    def __post_init__(self) -> None:
        """Validate the KCF affinity interval as well as scalar settings."""
        TrackerConfig.__post_init__(self)
        if self.appearance_lower > self.appearance_upper:
            raise ValueError("appearance_lower must not exceed appearance_upper.")


__all__ = ("MafHdaConfig",)
