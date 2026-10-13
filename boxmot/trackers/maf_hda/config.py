"""Canonical algorithm settings for MafHda."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class MafHdaConfig(TrackerConfig):
    """Configure mask association, KCF appearance, and tracklet recovery.

    Shared history and association settings are inherited from ``TrackerConfig``.
    Centroid association uses ``centroid_max_distance`` in place of overlap
    and area-ratio gates; appearance and motion affinity remain active.

    Args:
        max_age: Maximum frame gap for reconnecting a lost tracklet.
        min_hits: Observations required for confirmation; only current-frame
            observations are emitted.
        iou_threshold: Lower geometry-affinity bound for appearance gating
            and fusion.
        det_thresh: Minimum detection confidence.
        fuse_detection_score: Weight segment-to-track affinity by observation
            confidence while leaving tracklet recovery unchanged.
        velocity_alpha: Previous-velocity weight in the motion update.
        merge_iou_thresh: Mask IoU threshold for merging duplicate instances.
        appearance_lower: Minimum KCF affinity for track-to-track recovery.
        appearance_upper: Strong KCF affinity allowing overlap-based recovery.
            Ignored for centroid association, which uses ``appearance_lower``.
        appearance_gate: Gate KCF evaluation by geometry affinity.
        s2ta_mode: Segment-to-track affinity: ``motion``, ``appearance``, or
            ``maf``. Appearance modes require image pixels.
        t2ta_mode: Track-to-track affinity, using the same modes as
            ``s2ta_mode``.
        template_size: Maximum spatial dimension of the KCF feature template.
        kcf_padding: Context-window size relative to the tracked box.
        kcf_interp_factor: New-observation weight when interpolating KCF updates.
        kcf_sigma: Gaussian correlation-kernel bandwidth.
        kcf_regularization: Ridge regularization for KCF training.
        kcf_output_sigma_factor: Gaussian response bandwidth relative to the template.
        kcf_update_mode: Reinitialize appearance after each accepted match or
            interpolate it with the existing template.
        max_area_ratio: Strict upper ratio between detection and predicted box area.
        min_intersection_ratio: Minimum intersection divided by the smaller box
            area for segment-to-track motion matching.
        recovery_overlap_thresh: Geometry affinity that can rescue a zero-motion
            track-to-track match when appearance remains acceptable.
    """

    max_age: int = field(default=30, metadata={"ge": 1})
    min_hits: int = field(default=1, metadata={"ge": 1})
    iou_threshold: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    asso_func: str = field(default="iou", metadata={"choices": ("iou", "giou", "diou", "ciou", "hmiou", "centroid")})
    det_thresh: float = field(default=0.7, metadata={"ge": 0.0, "le": 1.0})
    fuse_detection_score: bool = False
    velocity_alpha: float = field(default=0.4, metadata={"ge": 0.0, "le": 1.0})
    merge_iou_thresh: float = field(default=0.4, metadata={"gt": 0.0, "le": 1.0})
    appearance_lower: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    appearance_upper: float = field(default=0.7, metadata={"ge": 0.0, "le": 1.0})
    appearance_gate: bool = True
    s2ta_mode: str = field(default="maf", metadata={"choices": ("motion", "appearance", "maf")})
    t2ta_mode: str = field(default="maf", metadata={"choices": ("motion", "appearance", "maf")})
    template_size: int = field(default=96, metadata={"ge": 16, "le": 512})
    kcf_padding: float = field(default=2.5, metadata={"ge": 1.0})
    kcf_interp_factor: float = field(default=0.005, metadata={"gt": 0.0, "le": 1.0})
    kcf_sigma: float = field(default=0.4, metadata={"gt": 0.0})
    kcf_regularization: float = field(default=1e-4, metadata={"gt": 0.0})
    kcf_output_sigma_factor: float = field(default=0.1, metadata={"gt": 0.0})
    kcf_update_mode: str = field(default="reinitialize", metadata={"choices": ("reinitialize", "interpolate")})
    max_area_ratio: float = field(default=2.0, metadata={"gt": 1.0})
    min_intersection_ratio: float = field(default=0.5, metadata={"ge": 0.0, "le": 1.0})
    recovery_overlap_thresh: float = field(default=0.9, metadata={"ge": 0.0, "le": 1.0})

    def __post_init__(self) -> None:
        """Validate the active KCF affinity interval as well as scalar settings."""
        TrackerConfig.__post_init__(self)
        if self.asso_func != "centroid" and self.appearance_lower > self.appearance_upper:
            raise ValueError("appearance_lower must not exceed appearance_upper.")


__all__ = ("MafHdaConfig",)
