"""Canonical algorithm settings for OcSort."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class OcSortConfig(TrackerConfig):
    """Configure observation history and optional low-confidence recovery.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        motion_fusion: First-pass Kalman association mode: off, gate, or blend.
        motion_fusion_lambda: Existing-cost weight when motion fusion blends costs.
        motion_fusion_only_position: Gate centers instead of full box geometry.
        fuse_detection_score: Weight first-pass geometry by detector confidence.
        min_conf: Minimum confidence for the low-score association pass
            when ``use_byte`` is enabled.
        delta_t: Observation lookback in frames for estimating motion
            direction.
        inertia: Weight of the observed velocity-direction term in matching.
        use_byte: Enable a second association pass for detections between
            ``min_conf`` and the shared detection threshold.
        det_thresh: Minimum detection confidence for the shared tracking kernel.
    """

    min_conf: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    delta_t: int = field(default=3, metadata={"gt": 0})
    inertia: float = field(default=0.1, metadata={"ge": 0})
    use_byte: bool = False
    det_thresh: float = field(default=0.6, metadata={"ge": 0.0, "le": 1.0})

    motion_fusion: str = field(default="off", metadata={"choices": ("off", "gate", "blend")})
    motion_fusion_lambda: float = field(default=0.98, metadata={"ge": 0.0, "le": 1.0})
    motion_fusion_only_position: bool = False
    fuse_detection_score: bool = False

    use_cmc: bool = False
    cmc_method: str = field(default="sof", metadata={"choices": ("ecc", "orb", "sof", "sift")})
    cmc_min_conf: float | None = field(default=None, metadata={"ge": 0.0, "le": 1.0})

__all__ = ("OcSortConfig",)
