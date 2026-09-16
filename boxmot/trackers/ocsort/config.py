"""Canonical algorithm settings for OcSort."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class OcSortConfig(TrackerConfig):
    """Configure observation history and optional low-confidence recovery.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
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


__all__ = ("OcSortConfig",)
