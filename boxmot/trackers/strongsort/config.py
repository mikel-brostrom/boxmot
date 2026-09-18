"""Canonical algorithm settings for StrongSort."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class StrongSortConfig(TrackerConfig):
    """Configure StrongSORT's appearance gallery, confirmation, and matching costs.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        min_conf: Minimum detection confidence accepted for tracking.
        max_cos_dist: Maximum cosine distance for appearance-gallery matching.
        max_iou_dist: Maximum geometry distance for fallback association.
        n_init: Consecutive hits required to confirm a track.
        nn_budget: Maximum number of appearance-gallery features stored per track.
        mc_lambda: Appearance-cost weight in the blend with Mahalanobis distance;
            the motion-distance weight is one minus this value.
        ema_alpha: Previous-embedding weight in exponential smoothing; higher values
            retain more history and adapt more slowly.
        det_thresh: Minimum detection confidence for the shared tracking kernel.
    """

    min_conf: float = field(default=0.6, metadata={"ge": 0.0, "le": 1.0})
    max_cos_dist: float = field(default=0.4, metadata={"ge": 0})
    max_iou_dist: float = field(default=0.7, metadata={"ge": 0})
    n_init: int = field(default=3, metadata={"gt": 0})
    nn_budget: int | None = field(default=100, metadata={"gt": 0})
    mc_lambda: float = field(default=0.98, metadata={"ge": 0.0, "le": 1.0})
    ema_alpha: float = field(default=0.9, metadata={"ge": 0.0, "le": 1.0})
    det_thresh: float = field(default=0.3, metadata={"ge": 0.0, "le": 1.0})


__all__ = ("StrongSortConfig",)
