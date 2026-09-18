"""Canonical algorithm settings for EagerMot."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class EagerMotConfig(TrackerConfig):
    """Configure fusion, 3D matching, and the image-only recovery stage.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        max_age: Consecutive updates without either sensor modality at which
            a track expires.
        min_hits: Observations needed for confirmation after initial warmup.
        iou_threshold: Minimum image IoU for second-stage recovery; 1 disables
            that stage.
        asso_func: Image association geometry; must be ``iou``.
        det_thresh: Minimum confidence for 2D image detections.
        det_thresh_3d: Minimum confidence for 3D detections.
        max_age_2d: Missing-image age at which track confidence starts decaying.
        fusion_iou_threshold: Minimum image IoU for fusing projected 3D boxes
            with 2D detections of the same class.
        first_matching_method: 3D association using ``dist_2d``,
            ``dist_2d_dims``, ``dist_2d_full``, or ``iou_3d``. The full
            distance includes center, dimension, and yaw disagreement.
        distance_threshold: Positive maximum distance for distance-based
            first-stage matching.
        iou_3d_threshold: Minimum volumetric IoU when using ``iou_3d`` matching.
    """

    max_age: int = field(default=3, metadata={"ge": 1})
    min_hits: int = field(default=1, metadata={"ge": 1})
    iou_threshold: float = field(default=0.3, metadata={"ge": 0.0, "le": 1.0})
    asso_func: str = field(default="iou", metadata={"choices": ("iou",)})
    det_thresh: float = field(default=0.0, metadata={"ge": 0.0, "le": 1.0})
    det_thresh_3d: float = field(default=0.0, metadata={"ge": 0.0, "le": 1.0})
    max_age_2d: int = field(default=3, metadata={"ge": 1})
    fusion_iou_threshold: float = field(default=0.01, metadata={"ge": 0.0, "le": 1.0})
    first_matching_method: str = field(
        default="dist_2d_full",
        metadata={"choices": ("dist_2d", "dist_2d_dims", "dist_2d_full", "iou_3d")},
    )
    distance_threshold: float = field(default=3.5, metadata={"gt": 0})
    iou_3d_threshold: float = field(default=0.01, metadata={"ge": 0.0, "le": 1.0})


__all__ = ("EagerMotConfig",)
