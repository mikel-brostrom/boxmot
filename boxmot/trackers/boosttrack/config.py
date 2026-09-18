"""Canonical algorithm settings for BoostTrack."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class BoostTrackConfig(TrackerConfig):
    """Configure confidence boosting, association, and optional appearance features.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        use_cmc: Enable camera-motion compensation; requires image frames.
        min_box_area: Minimum area of emitted track boxes in pixels squared.
        aspect_ratio_thresh: Maximum aspect ratio of emitted track boxes.
        cmc_method: Camera-motion compensation method used when CMC is enabled.
        lambda_iou: Weight of geometric similarity in association.
        lambda_mhd: Weight of Mahalanobis similarity in association.
        lambda_shape: Weight of shape similarity in association.
        use_dlo_boost: Boost confidence of detections similar to existing tracks.
        use_duo_boost: Boost confidence of detections far from existing tracks.
        dlo_boost_coef: Similarity multiplier for basic DLO boosting, used when
            soft boosting and varying thresholds are both disabled. Boosted
            confidence saturates at one.
        s_sim_corr: Use the corrected AABB shape-similarity formula.
        use_rich_s: Combine Mahalanobis, shape, and soft IoU similarities for DLO.
        use_sb: Blend detection confidence with DLO similarity for soft boosting.
        use_vt: Use track-age-dependent similarity thresholds for DLO boosting.
        use_embeddings: Use supplied appearance embeddings, generating missing
            embeddings from image frames with the configured ReID backend.
        det_thresh: Minimum detection confidence for the shared tracking kernel.
    """

    max_age: int = field(default=60, metadata={"ge": 0})
    use_cmc: bool = True
    min_box_area: int = field(default=10, metadata={"ge": 0})
    aspect_ratio_thresh: float = field(default=1.6, metadata={"gt": 0})
    cmc_method: str = field(default="ecc", metadata={"choices": ("ecc", "orb", "sof", "sift")})
    lambda_iou: float = field(default=0.5, metadata={"ge": 0})
    lambda_mhd: float = field(default=0.25, metadata={"ge": 0})
    lambda_shape: float = field(default=0.25, metadata={"ge": 0})
    use_dlo_boost: bool = True
    use_duo_boost: bool = True
    dlo_boost_coef: float = field(default=0.65, metadata={"ge": 0})
    s_sim_corr: bool = False
    use_rich_s: bool = True
    use_sb: bool = True
    use_vt: bool = True
    use_embeddings: bool = True
    det_thresh: float = field(default=0.6, metadata={"ge": 0.0, "le": 1.0})


__all__ = ("BoostTrackConfig",)
