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
        lambda_emb_multiplier: Multiplier for the appearance association weight.
        alpha_fixed_emb: Minimum previous-embedding weight in confidence-aware EMA.
        appearance_thresh: Minimum cosine similarity for first-pass appearance rescue.
        appearance_iou_fraction: Fraction of the geometry threshold required for
            appearance rescue; ignored for centroid association.
        use_dlo_boost: Boost confidence of detections similar to existing tracks.
        use_duo_boost: Boost confidence of detections far from existing tracks.
        dlo_boost_coef: Similarity multiplier for basic DLO boosting, used when
            soft boosting and varying thresholds are both disabled. Boosted
            confidence saturates at one.
        s_sim_corr: Use the corrected AABB shape-similarity formula.
        use_rich_s: Combine Mahalanobis, shape, and soft IoU similarities for DLO.
        use_sb: Blend detection confidence with DLO similarity for soft boosting.
        use_vt: Use track-age-dependent similarity thresholds for DLO boosting.
        dlo_soft_alpha: Original detection-confidence weight during soft DLO boosting.
        dlo_soft_power: Exponent applied to similarity during soft DLO boosting.
        dlo_threshold_start: DLO similarity threshold immediately after an observation.
        dlo_threshold_end: Minimum DLO similarity threshold as a track goes missing.
        duo_mahalanobis_thresh: Minimum squared motion distance for DUO candidates.
        duo_iou_thresh: IoU above which competing DUO candidates are grouped.
        track_confidence_decay: Per-frame factor used to discount track confidence.
        track_confidence_warmup: Track age at which the initial confidence ramp ends.
        use_embeddings: Use supplied appearance embeddings, generating missing
            embeddings from image frames with the configured ReID backend.
        motion_fusion: Additional first-pass Kalman gate or motion-cost blend.
        motion_fusion_lambda: Existing-cost weight in the optional motion blend.
        motion_fusion_only_position: Use only centers for the additional motion gate.
        fuse_iou: Fuse appearance similarity with IoU before appearance weighting.
        fuse_detection_score: Fuse detection confidence into first-pass geometry.
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
    lambda_emb_multiplier: float = field(default=1.5, metadata={"ge": 0})
    alpha_fixed_emb: float = field(default=0.95, metadata={"ge": 0.0, "le": 1.0})
    appearance_thresh: float = field(default=0.75, metadata={"ge": -1.0, "le": 1.0})
    appearance_iou_fraction: float = field(default=0.5, metadata={"ge": 0.0, "le": 1.0})
    use_dlo_boost: bool = True
    use_duo_boost: bool = True
    dlo_boost_coef: float = field(default=0.65, metadata={"ge": 0})
    s_sim_corr: bool = False
    use_rich_s: bool = True
    use_sb: bool = True
    use_vt: bool = True
    dlo_soft_alpha: float = field(default=0.65, metadata={"ge": 0.0, "le": 1.0})
    dlo_soft_power: float = field(default=1.5, metadata={"gt": 0})
    dlo_threshold_start: float = field(default=0.95, metadata={"ge": 0.0, "le": 1.0})
    dlo_threshold_end: float = field(default=0.8, metadata={"ge": 0.0, "le": 1.0})
    duo_mahalanobis_thresh: float = field(default=13.2767, metadata={"gt": 0})
    duo_iou_thresh: float = field(default=0.3, metadata={"ge": 0.0, "le": 1.0})
    track_confidence_decay: float = field(default=0.9, metadata={"gt": 0.0, "le": 1.0})
    track_confidence_warmup: int = field(default=7, metadata={"ge": 1})
    use_embeddings: bool = True
    motion_fusion: str = field(default="off", metadata={"choices": ("off", "gate", "blend")})
    motion_fusion_lambda: float = field(default=0.98, metadata={"ge": 0.0, "le": 1.0})
    motion_fusion_only_position: bool = False
    fuse_iou: bool = False
    fuse_detection_score: bool = False
    det_thresh: float = field(default=0.6, metadata={"ge": 0.0, "le": 1.0})

    cmc_min_conf: float | None = field(default=None, metadata={"ge": 0.0, "le": 1.0})

    def __post_init__(self) -> None:
        """Validate scalar fields and the decreasing DLO threshold interval."""
        TrackerConfig.__post_init__(self)
        if self.dlo_threshold_end > self.dlo_threshold_start:
            raise ValueError("dlo_threshold_end must not exceed dlo_threshold_start.")


__all__ = ("BoostTrackConfig",)
