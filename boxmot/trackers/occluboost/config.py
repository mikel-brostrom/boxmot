"""Canonical algorithm settings for OccluBoost."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.boosttrack.config import BoostTrackConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class OccluBoostConfig(BoostTrackConfig):
    """Configure recovery, track confirmation, and occlusion-aware motion updates.

    Shared history and association settings are inherited from ``BoostTrackConfig``.

    Args:
        recovery_appearance_thresh: Minimum cosine similarity for appearance recovery.
        recovery_iou_thresh: Minimum geometric similarity for appearance recovery.
        recovery_max_age: Maximum unmatched age after prediction for AABB recovery.
        feat_alpha: Previous-embedding weight for recovery and second-pass updates;
            higher values retain more history and adapt more slowly.
        track_low_thresh: Lower confidence bound for second-pass detections.
        second_iou_thresh: Minimum geometric similarity for the AABB second pass.
        second_appearance_thresh: Minimum cosine similarity for second-pass matching
            when appearance is enabled.
        second_pass_max_age: Maximum unmatched age for second-pass recovery.
        second_pass_min_hits: Minimum hit streak of tracks eligible for the second pass.
        use_second_pass: Enable low-confidence matching to eligible confirmed tracks.
        new_track_thresh: Minimum detection confidence to create an AABB track.
        confirm_hits: Consecutive matched updates needed to activate tentative tracks.
        instant_confirm_thresh: Confidence that immediately activates a new AABB track.
        tentative_max_age: Maximum unmatched age before a tentative track expires.
        duplicate_iou_thresh: Geometric similarity above which duplicate tracks
            are suppressed, retaining the older track.
        lambda_emb_multiplier: Appearance-weight multiplier in AABB first-pass matching.
        obb_det_thresh: Detection confidence threshold for OBB first-pass matching.
        obb_iou_threshold: Minimum geometric similarity for OBB first-pass matching.
        obb_new_track_thresh: Minimum detection confidence to create an OBB track.
        obb_instant_confirm_thresh: Confidence that immediately activates a new OBB track.
        obb_max_age: Maximum unmatched age before an OBB track expires.
        obb_recovery_max_age: Maximum unmatched age after prediction for OBB recovery.
        obb_second_iou_thresh: Minimum geometric similarity for the OBB second pass.
        det_thresh: Minimum detection confidence for the shared tracking kernel.
    """

    max_age: int = field(default=146, metadata={"ge": 0})
    min_hits: int = field(default=1, metadata={"ge": 0})
    iou_threshold: float = 0.2957128153631725
    min_box_area: int = field(default=73, metadata={"ge": 0})
    aspect_ratio_thresh: float = field(default=1.4888137942764672, metadata={"gt": 0})
    cmc_method: str = field(default="sof", metadata={"choices": ("ecc", "orb", "sof", "sift")})
    lambda_iou: float = field(default=1.0784558316374715, metadata={"ge": 0})
    lambda_mhd: float = field(default=0.304435887183232, metadata={"ge": 0})
    lambda_shape: float = field(default=1.6709449476805447, metadata={"ge": 0})
    use_duo_boost: bool = False
    dlo_boost_coef: float = field(default=1.2061962091907352, metadata={"ge": 0})
    use_rich_s: bool = False
    det_thresh: float = field(default=0.5678626013369781, metadata={"ge": 0.0, "le": 1.0})
    recovery_appearance_thresh: float = 0.6732855110134396
    recovery_iou_thresh: float = 0.24380051350243462
    recovery_max_age: int = field(default=113, metadata={"ge": 0})
    feat_alpha: float = field(default=0.8324072665785186, metadata={"ge": 0.0, "le": 1.0})
    track_low_thresh: float = field(default=0.04473431588067598, metadata={"ge": 0.0, "le": 1.0})
    second_iou_thresh: float = 0.8131671757478834
    second_appearance_thresh: float = 0.364089272226479
    second_pass_max_age: int = field(default=8, metadata={"ge": 0})
    second_pass_min_hits: int = field(default=7, metadata={"ge": 0})
    use_second_pass: bool = True
    new_track_thresh: float = field(default=0.7128242784621849, metadata={"ge": 0.0, "le": 1.0})
    confirm_hits: int = field(default=2, metadata={"gt": 0})
    instant_confirm_thresh: float = field(default=0.6783889178413256, metadata={"ge": 0.0, "le": 1.0})
    tentative_max_age: int = field(default=3, metadata={"ge": 0})
    duplicate_iou_thresh: float = 0.9571823233925608
    lambda_emb_multiplier: float = field(default=2.9476295884842885, metadata={"ge": 0})
    obb_det_thresh: float = field(default=0.2, metadata={"ge": 0.0, "le": 1.0})
    obb_iou_threshold: float = 0.15
    obb_new_track_thresh: float = field(default=0.3, metadata={"ge": 0.0, "le": 1.0})
    obb_instant_confirm_thresh: float = field(default=0.5, metadata={"ge": 0.0, "le": 1.0})
    obb_max_age: int = field(default=30, metadata={"ge": 0})
    obb_recovery_max_age: int = field(default=15, metadata={"ge": 0})
    obb_second_iou_thresh: float = 0.3


__all__ = ("OccluBoostConfig",)
