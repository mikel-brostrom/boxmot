"""Canonical algorithm settings for HybridSort."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class HybridSortConfig(TrackerConfig):
    """Configure HybridSORT association, confidence prediction, and appearance memory.

    Defaults match the historical MOT17 baseline. Shared history and association
    settings are inherited from ``TrackerConfig``; SportsMOT tuning is available
    through the ``hybridsort-sportsmot-val`` preset.

    Args:
        cmc_method: Camera-motion compensation method; None disables CMC.
        use_embeddings: Use supplied appearance embeddings, generating missing
            embeddings from image frames with the configured ReID backend.
        low_thresh: Lower detection confidence bound for second-pass matching.
        delta_t: Observation lookback in frames for estimating motion direction.
        inertia: Weight of observed velocity direction in AABB matching.
        use_byte: Enable a second association pass for low-confidence detections.
        longterm_bank_length: Number of appearance features retained per AABB track.
        alpha: Previous-embedding weight in feature smoothing; higher values
            retain more history. Adaptive smoothing also incorporates confidence.
        adapfs: Enable confidence-adaptive feature smoothing for AABB tracks.
        track_thresh: Clamp separating high and low AABB confidence predictions;
            does not select input detections.
        eg_weight_high_score: Appearance-distance weight for high-confidence matches.
        eg_weight_low_score: Appearance-distance weight for low-confidence AABB matches.
        tcm_first_step: Include motion-direction and confidence-consistency cues in
            the first AABB association pass; geometry and appearance remain active.
        tcm_byte_step: Add a confidence-difference penalty to low-score AABB matching.
        tcm_byte_step_weight: Weight of that low-score confidence-difference penalty.
        with_longterm_reid: Include the AABB long-term appearance bank during matching.
        longterm_reid_weight: Contribution of long-term appearance distance in AABB matching.
        with_longterm_reid_correction: Reject AABB matches using appearance-distance gates.
        longterm_reid_correction_thresh: Appearance-distance gate for high-score matches;
            in OBB mode, good appearance may rescue a poor geometry match.
        longterm_reid_correction_thresh_low: Appearance-distance gate for low-score
            AABB matches when correction is enabled.
        det_thresh: Minimum detection confidence for the shared tracking kernel.
    """

    cmc_method: str | None = field(default="ecc", metadata={"choices": ("ecc", "orb", "sof", "sift")})
    use_embeddings: bool = True
    low_thresh: float = field(default=0.1, metadata={"ge": 0.0, "le": 1.0})
    delta_t: int = field(default=3, metadata={"gt": 0})
    inertia: float = field(default=0.05, metadata={"ge": 0})
    use_byte: bool = True
    longterm_bank_length: int = field(default=30, metadata={"gt": 0})
    alpha: float = field(default=0.9, metadata={"ge": 0.0, "le": 1.0})
    adapfs: bool = False
    track_thresh: float = field(default=0.5, metadata={"ge": 0.0, "le": 1.0})
    eg_weight_high_score: float = field(default=4.6, metadata={"ge": 0})
    eg_weight_low_score: float = field(default=1.3, metadata={"ge": 0})
    tcm_first_step: bool = True
    tcm_byte_step: bool = True
    tcm_byte_step_weight: float = field(default=1.0, metadata={"ge": 0})
    with_longterm_reid: bool = True
    longterm_reid_weight: float = field(default=0.0, metadata={"ge": 0})
    with_longterm_reid_correction: bool = True
    longterm_reid_correction_thresh: float = field(default=0.4, metadata={"ge": 0})
    longterm_reid_correction_thresh_low: float = field(default=0.4, metadata={"ge": 0})
    det_thresh: float = field(default=0.3, metadata={"ge": 0.0, "le": 1.0})


__all__ = ("HybridSortConfig",)
