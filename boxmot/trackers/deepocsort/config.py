"""Canonical algorithm settings for DeepOcSort."""

from __future__ import annotations

from dataclasses import dataclass, field

from boxmot.trackers.common.algorithm_config import TrackerConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class DeepOcSortConfig(TrackerConfig):
    """Configure motion-direction matching and adaptive appearance weighting.

    Shared history and association settings are inherited from ``TrackerConfig``.

    Args:
        delta_t: Observation lookback in frames for estimating motion direction.
        inertia: Weight of the observed velocity-direction term in matching.
        w_association_emb: Base weight of appearance similarity during matching.
        alpha_fixed_emb: Previous-embedding weight for fully confident detections.
            Lower-confidence updates retain more of the previous embedding.
        aw_param: Similarity-ratio cutoff for reducing ambiguous appearance weights.
        use_embeddings: Use supplied appearance embeddings, generating missing
            embeddings from image frames with the configured ReID backend.
        cmc_off: Disable sparse-optical-flow camera-motion compensation.
        aw_off: Disable adaptive weighting of appearance similarity.
        det_thresh: Minimum detection confidence for the shared tracking kernel.
    """

    delta_t: int = field(default=3, metadata={"gt": 0})
    inertia: float = field(default=0.2, metadata={"ge": 0})
    w_association_emb: float = field(default=0.75, metadata={"ge": 0})
    alpha_fixed_emb: float = field(default=0.95, metadata={"ge": 0.0, "le": 1.0})
    aw_param: float = field(default=0.5, metadata={"ge": 0})
    use_embeddings: bool = True
    cmc_off: bool = False
    aw_off: bool = False
    det_thresh: float = field(default=0.5, metadata={"ge": 0.0, "le": 1.0})


__all__ = ("DeepOcSortConfig",)
