"""Shared association utilities grouped by association family."""

from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.association.matching import (
    apply_motion_fusion,
    chi2inv95,
    embedding_distance,
    feature_distance,
    fuse_iou,
    fuse_motion,
    fuse_score,
    gate_cost_matrix,
    iou_distance,
    linear_assignment,
    motion_gating_distance,
    motion_gating_threshold,
    solve_assignment,
)
from boxmot.trackers.common.association.stages import (
    AssociationStage,
    AssociationStageResult,
    all_indices,
    detection_track_similarity_assignment,
    run_association_stage,
)

__all__ = (
    "AssociationFunction",
    "AssociationStage",
    "AssociationStageResult",
    "all_indices",
    "apply_motion_fusion",
    "chi2inv95",
    "detection_track_similarity_assignment",
    "embedding_distance",
    "feature_distance",
    "fuse_iou",
    "fuse_motion",
    "fuse_score",
    "gate_cost_matrix",
    "iou_distance",
    "linear_assignment",
    "motion_gating_distance",
    "motion_gating_threshold",
    "run_association_stage",
    "solve_assignment",
)
