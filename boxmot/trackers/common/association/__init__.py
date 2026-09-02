"""Shared association utilities grouped by association family."""

from boxmot.trackers.common.association.iou import AssociationFunction
from boxmot.trackers.common.association.matching import (
    chi2inv95,
    embedding_distance,
    feature_distance,
    fuse_score,
    iou_distance,
    linear_assignment,
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
    "chi2inv95",
    "detection_track_similarity_assignment",
    "embedding_distance",
    "feature_distance",
    "fuse_score",
    "iou_distance",
    "linear_assignment",
    "run_association_stage",
    "solve_assignment",
)
