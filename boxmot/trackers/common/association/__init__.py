"""Shared association utilities grouped by association family."""

from boxmot.trackers.common.association.iou import AssociationFunction, iou_obb_pair
from boxmot.trackers.common.association.matching import (
    chi2inv95,
    embedding_distance,
    fuse_iou,
    fuse_motion,
    fuse_score,
    iou_distance,
    linear_assignment,
)
from boxmot.trackers.common.association.stages import (
    AssociationStage,
    AssociationStageResult,
    all_indices,
    detection_track_iou_assignment,
    detection_track_tuple_to_association_result,
    run_association_stage,
)

__all__ = (
    "AssociationFunction",
    "AssociationStage",
    "AssociationStageResult",
    "all_indices",
    "chi2inv95",
    "detection_track_iou_assignment",
    "detection_track_tuple_to_association_result",
    "embedding_distance",
    "fuse_iou",
    "fuse_motion",
    "fuse_score",
    "iou_distance",
    "iou_obb_pair",
    "linear_assignment",
    "run_association_stage",
)
