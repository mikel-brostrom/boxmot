"""Composable stages for offline perception materialization."""

from .base import FunctionStage, MaterializationContext, MaterializationStage, StageOutcome
from .detect import DetectStage, assign_instance_ids
from .embed import EmbedStage
from .finalize import FinalizeStage
from .segment import SegmentStage

__all__ = (
    "DetectStage",
    "EmbedStage",
    "FinalizeStage",
    "FunctionStage",
    "MaterializationContext",
    "MaterializationStage",
    "SegmentStage",
    "StageOutcome",
    "assign_instance_ids",
)
