"""Offline, resumable construction of immutable keyed dataset builds.

This package owns materialization workflow orchestration, planning, progress,
resumability, and publication. Reusable dataset serialization remains owned by
``boxmot.datasets``.
"""

from .builder import DatasetMaterializer, ExecutorFactory
from .executor import ExecutorSpec, InlineExecutor, PoolStageExecutor, StageExecutor, create_executor
from .finalize import FinalizeError, build_manifest, finalize_build
from .ids import fingerprint, make_build_id, make_instance_id, make_stage_fingerprint
from .metadata_cache import FileMetadataCache, default_source_metadata_cache_path
from .plan import BuildPlan, PublishOptions, StagePlan, default_build_root
from .source import BoundedFrameDecoder, SourceSample, decode_source_sample
from .stages import (
    DetectStage,
    EmbedStage,
    FinalizeStage,
    FunctionStage,
    MaterializationContext,
    MaterializationStage,
    SegmentStage,
    StageOutcome,
    assign_instance_ids,
)
from .state import BuildState, MaterializationStateStore, StageState, StateError

__all__ = (
    "BuildPlan",
    "BuildState",
    "BoundedFrameDecoder",
    "DatasetMaterializer",
    "DetectStage",
    "EmbedStage",
    "ExecutorFactory",
    "ExecutorSpec",
    "FileMetadataCache",
    "FinalizeError",
    "FinalizeStage",
    "FunctionStage",
    "InlineExecutor",
    "MaterializationContext",
    "MaterializationStage",
    "MaterializationStateStore",
    "PoolStageExecutor",
    "PublishOptions",
    "SegmentStage",
    "SourceSample",
    "StageExecutor",
    "StageOutcome",
    "StagePlan",
    "StageState",
    "StateError",
    "assign_instance_ids",
    "build_manifest",
    "create_executor",
    "default_build_root",
    "default_source_metadata_cache_path",
    "decode_source_sample",
    "finalize_build",
    "fingerprint",
    "make_build_id",
    "make_instance_id",
    "make_stage_fingerprint",
)
