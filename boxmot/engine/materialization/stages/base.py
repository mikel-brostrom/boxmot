"""Base protocol and context shared by materialization stages."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.schema import ARTIFACT_PATHS
from boxmot.datasets.storage import read_parquet_artifact

from ..executor import StageExecutor
from ..plan import BuildPlan, StagePlan
from ..state import MaterializationStateStore

if TYPE_CHECKING:
    from ..progress import MaterializationProgressReporter


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Small persisted-facing summary of a completed stage."""

    artifacts: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MaterializationContext:
    """Resources scoped to one stage execution."""

    build_plan: BuildPlan
    stage_plan: StagePlan
    staging_root: Path
    state: MaterializationStateStore
    executor: StageExecutor
    progress: MaterializationProgressReporter | None = None
    repaired_shards: bool = False

    @property
    def completed_shards(self) -> frozenset[str]:
        return frozenset(self.state.state.by_name[self.stage_plan.name].completed_shards)

    def record_shard(
        self,
        shard_id: str,
        *artifact_names: str,
        items: int | None = None,
        rows: int | None = None,
    ) -> None:
        hashes = {
            artifact_name: sha256_file(self.staging_root / ARTIFACT_PATHS[artifact_name] / f"part-{shard_id}.parquet")
            for artifact_name in artifact_names
        }
        self.state.record_shard(self.stage_plan.name, shard_id, hashes=hashes)
        if self.progress is not None:
            self.progress.shard_completed(
                self.stage_plan.name,
                shard_id,
                completed_shards=len(self.completed_shards),
                items=items,
                rows=rows,
            )

    def validate_completed_shards(self, *artifact_names: str) -> None:
        """Reject unsafe resume state before trusting previously written shards."""

        for shard_id in self.completed_shards:
            if not re.fullmatch(r"[0-9]{5,}", shard_id):
                raise ValueError(f"Invalid completed shard ID {shard_id!r}.")
            paths = [
                self.staging_root / ARTIFACT_PATHS[artifact_name] / f"part-{shard_id}.parquet"
                for artifact_name in artifact_names
            ]
            stage_state = self.state.state.by_name[self.stage_plan.name]
            recorded_hashes = {
                artifact: digest
                for recorded_shard, artifact, digest in stage_state.shard_hashes
                if recorded_shard == shard_id
            }
            try:
                for artifact_name, path in zip(artifact_names, paths, strict=True):
                    read_parquet_artifact(
                        path,
                        artifact_name=artifact_name,
                        box_type=self.build_plan.box_type,
                    )
                    if recorded_hashes.get(artifact_name) != sha256_file(path):
                        raise ValueError(
                            f"Completed shard {shard_id!r} for {artifact_name!r} failed its checkpoint hash."
                        )
            except (OSError, ValueError):
                for path in paths:
                    path.unlink(missing_ok=True)
                self.state.discard_shard(self.stage_plan.name, shard_id)
                self.repaired_shards = True


class MaterializationStage(Protocol):
    """A stage implementation bound to a named :class:`StagePlan`."""

    @property
    def name(self) -> str: ...

    def run(self, context: MaterializationContext) -> StageOutcome: ...


@dataclass(slots=True)
class FunctionStage:
    """Adapt a function into a stage for custom local workflows."""

    name: str
    function: Callable[[MaterializationContext], StageOutcome]

    def run(self, context: MaterializationContext) -> StageOutcome:
        return self.function(context)


__all__ = ("FunctionStage", "MaterializationContext", "MaterializationStage", "StageOutcome")
