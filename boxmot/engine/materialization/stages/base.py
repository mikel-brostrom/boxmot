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
    from boxmot.structures import Detections

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
        return self.state.completed_shards(self.stage_plan.name)

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
                completed_shards=self.state.completed_shard_count(self.stage_plan.name),
                items=items,
                rows=rows,
            )

    def validate_completed_shards(
        self,
        *artifact_names: str,
        expected_detections: Mapping[str, Detections] | None = None,
    ) -> None:
        """Reject unsafe resume state before trusting previously written shards."""

        for shard_id in self.completed_shards:
            if not re.fullmatch(r"[0-9]{5,}", shard_id):
                raise ValueError(f"Invalid completed shard ID {shard_id!r}.")
            paths = [
                self.staging_root / ARTIFACT_PATHS[artifact_name] / f"part-{shard_id}.parquet"
                for artifact_name in artifact_names
            ]
            recorded_hashes = self.state.shard_hashes(self.stage_plan.name, shard_id)
            try:
                for artifact_name, path in zip(artifact_names, paths, strict=True):
                    table = read_parquet_artifact(
                        path,
                        artifact_name=artifact_name,
                        box_type=self.build_plan.box_type,
                    )
                    if recorded_hashes.get(artifact_name) != sha256_file(path):
                        raise ValueError(
                            f"Completed shard {shard_id!r} for {artifact_name!r} failed its checkpoint hash."
                        )
                    if expected_detections is not None:
                        expected = expected_detections.get(shard_id)
                        if expected is None or table.num_rows != len(expected):
                            raise ValueError(f"Checkpoint {shard_id!r} does not match its image detections.")
                        sample_ids = set(table.column("sample_id").to_pylist())
                        instance_ids = table.column("instance_id").to_pylist()
                        if (
                            sample_ids != ({expected.sample_id} if len(expected) else set())
                            or len(set(instance_ids)) != len(instance_ids)
                            or set(instance_ids) != set(expected.instance_ids or ())
                        ):
                            raise ValueError(f"Checkpoint {shard_id!r} contains unexpected image or detection IDs.")
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
