"""Final materialization stage."""

from __future__ import annotations

from typing import Any, Mapping

from ..finalize import finalize_build
from .base import MaterializationContext, StageOutcome


class FinalizeStage:
    """Validate every key and atomically publish the staging directory."""

    name = "finalize"

    def __init__(
        self,
        *,
        embedding_metadata: Mapping[str, Any] | None = None,
        target_shard_rows: int = 50_000,
    ) -> None:
        self.embedding_metadata = embedding_metadata
        self.target_shard_rows = target_shard_rows

    def run(self, context: MaterializationContext) -> StageOutcome:
        if context.stage_plan.name != self.name:
            raise ValueError(f"FinalizeStage cannot execute plan stage {context.stage_plan.name!r}.")
        output = finalize_build(
            context.build_plan,
            embedding_metadata=self.embedding_metadata,
            target_shard_rows=self.target_shard_rows,
            before_publish=lambda: context.state.complete(
                context.stage_plan.name,
                artifacts=("manifest.json", "_SUCCESS"),
            ),
        )
        return StageOutcome(artifacts=(str(output),), metrics={"published": True})


__all__ = ("FinalizeStage",)
