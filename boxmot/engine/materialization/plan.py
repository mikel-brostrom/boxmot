"""Serializable execution plans for local dataset materialization."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from boxmot import __version__
from boxmot.datasets.manifest import frozen_json_mapping
from boxmot.datasets.schema import SCHEMA_ID, SCHEMA_VERSION, BoxType
from boxmot.utils.config import CONFIG_ID_PATTERN

from .executor import ExecutorKind
from .ids import make_build_id, make_stage_fingerprint

_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_EXECUTION_CONFIG_KEYS = frozenset(
    {
        "executor",
        "max_attempts",
        "prefetch",
        "retry_backoff_s",
        "shard_size",
        "target_shard_rows",
        "workers",
    }
)


def default_build_root() -> Path:
    """Resolve the immutable materialized-dataset root."""

    configured = os.environ.get("BOXMOT_BUILDS_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path("runs") / "materializations"


@dataclass(frozen=True, slots=True)
class PublishOptions:
    """Select which reusable perception products enter the final build."""

    image_references: bool = True
    masks: bool = False
    embeddings: bool = False

    def __post_init__(self) -> None:
        for name in ("image_references", "masks", "embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Publish option {name!r} must be a boolean.")


@dataclass(frozen=True, slots=True)
class StagePlan:
    """One resumable stage in a materialization DAG."""

    name: str
    fingerprint: str
    depends_on: tuple[str, ...] = ()
    batch_size: int = 1
    workers: int = 1
    executor: ExecutorKind = "inline"
    max_attempts: int = 1
    retry_backoff_s: float = 0.0
    config: Mapping[str, Any] = field(default_factory=dict)
    component: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"Invalid stage name {self.name!r}.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fingerprint):
            raise ValueError(f"Stage {self.name!r} has an invalid fingerprint.")
        counts = (self.batch_size, self.workers, self.max_attempts)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
            raise ValueError("Stage batch_size, workers, and max_attempts must be positive.")
        if self.executor not in {"inline", "thread", "process"}:
            raise ValueError(f"Unknown executor kind {self.executor!r}.")
        if (
            isinstance(self.retry_backoff_s, bool)
            or not isinstance(self.retry_backoff_s, (int, float))
            or self.retry_backoff_s < 0
        ):
            raise ValueError("Stage retry_backoff_s must be non-negative.")
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        reserved = set(self.config) & _EXECUTION_CONFIG_KEYS
        if reserved:
            raise ValueError(
                f"Stage config contains execution-only keys {sorted(reserved)!r}; "
                "set them on StagePlan instead so they do not affect content fingerprints."
            )
        object.__setattr__(self, "config", frozen_json_mapping(self.config))
        object.__setattr__(self, "component", frozen_json_mapping(self.component))

    @classmethod
    def create(
        cls,
        name: str,
        *,
        config: Mapping[str, Any] | None = None,
        component: Mapping[str, Any] | None = None,
        upstream_fingerprints: tuple[str, ...] = (),
        depends_on: tuple[str, ...] = (),
        batch_size: int = 1,
        workers: int = 1,
        executor: ExecutorKind = "inline",
        max_attempts: int = 1,
        retry_backoff_s: float = 0.0,
    ) -> "StagePlan":
        resolved_config = {} if config is None else config
        return cls(
            name=name,
            fingerprint=make_stage_fingerprint(
                name,
                config=resolved_config,
                component={} if component is None else component,
                upstream=upstream_fingerprints,
                batch_size=batch_size,
            ),
            depends_on=depends_on,
            batch_size=batch_size,
            workers=workers,
            executor=executor,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            config=resolved_config,
            component={} if component is None else component,
        )


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Complete immutable definition and filesystem layout of one build."""

    build_root: Path
    dataset_name: str
    build_id: str
    box_type: BoxType
    source_fingerprint: str
    publish: PublishOptions
    stages: tuple[StagePlan, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "build_root", Path(self.build_root).expanduser().resolve())
        object.__setattr__(self, "stages", tuple(self.stages))
        if not isinstance(self.publish, PublishOptions):
            raise TypeError("BuildPlan.publish must be PublishOptions.")
        if any(not isinstance(stage, StagePlan) for stage in self.stages):
            raise TypeError("BuildPlan.stages must contain only StagePlan values.")
        if not _NAME.fullmatch(self.dataset_name):
            raise ValueError(f"Invalid dataset name {self.dataset_name!r}.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.build_id):
            raise ValueError(f"Invalid deterministic build ID {self.build_id!r}.")
        if self.box_type not in {"aabb", "obb"}:
            raise ValueError(f"Unknown box type {self.box_type!r}.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_fingerprint):
            raise ValueError("source_fingerprint must be a full SHA-256 digest.")
        object.__setattr__(self, "metadata", frozen_json_mapping(self.metadata))
        self._validate_dag()

    @classmethod
    def create(
        cls,
        *,
        build_root: str | Path | None = None,
        dataset_name: str,
        box_type: BoxType,
        source_fingerprint: str,
        publish: PublishOptions,
        stages: tuple[StagePlan, ...],
        metadata: Mapping[str, Any] | None = None,
    ) -> "BuildPlan":
        resolved_metadata = {} if metadata is None else metadata
        experiment_id = resolved_metadata.get("experiment_id")
        if experiment_id is not None and (
            not isinstance(experiment_id, str) or CONFIG_ID_PATTERN.fullmatch(experiment_id) is None
        ):
            raise ValueError("Build experiment_id must be a lowercase kebab-case config identifier.")
        definition = {
            "schema": SCHEMA_ID,
            "schema_version": SCHEMA_VERSION,
            "code_version": __version__,
            "dataset_name": dataset_name,
            "box_type": box_type,
            "source_fingerprint": source_fingerprint,
            "publish": publish,
            "stages": [
                {
                    "name": stage.name,
                    "fingerprint": stage.fingerprint,
                    "depends_on": stage.depends_on,
                    "batch_size": stage.batch_size,
                    "config": stage.config,
                    "component": stage.component,
                }
                for stage in stages
            ],
        }
        if experiment_id is not None:
            # The filename-derived experiment identity is stable semantic
            # identity: eval, tune, and research require the selected build to
            # carry this exact value. Machine-local config paths and other
            # manifest provenance stay outside the content fingerprint below.
            definition["experiment_id"] = experiment_id
        # ``metadata`` is publication provenance, not an implicit identity
        # extension point.  It deliberately retains useful machine-local
        # locators (for example source roots and authored config paths) in the
        # manifest, while semantic identity is carried only by the explicit
        # fields above.  This keeps an otherwise identical plan stable when a
        # checkout or dataset root is relocated.
        return cls(
            build_root=default_build_root() if build_root is None else Path(build_root),
            dataset_name=dataset_name,
            build_id=make_build_id(definition),
            box_type=box_type,
            source_fingerprint=source_fingerprint,
            publish=publish,
            stages=stages,
            metadata=resolved_metadata,
        )

    @property
    def output_root(self) -> Path:
        return self.build_root / self.build_id

    @property
    def staging_root(self) -> Path:
        return self.build_root / ".staging" / self.build_id

    @property
    def state_path(self) -> Path:
        return self.staging_root / "_materialization_state.json"

    @property
    def stage_by_name(self) -> dict[str, StagePlan]:
        return {stage.name: stage for stage in self.stages}

    def ordered_stages(self) -> tuple[StagePlan, ...]:
        """Return a deterministic topological order."""

        remaining = {stage.name: stage for stage in self.stages}
        completed: set[str] = set()
        ordered: list[StagePlan] = []
        while remaining:
            ready = [stage for stage in self.stages if stage.name in remaining and set(stage.depends_on) <= completed]
            if not ready:
                raise ValueError("Materialization stage graph contains a cycle.")
            for stage in ready:
                ordered.append(stage)
                completed.add(stage.name)
                remaining.pop(stage.name)
        return tuple(ordered)

    def _validate_dag(self) -> None:
        names = [stage.name for stage in self.stages]
        if not names:
            raise ValueError("A materialization plan must contain at least one stage.")
        if len(set(names)) != len(names):
            raise ValueError("Materialization stage names must be unique.")
        known = set(names)
        for stage in self.stages:
            unknown = set(stage.depends_on) - known
            if unknown:
                raise ValueError(f"Stage {stage.name!r} has unknown dependencies: {sorted(unknown)!r}.")
            if stage.name in stage.depends_on:
                raise ValueError(f"Stage {stage.name!r} cannot depend on itself.")
        self.ordered_stages()


__all__ = ("BuildPlan", "PublishOptions", "StagePlan", "default_build_root")
