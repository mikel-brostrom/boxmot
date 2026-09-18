"""Atomic resumability state for local materialization runs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Mapping

from boxmot.datasets.manifest import utc_now_iso

from .plan import BuildPlan

StageStatus = Literal["pending", "running", "completed", "failed"]


class StateError(RuntimeError):
    """Raised when a persisted build state cannot be resumed safely."""


@dataclass(frozen=True, slots=True)
class StageState:
    """Persisted progress for one fingerprinted stage."""

    name: str
    fingerprint: str
    depends_on: tuple[str, ...] = ()
    status: StageStatus = "pending"
    completed_shards: tuple[str, ...] = ()
    shard_hashes: tuple[tuple[str, str, str], ...] = ()
    artifacts: tuple[str, ...] = ()
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.name) is None:
            raise StateError(f"Invalid persisted stage name {self.name!r}.")
        if not isinstance(self.fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", self.fingerprint) is None:
            raise StateError(f"Stage {self.name!r} has an invalid persisted fingerprint.")
        if not isinstance(self.depends_on, tuple) or any(
            not isinstance(name, str) or not name for name in self.depends_on
        ):
            raise StateError(f"Stage {self.name!r} has invalid persisted dependencies.")
        if len(set(self.depends_on)) != len(self.depends_on) or self.name in self.depends_on:
            raise StateError(f"Stage {self.name!r} has duplicate or self-referential dependencies.")
        if self.status not in {"pending", "running", "completed", "failed"}:
            raise StateError(f"Unknown persisted stage status {self.status!r}.")
        if not isinstance(self.completed_shards, tuple) or any(
            not isinstance(shard, str) or re.fullmatch(r"[0-9]{5,}", shard) is None for shard in self.completed_shards
        ):
            raise StateError(f"Stage {self.name!r} has invalid persisted shard IDs.")
        if len(set(self.completed_shards)) != len(self.completed_shards):
            raise StateError(f"Stage {self.name!r} has duplicate persisted shard IDs.")
        if not isinstance(self.shard_hashes, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 3
            or not isinstance(item[0], str)
            or item[0] not in self.completed_shards
            or not isinstance(item[1], str)
            or re.fullmatch(r"[a-z][a-z0-9_-]*", item[1]) is None
            or not isinstance(item[2], str)
            or re.fullmatch(r"[0-9a-f]{64}", item[2]) is None
            for item in self.shard_hashes
        ):
            raise StateError(f"Stage {self.name!r} has invalid persisted shard hashes.")
        hash_keys = [(shard_id, artifact) for shard_id, artifact, _ in self.shard_hashes]
        if len(set(hash_keys)) != len(hash_keys):
            raise StateError(f"Stage {self.name!r} has duplicate persisted shard hashes.")
        if not isinstance(self.artifacts, tuple) or any(
            not isinstance(artifact, str) or not artifact for artifact in self.artifacts
        ):
            raise StateError(f"Stage {self.name!r} has invalid persisted artifacts.")
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int) or self.attempts < 0:
            raise StateError(f"Stage {self.name!r} has an invalid persisted attempt count.")
        for field_name in ("started_at", "completed_at", "error"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise StateError(f"Stage {self.name!r} has an invalid persisted {field_name} value.")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "fingerprint": self.fingerprint,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "completed_shards": list(self.completed_shards),
            "shard_hashes": [
                {"shard_id": shard_id, "artifact": artifact, "sha256": digest}
                for shard_id, artifact, digest in self.shard_hashes
            ],
            "artifacts": list(self.artifacts),
            "attempts": self.attempts,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "StageState":
        status = value.get("status", "pending")
        if status not in {"pending", "running", "completed", "failed"}:
            raise StateError(f"Unknown persisted stage status {status!r}.")
        shard_hashes = value.get("shard_hashes", ())
        if not isinstance(shard_hashes, list | tuple):
            raise StateError("Persisted shard_hashes must be a sequence.")
        parsed_hashes: list[tuple[str, str, str]] = []
        for item in shard_hashes:
            if not isinstance(item, Mapping):
                raise StateError("Persisted shard hash entries must be mappings.")
            parsed_hashes.append((item.get("shard_id"), item.get("artifact"), item.get("sha256")))
        return cls(
            name=value["name"],
            fingerprint=value["fingerprint"],
            depends_on=tuple(value.get("depends_on", ())),
            status=status,  # type: ignore[arg-type]
            completed_shards=tuple(value.get("completed_shards", ())),
            shard_hashes=tuple(parsed_hashes),  # type: ignore[arg-type]
            artifacts=tuple(value.get("artifacts", ())),
            attempts=value.get("attempts", 0),
            started_at=value.get("started_at"),
            completed_at=value.get("completed_at"),
            error=value.get("error"),
        )


@dataclass(frozen=True, slots=True)
class BuildState:
    """Persisted state for all stages in one deterministic build."""

    build_id: str
    stages: tuple[StageState, ...]
    state_version: int = 1
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        if isinstance(self.state_version, bool) or not isinstance(self.state_version, int) or self.state_version != 1:
            raise StateError(f"Unsupported materialization state version {self.state_version!r}.")
        if not isinstance(self.build_id, str) or re.fullmatch(r"[0-9a-f]{64}", self.build_id) is None:
            raise StateError(f"Invalid persisted build ID {self.build_id!r}.")
        if not isinstance(self.updated_at, str) or not self.updated_at:
            raise StateError("Persisted materialization state requires an update timestamp.")
        if not isinstance(self.stages, tuple) or any(not isinstance(stage, StageState) for stage in self.stages):
            raise StateError("Persisted materialization state contains invalid stages.")
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise StateError("Persisted materialization stage names must be unique.")

    @property
    def by_name(self) -> dict[str, StageState]:
        return {stage.name: stage for stage in self.stages}

    def replace_stage(self, updated: StageState) -> "BuildState":
        stages = tuple(updated if stage.name == updated.name else stage for stage in self.stages)
        if all(stage.name != updated.name for stage in self.stages):
            raise StateError(f"Unknown materialization stage {updated.name!r}.")
        return replace(self, stages=stages, updated_at=utc_now_iso())

    def to_dict(self) -> dict:
        return {
            "state_version": self.state_version,
            "build_id": self.build_id,
            "updated_at": self.updated_at,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "BuildState":
        version = value.get("state_version", -1)
        if version != 1:
            raise StateError(f"Unsupported materialization state version {version}.")
        return cls(
            state_version=version,
            build_id=value["build_id"],
            updated_at=value["updated_at"],
            stages=tuple(StageState.from_dict(item) for item in value.get("stages", ())),
        )


class MaterializationStateStore:
    """Thread-safe state transitions persisted by atomic file replacement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state: BuildState | None = None

    @property
    def state(self) -> BuildState:
        if self._state is None:
            raise StateError("Materialization state has not been initialized.")
        return self._state

    def initialize(self, plan: BuildPlan) -> BuildState:
        """Create state or validate that persisted state belongs to this plan."""

        with self._lock:
            if self.path.exists():
                try:
                    raw = json.loads(self.path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise StateError(f"Unable to load materialization state: {self.path}") from exc
                try:
                    state = BuildState.from_dict(raw)
                except StateError:
                    raise
                except (KeyError, TypeError, ValueError) as exc:
                    raise StateError(f"Invalid materialization state: {self.path}") from exc
                self._validate_plan(state, plan)
                # A process death while running leaves completed shards reusable,
                # but the stage itself must be entered again.
                recovered = tuple(
                    replace(stage, status="pending", error=None) if stage.status == "running" else stage
                    for stage in state.stages
                )
                state = replace(state, stages=recovered, updated_at=utc_now_iso())
            else:
                state = BuildState(
                    build_id=plan.build_id,
                    stages=tuple(
                        StageState(name=stage.name, fingerprint=stage.fingerprint, depends_on=stage.depends_on)
                        for stage in plan.ordered_stages()
                    ),
                )
            self._state = state
            self._write()
            return state

    def is_completed(self, stage_name: str) -> bool:
        return self._stage(stage_name).status == "completed"

    def begin(self, stage_name: str) -> StageState:
        with self._lock:
            current = self._stage(stage_name)
            if current.status == "completed":
                return current
            incomplete = [name for name in current.depends_on if self._stage(name).status != "completed"]
            if incomplete:
                raise StateError(f"Stage {stage_name!r} has incomplete dependencies: {incomplete!r}.")
            updated = replace(
                current,
                status="running",
                attempts=current.attempts + 1,
                started_at=current.started_at or utc_now_iso(),
                completed_at=None,
                error=None,
            )
            self._replace(updated)
            return updated

    def record_shard(
        self,
        stage_name: str,
        shard_id: str,
        *,
        hashes: Mapping[str, str] | None = None,
    ) -> StageState:
        with self._lock:
            if not shard_id:
                raise StateError("shard_id must not be empty.")
            current = self._stage(stage_name)
            if current.status != "running":
                raise StateError(f"Cannot record a shard while stage {stage_name!r} is {current.status}.")
            completed = current.completed_shards
            if shard_id not in completed:
                completed = (*completed, shard_id)
            shard_hashes = tuple(item for item in current.shard_hashes if item[0] != shard_id)
            shard_hashes += tuple((shard_id, artifact, digest) for artifact, digest in sorted((hashes or {}).items()))
            updated = replace(current, completed_shards=completed, shard_hashes=shard_hashes)
            self._replace(updated)
            return updated

    def replace_shards(
        self,
        stage_name: str,
        shards: Mapping[str, Mapping[str, str]],
    ) -> StageState:
        """Atomically replace one running stage's complete shard checkpoints."""

        with self._lock:
            current = self._stage(stage_name)
            if current.status != "running":
                raise StateError(f"Cannot replace shards while stage {stage_name!r} is {current.status}.")
            completed: list[str] = []
            hashes: list[tuple[str, str, str]] = []
            for shard_id, artifact_hashes in sorted(shards.items()):
                if not isinstance(shard_id, str) or re.fullmatch(r"[0-9]{5,}", shard_id) is None:
                    raise StateError(f"Invalid replacement shard ID {shard_id!r}.")
                completed.append(shard_id)
                for artifact, digest in sorted(artifact_hashes.items()):
                    if not isinstance(artifact, str) or not artifact:
                        raise StateError("Replacement shard artifact names must be non-empty strings.")
                    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                        raise StateError(f"Invalid replacement shard digest for {artifact!r}.")
                    hashes.append((shard_id, artifact, digest))
            updated = replace(
                current,
                completed_shards=tuple(completed),
                shard_hashes=tuple(hashes),
                artifacts=(),
                completed_at=None,
                error=None,
            )
            self._replace(updated)
            return updated

    def discard_shard(self, stage_name: str, shard_id: str) -> StageState:
        """Forget an unreadable staging shard so the owning stage can rebuild it."""

        with self._lock:
            current = self._stage(stage_name)
            completed = tuple(item for item in current.completed_shards if item != shard_id)
            shard_hashes = tuple(item for item in current.shard_hashes if item[0] != shard_id)
            updated = replace(
                current,
                status="running",
                completed_shards=completed,
                shard_hashes=shard_hashes,
                artifacts=(),
                completed_at=None,
                error=None,
            )
            self._replace(updated)
            return updated

    def invalidate(self, stage_names: tuple[str, ...]) -> BuildState:
        """Reset transitive downstream stages after an upstream shard repair."""

        with self._lock:
            requested = set(stage_names)
            unknown = requested - set(self.state.by_name)
            if unknown:
                raise StateError(f"Cannot invalidate unknown stages: {sorted(unknown)!r}.")
            stages = tuple(
                replace(
                    stage,
                    status="pending",
                    completed_shards=(),
                    shard_hashes=(),
                    artifacts=(),
                    started_at=None,
                    completed_at=None,
                    error=None,
                )
                if stage.name in requested
                else stage
                for stage in self.state.stages
            )
            self._state = replace(self.state, stages=stages, updated_at=utc_now_iso())
            self._write()
            return self.state

    def complete(self, stage_name: str, *, artifacts: tuple[str, ...] = ()) -> StageState:
        with self._lock:
            current = self._stage(stage_name)
            if current.status not in {"running", "completed"}:
                raise StateError(f"Cannot complete stage {stage_name!r} while it is {current.status}.")
            updated = replace(
                current,
                status="completed",
                artifacts=tuple(artifacts),
                completed_at=current.completed_at or utc_now_iso(),
                error=None,
            )
            self._replace(updated)
            return updated

    def fail(self, stage_name: str, error: BaseException | str) -> StageState:
        """Persist a failure, retaining a diagnostic for exceptions without a message."""

        message = str(error)
        if not message and isinstance(error, BaseException):
            message = type(error).__name__
        with self._lock:
            current = self._stage(stage_name)
            updated = replace(current, status="failed", error=message, completed_at=None)
            self._replace(updated)
            return updated

    def _stage(self, stage_name: str) -> StageState:
        try:
            return self.state.by_name[stage_name]
        except KeyError as exc:
            raise StateError(f"Unknown materialization stage {stage_name!r}.") from exc

    def _replace(self, stage: StageState) -> None:
        self._state = self.state.replace_stage(stage)
        self._write()

    def _validate_plan(self, state: BuildState, plan: BuildPlan) -> None:
        if state.build_id != plan.build_id:
            raise StateError(f"State belongs to build {state.build_id!r}, not {plan.build_id!r}.")
        expected = {stage.name: (stage.fingerprint, stage.depends_on) for stage in plan.stages}
        actual = {stage.name: (stage.fingerprint, stage.depends_on) for stage in state.stages}
        if actual != expected:
            raise StateError("Persisted stage fingerprints/dependencies do not match the build plan.")

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.state.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, prefix=".state-", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
        try:
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is unavailable on some supported filesystems.
            pass


__all__ = (
    "BuildState",
    "MaterializationStateStore",
    "StageState",
    "StageStatus",
    "StateError",
)
