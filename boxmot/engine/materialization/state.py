"""Atomic state snapshots and incremental durable materialization checkpoints."""

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
    batch_size: int = 1
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
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise StateError(f"Stage {self.name!r} has an invalid persisted batch size.")
        if self.status not in {"pending", "running", "completed", "failed"}:
            raise StateError(f"Unknown persisted stage status {self.status!r}.")
        if not isinstance(self.completed_shards, tuple) or any(
            not isinstance(shard, str) or re.fullmatch(r"[0-9]{5,}", shard) is None for shard in self.completed_shards
        ):
            raise StateError(f"Stage {self.name!r} has invalid persisted shard IDs.")
        completed_shards = set(self.completed_shards)
        if len(completed_shards) != len(self.completed_shards):
            raise StateError(f"Stage {self.name!r} has duplicate persisted shard IDs.")
        if not isinstance(self.shard_hashes, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 3
            or not isinstance(item[0], str)
            or item[0] not in completed_shards
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
            "batch_size": self.batch_size,
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
            batch_size=value["batch_size"],
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
    """Persist stage transitions atomically and append individual checkpoints.

    The journal is never rotated during a build. Snapshots record the consumed
    byte offset, so a crash on either side of snapshot replacement cannot replay
    an invalidated checkpoint or lose an acknowledged image.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._state: BuildState | None = None
        self.journal_path = self.path.with_suffix(".jsonl")
        self._journal_offset = 0
        self._journal_usable = True
        self._checkpoints: dict[str, dict[str, dict[str, str]]] = {}
        self._dirty_stages: set[str] = set()

    @property
    def state(self) -> BuildState:
        """Materialize an immutable view only when a caller requests all state."""

        with self._lock:
            if self._state is None:
                raise StateError("Materialization state has not been initialized.")
            if self._dirty_stages:
                stages = tuple(
                    replace(
                        stage,
                        completed_shards=tuple(self._checkpoints[stage.name]),
                        shard_hashes=tuple(
                            (shard, artifact, digest)
                            for shard, hashes in self._checkpoints[stage.name].items()
                            for artifact, digest in hashes.items()
                        ),
                    )
                    if stage.name in self._dirty_stages
                    else stage
                    for stage in self._state.stages
                )
                self._state = replace(self._state, stages=stages)
                self._dirty_stages.clear()
            return self._state

    def completed_shards(self, stage_name: str) -> frozenset[str]:
        """Return checkpoint IDs without constructing or validating full state."""

        with self._lock:
            self._stage_metadata(stage_name)
            return frozenset(self._checkpoints[stage_name])

    def completed_shard_count(self, stage_name: str) -> int:
        """Return the durable checkpoint count in constant time."""

        with self._lock:
            self._stage_metadata(stage_name)
            return len(self._checkpoints[stage_name])

    def shard_hashes(self, stage_name: str, shard_id: str) -> dict[str, str]:
        """Look up one checkpoint's hashes without scanning previous images."""

        with self._lock:
            self._stage_metadata(stage_name)
            return dict(self._checkpoints[stage_name].get(shard_id, {}))

    def _index_checkpoints(self, state: BuildState) -> None:
        """Index a validated snapshot for incremental checkpoint updates."""

        self._checkpoints = {stage.name: {shard: {} for shard in stage.completed_shards} for stage in state.stages}
        for stage in state.stages:
            for shard, artifact, digest in stage.shard_hashes:
                self._checkpoints[stage.name][shard][artifact] = digest
        self._dirty_stages.clear()

    def initialize(self, plan: BuildPlan) -> BuildState:
        """Create state or validate that persisted state belongs to this plan."""

        with self._lock:
            self._journal_usable = True
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
                offset = raw.get("journal_offset", 0)
                if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                    raise StateError("Invalid checkpoint journal offset.")
                self._state = state
                self._index_checkpoints(state)
                self._journal_offset = offset
                self._replay_journal()
                state = self.state
                # A process death while running leaves completed shards reusable,
                # but the stage itself must be entered again.
                # Ordinal detection/segmentation checkpoints fix sample grouping
                # once committed. Embedding checkpoints each cover one image, so
                # their inference crop batches can change independently on resume.
                requested_stages = plan.stage_by_name
                recovered = tuple(
                    replace(
                        stage,
                        status="pending" if stage.status == "running" else stage.status,
                        error=None if stage.status == "running" else stage.error,
                        batch_size=(
                            requested_stages[stage.name].batch_size
                            if stage.status != "completed" and (stage.name == "embed" or not stage.completed_shards)
                            else stage.batch_size
                        ),
                    )
                    for stage in state.stages
                )
                state = replace(state, stages=recovered, updated_at=utc_now_iso())
            else:
                if self.journal_path.exists() and self.journal_path.stat().st_size:
                    raise StateError("Checkpoint journal has no materialization state snapshot.")
                self._journal_offset = 0
                state = BuildState(
                    build_id=plan.build_id,
                    stages=tuple(
                        StageState(
                            name=stage.name,
                            fingerprint=stage.fingerprint,
                            depends_on=stage.depends_on,
                            batch_size=stage.batch_size,
                        )
                        for stage in plan.ordered_stages()
                    ),
                )
            self._state = state
            self._index_checkpoints(state)
            self._write()
            return state

    def is_completed(self, stage_name: str) -> bool:
        return self._stage_metadata(stage_name).status == "completed"

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
    ) -> None:
        """Durably append one checkpoint without rewriting preceding images."""

        with self._lock:
            current = self._stage_metadata(stage_name)
            if current.status != "running":
                raise StateError(f"Cannot record a shard while stage {stage_name!r} is {current.status}.")
            resolved = dict(sorted((hashes or {}).items()))
            self._validate_checkpoint(shard_id, resolved)
            self._append_checkpoint(stage_name, shard_id, resolved)
            self._apply_checkpoint(stage_name, shard_id, resolved)

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

    def discard_shard(self, stage_name: str, shard_id: str) -> None:
        """Forget an unreadable staging shard so the owning stage can rebuild it."""

        with self._lock:
            self._stage_metadata(stage_name)
            self._validate_checkpoint(shard_id, None)
            self._append_checkpoint(stage_name, shard_id, None)
            self._apply_checkpoint(stage_name, shard_id, None)

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
            self._save_state(replace(self.state, stages=stages, updated_at=utc_now_iso()))
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

    def _stage_metadata(self, stage_name: str) -> StageState:
        """Read lifecycle fields without materializing checkpoint tuples."""

        if self._state is None:
            raise StateError("Materialization state has not been initialized.")
        try:
            return self._state.by_name[stage_name]
        except KeyError as exc:
            raise StateError(f"Unknown materialization stage {stage_name!r}.") from exc

    def _replace(self, stage: StageState) -> None:
        self._save_state(self.state.replace_stage(stage))

    def _save_state(self, updated: BuildState) -> None:
        """Restore the previous view if an atomic snapshot cannot be written."""

        previous = self.state
        checkpoints = self._checkpoints
        self._state = updated
        self._index_checkpoints(updated)
        try:
            self._write()
        except BaseException:
            self._state = previous
            self._checkpoints = checkpoints
            raise

    @staticmethod
    def _validate_checkpoint(shard_id: str, hashes: Mapping[str, str] | None) -> None:
        """Validate only the changed checkpoint, independently of build size."""

        if not isinstance(shard_id, str) or re.fullmatch(r"[0-9]{5,}", shard_id) is None:
            raise StateError(f"Invalid checkpoint shard ID {shard_id!r}.")
        if hashes is not None and (
            not isinstance(hashes, Mapping)
            or any(
                not isinstance(artifact, str)
                or re.fullmatch(r"[a-z][a-z0-9_-]*", artifact) is None
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for artifact, digest in hashes.items()
            )
        ):
            raise StateError("Invalid checkpoint artifact hashes.")

    def _apply_checkpoint(self, stage_name: str, shard_id: str, hashes: dict[str, str] | None) -> None:
        """Apply an acknowledged journal record to the in-memory index."""

        current = self._stage_metadata(stage_name)
        assert self._state is not None
        if hashes is None:
            self._checkpoints[stage_name].pop(shard_id, None)
            self._state = self._state.replace_stage(
                replace(
                    current,
                    status="running",
                    completed_shards=(),
                    shard_hashes=(),
                    artifacts=(),
                    completed_at=None,
                    error=None,
                )
            )
        else:
            self._checkpoints[stage_name][shard_id] = hashes
        self._dirty_stages.add(stage_name)
        self._state = replace(self._state, updated_at=utc_now_iso())

    def _append_checkpoint(self, stage_name: str, shard_id: str, hashes: dict[str, str] | None) -> None:
        """Flush one record before acknowledging it; roll back failed appends."""

        self._require_usable_journal()
        payload = (json.dumps([stage_name, shard_id, hashes], separators=(",", ":"), allow_nan=False) + "\n").encode()
        created = not self.journal_path.exists()
        with self.journal_path.open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() != self._journal_offset:
                raise StateError("Checkpoint journal changed outside the materialization store.")
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                if created:
                    self._fsync_directory()
            except BaseException as error:
                try:
                    stream.seek(self._journal_offset)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                except BaseException as rollback_error:
                    self._journal_usable = False
                    raise error from rollback_error
                raise
        self._journal_offset += len(payload)

    def _replay_journal(self) -> None:
        """Replay changes beyond the snapshot, discarding only a torn final line."""

        if not self.journal_path.exists():
            if self._journal_offset:
                raise StateError("Checkpoint journal is missing for its recorded offset.")
            return
        with self.journal_path.open("r+b") as stream:
            stream.seek(0, os.SEEK_END)
            if self._journal_offset > stream.tell():
                raise StateError("Checkpoint journal offset exceeds its size.")
            if self._journal_offset:
                stream.seek(self._journal_offset - 1)
                if stream.read(1) != b"\n":
                    raise StateError("Checkpoint journal offset is not a record boundary.")
            stream.seek(self._journal_offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    stream.seek(self._journal_offset)
                    stream.truncate()
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                try:
                    record = json.loads(line)
                    if not isinstance(record, list) or len(record) != 3 or not isinstance(record[0], str):
                        raise StateError("Invalid checkpoint journal record.")
                    stage_name, shard_id, hashes = record
                    current = self._stage_metadata(stage_name)
                    self._validate_checkpoint(shard_id, hashes)
                    if hashes is not None and current.status != "running":
                        raise StateError(f"Checkpoint journal records a shard for inactive stage {stage_name!r}.")
                    self._apply_checkpoint(stage_name, shard_id, hashes)
                except (UnicodeError, ValueError, TypeError) as exc:
                    raise StateError("Invalid checkpoint journal record.") from exc
                self._journal_offset = stream.tell()

    def _validate_plan(self, state: BuildState, plan: BuildPlan) -> None:
        if state.build_id != plan.build_id:
            raise StateError(f"State belongs to build {state.build_id!r}, not {plan.build_id!r}.")
        expected = {stage.name: (stage.fingerprint, stage.depends_on) for stage in plan.stages}
        actual = {stage.name: (stage.fingerprint, stage.depends_on) for stage in state.stages}
        if actual != expected:
            raise StateError("Persisted stage fingerprints/dependencies do not match the build plan.")

    def _write(self) -> None:
        self._require_usable_journal()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {**self.state.to_dict(), "journal_offset": self._journal_offset}
        payload = json.dumps(snapshot, separators=(",", ":"), sort_keys=True, allow_nan=False) + "\n"
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
        self._fsync_directory()

    def _require_usable_journal(self) -> None:
        """Prevent snapshots from acknowledging a journal with uncertain contents."""

        if not self._journal_usable:
            raise StateError(
                "Checkpoint journal recovery is required after a failed append rollback; reopen the store."
            )

    def _fsync_directory(self) -> None:
        """Persist new journal files and atomic snapshot replacements."""

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
