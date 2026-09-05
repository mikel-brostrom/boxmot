"""Manifest model for immutable materialized datasets."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

from .schema import (
    ARTIFACT_PATHS,
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MANIFEST_FILENAME,
    MASKS_ARTIFACT,
    SAMPLES_ARTIFACT,
    SCHEMA_ID,
    SCHEMA_VERSION,
    BoxType,
)

_BUILD_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ManifestError(ValueError):
    """Raised when a dataset manifest is invalid or unsupported."""


def _json_value(value: Any) -> Any:
    """Convert configuration-like values into deterministic JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_value({item.name: getattr(value, item.name) for item in fields(value)})
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError("Manifest values cannot contain NaN or infinity.")
        return value
    raise ManifestError(f"Value of type {type(value).__name__} is not JSON serializable.")


def _freeze_json_value(value: Any) -> Any:
    normalized = _json_value(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze_json_value(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze_json_value(item) for item in normalized)
    return normalized


def frozen_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return an immutable, detached snapshot of a JSON-like mapping."""

    frozen = _freeze_json_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by the annotation
        raise ManifestError("Expected a JSON mapping.")
    return frozen


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{name} must be a string.")
    return value


def _required_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{name} must be an integer.")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value to stable UTF-8 JSON suitable for fingerprints."""

    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular file."""

    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """Return a canonical UTC timestamp."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_relative_artifact_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ManifestError(f"Artifact paths must be safe paths relative to the build root, got {value!r}.")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class ShardRecord:
    """One immutable Parquet shard owned by an artifact."""

    path: str
    rows: int
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_relative_artifact_path(self.path))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.rows, self.size_bytes)):
            raise ManifestError("Shard row and byte counts must be integers.")
        if self.rows < 0 or self.size_bytes < 0:
            raise ManifestError("Shard row and byte counts must be non-negative.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ManifestError(f"Shard {self.path!r} has an invalid SHA-256 digest.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ShardRecord":
        return cls(
            path=_required_string(value["path"], "Shard path"),
            rows=_required_integer(value["rows"], "Shard rows"),
            sha256=_required_string(value["sha256"], "Shard SHA-256"),
            size_bytes=_required_integer(value["size_bytes"], "Shard size_bytes"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """One resolved, sharded artifact owned by a materialized build."""

    name: str
    path: str
    rows: int
    sha256: str
    size_bytes: int
    shards: tuple[ShardRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().lower():
            raise ManifestError(f"Artifact names must be canonical lowercase identifiers, got {self.name!r}.")
        object.__setattr__(self, "path", _validate_relative_artifact_path(self.path))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.rows, self.size_bytes)):
            raise ManifestError("Artifact row and byte counts must be integers.")
        if self.rows < 0 or self.size_bytes < 0:
            raise ManifestError("Artifact row and byte counts must be non-negative.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ManifestError(f"Artifact {self.name!r} has an invalid SHA-256 digest.")
        object.__setattr__(self, "shards", tuple(self.shards))
        if not self.shards:
            raise ManifestError(f"Artifact {self.name!r} must contain at least one shard.")
        if sum(shard.rows for shard in self.shards) != self.rows:
            raise ManifestError(f"Artifact {self.name!r} shard rows do not match its row count.")
        if sum(shard.size_bytes for shard in self.shards) != self.size_bytes:
            raise ManifestError(f"Artifact {self.name!r} shard bytes do not match its byte count.")
        object.__setattr__(self, "metadata", frozen_json_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "rows": self.rows,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "shards": [shard.to_dict() for shard in self.shards],
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(
            name=_required_string(value["name"], "Artifact name"),
            path=_required_string(value["path"], "Artifact path"),
            rows=_required_integer(value["rows"], "Artifact rows"),
            sha256=_required_string(value["sha256"], "Artifact SHA-256"),
            size_bytes=_required_integer(value["size_bytes"], "Artifact size_bytes"),
            shards=tuple(ShardRecord.from_dict(item) for item in value.get("shards", ())),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class PublishedContent:
    """Explicit completeness flags for the resolved build."""

    image_references: bool
    masks: bool
    embeddings: bool

    def __post_init__(self) -> None:
        for name in ("image_references", "masks", "embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ManifestError(f"Publish flag {name!r} must be a boolean.")

    def to_dict(self) -> dict[str, bool]:
        return {
            "image_references": self.image_references,
            "masks": self.masks,
            "embeddings": self.embeddings,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PublishedContent":
        try:
            return cls(
                image_references=value["image_references"],
                masks=value["masks"],
                embeddings=value["embeddings"],
            )
        except KeyError as exc:
            raise ManifestError(f"Manifest publish is missing flag {exc.args[0]!r}.") from exc


@dataclass(frozen=True, slots=True)
class StageProvenance:
    """Reproducibility record for one materialization stage."""

    name: str
    fingerprint: str
    batch_size: int = 1
    inputs: tuple[str, ...] = ()
    component: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip().lower():
            raise ManifestError(f"Stage names must be canonical lowercase identifiers, got {self.name!r}.")
        if not re.fullmatch(r"[0-9a-f]{64}", self.fingerprint):
            raise ManifestError(f"Stage {self.name!r} has an invalid fingerprint.")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ManifestError(f"Stage {self.name!r} has an invalid semantic batch size.")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "component", frozen_json_mapping(self.component))
        object.__setattr__(self, "config", frozen_json_mapping(self.config))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "fingerprint": self.fingerprint,
            "batch_size": self.batch_size,
            "inputs": list(self.inputs),
            "component": _json_value(self.component),
            "config": _json_value(self.config),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageProvenance":
        return cls(
            name=_required_string(value["name"], "Stage name"),
            fingerprint=_required_string(value["fingerprint"], "Stage fingerprint"),
            batch_size=_required_integer(value["batch_size"], "Stage batch_size"),
            inputs=tuple(str(item) for item in value.get("inputs", ())),
            component=value.get("component", {}),
            config=value.get("config", {}),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Complete description of one immutable dataset build."""

    build_id: str
    box_type: BoxType
    artifacts: tuple[ArtifactRecord, ...]
    publish: PublishedContent
    stages: tuple[StageProvenance, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    complete: bool = True
    created_at: str = field(default_factory=utc_now_iso)
    schema: str = SCHEMA_ID
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_ID or self.schema_version != SCHEMA_VERSION:
            raise ManifestError(
                f"Unsupported dataset schema {self.schema!r} version {self.schema_version}; "
                f"expected {SCHEMA_ID!r}. "
                "Legacy positional caches are not supported."
            )
        if not _BUILD_ID_PATTERN.fullmatch(self.build_id):
            raise ManifestError(f"Invalid build_id {self.build_id!r}.")
        if self.box_type not in {"aabb", "obb"}:
            raise ManifestError(f"Unknown box_type {self.box_type!r}.")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "stages", tuple(self.stages))
        object.__setattr__(self, "metadata", frozen_json_mapping(self.metadata))
        if not isinstance(self.publish, PublishedContent):
            raise ManifestError("Manifest publish must be PublishedContent.")
        if self.complete is not True:
            raise ManifestError("Only complete immutable dataset manifests are supported.")

        artifact_names = [artifact.name for artifact in self.artifacts]
        if len(set(artifact_names)) != len(artifact_names):
            raise ManifestError("Manifest artifact names must be unique.")
        unknown_artifacts = set(artifact_names) - set(ARTIFACT_PATHS)
        if unknown_artifacts:
            raise ManifestError(f"Manifest contains unknown v1 artifacts: {sorted(unknown_artifacts)!r}.")
        for required in (SAMPLES_ARTIFACT, INSTANCES_ARTIFACT):
            if required not in artifact_names:
                raise ManifestError(f"Manifest is missing required {required!r} artifact.")
        all_shard_paths: list[str] = []
        for artifact in self.artifacts:
            expected_path = ARTIFACT_PATHS[artifact.name]
            if artifact.path != expected_path:
                raise ManifestError(
                    f"Artifact {artifact.name!r} must use canonical path {expected_path!r}, got {artifact.path!r}."
                )
            shard_paths = [shard.path for shard in artifact.shards]
            expected_prefix = f"{expected_path}/part-"
            if any(
                not path.startswith(expected_prefix)
                or not re.fullmatch(r"part-[0-9]{5,}\.parquet", PurePosixPath(path).name)
                for path in shard_paths
            ):
                raise ManifestError(f"Artifact {artifact.name!r} contains a non-canonical shard path.")
            if shard_paths != sorted(shard_paths) or len(set(shard_paths)) != len(shard_paths):
                raise ManifestError(f"Artifact {artifact.name!r} shard paths must be unique and sorted.")
            all_shard_paths.extend(shard_paths)
        if len(set(all_shard_paths)) != len(all_shard_paths):
            raise ManifestError("Manifest shard paths must be globally unique.")
        if self.publish.masks != (MASKS_ARTIFACT in artifact_names):
            raise ManifestError("Manifest mask publish flag does not match resolved artifacts.")
        if self.publish.embeddings != (EMBEDDINGS_ARTIFACT in artifact_names):
            raise ManifestError("Manifest embedding publish flag does not match resolved artifacts.")
        if self.publish.embeddings:
            embedding_metadata = self.artifact(EMBEDDINGS_ARTIFACT).metadata
            encoder_fingerprint = embedding_metadata.get("encoder_fingerprint")
            dim = embedding_metadata.get("dim")
            if not isinstance(encoder_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", encoder_fingerprint):
                raise ManifestError("Embedding artifact metadata requires a full encoder_fingerprint.")
            if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
                raise ManifestError("Embedding artifact metadata requires a positive integer dim.")
        resolved_counts = {artifact.name: artifact.rows for artifact in self.artifacts}
        if self.counts and dict(self.counts) != resolved_counts:
            raise ManifestError("Manifest counts do not match resolved artifact row counts.")
        object.__setattr__(self, "counts", MappingProxyType(resolved_counts))
        stage_names = [stage.name for stage in self.stages]
        if not stage_names:
            raise ManifestError("Manifest must record at least one materialization stage fingerprint.")
        if len(set(stage_names)) != len(stage_names):
            raise ManifestError("Manifest stage names must be unique.")
        source_fingerprint = self.metadata.get("source_fingerprint")
        if not isinstance(source_fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint):
            raise ManifestError("Manifest metadata requires a full source_fingerprint SHA-256 digest.")

    @property
    def artifacts_by_name(self) -> dict[str, ArtifactRecord]:
        return {artifact.name: artifact for artifact in self.artifacts}

    def artifact(self, name: str) -> ArtifactRecord:
        try:
            return self.artifacts_by_name[name]
        except KeyError as exc:
            raise ManifestError(f"Dataset build {self.build_id!r} does not contain artifact {name!r}.") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "build_id": self.build_id,
            "created_at": self.created_at,
            "box_type": self.box_type,
            "complete": self.complete,
            "publish": self.publish.to_dict(),
            "counts": dict(self.counts),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": _json_value(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifest":
        if value.get("schema") != SCHEMA_ID:
            raise ManifestError(
                f"Unsupported dataset schema {value.get('schema')!r}; expected {SCHEMA_ID!r}. "
                "Legacy positional caches are not supported."
            )
        complete = value.get("complete")
        if not isinstance(complete, bool):
            raise ManifestError("Manifest complete must be a boolean.")
        return cls(
            schema=_required_string(value.get("schema"), "Manifest schema"),
            schema_version=_required_integer(value.get("schema_version"), "Manifest schema_version"),
            build_id=_required_string(value["build_id"], "Manifest build_id"),
            created_at=_required_string(value["created_at"], "Manifest created_at"),
            box_type=_required_string(value["box_type"], "Manifest box_type"),  # type: ignore[arg-type]
            complete=complete,
            publish=PublishedContent.from_dict(value.get("publish", {})),
            counts=value.get("counts", {}),
            artifacts=tuple(ArtifactRecord.from_dict(item) for item in value.get("artifacts", ())),
            stages=tuple(StageProvenance.from_dict(item) for item in value.get("stages", ())),
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def load(cls, root_or_path: str | Path) -> "DatasetManifest":
        path = Path(root_or_path)
        if path.is_dir():
            manifest_path = path / MANIFEST_FILENAME
            if not manifest_path.is_file() and any(
                candidate.is_file() and candidate.suffix.lower() in {".npy", ".npz", ".txt"}
                for candidate in path.rglob("*")
            ):
                raise ManifestError(
                    f"Unsupported dataset schema at {path}: legacy NPY/NPZ/text caches are not supported."
                )
            path = manifest_path
        elif path.suffix.lower() in {".npy", ".npz", ".txt"}:
            raise ManifestError(
                f"Unsupported dataset schema at {path}: legacy NPY/NPZ/text caches are not supported."
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ManifestError(f"Dataset manifest does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Dataset manifest is not valid JSON: {path}") from exc
        if not isinstance(value, dict):
            raise ManifestError("Dataset manifest must contain a JSON object.")
        try:
            return cls.from_dict(value)
        except ManifestError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ManifestError(f"Dataset manifest has invalid fields: {path}") from exc

    def write(self, root_or_path: str | Path) -> Path:
        """Atomically write the manifest, which must be published last."""

        path = Path(root_or_path)
        if path.suffix.lower() != ".json":
            path = path / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, prefix=".manifest-", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is unavailable on some supported filesystems.
            pass
        return path


__all__ = (
    "ArtifactRecord",
    "DatasetManifest",
    "ManifestError",
    "PublishedContent",
    "ShardRecord",
    "StageProvenance",
    "canonical_json_bytes",
    "frozen_json_mapping",
    "sha256_file",
    "utc_now_iso",
)
