"""Integrity validation for immutable keyed dataset builds."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .manifest import DatasetManifest, ManifestError, sha256_file
from .masks import MASK_CODEC, MaskCodecError, _validate_mask_payload
from .schema import (
    ARTIFACT_PATHS,
    ARTIFACT_SCHEMAS,
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MASKS_ARTIFACT,
    SAMPLES_ARTIFACT,
    SCHEMA_ID,
    SUCCESS_FILENAME,
    BoxType,
    embeddings_schema,
    instances_schema,
)
from .storage import (
    ENCODER_FINGERPRINT_METADATA_KEY,
    PARQUET_ROW_GROUP_ROWS,
    artifact_files,
    read_parquet_artifact,
    resolve_artifact_path,
    resolve_embedding_metadata,
)

if TYPE_CHECKING:
    import pyarrow as pa


class DatasetValidationError(ValueError):
    """Raised when a materialized dataset is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Validated row counts for a dataset build."""

    samples: int
    instances: int
    masks: int = 0
    embeddings: int = 0


def expected_instance_id(build_id: str, sample_id: str, detection_index: int) -> str:
    """Return the stable instance key used by all downstream tables."""

    if not sample_id:
        raise DatasetValidationError("sample_id must not be empty.")
    if detection_index < 0:
        raise DatasetValidationError("detection_index must be non-negative.")
    return f"{build_id}:{sample_id}:{detection_index}"


def _validate_publication_marker(build_root: Path, manifest: DatasetManifest) -> None:
    """Validate that a directory is an atomically published build."""

    if build_root.resolve().parent.name == ".staging":
        raise DatasetValidationError("Dataset staging directories are resumable state, not published builds.")
    success_path = build_root / SUCCESS_FILENAME
    if not success_path.is_file():
        raise DatasetValidationError(f"Dataset build is not published: missing {SUCCESS_FILENAME}.")
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Dataset publication marker is invalid: {success_path}") from exc
    if success != {"schema": SCHEMA_ID, "build_id": manifest.build_id}:
        raise DatasetValidationError("Dataset publication marker does not match its manifest.")


def _artifact_sha256_from_shards(
    path: Path,
    files: tuple[Path, ...],
    shard_hashes: tuple[str, ...],
) -> str:
    """Return an artifact digest without reading already-hashed shards twice."""

    if path.is_file():
        return shard_hashes[0]
    digest = hashlib.sha256()
    for file, shard_hash in zip(files, shard_hashes, strict=True):
        relative = file.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(shard_hash))
    return digest.hexdigest()


def _expected_schema(name: str, box_type: BoxType, actual_schema: "pa.Schema") -> "pa.Schema":
    """Resolve the canonical full schema from one physical shard schema."""

    if name == EMBEDDINGS_ARTIFACT:
        try:
            dimension = int(actual_schema.field("values").type.list_size)
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Embedding values must use a fixed-size float32 list type.") from exc
        return embeddings_schema(dimension)
    if name == INSTANCES_ARTIFACT:
        return instances_schema(box_type)
    try:
        return ARTIFACT_SCHEMAS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown materialized dataset artifact {name!r}.") from exc


def _validate_embedding_shard_metadata(
    schema: "pa.Schema",
    *,
    shard_path: str,
    declared_dimension: int,
    declared_encoder: str,
) -> None:
    """Bind an embedding shard's physical schema metadata to its manifest."""

    try:
        dimension = int(schema.field("values").type.list_size)
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise DatasetValidationError("Embedding values must use a fixed-size float32 list type.") from exc
    if dimension != declared_dimension:
        raise DatasetValidationError(f"Embedding shard {shard_path!r} dimension differs from its manifest.")
    encoded = (schema.metadata or {}).get(ENCODER_FINGERPRINT_METADATA_KEY)
    try:
        shard_encoder = None if encoded is None else encoded.decode("ascii")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError(f"Embedding shard {shard_path!r} has an invalid encoder fingerprint.") from exc
    if shard_encoder != declared_encoder:
        raise DatasetValidationError(f"Embedding shard {shard_path!r} encoder fingerprint differs from its manifest.")


def _validate_artifact_metadata(
    root: Path,
    manifest: DatasetManifest,
    name: str,
    *,
    verify_hashes: bool,
) -> int:
    """Validate one artifact without decoding any Parquet column payload."""

    artifact = manifest.artifact(name)
    try:
        path = resolve_artifact_path(root, artifact.path)
    except ManifestError as exc:
        raise DatasetValidationError(str(exc)) from exc
    if not path.exists():
        raise DatasetValidationError(f"Dataset artifact {name!r} is missing: {path}")
    try:
        files = artifact_files(path)
    except FileNotFoundError as exc:
        raise DatasetValidationError(str(exc)) from exc
    actual_shards = tuple(file.relative_to(root.resolve()).as_posix() for file in files)
    declared_shards = tuple(shard.path for shard in artifact.shards)
    if actual_shards != declared_shards:
        raise DatasetValidationError(f"Dataset artifact {name!r} shard list differs from its manifest.")

    # Resolve and contain every declared shard before opening any of them.
    # In particular, do not hash a canonical-looking symlink whose target
    # escapes the immutable build root.
    for file, shard in zip(files, artifact.shards, strict=True):
        try:
            resolved_file = resolve_artifact_path(root, shard.path)
        except ManifestError as exc:
            raise DatasetValidationError(str(exc)) from exc
        if resolved_file != file.resolve():
            raise DatasetValidationError(f"Dataset shard {shard.path!r} does not resolve to its declared path.")

    actual_size = sum(file.stat().st_size for file in files)
    if actual_size != artifact.size_bytes:
        raise DatasetValidationError(f"Dataset artifact {name!r} byte count differs from its manifest.")

    # Read each potentially large shard exactly once for SHA-256. The
    # aggregate digest is derived from these bytes rather than reopening all
    # shards through the directory hashing helper.
    shard_hashes = tuple(sha256_file(file) for file in files) if verify_hashes else ()
    reference_schema = None
    actual_rows = 0
    for shard_index, (file, shard) in enumerate(zip(files, artifact.shards, strict=True)):
        if file.stat().st_size != shard.size_bytes:
            raise DatasetValidationError(f"Dataset shard {shard.path!r} byte count differs from its manifest.")
        if verify_hashes and shard_hashes[shard_index] != shard.sha256:
            raise DatasetValidationError(f"Dataset shard {shard.path!r} failed its SHA-256 check.")
        import pyarrow.parquet as pq

        try:
            parquet = pq.ParquetFile(file)
        except (OSError, ValueError) as exc:
            raise DatasetValidationError(f"Dataset shard {shard.path!r} is not valid Parquet.") from exc
        actual_schema = parquet.schema_arrow
        try:
            expected_schema = _expected_schema(name, manifest.box_type, actual_schema)
        except ValueError as exc:
            raise DatasetValidationError(str(exc)) from exc
        if not actual_schema.equals(expected_schema, check_metadata=False):
            raise DatasetValidationError(
                f"Parquet schema mismatch for {name!r}: expected {expected_schema}, got {actual_schema}. "
                "Legacy or positional caches are not supported."
            )
        if reference_schema is None:
            reference_schema = actual_schema
        elif not actual_schema.equals(reference_schema, check_metadata=False):
            raise DatasetValidationError(f"Dataset artifact {name!r} shards use inconsistent schemas.")
        if parquet.metadata.num_rows != shard.rows:
            raise DatasetValidationError(f"Dataset shard {shard.path!r} row count differs from its manifest.")
        actual_rows += parquet.metadata.num_rows
        for row_group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.metadata.row_group(row_group_index)
            for column_index in range(row_group.num_columns):
                if row_group.column(column_index).compression.upper() != "ZSTD":
                    raise DatasetValidationError(
                        f"Dataset shard {shard.path!r} must use zstd compression for every column chunk."
                    )
        if name == EMBEDDINGS_ARTIFACT:
            _validate_embedding_shard_metadata(
                actual_schema,
                shard_path=shard.path,
                declared_dimension=int(artifact.metadata["dim"]),
                declared_encoder=str(artifact.metadata["encoder_fingerprint"]),
            )

    if actual_rows != artifact.rows:
        raise DatasetValidationError(f"Dataset artifact {name!r} row count differs from its manifest.")
    if verify_hashes and _artifact_sha256_from_shards(path, files, shard_hashes) != artifact.sha256:
        raise DatasetValidationError(f"Dataset artifact {name!r} failed its SHA-256 check.")
    return actual_rows


def _validate_artifact(
    root: Path,
    manifest: DatasetManifest,
    name: str,
    *,
    verify_hashes: bool,
    columns: list[str] | None = None,
) -> "pa.Table":
    """Exhaustively validate metadata and return the requested payload."""

    _validate_artifact_metadata(root, manifest, name, verify_hashes=verify_hashes)
    artifact = manifest.artifact(name)
    path = resolve_artifact_path(root, artifact.path)
    try:
        table = read_parquet_artifact(
            path,
            artifact_name=name,
            box_type=manifest.box_type,
            columns=columns,
        )
    except (OSError, ValueError) as exc:
        raise DatasetValidationError(str(exc)) from exc
    if table.num_rows != artifact.rows:
        raise DatasetValidationError(f"Dataset artifact {name!r} row count differs from its manifest.")
    return table


def _iter_artifact_batches(
    root: Path,
    manifest: DatasetManifest,
    name: str,
    *,
    columns: list[str],
) -> Iterator["pa.RecordBatch"]:
    """Yield bounded record batches in deterministic manifest shard order."""

    import pyarrow.parquet as pq

    for shard in manifest.artifact(name).shards:
        parquet = pq.ParquetFile(resolve_artifact_path(root, shard.path))
        yield from parquet.iter_batches(
            batch_size=PARQUET_ROW_GROUP_ROWS,
            columns=columns,
        )


def _duplicates(values: list[str | int]) -> set[str | int]:
    seen: set[str | int] = set()
    return {value for value in values if value in seen or seen.add(value)}


def validate_published_build(
    root: str | Path,
    *,
    manifest: DatasetManifest | None = None,
) -> ValidationReport:
    """Validate an immutable published build for materialization reuse.

    The pre-publication finalizer already performs exhaustive value, key, and
    referential validation before its atomic rename. Reuse therefore verifies
    the publication marker, declared artifact set, file hashes, exact Arrow
    schemas, and Parquet footer metadata without inflating column payloads.
    """

    build_root = Path(root)
    try:
        resolved_manifest = manifest or DatasetManifest.load(build_root)
    except ManifestError as exc:
        raise DatasetValidationError(str(exc)) from exc
    _validate_publication_marker(build_root, resolved_manifest)

    expected_artifacts = {SAMPLES_ARTIFACT, INSTANCES_ARTIFACT}
    if resolved_manifest.publish.masks:
        expected_artifacts.add(MASKS_ARTIFACT)
    if resolved_manifest.publish.embeddings:
        expected_artifacts.add(EMBEDDINGS_ARTIFACT)
    actual_artifacts = set(resolved_manifest.artifacts_by_name)
    if actual_artifacts != expected_artifacts:
        raise DatasetValidationError("Dataset manifest artifact set does not match its declared published content.")

    for optional_name in (MASKS_ARTIFACT, EMBEDDINGS_ARTIFACT):
        try:
            optional_path = resolve_artifact_path(build_root, ARTIFACT_PATHS[optional_name])
        except ManifestError as exc:
            raise DatasetValidationError(str(exc)) from exc
        if optional_name not in actual_artifacts and optional_path.exists():
            raise DatasetValidationError(f"Dataset build contains unpublished {optional_name!r} artifact data.")

    counts = {
        name: _validate_artifact_metadata(
            build_root,
            resolved_manifest,
            name,
            verify_hashes=True,
        )
        for name in sorted(actual_artifacts)
    }
    source_count = resolved_manifest.metadata.get("source_count")
    if source_count is not None:
        if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 0:
            raise DatasetValidationError("Manifest source_count must be a non-negative integer.")
        if source_count != counts[SAMPLES_ARTIFACT]:
            raise DatasetValidationError(
                f"Sample row count {counts[SAMPLES_ARTIFACT]} does not match manifest source_count {source_count}."
            )
    for optional_name in (MASKS_ARTIFACT, EMBEDDINGS_ARTIFACT):
        if optional_name in counts and counts[optional_name] != counts[INSTANCES_ARTIFACT]:
            raise DatasetValidationError(f"Published {optional_name} row count must match the instance row count.")

    return ValidationReport(
        samples=counts[SAMPLES_ARTIFACT],
        instances=counts[INSTANCES_ARTIFACT],
        masks=counts.get(MASKS_ARTIFACT, 0),
        embeddings=counts.get(EMBEDDINGS_ARTIFACT, 0),
    )


def validate_dataset(
    root: str | Path,
    *,
    manifest: DatasetManifest | None = None,
    verify_hashes: bool = True,
    require_success: bool = True,
) -> ValidationReport:
    """Validate schemas, hashes, keys, geometry, and optional table alignment."""

    build_root = Path(root)
    try:
        resolved_manifest = manifest or DatasetManifest.load(build_root)
    except ManifestError as exc:
        raise DatasetValidationError(str(exc)) from exc

    if require_success:
        _validate_publication_marker(build_root, resolved_manifest)

    samples_table = _validate_artifact(build_root, resolved_manifest, SAMPLES_ARTIFACT, verify_hashes=verify_hashes)
    instances_table = _validate_artifact(build_root, resolved_manifest, INSTANCES_ARTIFACT, verify_hashes=verify_hashes)
    samples = samples_table.to_pylist()
    instances = instances_table.to_pylist()

    sample_ids = [row["sample_id"] for row in samples]
    duplicate_samples = _duplicates(sample_ids)
    if duplicate_samples:
        raise DatasetValidationError(f"Duplicate sample IDs: {sorted(duplicate_samples)!r}.")
    sample_by_id = {row["sample_id"]: row for row in samples}
    source_count = resolved_manifest.metadata.get("source_count")
    if source_count is not None:
        if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 0:
            raise DatasetValidationError("Manifest source_count must be a non-negative integer.")
        if source_count != len(samples):
            raise DatasetValidationError(
                f"Sample row count {len(samples)} does not match manifest source_count {source_count}."
            )
    sequence_keys: set[tuple[str, str, int]] = set()
    for row in samples:
        if not row["sample_id"] or not row["split"] or not row["sequence_id"] or row["frame_index"] < 0:
            raise DatasetValidationError(f"Sample {row['sample_id']!r} has invalid sequence metadata.")
        if row["timestamp_s"] is not None and not math.isfinite(row["timestamp_s"]):
            raise DatasetValidationError(f"Sample {row['sample_id']!r} has an invalid timestamp.")
        if row["height"] <= 0 or row["width"] <= 0:
            raise DatasetValidationError(f"Sample {row['sample_id']!r} has invalid dimensions.")
        sequence_key = (row["split"], row["sequence_id"], row["frame_index"])
        if sequence_key in sequence_keys:
            raise DatasetValidationError("Sample rows must have unique (split, sequence_id, frame_index) identities.")
        sequence_keys.add(sequence_key)
        if resolved_manifest.publish.image_references and not row["image_ref"]:
            raise DatasetValidationError(f"Sample {row['sample_id']!r} is missing a published image reference.")
        if not resolved_manifest.publish.image_references and row["image_ref"] is not None:
            raise DatasetValidationError(f"Sample {row['sample_id']!r} contains an unpublished image reference.")

    instance_ids = [row["instance_id"] for row in instances]
    duplicate_instances = _duplicates(instance_ids)
    if duplicate_instances:
        raise DatasetValidationError(f"Duplicate instance IDs: {sorted(duplicate_instances)!r}.")

    indices_by_sample: dict[str, list[int]] = {sample_id: [] for sample_id in sample_ids}
    for row in instances:
        sample_id = row["sample_id"]
        if sample_id not in sample_by_id:
            raise DatasetValidationError(f"Instance {row['instance_id']!r} refers to unknown sample {sample_id!r}.")
        detection_index = row["detection_index"]
        indices_by_sample[sample_id].append(detection_index)
        expected_id = expected_instance_id(resolved_manifest.build_id, sample_id, detection_index)
        if row["instance_id"] != expected_id:
            raise DatasetValidationError(
                f"Instance ID {row['instance_id']!r} does not match its build/sample/detection key {expected_id!r}."
            )
        if not math.isfinite(row["score"]) or not 0.0 <= row["score"] <= 1.0:
            raise DatasetValidationError(f"Instance {row['instance_id']!r} has an invalid score.")
        if row["class_id"] < 0:
            raise DatasetValidationError(f"Instance {row['instance_id']!r} has a negative class ID.")

        if resolved_manifest.box_type == "aabb":
            geometry = [row[key] for key in ("x1", "y1", "x2", "y2")]
            if any(value is None or not math.isfinite(value) for value in geometry):
                raise DatasetValidationError(f"Instance {row['instance_id']!r} has invalid AABB geometry.")
            if geometry[2] <= geometry[0] or geometry[3] <= geometry[1]:
                raise DatasetValidationError(f"Instance {row['instance_id']!r} has a degenerate AABB.")
        else:
            geometry = [row[key] for key in ("cx", "cy", "w", "h", "angle")]
            if any(value is None or not math.isfinite(value) for value in geometry):
                raise DatasetValidationError(f"Instance {row['instance_id']!r} has invalid OBB geometry.")
            if geometry[2] <= 0 or geometry[3] <= 0:
                raise DatasetValidationError(f"Instance {row['instance_id']!r} has a degenerate OBB.")

    for sample_id, detection_indices in indices_by_sample.items():
        if sorted(detection_indices) != list(range(len(detection_indices))):
            raise DatasetValidationError(
                f"Detection indices for sample {sample_id!r} must be unique and dense from zero."
            )

    artifacts = resolved_manifest.artifacts_by_name
    if require_success:
        for optional_name in (MASKS_ARTIFACT, EMBEDDINGS_ARTIFACT):
            optional_path = resolve_artifact_path(build_root, optional_name)
            if optional_name not in artifacts and optional_path.exists():
                raise DatasetValidationError(f"Dataset build contains unpublished {optional_name!r} artifact data.")
    expected_keys = set(instance_ids)
    mask_count = 0
    if MASKS_ARTIFACT in artifacts:
        masks_table = _validate_artifact(
            build_root,
            resolved_manifest,
            MASKS_ARTIFACT,
            verify_hashes=verify_hashes,
            columns=["instance_id"],
        )
        mask_keys = masks_table.column("instance_id").to_pylist()
        if set(mask_keys) != expected_keys or len(mask_keys) != len(expected_keys):
            raise DatasetValidationError("Mask keys must match instance keys exactly once.")
        instance_to_sample = {row["instance_id"]: row["sample_id"] for row in instances}
        mask_count = len(mask_keys)
        for batch in _iter_artifact_batches(
            build_root,
            resolved_manifest,
            MASKS_ARTIFACT,
            columns=["sample_id", "instance_id", "height", "width", "codec", "data"],
        ):
            sample_values = batch.column("sample_id")
            instance_values = batch.column("instance_id")
            height_values = batch.column("height")
            width_values = batch.column("width")
            codec_values = batch.column("codec")
            payload_values = batch.column("data")
            for row_index in range(batch.num_rows):
                instance_id = instance_values[row_index].as_py()
                sample_id = sample_values[row_index].as_py()
                expected_sample_id = instance_to_sample[instance_id]
                if sample_id != expected_sample_id:
                    raise DatasetValidationError(f"Mask {instance_id!r} has the wrong sample key.")
                sample = sample_by_id[expected_sample_id]
                height = height_values[row_index].as_py()
                width = width_values[row_index].as_py()
                if (height, width) != (sample["height"], sample["width"]):
                    raise DatasetValidationError(f"Mask {instance_id!r} is not full-frame sized.")
                if codec_values[row_index].as_py() != MASK_CODEC:
                    raise DatasetValidationError(f"Mask {instance_id!r} uses an unsupported codec.")
                try:
                    _validate_mask_payload(payload_values[row_index].as_py(), height, width)
                except MaskCodecError as exc:
                    raise DatasetValidationError(f"Mask {instance_id!r} has an invalid payload: {exc}") from exc

    embedding_count = 0
    if EMBEDDINGS_ARTIFACT in artifacts:
        embeddings_table = _validate_artifact(
            build_root,
            resolved_manifest,
            EMBEDDINGS_ARTIFACT,
            verify_hashes=verify_hashes,
            columns=["sample_id", "instance_id", "encoder_fingerprint", "dim"],
        )
        embedding_keys = embeddings_table.column("instance_id").to_pylist()
        if set(embedding_keys) != expected_keys or len(embedding_keys) != len(expected_keys):
            raise DatasetValidationError("Embedding keys must match instance keys exactly once.")
        import pyarrow.parquet as pq

        first_shard = artifacts[EMBEDDINGS_ARTIFACT].shards[0]
        embedding_schema = pq.read_schema(resolve_artifact_path(build_root, first_shard.path))
        try:
            resolved_embedding_metadata = resolve_embedding_metadata(
                embeddings_table,
                declared=artifacts[EMBEDDINGS_ARTIFACT].metadata,
                schema=embedding_schema,
            )
        except ValueError as exc:
            raise DatasetValidationError(str(exc)) from exc
        declared_encoder = resolved_embedding_metadata["encoder_fingerprint"]
        embedding_dimension = resolved_embedding_metadata["dim"]

        for shard in artifacts[EMBEDDINGS_ARTIFACT].shards:
            shard_schema = pq.read_schema(resolve_artifact_path(build_root, shard.path))
            shard_metadata = shard_schema.metadata or {}
            encoded = shard_metadata.get(ENCODER_FINGERPRINT_METADATA_KEY)
            try:
                shard_encoder = None if encoded is None else encoded.decode("ascii")
            except UnicodeDecodeError as exc:
                raise DatasetValidationError(
                    f"Embedding shard {shard.path!r} has an invalid encoder fingerprint."
                ) from exc
            if shard_encoder != declared_encoder:
                raise DatasetValidationError(
                    f"Embedding shard {shard.path!r} encoder fingerprint differs from its manifest."
                )
        instance_to_sample = {row["instance_id"]: row["sample_id"] for row in instances}
        embedding_count = len(embedding_keys)
        for batch in _iter_artifact_batches(
            build_root,
            resolved_manifest,
            EMBEDDINGS_ARTIFACT,
            columns=["sample_id", "instance_id", "dim", "values"],
        ):
            sample_values = batch.column("sample_id")
            instance_values = batch.column("instance_id")
            dimension_values = batch.column("dim")
            embedding_values = batch.column("values")
            flattened = embedding_values.flatten().to_numpy(zero_copy_only=False)
            finite_rows = np.isfinite(flattened).reshape(batch.num_rows, embedding_dimension).all(axis=1)
            for row_index in range(batch.num_rows):
                instance_id = instance_values[row_index].as_py()
                sample_id = sample_values[row_index].as_py()
                if sample_id != instance_to_sample[instance_id]:
                    raise DatasetValidationError(f"Embedding {instance_id!r} has the wrong sample key.")
                row_dimension = dimension_values[row_index].as_py()
                if row_dimension != embedding_dimension:
                    raise DatasetValidationError(f"Embedding {instance_id!r} has the wrong vector length.")
                if not finite_rows[row_index]:
                    raise DatasetValidationError(f"Embedding {instance_id!r} contains non-finite values.")

    return ValidationReport(
        samples=len(samples),
        instances=len(instances),
        masks=mask_count,
        embeddings=embedding_count,
    )


__all__ = (
    "DatasetValidationError",
    "ValidationReport",
    "expected_instance_id",
    "validate_dataset",
    "validate_published_build",
)
