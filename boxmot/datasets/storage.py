"""Low-level atomic Parquet I/O used by loaders and materializers."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes

from .manifest import ArtifactRecord, ManifestError, ShardRecord, sha256_file
from .masks import MASK_CODEC, pack_mask_batch
from .schema import (
    ARTIFACT_PATHS,
    ARTIFACT_SCHEMAS,
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MASKS_ARTIFACT,
    BoxType,
    embeddings_schema,
    instances_schema,
    masks_schema,
    require_pyarrow,
)

if TYPE_CHECKING:
    import pyarrow as pa

ENCODER_FINGERPRINT_METADATA_KEY = b"boxmot.encoder_fingerprint"

# Keep column chunks small enough that keyed reads do not have to inflate an
# entire 50,000-row publication shard.  For a 3,584-wide float32 embedding,
# 128 rows are about 1.75 MiB before Parquet encoding.  Full-frame bit-packed
# masks are wider, but remain bounded to the same deterministic row count.
# This is a physical storage policy and therefore deliberately does not
# participate in dataset or stage fingerprints.
PARQUET_ROW_GROUP_ROWS = 128
_PAYLOAD_COMPACTION_BATCH_BYTES = 64 * 1024 * 1024

_ARTIFACT_SORT_KEYS = {
    "samples": ("split", "sequence_id", "frame_index", "sample_id"),
    "instances": ("sample_id", "detection_index", "instance_id"),
    "masks": ("sample_id", "instance_id"),
    "embeddings": ("encoder_fingerprint", "sample_id", "instance_id"),
}

# Embedding vectors are high-cardinality floating-point data. Dictionary
# encoding them adds a dictionary plus indices for almost every unique value,
# making the files larger and substantially slower to write. Keep dictionary
# encoding for the repeated key/metadata columns and use byte-stream split for
# the float leaf so Zstandard can compress like-significance bytes together.
_EMBEDDING_DICTIONARY_COLUMNS = (
    "sample_id",
    "instance_id",
    "encoder_fingerprint",
    "dim",
)
_EMBEDDING_BYTE_STREAM_SPLIT_COLUMNS = ("values.list.element",)


def _parquet_encoding_options(artifact_name: str) -> dict[str, Any]:
    """Return artifact-specific physical encodings for Parquet writers."""

    if artifact_name != EMBEDDINGS_ARTIFACT:
        return {}
    return {
        "use_dictionary": list(_EMBEDDING_DICTIONARY_COLUMNS),
        "use_byte_stream_split": list(_EMBEDDING_BYTE_STREAM_SPLIT_COLUMNS),
    }


def sample_record(
    frame: Frame,
    *,
    split: str,
    image_ref: str | None,
) -> dict[str, Any]:
    """Convert one canonical frame to a sample table row."""

    if not isinstance(split, str) or not split or split != split.strip():
        raise ValueError("split must be a non-empty canonical string.")
    if image_ref is not None and (not isinstance(image_ref, str) or not image_ref or image_ref != image_ref.strip()):
        raise ValueError("image_ref must be a non-empty canonical string or None.")
    if frame.sequence_id is None or frame.frame_index is None:
        raise ValueError("Materialized frames require sequence_id and frame_index.")
    return {
        "sample_id": frame.sample_id,
        "split": split,
        "sequence_id": frame.sequence_id,
        "frame_index": frame.frame_index,
        "timestamp_s": frame.timestamp_s,
        "image_ref": image_ref,
        "height": frame.height,
        "width": frame.width,
    }


def instance_records(detections: Detections, *, build_id: str) -> list[dict[str, Any]]:
    """Convert detections to stable, explicitly typed instance rows."""

    expected_ids = tuple(f"{build_id}:{detections.sample_id}:{index}" for index in range(len(detections)))
    if detections.instance_ids != expected_ids:
        raise ValueError("Detections must carry the canonical build/sample/detection instance IDs before writing.")

    geometry = detections.geometry.values.tolist()
    scores = detections.scores.tolist()
    classes = detections.class_ids.tolist()
    records: list[dict[str, Any]] = []
    for index, (instance_id, values, score, class_id) in enumerate(
        zip(expected_ids, geometry, scores, classes, strict=True)
    ):
        record = {
            "instance_id": instance_id,
            "sample_id": detections.sample_id,
            "detection_index": index,
            "score": score,
            "class_id": class_id,
        }
        if isinstance(detections.geometry, Boxes):
            record.update(dict(zip(("x1", "y1", "x2", "y2"), values, strict=True)))
        elif isinstance(detections.geometry, OrientedBoxes):
            record.update(dict(zip(("cx", "cy", "w", "h", "angle"), values, strict=True)))
        else:  # Detections validates this; retain a defensive error at persistence boundary.
            raise TypeError(f"Unsupported geometry type {type(detections.geometry).__name__}.")
        records.append(record)
    return records


def mask_records(detections: Detections) -> list[dict[str, Any]]:
    """Encode row-aligned full-frame masks using stable keys."""

    if detections.instance_ids is None:
        raise ValueError("Mask rows require stable instance IDs.")
    if detections.masks is None:
        raise ValueError("Detections do not contain masks.")
    height, width = detections.masks.image_size
    payloads = pack_mask_batch(detections.masks.values)
    return [
        {
            "sample_id": detections.sample_id,
            "instance_id": instance_id,
            "height": height,
            "width": width,
            "codec": MASK_CODEC,
            "data": payload,
        }
        for instance_id, payload in zip(detections.instance_ids, payloads, strict=True)
    ]


def embedding_records(detections: Detections, *, encoder_fingerprint: str) -> list[dict[str, Any]]:
    """Convert row-aligned appearance embeddings to keyed records."""

    if not encoder_fingerprint:
        raise ValueError("encoder_fingerprint must not be empty.")
    if detections.instance_ids is None:
        raise ValueError("Embedding rows require stable instance IDs.")
    if detections.embeddings is None:
        raise ValueError("Detections do not contain embeddings.")
    dim = int(detections.embeddings.shape[1])
    return [
        {
            "sample_id": detections.sample_id,
            "instance_id": instance_id,
            "encoder_fingerprint": encoder_fingerprint,
            "dim": dim,
            "values": values,
        }
        for instance_id, values in zip(detections.instance_ids, detections.embeddings.tolist(), strict=True)
    ]


def resolve_artifact_path(root: str | Path, relative_path: str) -> Path:
    """Resolve a manifest-owned artifact without permitting root traversal."""

    build_root = Path(root).resolve()
    candidate = (build_root / relative_path).resolve()
    try:
        candidate.relative_to(build_root)
    except ValueError as exc:
        raise ManifestError(f"Artifact path escapes the dataset root: {relative_path!r}.") from exc
    return candidate


def write_parquet_records(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    artifact_name: str,
    box_type: BoxType | None = None,
    embedding_dim: int | None = None,
    encoder_fingerprint: str | None = None,
    compression: str = "zstd",
) -> Path:
    """Atomically write records using the exact versioned artifact schema."""

    materialized_records = list(records)
    if artifact_name == EMBEDDINGS_ARTIFACT:
        if embedding_dim is None:
            raise ValueError("embedding_dim is required when writing embeddings.")
        schema = embeddings_schema(embedding_dim)
        row_fingerprints = {str(record["encoder_fingerprint"]) for record in materialized_records}
        if encoder_fingerprint is not None:
            row_fingerprints.add(encoder_fingerprint)
        if len(row_fingerprints) != 1:
            raise ValueError("Embedding shards require exactly one encoder fingerprint, including when empty.")
        resolved_fingerprint = next(iter(row_fingerprints))
        if not re.fullmatch(r"[0-9a-f]{64}", resolved_fingerprint):
            raise ValueError("encoder_fingerprint must be a full SHA-256 digest.")
        schema = schema.with_metadata({ENCODER_FINGERPRINT_METADATA_KEY: resolved_fingerprint.encode("ascii")})
    elif artifact_name == INSTANCES_ARTIFACT:
        if box_type is None:
            raise ValueError("box_type is required when writing instances.")
        schema = instances_schema(box_type)
    else:
        try:
            schema = ARTIFACT_SCHEMAS[artifact_name]()
        except KeyError as exc:
            raise ValueError(f"Unknown materialized dataset artifact {artifact_name!r}.") from exc

    pa = require_pyarrow()
    import pyarrow.parquet as pq

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(materialized_records, schema=schema)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}-", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(
            table,
            tmp_path,
            compression=compression,
            row_group_size=PARQUET_ROW_GROUP_ROWS,
            **_parquet_encoding_options(artifact_name),
        )
        with tmp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(tmp_path, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Directory fsync is unavailable on some supported filesystems.
            pass
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def read_parquet_artifact(
    path: str | Path,
    *,
    artifact_name: str,
    box_type: BoxType | None = None,
    columns: list[str] | None = None,
    filters: Any | None = None,
):
    """Read an artifact and reject schema drift before returning its table."""

    import pyarrow.parquet as pq

    resolved = Path(path)
    table = pq.read_table(resolved, columns=columns, filters=filters)
    # A projected read intentionally has a projected schema. Callers that
    # request columns validate the complete physical schema separately before
    # using this low-level reader.
    if columns is not None:
        return table
    if artifact_name == EMBEDDINGS_ARTIFACT:
        try:
            values_type = table.schema.field("values").type
            dimension = int(values_type.list_size)
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Embedding values must use a fixed-size float32 list type.") from exc
        expected = embeddings_schema(dimension)
    elif artifact_name == INSTANCES_ARTIFACT:
        if box_type is None:
            raise ValueError("box_type is required when reading instances.")
        expected = instances_schema(box_type)
    else:
        try:
            expected = ARTIFACT_SCHEMAS[artifact_name]()
        except KeyError as exc:
            raise ValueError(f"Unknown materialized dataset artifact {artifact_name!r}.") from exc
    if not table.schema.equals(expected, check_metadata=False):
        raise ValueError(
            f"Parquet schema mismatch for {artifact_name!r}: expected {expected}, got {table.schema}. "
            "Legacy or positional caches are not supported."
        )
    return table


def _iter_sorted_payload_tables(
    source_files: tuple[Path, ...], *, artifact_name: str,
) -> Iterator["pa.Table"]:
    """Sort wide artifact keys globally while decoding bounded payload batches.

    Frame-number strings can interleave shard key ranges. Sorting the complete
    payload can exhaust memory; binary masks can also overflow Arrow's 32-bit
    offsets. Keep only keys and source locations in the global sort instead.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    physical_schema = pq.read_schema(source_files[0])
    is_mask = artifact_name == MASKS_ARTIFACT
    schema = masks_schema() if is_mask else embeddings_schema(physical_schema.field("values").type.list_size)
    schema = schema.with_metadata(physical_schema.metadata)
    sort_keys = _ARTIFACT_SORT_KEYS[artifact_name]
    locations: list[tuple[Path, int, int, int]] = []
    key_tables = []
    maximum_payload_bytes = 1
    for source_file in source_files:
        with pq.ParquetFile(source_file, pre_buffer=False) as parquet:
            if not parquet.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError(
                    f"Parquet schema mismatch for {artifact_name!r}: expected {schema}, got {parquet.schema_arrow}."
                )
            for row_group in range(parquet.num_row_groups):
                keys = parquet.read_row_group(
                    row_group, columns=[*sort_keys, "height", "width"] if is_mask else list(sort_keys), use_threads=False,
                )
                if is_mask:
                    largest_payload = max(
                        ((height * width + 7) // 8 for height, width in zip(
                            keys.column("height").to_pylist(), keys.column("width").to_pylist(), strict=True,
                        )),
                        default=1,
                    )
                else:
                    largest_payload = schema.field("values").type.list_size * 4
                maximum_payload_bytes = max(maximum_payload_bytes, largest_payload)
                batch_rows = min(
                    PARQUET_ROW_GROUP_ROWS, max(1, _PAYLOAD_COMPACTION_BATCH_BYTES // max(1, largest_payload)),
                )
                keys = keys.select(sort_keys)
                for batch_index, offset in enumerate(range(0, keys.num_rows, batch_rows)):
                    batch_keys = keys.slice(offset, batch_rows)
                    location = len(locations)
                    locations.append((source_file, row_group, batch_index, batch_rows))
                    batch_keys = batch_keys.append_column(
                        "_location", pa.repeat(pa.scalar(location, type=pa.int64()), batch_keys.num_rows),
                    ).append_column("_row", pa.array(range(batch_keys.num_rows), type=pa.int64()))
                    key_tables.append(batch_keys)

    if not key_tables:
        yield pa.Table.from_batches([], schema=schema)
        return

    keys = pa.concat_tables(key_tables)
    del key_tables
    order = keys.sort_by([(key, "ascending") for key in (*sort_keys, "_location", "_row")])
    order = order.select(("_location", "_row"))
    del keys

    @lru_cache(maxsize=4)
    def read_batch(location: int) -> "pa.Table":
        """Borrow a small decoded batch, keeping file handles scoped to reads."""
        source_file, row_group, batch_index, batch_rows = locations[location]
        with pq.ParquetFile(source_file, pre_buffer=False) as parquet:
            for index, batch in enumerate(parquet.iter_batches(
                batch_size=batch_rows, row_groups=[row_group], use_threads=False,
            )):
                if index == batch_index:
                    return pa.Table.from_batches([batch], schema=schema)
        raise ValueError(f"Artifact source changed during compaction: {source_file}")

    output_rows = min(PARQUET_ROW_GROUP_ROWS, max(1, _PAYLOAD_COMPACTION_BATCH_BYTES // maximum_payload_bytes))
    for offset in range(0, order.num_rows, output_rows):
        selected = order.slice(offset, output_rows)
        groups: dict[int, list[tuple[int, int]]] = {}
        for position, (location, row) in enumerate(zip(
            selected.column("_location").to_pylist(), selected.column("_row").to_pylist(), strict=True,
        )):
            groups.setdefault(location, []).append((position, row))
        parts = []
        positions = []
        for location, rows in groups.items():
            parts.append(read_batch(location).take(pa.array([row for _, row in rows], type=pa.int64())))
            positions.extend(position for position, _ in rows)
        restore_order = sorted(range(len(positions)), key=positions.__getitem__)
        yield pa.concat_tables(parts).take(pa.array(restore_order, type=pa.int64()))


def _write_compacted_tables(
    tables: Iterable["pa.Table"], destination: str | Path, *, artifact_name: str, target_rows: int,
) -> tuple[Path, ...]:
    """Write a sorted table stream to deterministic shards and small row groups."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(destination)
    output.mkdir(parents=True)
    paths: list[Path] = []
    writer = None
    rows_in_shard = 0
    schema = None
    try:
        for table in tables:
            schema = table.schema if schema is None else schema
            offset = 0
            while offset < table.num_rows:
                if writer is None:
                    path = output / f"part-{len(paths):05d}.parquet"
                    writer = pq.ParquetWriter(
                        path, schema, compression="zstd", **_parquet_encoding_options(artifact_name),
                    )
                    paths.append(path)
                count = min(table.num_rows - offset, target_rows - rows_in_shard)
                writer.write_table(table.slice(offset, count), row_group_size=PARQUET_ROW_GROUP_ROWS)
                offset += count
                rows_in_shard += count
                if rows_in_shard == target_rows:
                    writer.close()
                    writer = None
                    rows_in_shard = 0
        if writer is not None:
            writer.close()
            writer = None
        if not paths:
            if schema is None:
                raise FileNotFoundError(f"Parquet artifact has no input tables: {destination}")
            path = output / "part-00000.parquet"
            pq.write_table(
                pa.Table.from_batches([], schema=schema), path, compression="zstd",
                row_group_size=PARQUET_ROW_GROUP_ROWS, **_parquet_encoding_options(artifact_name),
            )
            paths.append(path)
        return tuple(paths)
    finally:
        if writer is not None:
            writer.close()


def write_compacted_parquet_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    artifact_name: str,
    box_type: BoxType,
    target_rows: int = 50_000,
) -> tuple[Path, ...]:
    """Write globally sorted, bounded canonical shards to a new directory.

    The caller owns durability and directory publication. This helper owns only
    the schema-specific ordering and Parquet representation. Masks and
    embeddings sort keys globally and gather bounded payload batches, including
    overlapping key ranges. Narrow artifacts merge disjoint source runs and
    fall back to a whole-table sort when their key ranges overlap.
    """

    if target_rows <= 0:
        raise ValueError("target_rows must be positive.")
    try:
        sort_keys = _ARTIFACT_SORT_KEYS[artifact_name]
    except KeyError as exc:
        raise ValueError(f"Unknown materialized dataset artifact {artifact_name!r}.") from exc

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source_files = artifact_files(source)
    if artifact_name in {MASKS_ARTIFACT, EMBEDDINGS_ARTIFACT}:
        return _write_compacted_tables(
            _iter_sorted_payload_tables(source_files, artifact_name=artifact_name),
            destination, artifact_name=artifact_name, target_rows=target_rows,
        )
    runs = []
    for source_file in source_files:
        keys = pq.read_table(source_file, columns=list(sort_keys))
        indices = pc.sort_indices(keys, sort_keys=[(key, "ascending") for key in sort_keys])
        if keys.num_rows:
            first_index = indices[0].as_py()
            last_index = indices[-1].as_py()
            first_key = tuple(keys.column(key)[first_index].as_py() for key in sort_keys)
            last_key = tuple(keys.column(key)[last_index].as_py() for key in sort_keys)
        else:
            first_key = last_key = None
        identity = indices.equals(pa.array(range(keys.num_rows), type=indices.type))
        runs.append((source_file, indices, identity, first_key, last_key))

    populated = sorted((run for run in runs if run[3] is not None), key=lambda run: run[3])
    ranges_do_not_overlap = all(left[4] < right[3] for left, right in zip(populated, populated[1:], strict=False))
    if ranges_do_not_overlap:
        ordered = [*populated, *(run for run in runs if run[3] is None)]

        def sorted_tables():
            for source_file, indices, identity, _first_key, _last_key in ordered:
                table = read_parquet_artifact(
                    source_file,
                    artifact_name=artifact_name,
                    box_type=box_type,
                )
                yield table if identity else table.take(indices)

        return _write_compacted_tables(
            sorted_tables(), destination, artifact_name=artifact_name, target_rows=target_rows,
        )

    # Arbitrary callers may provide shards whose key ranges overlap. Retain
    # the general full-table fallback for narrow inputs. Wide payloads use
    # bounded gathering above even when frame-number strings interleave shards.
    table = read_parquet_artifact(source, artifact_name=artifact_name, box_type=box_type)
    table = table.sort_by([(key, "ascending") for key in sort_keys])
    return _write_compacted_tables(
        (table,), destination, artifact_name=artifact_name, target_rows=target_rows,
    )


def _iter_rekeyed_instance_tables(
    source: str | Path,
    *,
    box_type: BoxType,
    source_build_id: str,
    target_build_id: str,
    batch_rows: int = 8192,
) -> Iterator["pa.Table"]:
    """Yield schema-checked instance batches with canonical target IDs."""

    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive.")

    pa = require_pyarrow()
    import pyarrow.parquet as pq

    expected_schema = instances_schema(box_type)
    instance_column = expected_schema.get_field_index("instance_id")
    sample_column = expected_schema.get_field_index("sample_id")
    detection_column = expected_schema.get_field_index("detection_index")
    for source_file in artifact_files(source):
        parquet = pq.ParquetFile(source_file)
        if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
            raise ValueError(f"Instance artifact schema mismatch: {source_file}")
        for batch in parquet.iter_batches(batch_size=batch_rows):
            table = pa.Table.from_batches((batch,))
            sample_ids = table.column(sample_column).to_pylist()
            detection_indices = table.column(detection_column).to_pylist()
            instance_ids = table.column(instance_column).to_pylist()
            expected_ids = [
                f"{source_build_id}:{sample_id}:{detection_index}"
                for sample_id, detection_index in zip(sample_ids, detection_indices, strict=True)
            ]
            if instance_ids != expected_ids:
                raise ValueError("Instance artifact contains non-canonical instance IDs.")
            target_ids = [
                f"{target_build_id}:{sample_id}:{detection_index}"
                for sample_id, detection_index in zip(sample_ids, detection_indices, strict=True)
            ]
            yield table.set_column(
                instance_column,
                expected_schema.field(instance_column),
                pa.array(target_ids, type=pa.string()),
            )


def _fsync_directory(path: Path) -> None:
    """Best-effort fsync for a directory containing completed shards."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Directory fsync is unavailable on some supported filesystems.
        pass


def write_rekeyed_instance_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    box_type: BoxType,
    source_build_id: str,
    target_build_id: str,
    target_rows: int = 50_000,
) -> tuple[Path, ...]:
    """Stream instance rows into bounded shards with target-build IDs.

    ``destination`` must be a new artifact directory. The caller owns its
    eventual atomic publication and cleanup if this operation fails.
    """

    if target_rows <= 0:
        raise ValueError("target_rows must be positive.")

    pa = require_pyarrow()
    import pyarrow.parquet as pq

    expected_schema = instances_schema(box_type)
    output = Path(destination)
    output.mkdir(parents=True)
    writer = None
    paths: list[Path] = []
    rows_in_shard = 0

    def open_writer():
        path = output / f"part-{len(paths):05d}.parquet"
        paths.append(path)
        return pq.ParquetWriter(path, expected_schema, compression="zstd")

    def close_writer() -> None:
        nonlocal writer, rows_in_shard
        if writer is None:
            return
        active_writer = writer
        writer = None
        active_writer.close()
        with paths[-1].open("rb") as stream:
            os.fsync(stream.fileno())
        rows_in_shard = 0

    try:
        for table in _iter_rekeyed_instance_tables(
            source,
            box_type=box_type,
            source_build_id=source_build_id,
            target_build_id=target_build_id,
        ):
            offset = 0
            while offset < table.num_rows:
                if writer is None:
                    writer = open_writer()
                count = min(target_rows - rows_in_shard, table.num_rows - offset)
                writer.write_table(table.slice(offset, count), row_group_size=PARQUET_ROW_GROUP_ROWS)
                rows_in_shard += count
                offset += count
                if rows_in_shard == target_rows:
                    close_writer()
        if not paths:
            writer = open_writer()
            writer.write_table(pa.Table.from_pylist([], schema=expected_schema))
        close_writer()
        _fsync_directory(output)
    except BaseException as exc:
        try:
            close_writer()
        except BaseException as close_error:
            add_note = getattr(exc, "add_note", None)
            if callable(add_note):
                add_note(f"Instance artifact writer cleanup also failed: {close_error}")
        raise
    return tuple(paths)


def write_repartitioned_instance_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    box_type: BoxType,
    source_build_id: str,
    target_build_id: str,
    shard_by_sample: Mapping[str, int],
    shard_count: int,
) -> tuple[Path, ...]:
    """Stream and rekey instances into caller-defined sample partitions.

    This preserves checkpoint shard semantics when a compact published
    artifact is restored into a resumable materialization stage.
    ``destination`` must be a new artifact directory.
    """

    if shard_count <= 0:
        raise ValueError("shard_count must be positive.")
    if any(
        not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < shard_count
        for index in shard_by_sample.values()
    ):
        raise ValueError("Sample shard indices must be integers within shard_count.")

    pa = require_pyarrow()
    import pyarrow.parquet as pq

    expected_schema = instances_schema(box_type)
    sample_column = expected_schema.get_field_index("sample_id")
    output = Path(destination)
    output.mkdir(parents=True)
    fragment_root = Path(tempfile.mkdtemp(prefix=".instance-fragments-", dir=output.parent))
    paths: list[Path] = []
    try:
        for batch_index, table in enumerate(
            _iter_rekeyed_instance_tables(
                source,
                box_type=box_type,
                source_build_id=source_build_id,
                target_build_id=target_build_id,
            )
        ):
            rows_by_shard: dict[int, list[int]] = {}
            for row_index, sample_id in enumerate(table.column(sample_column).to_pylist()):
                try:
                    shard_index = shard_by_sample[sample_id]
                except KeyError as exc:
                    raise ValueError(f"Instance artifact contains unknown sample {sample_id!r}.") from exc
                rows_by_shard.setdefault(shard_index, []).append(row_index)
            for shard_index, row_indices in rows_by_shard.items():
                fragment_directory = fragment_root / f"{shard_index:05d}"
                fragment_directory.mkdir(exist_ok=True)
                pq.write_table(
                    table.take(pa.array(row_indices, type=pa.int64())),
                    fragment_directory / f"part-{batch_index:08d}.parquet",
                    compression="zstd",
                    row_group_size=PARQUET_ROW_GROUP_ROWS,
                )

        for shard_index in range(shard_count):
            instance_path = output / f"part-{shard_index:05d}.parquet"
            writer = pq.ParquetWriter(instance_path, expected_schema, compression="zstd")
            try:
                fragments = sorted((fragment_root / f"{shard_index:05d}").glob("*.parquet"))
                if fragments:
                    for fragment in fragments:
                        writer.write_table(pq.read_table(fragment), row_group_size=PARQUET_ROW_GROUP_ROWS)
                else:
                    writer.write_table(pa.Table.from_pylist([], schema=expected_schema))
            finally:
                writer.close()
            with instance_path.open("rb") as stream:
                os.fsync(stream.fileno())
            paths.append(instance_path)
        _fsync_directory(output)
    finally:
        shutil.rmtree(fragment_root, ignore_errors=True)
    return tuple(paths)


def resolve_embedding_metadata(
    table: "pa.Table",
    *,
    declared: Mapping[str, Any] | None = None,
    schema: "pa.Schema | None" = None,
) -> dict[str, Any]:
    """Resolve and validate one embedding artifact's dimension and encoder.

    ``schema`` allows callers that scan only the narrow metadata columns to
    validate the fixed-size embedding type without materializing the vector
    payload.  The default retains the established full-table behavior.
    """

    resolved_schema = table.schema if schema is None else schema

    try:
        dimension = int(resolved_schema.field("values").type.list_size)
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Embedding values must use a fixed-size float32 list type.") from exc
    if dimension <= 0:
        raise ValueError("Embedding metadata dim must be a positive integer.")

    schema_metadata = resolved_schema.metadata or {}
    encoded_fingerprint = schema_metadata.get(ENCODER_FINGERPRINT_METADATA_KEY)
    try:
        schema_fingerprint = None if encoded_fingerprint is None else encoded_fingerprint.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Embedding schema contains a non-ASCII encoder fingerprint.") from exc

    row_dimensions = {int(value) for value in table.column("dim").to_pylist()}
    dimensions = {dimension, *row_dimensions}
    fingerprints = {str(value) for value in table.column("encoder_fingerprint").to_pylist()}
    if schema_fingerprint is not None:
        fingerprints.add(schema_fingerprint)

    if declared is not None:
        declared_dimension = declared.get("dim")
        if declared_dimension is not None:
            if isinstance(declared_dimension, bool) or not isinstance(declared_dimension, int):
                raise ValueError("Embedding metadata dim must be a positive integer.")
            if declared_dimension != dimension:
                raise ValueError("Embedding metadata dim does not match the fixed-size Parquet schema.")
        declared_fingerprint = declared.get("encoder_fingerprint")
        if declared_fingerprint is not None:
            fingerprints.add(str(declared_fingerprint))

    if len(dimensions) != 1 or next(iter(dimensions)) <= 0:
        raise ValueError("Embedding dimensions must be one consistent positive value.")
    if len(fingerprints) != 1:
        raise ValueError("Published embeddings require exactly one resolved encoder fingerprint.")
    encoder_fingerprint = next(iter(fingerprints))
    if not re.fullmatch(r"[0-9a-f]{64}", encoder_fingerprint):
        raise ValueError("Embedding metadata encoder_fingerprint must be a full SHA-256 digest.")
    return {"encoder_fingerprint": encoder_fingerprint, "dim": dimension}


def read_embedding_metadata(
    path: str | Path,
    *,
    declared: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve embedding metadata without materializing vector payloads."""

    import pyarrow.parquet as pq

    resolved = Path(path)
    table = read_parquet_artifact(
        resolved,
        artifact_name=EMBEDDINGS_ARTIFACT,
        columns=["encoder_fingerprint", "dim"],
    )
    schema = pq.read_schema(artifact_files(resolved)[0])
    return resolve_embedding_metadata(table, declared=declared, schema=schema)


def artifact_files(path: str | Path) -> tuple[Path, ...]:
    """Return deterministic Parquet shards owned by a file or directory artifact."""

    resolved = Path(path)
    if resolved.is_file():
        return (resolved,)
    if resolved.is_dir():
        files = tuple(sorted(item for item in resolved.rglob("*.parquet") if item.is_file()))
        if files:
            return files
    raise FileNotFoundError(f"Parquet artifact has no shards: {resolved}")


def artifact_size(path: str | Path) -> int:
    """Return the total bytes across deterministic Parquet shards."""

    return sum(item.stat().st_size for item in artifact_files(path))


def sha256_artifact(path: str | Path) -> str:
    """Hash shard names and contents so directory artifacts are immutable."""

    root = Path(path)
    files = artifact_files(root)
    if root.is_file():
        return sha256_file(root)
    digest = hashlib.sha256()
    for file in files:
        relative = file.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


def _sha256_from_shard_records(root: Path, files: tuple[Path, ...], shards: tuple[ShardRecord, ...]) -> str:
    """Derive an artifact digest from shard digests already read from disk."""

    if root.is_file():
        return shards[0].sha256
    digest = hashlib.sha256()
    for file, shard in zip(files, shards, strict=True):
        relative = file.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(shard.sha256))
    return digest.hexdigest()


def describe_parquet_artifact(
    root: str | Path,
    *,
    name: str,
    relative_path: str,
    metadata: Mapping[str, Any] | None = None,
) -> ArtifactRecord:
    """Build a checksummed manifest record for an existing Parquet file."""

    import pyarrow.parquet as pq

    resolved_root = Path(root).resolve()
    path = resolve_artifact_path(resolved_root, relative_path)
    files = artifact_files(path)
    shards = tuple(
        ShardRecord(
            path=file.relative_to(resolved_root).as_posix(),
            rows=pq.ParquetFile(file).metadata.num_rows,
            sha256=sha256_file(file),
            size_bytes=file.stat().st_size,
        )
        for file in files
    )
    rows = sum(shard.rows for shard in shards)
    return ArtifactRecord(
        name=name,
        path=relative_path,
        rows=rows,
        sha256=_sha256_from_shard_records(path, files, shards),
        size_bytes=sum(shard.size_bytes for shard in shards),
        shards=shards,
        metadata={} if metadata is None else metadata,
    )


class ParquetShardWriter:
    """Write deterministic ``part-NNNNN.parquet`` files into a dataset root."""

    def __init__(self, root: str | Path, *, box_type: BoxType, compression: str = "zstd") -> None:
        self.root = Path(root)
        self.box_type = box_type
        self.compression = compression

    def write(
        self,
        artifact_name: str,
        records: Iterable[Mapping[str, Any]],
        *,
        shard_index: int,
        embedding_dim: int | None = None,
        encoder_fingerprint: str | None = None,
    ) -> Path:
        if shard_index < 0:
            raise ValueError("shard_index must be non-negative.")
        try:
            directory = ARTIFACT_PATHS[artifact_name]
        except KeyError as exc:
            raise ValueError(f"Unknown materialization artifact {artifact_name!r}.") from exc
        relative = Path(directory) / f"part-{shard_index:05d}.parquet"
        return write_parquet_records(
            self.root / relative,
            records,
            artifact_name=artifact_name,
            box_type=self.box_type,
            embedding_dim=embedding_dim,
            encoder_fingerprint=encoder_fingerprint,
            compression=self.compression,
        )

    def describe(self, artifact_name: str, *, metadata: Mapping[str, Any] | None = None):
        try:
            relative = ARTIFACT_PATHS[artifact_name]
        except KeyError as exc:
            raise ValueError(f"Unknown materialization artifact {artifact_name!r}.") from exc
        resolved_metadata = {} if metadata is None else metadata
        if artifact_name == EMBEDDINGS_ARTIFACT:
            if "dim" not in resolved_metadata or "encoder_fingerprint" not in resolved_metadata:
                raise ValueError("Embedding artifact metadata requires dim and encoder_fingerprint.")
        return describe_parquet_artifact(
            self.root,
            name=artifact_name,
            relative_path=relative,
            metadata=resolved_metadata,
        )


__all__ = (
    "ParquetShardWriter",
    "PARQUET_ROW_GROUP_ROWS",
    "describe_parquet_artifact",
    "ENCODER_FINGERPRINT_METADATA_KEY",
    "artifact_files",
    "artifact_size",
    "embedding_records",
    "instance_records",
    "mask_records",
    "read_embedding_metadata",
    "read_parquet_artifact",
    "resolve_embedding_metadata",
    "resolve_artifact_path",
    "sample_record",
    "sha256_artifact",
    "write_compacted_parquet_artifact",
    "write_parquet_records",
    "write_rekeyed_instance_artifact",
    "write_repartitioned_instance_artifact",
)
