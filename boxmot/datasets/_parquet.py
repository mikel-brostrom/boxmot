"""Conservative row-group selection for sequence-scoped Parquet readers."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import MANIFEST_FILENAME, SUCCESS_FILENAME
from .storage import artifact_files, resolve_artifact_path

if TYPE_CHECKING:
    from .manifest import DatasetManifest


def source_snapshot(build: Path, manifest: DatasetManifest) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Identify changes to publication metadata and immutable artifact files.

    This is a cheap reuse check, not a substitute for dataset validation. Both
    file identity and mutation timestamps are retained, including the current
    shard listing so added files also invalidate an in-process sequence view.
    """

    paths = [build / MANIFEST_FILENAME, build / SUCCESS_FILENAME]
    for artifact in manifest.artifacts:
        paths.extend(artifact_files(resolve_artifact_path(build, artifact.path)))
    result = []
    for path in paths:
        stat = path.stat()
        result.append(
            (
                path.relative_to(build).as_posix(),
                (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns),
            )
        )
    return tuple(result)


def row_group_may_contain_key(
    parquet: Any,
    row_group: int,
    *,
    column_name: str,
    keys: tuple[str, ...],
) -> bool:
    """Prune impossible string-key ranges without relying on physical ordering."""

    try:
        column_index = parquet.schema.names.index(column_name)
        statistics = parquet.metadata.row_group(row_group).column(column_index).statistics
        minimum = statistics.min
        maximum = statistics.max
        if isinstance(minimum, bytes):
            minimum = minimum.decode("utf-8")
        if isinstance(maximum, bytes):
            maximum = maximum.decode("utf-8")
        if not isinstance(minimum, str) or not isinstance(maximum, str):
            return True
    except (AttributeError, UnicodeDecodeError, ValueError):
        return True
    candidate = bisect_left(keys, minimum)
    return candidate < len(keys) and keys[candidate] <= maximum


def iter_selected_batches(
    build: Path,
    manifest: DatasetManifest,
    name: str,
    *,
    sample_ids: set[str],
    expected_schema: Any,
    columns: tuple[str, ...] | None = None,
    batch_size: int = 128,
) -> Iterator[Any]:
    """Yield selected rows with bounded decoding and deterministic file cleanup.

    Check the complete physical schema before projecting columns. Row-group
    statistics only prune impossible ranges; exact selection preserves joins
    for unsorted shards. Full selected batches retain their existing buffers.
    """

    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    artifact = manifest.artifact(name)
    path = resolve_artifact_path(build, artifact.path)
    sorted_sample_ids = tuple(sorted(sample_ids))
    selected_ids = pa.array(sorted_sample_ids, type=pa.string())
    for shard in artifact_files(path):
        with pq.ParquetFile(shard, pre_buffer=False) as parquet:
            actual_schema = parquet.schema_arrow
            if not actual_schema.equals(expected_schema, check_metadata=False):
                raise ValueError(
                    f"Parquet schema mismatch for {name!r}: expected {expected_schema}, got {actual_schema}. "
                    "Legacy or positional caches are not supported."
                )
            row_groups = [
                index
                for index in range(parquet.num_row_groups)
                if row_group_may_contain_key(parquet, index, column_name="sample_id", keys=sorted_sample_ids)
            ]
            if not row_groups:
                continue
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=columns,
                row_groups=row_groups,
                use_threads=False,
            ):
                sample_id_index = batch.schema.get_field_index("sample_id")
                selected = pc.is_in(batch.column(sample_id_index), value_set=selected_ids)
                if pc.any(selected).as_py():
                    yield batch if pc.all(selected).as_py() else batch.filter(selected)
