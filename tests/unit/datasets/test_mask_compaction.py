from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import boxmot.datasets.storage as storage_module
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.masks import MASK_CODEC
from boxmot.datasets.schema import MASKS_ARTIFACT, masks_schema
from boxmot.datasets.storage import PARQUET_ROW_GROUP_ROWS, write_compacted_parquet_artifact


def _mask_rows(count: int) -> list[dict[str, Any]]:
    """Produce distinct keyed payloads with realistic, unpadded frame IDs."""

    rows = []
    for index in range(count):
        sample_id = f"validation:MOT17-02-FRCNN:{index // 3}"
        rows.append(
            {
                "sample_id": sample_id,
                "instance_id": f"build:{sample_id}:{index % 3}",
                "height": 8,
                "width": 8,
                "codec": MASK_CODEC,
                "data": index.to_bytes(8, byteorder="little"),
            }
        )
    return rows


def _write_source_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    """Create a canonical shard, including legacy larger physical row groups."""

    pq.write_table(
        pa.Table.from_pylist(rows, schema=masks_schema()),
        path,
        compression="zstd",
        row_group_size=max(1, len(rows)),
    )


@pytest.mark.parametrize("layout", ["overlapping", "disjoint", "single-unsorted"])
def test_mask_compaction_streams_payloads_and_preserves_sorted_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    """Sorting mask keys must not gather the whole artifact's binary payload."""

    rows = _mask_rows(PARQUET_ROW_GROUP_ROWS * 2 + 17)
    sorted_rows = sorted(rows, key=lambda row: (row["sample_id"], row["instance_id"]))
    if layout == "overlapping":
        partitions = [rows[::2], list(reversed(rows[1::2]))]
    elif layout == "disjoint":
        midpoint = len(sorted_rows) // 2
        partitions = [list(reversed(sorted_rows[midpoint:])), list(reversed(sorted_rows[:midpoint]))]
    else:
        partitions = [list(reversed(rows))]

    source = tmp_path / "source"
    source.mkdir()
    _write_source_shard(source / "part-00000.parquet", [])
    for index, partition in enumerate(partitions, start=1):
        _write_source_shard(source / f"part-{index:05d}.parquet", partition)

    real_read_table = pq.read_table
    real_iter_batches = pq.ParquetFile.iter_batches
    payload_batch_rows = []

    def projected_read_only(path: Any, *args: Any, **kwargs: Any) -> pa.Table:
        """Allow full key reads while forbidding full mask payload reads."""

        columns = kwargs.get("columns")
        assert columns is not None and "data" not in columns, "Mask payloads must be streamed."
        return real_read_table(path, *args, **kwargs)

    def reject_full_file_read(*args: Any, **kwargs: Any) -> None:
        """Reject the alternative API for loading complete binary shards."""

        pytest.fail("Mask compaction must not load an entire Parquet file's payload.")

    def recording_iter_batches(parquet: pq.ParquetFile, *args: Any, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        """Observe the actual payload batches without changing their content."""

        for batch in real_iter_batches(parquet, *args, **kwargs):
            if "data" in batch.schema.names:
                payload_batch_rows.append(batch.num_rows)
            yield batch

    target_rows = PARQUET_ROW_GROUP_ROWS + 7
    with monkeypatch.context() as patch:
        patch.setattr(pq, "read_table", projected_read_only)
        patch.setattr(pq.ParquetFile, "read", reject_full_file_read)
        patch.setattr(pq.ParquetFile, "iter_batches", recording_iter_batches)
        outputs = [
            write_compacted_parquet_artifact(
                source,
                tmp_path / name,
                artifact_name=MASKS_ARTIFACT,
                box_type="aabb",
                target_rows=target_rows,
            )
            for name in ("first", "second")
        ]

    assert payload_batch_rows
    assert max(payload_batch_rows) <= PARQUET_ROW_GROUP_ROWS
    assert [sha256_file(path) for path in outputs[0]] == [sha256_file(path) for path in outputs[1]]
    assert [pq.ParquetFile(path).metadata.num_rows for path in outputs[0]] == [target_rows, target_rows, 3]
    for paths in outputs:
        actual_rows = []
        for path in paths:
            with pq.ParquetFile(path) as parquet:
                assert parquet.schema_arrow.equals(masks_schema(), check_metadata=False)
                assert parquet.schema_arrow.field("data").type == pa.binary()
                assert all(
                    parquet.metadata.row_group(index).num_rows <= PARQUET_ROW_GROUP_ROWS
                    for index in range(parquet.metadata.num_row_groups)
                )
            actual_rows.extend(pq.read_table(path).to_pylist())
        assert actual_rows == sorted_rows


def test_mask_compaction_preserves_an_empty_artifact(tmp_path: Path) -> None:
    """An empty mask artifact still publishes one shard with its exact schema."""

    source = tmp_path / "source"
    source.mkdir()
    for index in range(2):
        _write_source_shard(source / f"part-{index:05d}.parquet", [])

    paths = write_compacted_parquet_artifact(
        source,
        tmp_path / "compacted",
        artifact_name=MASKS_ARTIFACT,
        box_type="aabb",
        target_rows=3,
    )

    assert len(paths) == 1
    table = pq.read_table(paths[0])
    assert table.num_rows == 0
    assert table.schema.equals(masks_schema(), check_metadata=False)


def test_mask_compaction_bounds_variable_size_payloads_by_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mask dimensions must bound decoded and sorted bytes, not just row counts."""

    byte_budget = 64
    rows = _mask_rows(31)
    for index, row in enumerate(rows):
        row["width"] = (index % 3 + 1) * 8
        row["data"] = bytes([index]) * row["width"]
    assert sum(len(row["data"]) for row in rows) > byte_budget
    source = tmp_path / "source"
    source.mkdir()
    _write_source_shard(source / "part-00000.parquet", list(reversed(rows[::2])))
    _write_source_shard(source / "part-00001.parquet", rows[1::2])

    real_iter_batches = pq.ParquetFile.iter_batches
    real_write_table = pq.ParquetWriter.write_table
    decoded_bytes = []
    written_bytes = []

    def bounded_iter_batches(parquet: pq.ParquetFile, *args: Any, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        """Check actual decoded payload size using a small substitute budget."""

        for batch in real_iter_batches(parquet, *args, **kwargs):
            if "data" in batch.schema.names:
                size = sum(len(value) for value in batch.column("data").to_pylist())
                assert size <= byte_budget
                decoded_bytes.append(size)
            yield batch

    def bounded_write_table(writer: pq.ParquetWriter, table: pa.Table, *args: Any, **kwargs: Any) -> None:
        """Check sorted output stays bounded before Parquet encoding begins."""

        size = sum(len(value) for value in table.column("data").to_pylist())
        assert size <= byte_budget
        written_bytes.append(size)
        real_write_table(writer, table, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(storage_module, "_PAYLOAD_COMPACTION_BATCH_BYTES", byte_budget)
        patch.setattr(pq.ParquetFile, "iter_batches", bounded_iter_batches)
        patch.setattr(pq.ParquetWriter, "write_table", bounded_write_table)
        paths = write_compacted_parquet_artifact(
            source,
            tmp_path / "compacted",
            artifact_name=MASKS_ARTIFACT,
            box_type="aabb",
            target_rows=11,
        )

    assert decoded_bytes and written_bytes
    assert [row for path in paths for row in pq.read_table(path).to_pylist()] == sorted(
        rows, key=lambda row: (row["sample_id"], row["instance_id"])
    )
