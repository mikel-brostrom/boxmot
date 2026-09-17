from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, embeddings_schema
from boxmot.datasets.storage import (
    ENCODER_FINGERPRINT_METADATA_KEY,
    PARQUET_ROW_GROUP_ROWS,
    write_compacted_parquet_artifact,
)


@pytest.mark.parametrize("layout", ["overlapping", "disjoint", "single-unsorted", "empty"])
def test_wide_embeddings_compact_in_bounded_batches(tmp_path: Path, monkeypatch, layout: str) -> None:
    """Wide vectors must never be read or sorted as a whole artifact or shard."""
    dimension = 3584
    schema = embeddings_schema(dimension).with_metadata({ENCODER_FINGERPRINT_METADATA_KEY: b"e" * 64})
    count = 0 if layout == "empty" else PARQUET_ROW_GROUP_ROWS * 2 + 17
    rows = [
        {
            "sample_id": f"mot20:{index // 3}",
            "instance_id": f"build:mot20:{index // 3}:{index % 3}",
            "encoder_fingerprint": "e" * 64,
            "dim": dimension,
            "values": [float(index)] * dimension,
        }
        for index in range(count)
    ]
    expected = pa.Table.from_pylist(rows, schema=schema).sort_by([
        ("encoder_fingerprint", "ascending"), ("sample_id", "ascending"), ("instance_id", "ascending"),
    ])
    if layout == "overlapping":
        partitions = [rows[::2], rows[1::2]]
    elif layout == "disjoint":
        ordered = expected.to_pylist()
        partitions = [ordered[count // 2:], ordered[:count // 2]]
    else:
        partitions = [rows]
    source = tmp_path / "source"
    source.mkdir()
    for index, partition in enumerate(partitions):
        pq.write_table(
            pa.Table.from_pylist(list(reversed(partition)), schema=schema),
            source / f"part-{index:05d}.parquet",
            row_group_size=max(1, len(partition)),
        )

    real_iter = pq.ParquetFile.iter_batches
    real_read_group = pq.ParquetFile.read_row_group
    decoded_rows = []

    def forbid_whole_payload(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Embedding payloads must be streamed through bounded batches.")

    def keys_only(parquet, *args: Any, **kwargs: Any) -> pa.Table:
        assert kwargs.get("columns") is not None and "values" not in kwargs["columns"]
        return real_read_group(parquet, *args, **kwargs)

    def bounded_batches(parquet, *args: Any, **kwargs: Any) -> Iterator[pa.RecordBatch]:
        for batch in real_iter(parquet, *args, **kwargs):
            assert batch.num_rows <= PARQUET_ROW_GROUP_ROWS
            decoded_rows.append(batch.num_rows)
            yield batch

    with monkeypatch.context() as patch:
        patch.setattr(pq, "read_table", forbid_whole_payload)
        patch.setattr(pq.ParquetFile, "read", forbid_whole_payload)
        patch.setattr(pq.ParquetFile, "read_row_group", keys_only)
        patch.setattr(pq.ParquetFile, "iter_batches", bounded_batches)
        outputs = [
            write_compacted_parquet_artifact(
                source, tmp_path / name, artifact_name=EMBEDDINGS_ARTIFACT,
                box_type="aabb", target_rows=PARQUET_ROW_GROUP_ROWS + 7,
            )
            for name in ("first", "second")
        ]

    assert bool(decoded_rows) == bool(count)
    assert [sha256_file(path) for path in outputs[0]] == [sha256_file(path) for path in outputs[1]]
    for paths in outputs:
        actual = pa.concat_tables([pq.read_table(path) for path in paths])
        assert actual.schema.equals(schema, check_metadata=False)
        assert actual.schema.metadata == schema.metadata
        assert actual.cast(schema).equals(expected, check_metadata=True)
        for path in paths:
            metadata = pq.ParquetFile(path).metadata
            assert metadata.num_rows <= PARQUET_ROW_GROUP_ROWS + 7
            assert all(
                metadata.row_group(index).num_rows <= PARQUET_ROW_GROUP_ROWS
                for index in range(metadata.num_row_groups)
            )
