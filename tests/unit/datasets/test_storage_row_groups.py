from __future__ import annotations

from itertools import pairwise

import pyarrow as pa
import pyarrow.parquet as pq

import boxmot.datasets.storage as storage_module
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, MASKS_ARTIFACT, SAMPLES_ARTIFACT, samples_schema
from boxmot.datasets.storage import (
    PARQUET_ROW_GROUP_ROWS,
    describe_parquet_artifact,
    read_parquet_artifact,
    write_compacted_parquet_artifact,
    write_parquet_records,
)


def _row_group_sizes(path) -> list[int]:
    metadata = pq.ParquetFile(path).metadata
    return [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]


def _column_encodings(path, column: str) -> list[set[str]]:
    metadata = pq.ParquetFile(path).metadata
    encodings = []
    for row_group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_index)
        matches = [
            row_group.column(column_index)
            for column_index in range(row_group.num_columns)
            if row_group.column(column_index).path_in_schema == column
        ]
        assert len(matches) == 1
        encodings.append(set(matches[0].encodings))
    return encodings


def _assert_embedding_encodings(path) -> None:
    assert all("RLE_DICTIONARY" in encodings for encodings in _column_encodings(path, "sample_id"))
    for encodings in _column_encodings(path, "values.list.element"):
        assert "BYTE_STREAM_SPLIT" in encodings
        assert "RLE_DICTIONARY" not in encodings


def test_embedding_writes_use_bounded_row_groups_and_preserve_content(tmp_path) -> None:
    count = PARQUET_ROW_GROUP_ROWS + 1
    fingerprint = "a" * 64
    records = [
        {
            "sample_id": f"sample-{index:05d}",
            "instance_id": f"instance-{index:05d}",
            "encoder_fingerprint": fingerprint,
            "dim": 3,
            "values": [float(index), 1.0, 2.0],
        }
        for index in range(count)
    ]
    path = tmp_path / "embeddings.parquet"
    second_path = tmp_path / "embeddings-second.parquet"

    write_parquet_records(
        path,
        records,
        artifact_name=EMBEDDINGS_ARTIFACT,
        embedding_dim=3,
    )
    write_parquet_records(
        second_path,
        records,
        artifact_name=EMBEDDINGS_ARTIFACT,
        embedding_dim=3,
    )

    assert _row_group_sizes(path) == [PARQUET_ROW_GROUP_ROWS, 1]
    _assert_embedding_encodings(path)
    assert sha256_file(path) == sha256_file(second_path)
    assert read_parquet_artifact(path, artifact_name=EMBEDDINGS_ARTIFACT).to_pylist() == records


def test_mask_writes_use_bounded_row_groups_and_preserve_content(tmp_path) -> None:
    count = PARQUET_ROW_GROUP_ROWS + 1
    records = [
        {
            "sample_id": f"sample-{index:05d}",
            "instance_id": f"instance-{index:05d}",
            "height": 8,
            "width": 8,
            "codec": "bitpack-row-major-v1",
            "data": bytes((index % 256 for _ in range(8))),
        }
        for index in range(count)
    ]
    path = tmp_path / "masks.parquet"

    write_parquet_records(path, records, artifact_name=MASKS_ARTIFACT)

    assert _row_group_sizes(path) == [PARQUET_ROW_GROUP_ROWS, 1]
    assert read_parquet_artifact(path, artifact_name=MASKS_ARTIFACT).to_pylist() == records


def test_compaction_has_deterministic_bounded_prunable_row_groups(tmp_path) -> None:
    count = PARQUET_ROW_GROUP_ROWS * 2 + 7
    target_rows = PARQUET_ROW_GROUP_ROWS + 17
    fingerprint = "b" * 64
    records = [
        {
            "sample_id": f"sample-{index:05d}",
            "instance_id": f"instance-{index:05d}",
            "encoder_fingerprint": fingerprint,
            "dim": 2,
            "values": [float(index), 1.0],
        }
        for index in reversed(range(count))
    ]
    source = tmp_path / "source"
    source.mkdir()
    write_parquet_records(
        source / "part-00000.parquet",
        records,
        artifact_name=EMBEDDINGS_ARTIFACT,
        embedding_dim=2,
    )

    first_paths = write_compacted_parquet_artifact(
        source,
        tmp_path / "first",
        artifact_name=EMBEDDINGS_ARTIFACT,
        box_type="aabb",
        target_rows=target_rows,
    )
    second_paths = write_compacted_parquet_artifact(
        source,
        tmp_path / "second",
        artifact_name=EMBEDDINGS_ARTIFACT,
        box_type="aabb",
        target_rows=target_rows,
    )

    assert [path.name for path in first_paths] == ["part-00000.parquet", "part-00001.parquet"]
    assert [_row_group_sizes(path) for path in first_paths] == [
        [PARQUET_ROW_GROUP_ROWS, 17],
        [PARQUET_ROW_GROUP_ROWS - 10],
    ]
    assert [sha256_file(path) for path in first_paths] == [sha256_file(path) for path in second_paths]
    for path in first_paths:
        _assert_embedding_encodings(path)

    sample_ranges = []
    for path in first_paths:
        parquet = pq.ParquetFile(path)
        for index in range(parquet.metadata.num_row_groups):
            statistics = parquet.metadata.row_group(index).column(0).statistics
            assert statistics is not None and statistics.has_min_max
            sample_ranges.append((statistics.min, statistics.max))
    assert all(left[1] < right[0] for left, right in pairwise(sample_ranges))

    rows = read_parquet_artifact(tmp_path / "first", artifact_name=EMBEDDINGS_ARTIFACT).to_pylist()
    assert [row["sample_id"] for row in rows] == [f"sample-{index:05d}" for index in range(count)]


def test_compaction_merges_disjoint_sorted_runs_without_loading_the_full_artifact(tmp_path, monkeypatch) -> None:
    fingerprint = "c" * 64
    source = tmp_path / "source"
    source.mkdir()
    for shard_index, sample_id in enumerate(("sample-b", "sample-a")):
        records = [
            {
                "sample_id": sample_id,
                "instance_id": f"{sample_id}:{index}",
                "encoder_fingerprint": fingerprint,
                "dim": 2,
                "values": [float(index), 1.0],
            }
            for index in reversed(range(12))
        ]
        write_parquet_records(
            source / f"part-{shard_index:05d}.parquet",
            records,
            artifact_name=EMBEDDINGS_ARTIFACT,
            embedding_dim=2,
        )

    reads = []
    real_read = storage_module.read_parquet_artifact

    def recording_read(path, **kwargs):
        reads.append(path)
        return real_read(path, **kwargs)

    monkeypatch.setattr(storage_module, "read_parquet_artifact", recording_read)
    destination = tmp_path / "compacted"
    write_compacted_parquet_artifact(
        source,
        destination,
        artifact_name=EMBEDDINGS_ARTIFACT,
        box_type="aabb",
        target_rows=13,
    )

    assert {item.resolve() for item in reads} == {item.resolve() for item in source.glob("*.parquet")}
    rows = real_read(destination, artifact_name=EMBEDDINGS_ARTIFACT).to_pylist()
    assert [(row["sample_id"], row["instance_id"]) for row in rows] == sorted(
        (row["sample_id"], row["instance_id"]) for row in rows
    )


def test_compaction_falls_back_for_overlapping_shard_key_ranges(tmp_path, monkeypatch) -> None:
    fingerprint = "d" * 64
    source = tmp_path / "source"
    source.mkdir()
    for shard_index, sample_ids in enumerate((("sample-a", "sample-c"), ("sample-b", "sample-d"))):
        write_parquet_records(
            source / f"part-{shard_index:05d}.parquet",
            [
                {
                    "sample_id": sample_id,
                    "instance_id": f"{sample_id}:0",
                    "encoder_fingerprint": fingerprint,
                    "dim": 2,
                    "values": [float(shard_index), 1.0],
                }
                for sample_id in sample_ids
            ],
            artifact_name=EMBEDDINGS_ARTIFACT,
            embedding_dim=2,
        )

    reads = []
    real_read = storage_module.read_parquet_artifact

    def recording_read(path, **kwargs):
        reads.append(path)
        return real_read(path, **kwargs)

    monkeypatch.setattr(storage_module, "read_parquet_artifact", recording_read)
    destination = tmp_path / "compacted"
    write_compacted_parquet_artifact(
        source,
        destination,
        artifact_name=EMBEDDINGS_ARTIFACT,
        box_type="aabb",
        target_rows=3,
    )

    assert reads == [source]
    for path in destination.glob("*.parquet"):
        _assert_embedding_encodings(path)
    rows = real_read(destination, artifact_name=EMBEDDINGS_ARTIFACT).to_pylist()
    assert [row["sample_id"] for row in rows] == ["sample-a", "sample-b", "sample-c", "sample-d"]


def test_describe_parquet_artifact_hashes_each_shard_once(tmp_path, monkeypatch) -> None:
    source = tmp_path / "samples"
    source.mkdir()
    for shard_index in range(2):
        write_parquet_records(
            source / f"part-{shard_index:05d}.parquet",
            [
                {
                    "sample_id": f"sample-{shard_index}",
                    "split": "validation",
                    "sequence_id": "sequence",
                    "frame_index": shard_index,
                    "timestamp_s": None,
                    "image_ref": None,
                    "height": 8,
                    "width": 8,
                }
            ],
            artifact_name=SAMPLES_ARTIFACT,
        )

    calls = []
    real_sha256 = storage_module.sha256_file

    def recording_sha256(path):
        calls.append(path)
        return real_sha256(path)

    monkeypatch.setattr(storage_module, "sha256_file", recording_sha256)
    record = describe_parquet_artifact(tmp_path, name=SAMPLES_ARTIFACT, relative_path="samples")

    assert len(record.shards) == 2
    assert {item.resolve() for item in calls} == {item.resolve() for item in source.glob("*.parquet")}
    assert len(calls) == 2


def test_reader_accepts_legacy_single_row_group_artifacts(tmp_path) -> None:
    """The physical optimization must not become a loader schema requirement."""

    count = PARQUET_ROW_GROUP_ROWS + 1
    records = [
        {
            "sample_id": f"sample-{index:05d}",
            "split": "validation",
            "sequence_id": "sequence",
            "frame_index": index,
            "timestamp_s": None,
            "image_ref": None,
            "height": 8,
            "width": 8,
        }
        for index in range(count)
    ]
    path = tmp_path / "legacy-single-group.parquet"
    pq.write_table(
        pa.Table.from_pylist(records, schema=samples_schema()),
        path,
        compression="zstd",
        row_group_size=count,
    )

    assert _row_group_sizes(path) == [count]
    assert read_parquet_artifact(path, artifact_name=SAMPLES_ARTIFACT).to_pylist() == records
