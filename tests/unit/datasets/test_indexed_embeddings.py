"""Indexed replay must preserve key joins without retaining a sequence of vectors."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import boxmot.datasets._embedding_reader as embedding_reader
from boxmot.datasets._embedding_reader import IndexedEmbeddingReader
from boxmot.datasets.cached import CachedVisionDataset
from boxmot.datasets.schema import embeddings_schema
from boxmot.datasets.validation import DatasetValidationError


def _dataset(tmp_path, rows, *, row_group_size=128, shards=1, dimension=7, samples=None):
    """Write genuine Parquet shards around the reader's small dataset interface."""

    directory = tmp_path / "embeddings"
    directory.mkdir(exist_ok=True)
    for shard in range(shards):
        table = pa.Table.from_pylist(rows[shard::shards], schema=embeddings_schema(dimension))
        pq.write_table(table, directory / f"part-{shard:05}.parquet", row_group_size=row_group_size)
    instances = defaultdict(list)
    for row in rows:
        instances[row["sample_id"]].append({"instance_id": row["instance_id"]})
    return SimpleNamespace(
        build=tmp_path,
        manifest=SimpleNamespace(
            artifact=lambda _: SimpleNamespace(
                path="embeddings", metadata={"dim": dimension, "encoder_fingerprint": "encoder"}
            )
        ),
        sample_ids=tuple(instances) if samples is None else tuple(samples),
        _instances_by_sample=instances,
    )


def _rows(*, frames=13, detections=12, dimension=7):
    return [
        {
            "sample_id": f"sequence:{frame}",
            "instance_id": f"sequence:{frame}:{detection}",
            "encoder_fingerprint": "encoder",
            "dim": dimension,
            "values": [float(frame * 100 + detection + channel) for channel in range(dimension)],
        }
        for frame in range(frames)
        for detection in range(detections)
    ]


@pytest.mark.parametrize("row_group_size,shards", [(1, 1), (5, 3), (1000, 2)])
@pytest.mark.parametrize("order", ["lexical", "random"])
def test_numeric_frame_order_joins_independently_ordered_shards(
    tmp_path, monkeypatch, row_group_size, shards, order
) -> None:
    rows = _rows()
    expected = {(row["sample_id"], row["instance_id"]): row["values"] for row in rows}
    if order == "lexical":
        physical = sorted(rows, key=lambda row: (row["sample_id"], row["instance_id"]))
    else:
        physical = [rows[index] for index in np.random.default_rng(12).permutation(len(rows))]
    dataset = _dataset(tmp_path, physical, row_group_size=row_group_size, shards=shards)
    # Force projected batches to cross several row-group boundaries as well.
    monkeypatch.setattr(embedding_reader, "_KEY_BATCH_ROWS", 7)
    with IndexedEmbeddingReader(dataset, cache_bytes=7 * 4 * 3) as reader:
        for frame in range(13):
            keys = tuple((f"sequence:{frame}", f"sequence:{frame}:{index}") for index in range(12))
            values = reader.take(keys)
            assert values.dtype == torch.float32
            assert values.is_contiguous()
            assert values.tolist() == [expected[key] for key in keys]
            assert reader._cache_bytes <= 7 * 4 * 3
        reader.finish()


def test_large_old_row_group_never_decodes_one_full_values_table(tmp_path, monkeypatch) -> None:
    rows = _rows(frames=1, detections=4096, dimension=16)
    dataset = _dataset(tmp_path, rows, row_group_size=10_000, dimension=16)
    original_batches = pq.ParquetFile.iter_batches
    payload_sizes = []

    def checked_batches(self, **kwargs):
        for batch in original_batches(self, **kwargs):
            if kwargs.get("columns") == ["values"]:
                payload_sizes.append(batch.num_rows)
                assert batch.num_rows <= 16
            yield batch

    def forbidden_full_group(*args, **kwargs):
        pytest.fail("A values row group must not be decoded into a full table.")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", checked_batches)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", forbidden_full_group)
    with IndexedEmbeddingReader(dataset, cache_bytes=1024) as reader:
        # Reading the end and then the beginning forces a bounded decoder restart.
        for index in (4095, 0, 2000, 1):
            row = rows[index]
            values = reader.take(((row["sample_id"], row["instance_id"]),))
            assert values.tolist() == [row["values"]]
            assert reader._cache_bytes <= 1024
    assert payload_sizes and max(payload_sizes) == 16


@pytest.mark.parametrize("field,value,message", [("dim", 8, "dimensions"), ("encoder_fingerprint", "other", "encoder")])
def test_metadata_errors_are_detected_before_any_payload_is_decoded(
    tmp_path, monkeypatch, field, value, message
) -> None:
    rows = _rows(frames=3, detections=2)
    rows[-1][field] = value
    dataset = _dataset(tmp_path, rows, row_group_size=2)
    original_batches = pq.ParquetFile.iter_batches

    def checked_batches(self, **kwargs):
        assert kwargs.get("columns") != ["values"]
        yield from original_batches(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", checked_batches)
    with pytest.raises(DatasetValidationError, match=message):
        IndexedEmbeddingReader(dataset)


def test_schema_checks_include_unselected_shards_and_payload_columns(tmp_path) -> None:
    dataset = _dataset(tmp_path, _rows(frames=1, detections=1))
    wrong = pa.Table.from_pylist(_rows(frames=1, detections=1)).drop(["values"])
    pq.write_table(wrong, tmp_path / "embeddings" / "part-00001.parquet")
    with pytest.raises(ValueError, match="schema mismatch"):
        IndexedEmbeddingReader(dataset)


def test_frame_outputs_own_writable_values(tmp_path) -> None:
    rows = _rows(frames=2, detections=1)
    dataset = _dataset(tmp_path, rows)
    with IndexedEmbeddingReader(dataset) as reader:
        first = reader.take(((rows[0]["sample_id"], rows[0]["instance_id"]),))
        first.fill_(-999)
        second = reader.take(((rows[1]["sample_id"], rows[1]["instance_id"]),))
        assert second.tolist() == [rows[1]["values"]]
        reader.finish()
    assert pq.read_table(tmp_path / "embeddings").column("values")[0].as_py() == rows[0]["values"]


def test_no_detections_yields_empty_embedding_tensor_without_payload_decode(tmp_path, monkeypatch) -> None:
    dataset = _dataset(tmp_path, [], samples=("empty-frame",))
    original_batches = pq.ParquetFile.iter_batches

    def checked_batches(self, **kwargs):
        assert kwargs.get("columns") != ["values"]
        yield from original_batches(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", checked_batches)
    with IndexedEmbeddingReader(dataset) as reader:
        assert reader.take(()).shape == (0, 7)
        reader.finish()


def test_stream_early_close_releases_decoder_and_files(materialized_build, monkeypatch) -> None:
    opened = []
    original_close = IndexedEmbeddingReader.close

    def recording_close(self):
        opened.extend(self._files)
        original_close(self)

    monkeypatch.setattr(IndexedEmbeddingReader, "close", recording_close)
    stream = CachedVisionDataset._stream_sequence(materialized_build["root"], sequence_id="seq-a", load_embeddings=True)
    iterator = iter(stream)
    first = next(iterator)
    iterator.close()
    assert opened and all(parquet.closed for parquet in opened)
    # A repeated replay creates independent decoder state and has identical values.
    assert torch.equal(next(iter(stream)).detections.embeddings, first.detections.embeddings)


@pytest.mark.parametrize("source", ["manifest.json", "_SUCCESS", "instances/part-00000.parquet"])
def test_retained_sequence_metadata_detects_source_mutation(materialized_build, source) -> None:
    root = materialized_build["root"]
    stream = CachedVisionDataset._stream_sequence(root, sequence_id="seq-a", load_embeddings=True)
    assert stream.source_is_current()
    path = root / source
    path.write_bytes(path.read_bytes() + b" ")
    assert not stream.source_is_current()


def test_retained_sequence_metadata_detects_added_shards(materialized_build) -> None:
    root = materialized_build["root"]
    stream = CachedVisionDataset._stream_sequence(root, sequence_id="seq-a")
    assert stream.source_is_current()
    path = root / "instances" / "extra.parquet"
    path.write_bytes((root / "instances" / "part-00000.parquet").read_bytes())
    assert not stream.source_is_current()
