"""Mask replay follows instance keys while bounding future-frame payloads."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import boxmot.datasets._mask_reader as mask_reader
from boxmot.datasets._mask_reader import IndexedMaskReader
from boxmot.datasets.cached import CachedVisionDataset
from boxmot.datasets.masks import MASK_CODEC, MaskCodecError, pack_mask, unpack_mask_batch
from boxmot.datasets.schema import masks_schema
from boxmot.datasets.validation import DatasetValidationError


def _dataset(tmp_path, rows, *, row_group_size=128, shards=1, frames=13, height=8, width=11):
    """Write genuine shards against the reader's small dataset interface."""
    directory = tmp_path / "masks"
    directory.mkdir(exist_ok=True)
    for shard in range(shards):
        table = pa.Table.from_pylist(rows[shard::shards], schema=masks_schema())
        pq.write_table(table, directory / f"part-{shard:05}.parquet", row_group_size=row_group_size)
    instances = defaultdict(list)
    for row in rows:
        instances[row["sample_id"]].append({"instance_id": row["instance_id"]})
    samples = [{"sample_id": f"sequence:{frame}", "height": height, "width": width} for frame in range(frames)]
    return SimpleNamespace(
        build=tmp_path,
        manifest=SimpleNamespace(artifact=lambda _: SimpleNamespace(path="masks")),
        sample_ids=tuple(sample["sample_id"] for sample in samples),
        _instances_by_sample=instances,
        _samples=samples,
    )


def _rows(*, frames=13, detections=12, height=8, width=11):
    rows = []
    for frame in range(frames):
        for detection in range(detections):
            mask = np.random.default_rng(frame * detections + detection).random((height, width)) > 0.7
            rows.append(
                {
                    "sample_id": f"sequence:{frame}",
                    "instance_id": f"sequence:{frame}:{detection}",
                    "height": height,
                    "width": width,
                    "codec": MASK_CODEC,
                    "data": pack_mask(mask),
                }
            )
    return rows


@pytest.mark.parametrize("row_group_size,shards", [(1, 1), (5, 3), (1000, 2)])
@pytest.mark.parametrize("order", ["lexical", "random"])
def test_numeric_frames_join_masks_by_key_with_bounded_packed_batches(
    tmp_path, monkeypatch, row_group_size, shards, order
) -> None:
    rows = _rows()
    expected = {(row["sample_id"], row["instance_id"]): row["data"] for row in rows}
    physical = (
        sorted(rows, key=lambda row: (row["sample_id"], row["instance_id"]))
        if order == "lexical"
        else [rows[index] for index in np.random.default_rng(14).permutation(len(rows))]
    )
    dataset = _dataset(tmp_path, physical, row_group_size=row_group_size, shards=shards)
    monkeypatch.setattr(mask_reader, "_KEY_BATCH_ROWS", 7)
    with IndexedMaskReader(dataset, cache_bytes=45) as reader:
        for frame in range(13):
            keys = tuple((f"sequence:{frame}", f"sequence:{frame}:{detection}") for detection in range(12))
            actual = reader.take(keys, height=8, width=11)
            assert actual.dtype == torch.bool and actual.is_contiguous()
            assert torch.equal(actual, unpack_mask_batch((expected[key] for key in keys), 8, 11))
            assert reader._cache_bytes <= 45
        reader.finish()


def test_large_legacy_groups_decode_only_bounded_payload_batches(tmp_path, monkeypatch) -> None:
    rows = _rows(frames=1, detections=1024)
    dataset = _dataset(tmp_path, rows, frames=1, row_group_size=10_000)
    original_batches = pq.ParquetFile.iter_batches
    payload_sizes = []

    def checked_batches(self, **kwargs):
        for batch in original_batches(self, **kwargs):
            if kwargs.get("columns") == ["data"]:
                payload_sizes.append(batch.num_rows)
                assert batch.num_rows <= 3
                assert batch.nbytes <= 45
            yield batch

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", checked_batches)
    monkeypatch.setattr(
        pq.ParquetFile, "read_row_group", lambda *_args, **_kwargs: pytest.fail("Decoded a whole payload row group")
    )
    with IndexedMaskReader(dataset, cache_bytes=45) as reader:
        for index in (1023, 0, 500, 1):
            row = rows[index]
            actual = reader.take(((row["sample_id"], row["instance_id"]),), height=8, width=11)
            assert torch.equal(actual, unpack_mask_batch([row["data"]], 8, 11))
            assert reader._cache_bytes <= 45
    assert payload_sizes and max(payload_sizes) == 3


@pytest.mark.parametrize("change", ["missing", "duplicate", "foreign", "dimensions", "codec"])
def test_selected_keys_and_metadata_are_validated_before_pixels(tmp_path, monkeypatch, change) -> None:
    rows = _rows(frames=2, detections=2)
    dataset = _dataset(tmp_path, rows, frames=2)
    changed = [dict(row) for row in rows]
    if change == "missing":
        changed.pop()
    elif change == "duplicate":
        changed.append(changed[0])
    elif change == "foreign":
        changed[0]["instance_id"] = "foreign"
    elif change == "dimensions":
        changed[0]["height"] = 9
    else:
        changed[0]["codec"] = "other"
    pq.write_table(pa.Table.from_pylist(changed, schema=masks_schema()), tmp_path / "masks" / "part-00000.parquet")
    original_batches = pq.ParquetFile.iter_batches

    def checked_batches(self, **kwargs):
        assert kwargs.get("columns") != ["data"]
        yield from original_batches(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", checked_batches)
    with pytest.raises(DatasetValidationError, match="Mask"):
        IndexedMaskReader(dataset)


def test_reader_rejects_corrupt_payload_padding_and_wrong_schema(tmp_path) -> None:
    rows = _rows(frames=1, detections=1, height=3, width=5)
    rows[0]["data"] = bytes([0, 128])
    dataset = _dataset(tmp_path, rows, frames=1, height=3, width=5)
    with IndexedMaskReader(dataset) as reader:
        with pytest.raises(MaskCodecError, match="outside"):
            reader.take(((rows[0]["sample_id"], rows[0]["instance_id"]),), height=3, width=5)
    pq.write_table(pa.Table.from_pylist(rows).drop(["data"]), tmp_path / "masks" / "extra.parquet")
    with pytest.raises(ValueError, match="schema mismatch"):
        IndexedMaskReader(dataset)


def test_empty_frames_have_no_pixel_reads_and_outputs_remain_writable(tmp_path, monkeypatch) -> None:
    dataset = _dataset(tmp_path, [], frames=2)
    original_batches = pq.ParquetFile.iter_batches

    def checked_batches(self, **kwargs):
        assert kwargs.get("columns") != ["data"]
        yield from original_batches(self, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", checked_batches)
    with IndexedMaskReader(dataset) as reader:
        result = reader.take((), height=8, width=11)
        assert result.shape == (0, 8, 11) and result.dtype == torch.bool
        reader.finish()


def test_stream_early_close_releases_mask_files_and_replays_independently(materialized_build, monkeypatch) -> None:
    opened = []
    original_close = IndexedMaskReader.close

    def recording_close(self):
        opened.extend(self._files)
        original_close(self)

    monkeypatch.setattr(IndexedMaskReader, "close", recording_close)
    stream = CachedVisionDataset._stream_sequence(materialized_build["root"], sequence_id="seq-a", load_masks=True)
    iterator = iter(stream)
    first = next(iterator).detections.masks.values
    expected = first.clone()
    first.logical_not_()
    iterator.close()
    assert opened and all(parquet.closed for parquet in opened)
    repeated = iter(stream)
    try:
        assert torch.equal(next(repeated).detections.masks.values, expected)
    finally:
        repeated.close()


def test_batch_decoder_allocates_one_owned_dense_output_without_stacking(monkeypatch) -> None:
    masks = np.random.default_rng(8).random((7, 13, 17)) > 0.6
    payloads = [pack_mask(mask) for mask in masks]
    monkeypatch.setattr(torch, "stack", lambda *_args, **_kwargs: pytest.fail("Allocated a second dense mask batch"))
    actual = unpack_mask_batch(iter(payloads), 13, 17)
    assert actual.dtype == torch.bool and actual.is_contiguous()
    assert np.array_equal(actual.numpy(), masks)
    actual[0].logical_not_()
    assert np.array_equal(unpack_mask_batch(payloads, 13, 17).numpy(), masks)
