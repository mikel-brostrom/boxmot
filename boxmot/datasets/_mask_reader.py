"""Bounded indexed reads of keyed Parquet masks in tracker frame order."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch

from ._parquet import row_group_may_contain_key
from .masks import MASK_CODEC, _validate_mask_payload, packed_mask_size, unpack_mask_batch
from .schema import MASKS_ARTIFACT, masks_schema
from .storage import artifact_files, resolve_artifact_path
from .validation import DatasetValidationError

if TYPE_CHECKING:
    from .cached import CachedVisionDataset

MASK_CACHE_BYTES = 64 * 1024 * 1024
_PAYLOAD_BATCH_ROWS = 128
_KEY_BATCH_ROWS = 65_536
_InstanceKey = tuple[str, str]
_BatchKey = tuple[int, int, int]


class IndexedMaskReader:
    """Join masks by identity without buffering later frames' packed pixels.

    The key index validates metadata before reading payloads. An LRU retains
    at most ``cache_bytes`` of decoded packed batches, regardless of physical
    shard order. The current frame's packed and dense masks, plus Arrow's
    compressed pages, are additional. One mask larger than the budget occupies
    a single batch because its payload cannot be split by the Parquet reader.
    """

    def __init__(self, dataset: CachedVisionDataset, *, cache_bytes: int = MASK_CACHE_BYTES) -> None:
        import pyarrow.parquet as pq

        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes <= 0:
            raise ValueError("cache_bytes must be a positive integer.")
        self._cache_limit = cache_bytes
        self._files: list[Any] = []
        self._locations: dict[_InstanceKey, tuple[_BatchKey, int]] = {}
        self._remaining: dict[_BatchKey, int] = defaultdict(int)
        self._batch_rows: dict[tuple[int, int], int] = {}
        self._batch_bytes: dict[tuple[int, int], int] = {}
        self._cache: OrderedDict[_BatchKey, Any] = OrderedDict()
        self._cache_bytes = 0
        self._active_group: tuple[int, int] | None = None
        self._active_batches: Iterator[Any] | None = None
        self._next_batch_index = 0
        self._closed = False
        try:
            artifact = dataset.manifest.artifact(MASKS_ARTIFACT)
            for path in artifact_files(resolve_artifact_path(dataset.build, artifact.path)):
                parquet = pq.ParquetFile(path, pre_buffer=False)
                self._files.append(parquet)
                if not parquet.schema_arrow.equals(masks_schema(), check_metadata=False):
                    raise ValueError(
                        f"Parquet schema mismatch for 'masks': expected {masks_schema()}, got {parquet.schema_arrow}."
                    )
            self._index(dataset)
        except BaseException:
            self.close()
            raise

    def _index(self, dataset: CachedVisionDataset) -> None:
        """Read only keys and dimensions to locate each selected packed mask."""
        expected = {
            (sample_id, row["instance_id"]) for sample_id, rows in dataset._instances_by_sample.items() for row in rows
        }
        sizes = {sample["sample_id"]: (sample["height"], sample["width"]) for sample in dataset._samples}
        sorted_ids = tuple(sorted(dataset.sample_ids))
        for file_index, parquet in enumerate(self._files):
            for group_index in range(parquet.num_row_groups):
                if not row_group_may_contain_key(parquet, group_index, column_name="sample_id", keys=sorted_ids):
                    continue
                selected: list[tuple[_InstanceKey, int]] = []
                largest_payload = 0
                offset = 0
                for batch in parquet.iter_batches(
                    batch_size=_KEY_BATCH_ROWS,
                    row_groups=[group_index],
                    columns=["sample_id", "instance_id", "height", "width", "codec"],
                    use_threads=False,
                ):
                    for row in batch.to_pylist():
                        height, width = row["height"], row["width"]
                        largest_payload = max(largest_payload, packed_mask_size(height, width))
                        sample_id = row["sample_id"]
                        if sample_id in sizes:
                            key = (sample_id, row["instance_id"])
                            if key not in expected:
                                raise DatasetValidationError("Mask artifacts contain a foreign sample/instance key.")
                            if key in self._locations:
                                raise DatasetValidationError("Mask artifacts contain duplicate sample/instance keys.")
                            if (height, width) != sizes[sample_id] or row["codec"] != MASK_CODEC:
                                raise DatasetValidationError(
                                    "Mask rows must use the selected sample dimensions and canonical codec."
                                )
                            # Reserve the key immediately so duplicates inside a group also fail.
                            self._locations[key] = ((file_index, group_index, 0), offset)
                            selected.append((key, offset))
                        offset += 1
                if not selected:
                    continue
                group = (file_index, group_index)
                batch_rows = max(1, min(_PAYLOAD_BATCH_ROWS, self._cache_limit // (largest_payload + 4)))
                self._batch_rows[group] = batch_rows
                self._batch_bytes[group] = batch_rows * (largest_payload + 4)
                for key, row_offset in selected:
                    batch_index, batch_offset = divmod(row_offset, batch_rows)
                    batch_key = (*group, batch_index)
                    self._locations[key] = (batch_key, batch_offset)
                    self._remaining[batch_key] += 1
        if self._locations.keys() != expected:
            raise DatasetValidationError("Mask keys must match selected instance keys exactly once.")

    def _close_batches(self) -> None:
        """Release an active Arrow decoder before changing physical groups."""
        if self._active_batches is not None:
            close = getattr(self._active_batches, "close", None)
            if callable(close):
                close()
        self._active_batches = None
        self._active_group = None
        self._next_batch_index = 0

    def _values(self, key: _BatchKey) -> Any:
        """Read one bounded packed batch, restarting only after backwards misses."""
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        file_index, group_index, batch_index = key
        group = (file_index, group_index)
        if group != self._active_group or batch_index < self._next_batch_index:
            self._close_batches()
            self._active_group = group
            self._active_batches = iter(
                self._files[file_index].iter_batches(
                    batch_size=self._batch_rows[group],
                    row_groups=[group_index],
                    columns=["data"],
                    use_threads=False,
                )
            )
        assert self._active_batches is not None
        while self._cache and self._cache_bytes + self._batch_bytes[group] > self._cache_limit:
            _, dropped = self._cache.popitem(last=False)
            self._cache_bytes -= dropped.nbytes
        batch = None
        while self._next_batch_index <= batch_index:
            batch = next(self._active_batches)
            self._next_batch_index += 1
        assert batch is not None
        values = batch.column("data")
        self._cache[key] = values
        self._cache_bytes += values.nbytes
        return values

    def take_packed(self, keys: tuple[_InstanceKey, ...], *, height: int, width: int) -> tuple[bytes, ...]:
        """Return validated packed masks aligned to one frame's detection keys."""
        if self._closed:
            raise RuntimeError("The mask reader is closed.")
        payloads: list[bytes] = [b""] * len(keys)
        by_batch: dict[_BatchKey, list[tuple[int, int]]] = defaultdict(list)
        for output_index, key in enumerate(keys):
            batch, row = self._locations[key]
            by_batch[batch].append((output_index, row))
        for batch, positions in by_batch.items():
            values = self._values(batch)
            for output_index, row in positions:
                payloads[output_index] = values[row].as_py()
            self._remaining[batch] -= len(positions)
            if self._remaining[batch] == 0:
                released = self._cache.pop(batch)
                self._cache_bytes -= released.nbytes
        return tuple(_validate_mask_payload(payload, height, width) for payload in payloads)

    def take(self, keys: tuple[_InstanceKey, ...], *, height: int, width: int) -> torch.Tensor:
        """Decode one frame into writable masks aligned to its detection keys."""
        return unpack_mask_batch(self.take_packed(keys, height=height, width=width), height, width)

    def finish(self) -> None:
        """Prove all selected keys were consumed without decoding unselected tails."""
        if any(self._remaining.values()):
            raise DatasetValidationError("Mask keys must match selected instance keys exactly once.")

    def close(self) -> None:
        """Release packed payloads, active iterators, and opened Parquet files."""
        if self._closed:
            return
        self._closed = True
        self._close_batches()
        self._cache.clear()
        self._cache_bytes = 0
        for parquet in self._files:
            parquet.close()
        self._files.clear()

    def __enter__(self) -> IndexedMaskReader:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
