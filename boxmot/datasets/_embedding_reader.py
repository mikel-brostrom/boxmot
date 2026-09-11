"""Bounded indexed reads of keyed Parquet embeddings in tracker frame order."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ._parquet import row_group_may_contain_key
from .schema import EMBEDDINGS_ARTIFACT, embeddings_schema
from .storage import artifact_files, resolve_artifact_path
from .validation import DatasetValidationError

if TYPE_CHECKING:
    from .cached import CachedVisionDataset

EMBEDDING_CACHE_BYTES = 64 * 1024 * 1024
_PAYLOAD_BATCH_ROWS = 128
_KEY_BATCH_ROWS = 65_536
_InstanceKey = tuple[str, str]
_BatchKey = tuple[int, int, int]


class IndexedEmbeddingReader:
    """Read selected embeddings with a key index and bounded decoded-batch LRU.

    The index validates selected keys and metadata before any values are read.
    Physical sample and instance order is irrelevant. Values from even a large
    existing row group are decoded in bounded batches, with at most one active
    Parquet batch iterator. The cache budget covers decoded embedding arrays;
    Arrow's compressed pages and the current frame allocation are additional.
    A single embedding wider than the budget necessarily occupies one batch.
    """

    def __init__(
        self,
        dataset: CachedVisionDataset,
        *,
        cache_bytes: int = EMBEDDING_CACHE_BYTES,
    ) -> None:
        import pyarrow.parquet as pq

        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes <= 0:
            raise ValueError("cache_bytes must be a positive integer.")
        self._cache_limit = cache_bytes
        self._files: list[Any] = []
        self._locations: dict[_InstanceKey, tuple[_BatchKey, int]] = {}
        self._remaining: dict[_BatchKey, int] = defaultdict(int)
        self._cache: OrderedDict[_BatchKey, np.ndarray] = OrderedDict()
        self._cache_bytes = 0
        self._active_group: tuple[int, int] | None = None
        self._active_batches: Iterator[Any] | None = None
        self._next_batch_index = 0
        self._closed = False
        try:
            artifact = dataset.manifest.artifact(EMBEDDINGS_ARTIFACT)
            paths = artifact_files(resolve_artifact_path(dataset.build, artifact.path))
            for path in paths:
                self._files.append(pq.ParquetFile(path, pre_buffer=False))
            try:
                self.dimension = int(self._files[0].schema_arrow.field("values").type.list_size)
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                raise ValueError("Embedding values must use a fixed-size float32 list type.") from exc
            declared_dim = artifact.metadata.get("dim")
            if declared_dim is not None and int(declared_dim) != self.dimension:
                raise DatasetValidationError("Embedding row dimensions differ from manifest metadata.")
            expected_schema = embeddings_schema(self.dimension)
            self._batch_rows = max(1, min(_PAYLOAD_BATCH_ROWS, cache_bytes // (self.dimension * 4)))
            for parquet in self._files:
                if not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
                    raise ValueError(
                        f"Parquet schema mismatch for 'embeddings': expected {expected_schema}, "
                        f"got {parquet.schema_arrow}. Legacy or positional caches are not supported."
                    )
            self._index(dataset, metadata=artifact.metadata)
        except BaseException:
            self.close()
            raise

    def _index(self, dataset: CachedVisionDataset, *, metadata: Any) -> None:
        """Index physical offsets using bounded projections of small key columns."""

        expected_keys = {
            (sample_id, row["instance_id"]) for sample_id, rows in dataset._instances_by_sample.items() for row in rows
        }
        sample_ids = set(dataset.sample_ids)
        sorted_ids = tuple(sorted(sample_ids))
        declared_encoder = metadata.get("encoder_fingerprint")
        encoders = set() if declared_encoder is None else {str(declared_encoder)}
        dimensions: set[int] = set()
        for file_index, parquet in enumerate(self._files):
            groups = [
                group
                for group in range(parquet.num_row_groups)
                if row_group_may_contain_key(parquet, group, column_name="sample_id", keys=sorted_ids)
                and parquet.metadata.row_group(group).num_rows
            ]
            if not groups:
                continue
            group_sizes = [parquet.metadata.row_group(group).num_rows for group in groups]
            group_index = 0
            group_offset = 0
            for batch in parquet.iter_batches(
                batch_size=_KEY_BATCH_ROWS,
                row_groups=groups,
                columns=["sample_id", "instance_id", "encoder_fingerprint", "dim"],
                use_threads=False,
            ):
                rows = zip(
                    batch.column("sample_id").to_pylist(),
                    batch.column("instance_id").to_pylist(),
                    batch.column("encoder_fingerprint").to_pylist(),
                    batch.column("dim").to_pylist(),
                    strict=True,
                )
                for sample_id, instance_id, encoder, dimension in rows:
                    group = groups[group_index]
                    if sample_id in sample_ids:
                        key = (sample_id, instance_id)
                        if key not in expected_keys:
                            raise DatasetValidationError("Embedding artifacts contain a foreign sample/instance key.")
                        if key in self._locations:
                            raise DatasetValidationError("Embedding artifacts contain duplicate sample/instance keys.")
                        batch_index, batch_offset = divmod(group_offset, self._batch_rows)
                        batch_key = (file_index, group, batch_index)
                        self._locations[key] = (batch_key, batch_offset)
                        self._remaining[batch_key] += 1
                        encoders.add(str(encoder))
                        dimensions.add(int(dimension))
                    group_offset += 1
                    if group_offset == group_sizes[group_index]:
                        group_index += 1
                        group_offset = 0
        if self._locations.keys() != expected_keys:
            raise DatasetValidationError("Embedding keys must match selected instance keys exactly once.")
        if len(encoders) != 1:
            raise DatasetValidationError("The embedding artifact must identify exactly one encoder.")
        if dimensions and dimensions != {self.dimension}:
            raise DatasetValidationError("Embedding row dimensions differ from manifest metadata.")

    def _close_batches(self) -> None:
        """Release the active Arrow decoder before switching groups or closing."""

        if self._active_batches is not None:
            close = getattr(self._active_batches, "close", None)
            if callable(close):
                close()
        self._active_batches = None
        self._active_group = None
        self._next_batch_index = 0

    def _values(self, key: _BatchKey) -> np.ndarray:
        """Get one bounded values batch, restarting decoding only for a backwards miss."""

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
                    batch_size=self._batch_rows,
                    row_groups=[group_index],
                    columns=["values"],
                    use_threads=False,
                )
            )
        assert self._active_batches is not None
        # Reserve space before decoding. A group may contain many batches;
        # never read it into one table just to obtain an offset near its end.
        maximum_bytes = self._batch_rows * self.dimension * 4
        while self._cache and self._cache_bytes + maximum_bytes > self._cache_limit:
            _, dropped = self._cache.popitem(last=False)
            self._cache_bytes -= dropped.nbytes
        batch = None
        while self._next_batch_index <= batch_index:
            batch = next(self._active_batches)
            self._next_batch_index += 1
        assert batch is not None
        values = batch.column("values")
        flat = values.values.slice(
            values.offset * self.dimension,
            len(values) * self.dimension,
        ).to_numpy(zero_copy_only=False)
        matrix = flat.reshape(len(values), self.dimension)
        self._cache[key] = matrix
        self._cache_bytes += matrix.nbytes
        return matrix

    def take(self, keys: tuple[_InstanceKey, ...]) -> torch.Tensor:
        """Gather selected keys into one writable contiguous array in detection order."""

        if self._closed:
            raise RuntimeError("The embedding reader is closed.")
        result = np.empty((len(keys), self.dimension), dtype=np.float32)
        by_batch: dict[_BatchKey, list[tuple[int, int]]] = defaultdict(list)
        for output_index, key in enumerate(keys):
            batch, row = self._locations[key]
            by_batch[batch].append((output_index, row))
        for batch, positions in by_batch.items():
            values = self._values(batch)
            run_start = 0
            for run_end in range(1, len(positions) + 1):
                if run_end < len(positions) and positions[run_end][0] == positions[run_end - 1][0] + 1:
                    continue
                run = positions[run_start:run_end]
                start, stop = run[0][0], run[-1][0] + 1
                np.take(values, [row for _, row in run], axis=0, out=result[start:stop])
                run_start = run_end
            self._remaining[batch] -= len(positions)
            if self._remaining[batch] == 0:
                released = self._cache.pop(batch)
                self._cache_bytes -= released.nbytes
        return torch.from_numpy(result)

    def finish(self) -> None:
        """Prove all selected rows were consumed without decoding unused payload tails."""

        if any(self._remaining.values()):
            raise DatasetValidationError("Embedding keys must match selected instance keys exactly once.")

    def close(self) -> None:
        """Release decoded arrays, active iterators, and all opened Parquet files."""

        if self._closed:
            return
        self._closed = True
        self._close_batches()
        self._cache.clear()
        self._cache_bytes = 0
        for parquet in self._files:
            parquet.close()
        self._files.clear()

    def __enter__(self) -> IndexedEmbeddingReader:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()
