"""Key-joined loader for immutable Parquet vision datasets."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes

from ._embedding_reader import IndexedEmbeddingReader
from ._parquet import iter_selected_batches, row_group_may_contain_key, source_snapshot
from .manifest import DatasetManifest, ManifestError
from .masks import MASK_CODEC, unpack_mask_batch
from .readers import read_rgb_chw_uint8
from .schema import (
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MASKS_ARTIFACT,
    SAMPLES_ARTIFACT,
    SCHEMA_ID,
    SUCCESS_FILENAME,
    embeddings_schema,
    instances_schema,
    masks_schema,
    samples_schema,
)
from .storage import artifact_files, read_parquet_artifact, resolve_artifact_path
from .validation import DatasetValidationError, validate_dataset

_SELECTED_ARTIFACT_BATCH_ROWS = 128
_SELECTED_KEY_BATCH_ROWS = 65_536


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One canonical sample loaded from a materialized build."""

    sample_id: str
    split: str
    sequence_id: str
    frame_index: int
    timestamp_s: float | None
    image_size: tuple[int, int]
    image_ref: str | None
    frame: Frame | None
    detections: Detections


class CachedVisionDataset(Sequence[DatasetSample]):
    """Read a versioned build and join optional data by stable instance key.

    Args:
        build: Explicit path to the immutable dataset build.
        split: Optional exact split filter.
        load_images: Decode image references into canonical :class:`Frame`s.
        load_masks: Decode the keyed bit-packed mask table.
        load_embeddings: Load the keyed embedding table.
    """

    def __init__(
        self,
        build: str | Path,
        *,
        split: str | None = None,
        load_images: bool = False,
        load_masks: bool = False,
        load_embeddings: bool = False,
    ) -> None:
        self._initialize(
            build,
            split=split,
            load_images=load_images,
            load_masks=load_masks,
            load_embeddings=load_embeddings,
            sequence_ids=None,
            validate=True,
        )

    @classmethod
    def _for_sequence(
        cls,
        build: str | Path,
        *,
        sequence_id: str,
        split: str | None = None,
        load_images: bool = False,
        load_masks: bool = False,
        load_embeddings: bool = False,
    ) -> CachedVisionDataset:
        """Open only one sequence for an engine-owned replay worker.

        Published builds have already passed the full finalize validation. This
        private constructor rechecks the publication marker and schemas, then
        uses Parquet predicates so a sequence worker never expands every
        embedding in the build into its own address space.
        """

        if not isinstance(sequence_id, str) or not sequence_id or sequence_id != sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty canonical string.")
        dataset = cls.__new__(cls)
        dataset._initialize(
            build,
            split=split,
            load_images=load_images,
            load_masks=load_masks,
            load_embeddings=load_embeddings,
            sequence_ids=frozenset({sequence_id}),
            validate=False,
        )
        if not dataset._samples:
            split_suffix = "" if split is None else f" in split {split!r}"
            raise ValueError(f"Build does not contain sequence {sequence_id!r}{split_suffix}.")
        return dataset

    @classmethod
    def _stream_sequence(
        cls,
        build: str | Path,
        *,
        sequence_id: str,
        split: str | None = None,
        load_images: bool = False,
        load_masks: bool = False,
        load_embeddings: bool = False,
    ) -> _SequenceDatasetStream:
        """Open a replay-only sequence view with lazy optional payload joins.

        Samples and detection rows are loaded up front because they are small
        and define canonical frame/detection order. Large keyed masks and
        embeddings are streamed only as iteration advances. The ordinary
        constructor and :meth:`_for_sequence` retain their eager, random-access
        behavior.
        """

        dataset = cls._for_sequence(
            build,
            sequence_id=sequence_id,
            split=split,
            load_images=load_images,
            load_masks=False,
            load_embeddings=False,
        )
        return _SequenceDatasetStream(
            dataset,
            load_masks=load_masks,
            load_embeddings=load_embeddings,
        )

    def _initialize(
        self,
        build: str | Path,
        *,
        split: str | None,
        load_images: bool,
        load_masks: bool,
        load_embeddings: bool,
        sequence_ids: frozenset[str] | None,
        validate: bool,
    ) -> None:
        for name, value in (
            ("load_images", load_images),
            ("load_masks", load_masks),
            ("load_embeddings", load_embeddings),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")
        if split is not None and (not isinstance(split, str) or not split or split != split.strip()):
            raise ValueError("split must be a non-empty canonical string or None.")

        self.build = Path(build)
        try:
            self.manifest = DatasetManifest.load(self.build)
        except ManifestError as exc:
            raise DatasetValidationError(str(exc)) from exc
        if validate:
            validate_dataset(self.build, manifest=self.manifest)
        else:
            self._validate_publication_marker()

        artifacts = self.manifest.artifacts_by_name
        if load_images and not self.manifest.publish.image_references:
            raise DatasetValidationError(
                "Images were requested, but this dataset build does not publish image references."
            )
        if load_masks and MASKS_ARTIFACT not in artifacts:
            raise DatasetValidationError("Masks were requested, but this dataset build does not publish masks.")
        if load_embeddings and EMBEDDINGS_ARTIFACT not in artifacts:
            raise DatasetValidationError(
                "Embeddings were requested, but this dataset build does not publish embeddings."
            )

        self.load_images = load_images
        self.load_masks = load_masks
        self.load_embeddings = load_embeddings

        self._image_root: str | Path = self.manifest.metadata.get("source_root_uri", self.build)

        sample_filters: list[tuple[str, str, Any]] = []
        if split is not None:
            sample_filters.append(("split", "=", split))
        if sequence_ids is not None:
            sample_filters.append(("sequence_id", "in", sorted(sequence_ids)))
        samples = (
            self._read(SAMPLES_ARTIFACT, filters=sample_filters or None).to_pylist()
            if sequence_ids is None
            else self._read_sequence_rows(
                SAMPLES_ARTIFACT,
                column_name="sequence_id",
                keys=sequence_ids,
                split=split,
            )
        )
        samples.sort(key=lambda row: (row["split"], row["sequence_id"], row["frame_index"], row["sample_id"]))
        self._samples = samples

        selected_ids = {row["sample_id"] for row in samples}
        instances_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        instance_filters = [("sample_id", "in", sorted(selected_ids))] if selected_ids else None
        instance_rows = []
        if selected_ids:
            instance_rows = (
                self._read(INSTANCES_ARTIFACT, filters=instance_filters).to_pylist()
                if sequence_ids is None
                else self._read_sequence_rows(
                    INSTANCES_ARTIFACT,
                    column_name="sample_id",
                    keys=selected_ids,
                )
            )
        instance_keys: set[tuple[str, str]] = set()
        for row in instance_rows:
            key = (row["sample_id"], row["instance_id"])
            if key in instance_keys:
                raise DatasetValidationError("Instance artifacts contain duplicate sample/instance keys.")
            instance_keys.add(key)
            instances_by_sample[row["sample_id"]].append(row)
        for rows in instances_by_sample.values():
            rows.sort(key=lambda row: row["detection_index"])
        self._instances_by_sample = dict(instances_by_sample)

        self._masks_by_instance: dict[tuple[str, str], dict[str, Any]] = {}
        if self.load_masks:
            if sequence_ids is None:
                mask_rows = self._read(MASKS_ARTIFACT, filters=instance_filters).to_pylist() if selected_ids else []
            else:
                mask_rows = [
                    row
                    for batch in self._iter_selected_batches(
                        MASKS_ARTIFACT,
                        sample_ids=selected_ids,
                        expected_schema=masks_schema(),
                    )
                    for row in batch.to_pylist()
                ]
            self._masks_by_instance = {(row["sample_id"], row["instance_id"]): row for row in mask_rows}
            if len(self._masks_by_instance) != len(mask_rows):
                raise DatasetValidationError("Mask artifacts contain duplicate sample/instance keys.")
            self._validate_selected_keys(self._masks_by_instance, artifact="Mask")

        self._embedding_index_by_instance: dict[tuple[str, str], int] = {}
        self._embedding_values = torch.empty((0, 0), dtype=torch.float32)
        self._embedding_dim: int | None = None
        self.encoder_fingerprint: str | None = None
        if self.load_embeddings:
            embedding_artifact = artifacts[EMBEDDINGS_ARTIFACT]
            declared = embedding_artifact.metadata.get("dim")
            self._embedding_dim = None if declared is None else int(declared)
            if sequence_ids is None:
                embedding_table = (
                    self._read(EMBEDDINGS_ARTIFACT, filters=instance_filters)
                    if selected_ids
                    else self._read(EMBEDDINGS_ARTIFACT, filters=[("sample_id", "=", "")])
                )
                self._load_embedding_table(embedding_table, embedding_artifact.metadata)
            else:
                self._load_selected_embeddings(selected_ids, embedding_artifact.metadata)

    def _validate_publication_marker(self) -> None:
        success_path = self.build / SUCCESS_FILENAME
        if not success_path.is_file():
            raise DatasetValidationError(f"Dataset build is not published: missing {SUCCESS_FILENAME}.")
        try:
            payload = json.loads(success_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetValidationError(f"Dataset publication marker is invalid: {success_path}") from exc
        if payload != {"schema": SCHEMA_ID, "build_id": self.manifest.build_id}:
            raise DatasetValidationError("Dataset publication marker does not match its manifest.")

    def _validate_selected_keys(self, values: dict[tuple[str, str], Any], *, artifact: str) -> None:
        expected = {
            (sample_id, row["instance_id"]) for sample_id, rows in self._instances_by_sample.items() for row in rows
        }
        if set(values) != expected:
            raise DatasetValidationError(f"{artifact} keys must match selected instance keys exactly once.")

    def _read(self, name: str, *, filters: Any | None = None):
        artifact = self.manifest.artifact(name)
        path = resolve_artifact_path(self.build, artifact.path)
        return read_parquet_artifact(
            path,
            artifact_name=name,
            box_type=self.manifest.box_type,
            filters=filters,
        )

    def _read_sequence_rows(
        self,
        name: str,
        *,
        column_name: str,
        keys: set[str] | frozenset[str],
        split: str | None = None,
    ) -> list[dict[str, Any]]:
        """Read small sequence metadata without initializing Arrow's dataset scanner.

        Direct Parquet reads avoid the dataset and pandas imports triggered by
        generic filter expressions in each spawned replay worker. Statistics
        only prune impossible row groups; exact Python key and split checks
        preserve selection even for unsorted or unindexed publication shards.
        """

        import pyarrow.parquet as pq

        expected_schema = samples_schema() if name == SAMPLES_ARTIFACT else instances_schema(self.manifest.box_type)
        artifact = self.manifest.artifact(name)
        path = resolve_artifact_path(self.build, artifact.path)
        sorted_keys = tuple(sorted(keys))
        rows: list[dict[str, Any]] = []
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
                    if row_group_may_contain_key(
                        parquet,
                        index,
                        column_name=column_name,
                        keys=sorted_keys,
                    )
                ]
                if not row_groups:
                    continue
                for batch in parquet.iter_batches(
                    batch_size=_SELECTED_KEY_BATCH_ROWS,
                    row_groups=row_groups,
                    use_threads=False,
                ):
                    rows.extend(
                        row
                        for row in batch.to_pylist()
                        if row[column_name] in keys and (split is None or row["split"] == split)
                    )
        return rows

    def _iter_selected_batches(
        self,
        name: str,
        *,
        sample_ids: set[str],
        expected_schema: Any,
        columns: tuple[str, ...] | None = None,
        batch_size: int = _SELECTED_ARTIFACT_BATCH_ROWS,
    ) -> Iterator[Any]:
        """Yield only selected rows while keeping large Parquet values batch-bounded."""

        yield from iter_selected_batches(
            self.build,
            self.manifest,
            name,
            sample_ids=sample_ids,
            expected_schema=expected_schema,
            columns=columns,
            batch_size=batch_size,
        )

    def _load_embedding_table(self, embedding_table: Any, metadata: dict[str, Any]) -> None:
        """Load an eagerly read embedding table for the public all-build loader."""

        sample_keys = embedding_table.column("sample_id").to_pylist()
        instance_keys = embedding_table.column("instance_id").to_pylist()
        available_encoders = {
            str(value) for value in embedding_table.column("encoder_fingerprint").unique().to_pylist()
        }
        declared_encoder = metadata.get("encoder_fingerprint")
        if declared_encoder is not None:
            available_encoders.add(str(declared_encoder))
        if len(available_encoders) != 1:
            raise DatasetValidationError("The embedding artifact must identify exactly one encoder.")
        self.encoder_fingerprint = next(iter(available_encoders))
        dimensions = {int(value) for value in embedding_table.column("dim").unique().to_pylist()}
        self._resolve_embedding_dimension(dimensions)

        keys = list(zip(sample_keys, instance_keys, strict=True))
        self._embedding_index_by_instance = {key: index for index, key in enumerate(keys)}
        if len(self._embedding_index_by_instance) != len(keys):
            raise DatasetValidationError("Embedding artifacts contain duplicate sample/instance keys.")
        self._validate_selected_keys(self._embedding_index_by_instance, artifact="Embedding")

        values = embedding_table.column("values").combine_chunks()
        flat = values.values.to_numpy(zero_copy_only=False)
        self._embedding_values = torch.from_numpy(flat.copy()).reshape(-1, self._embedding_dim).contiguous()

    def _load_selected_embeddings(self, sample_ids: set[str], metadata: dict[str, Any]) -> None:
        """Stream one sequence's keyed embeddings without materializing full shards."""

        import pyarrow.parquet as pq

        artifact = self.manifest.artifact(EMBEDDINGS_ARTIFACT)
        path = resolve_artifact_path(self.build, artifact.path)
        shards = artifact_files(path)
        try:
            storage_dim = int(pq.ParquetFile(shards[0], pre_buffer=False).schema_arrow.field("values").type.list_size)
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("Embedding values must use a fixed-size float32 list type.") from exc
        expected_schema = embeddings_schema(storage_dim)

        expected_keys = tuple(
            (sample["sample_id"], row["instance_id"])
            for sample in self._samples
            for row in self._instances_by_sample.get(sample["sample_id"], ())
        )
        expected_index = {key: index for index, key in enumerate(expected_keys)}
        values = torch.empty((len(expected_keys), storage_dim), dtype=torch.float32)
        loaded: dict[tuple[str, str], int] = {}
        available_encoders: set[str] = set()
        dimensions: set[int] = set()

        declared_encoder = metadata.get("encoder_fingerprint")
        if declared_encoder is not None:
            available_encoders.add(str(declared_encoder))

        for batch in self._iter_selected_batches(
            EMBEDDINGS_ARTIFACT,
            sample_ids=sample_ids,
            expected_schema=expected_schema,
        ):
            sample_keys = batch.column("sample_id").to_pylist()
            instance_keys = batch.column("instance_id").to_pylist()
            available_encoders.update(str(value) for value in batch.column("encoder_fingerprint").unique().to_pylist())
            dimensions.update(int(value) for value in batch.column("dim").unique().to_pylist())
            keys = list(zip(sample_keys, instance_keys, strict=True))
            target_indices: list[int] = []
            for key in keys:
                if key in loaded:
                    raise DatasetValidationError("Embedding artifacts contain duplicate sample/instance keys.")
                try:
                    target_index = expected_index[key]
                except KeyError as exc:
                    raise DatasetValidationError(
                        "Embedding keys must match selected instance keys exactly once."
                    ) from exc
                loaded[key] = target_index
                target_indices.append(target_index)

            embedding_array = batch.column("values")
            flat = embedding_array.values.slice(
                embedding_array.offset * storage_dim,
                len(embedding_array) * storage_dim,
            ).to_numpy(zero_copy_only=False)
            batch_values = torch.from_numpy(flat.copy()).reshape(len(keys), storage_dim)
            values.index_copy_(0, torch.tensor(target_indices, dtype=torch.int64), batch_values)

        if len(available_encoders) != 1:
            raise DatasetValidationError("The embedding artifact must identify exactly one encoder.")
        self.encoder_fingerprint = next(iter(available_encoders))
        self._resolve_embedding_dimension(dimensions)
        if self._embedding_dim != storage_dim:
            raise DatasetValidationError("Embedding row dimensions differ from manifest metadata.")
        self._embedding_index_by_instance = loaded
        self._validate_selected_keys(self._embedding_index_by_instance, artifact="Embedding")
        self._embedding_values = values.contiguous()

    def _resolve_embedding_dimension(self, dimensions: set[int]) -> None:
        if self._embedding_dim is None and dimensions:
            if len(dimensions) != 1:
                raise DatasetValidationError("Embedding rows must use exactly one dimension.")
            self._embedding_dim = next(iter(dimensions))
        if self._embedding_dim is None:
            raise DatasetValidationError(
                "An empty embedding artifact must declare its embedding dimension in manifest metadata."
            )
        if dimensions and dimensions != {self._embedding_dim}:
            raise DatasetValidationError("Embedding row dimensions differ from manifest metadata.")

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row["sample_id"] for row in self._samples)

    def __getitem__(self, index: int) -> DatasetSample:
        if not isinstance(index, int):
            raise TypeError(f"Dataset indices must be integers, got {type(index).__name__}.")
        sample = self._samples[index]
        sample_id = sample["sample_id"]
        instance_rows = self._instances_by_sample.get(sample_id, [])
        instance_ids = tuple(row["instance_id"] for row in instance_rows)

        if self.manifest.box_type == "aabb":
            geometry_values = [[row[key] for key in ("x1", "y1", "x2", "y2")] for row in instance_rows]
            geometry = Boxes(torch.tensor(geometry_values, dtype=torch.float32).reshape(-1, 4).contiguous())
        else:
            geometry_values = [[row[key] for key in ("cx", "cy", "w", "h", "angle")] for row in instance_rows]
            geometry = OrientedBoxes(torch.tensor(geometry_values, dtype=torch.float32).reshape(-1, 5).contiguous())

        scores = torch.tensor([row["score"] for row in instance_rows], dtype=torch.float32).contiguous()
        class_ids = torch.tensor([row["class_id"] for row in instance_rows], dtype=torch.int64).contiguous()

        masks = None
        if self.load_masks:
            payloads = [self._masks_by_instance[(sample_id, instance_id)]["data"] for instance_id in instance_ids]
            mask_values = unpack_mask_batch(payloads, sample["height"], sample["width"])
            masks = MaskBatch(mask_values)

        embeddings = None
        if self.load_embeddings:
            indices = [self._embedding_index_by_instance[(sample_id, instance_id)] for instance_id in instance_ids]
            if indices:
                embeddings = self._embedding_values[torch.tensor(indices, dtype=torch.int64)].contiguous()
            else:
                embeddings = torch.empty((0, self._embedding_dim), dtype=torch.float32)

        detections = Detections(
            geometry=geometry,
            scores=scores,
            class_ids=class_ids,
            sample_id=sample_id,
            instance_ids=instance_ids,
            masks=masks,
            embeddings=embeddings,
        )

        frame = None
        image_ref = sample["image_ref"]
        if self.load_images:
            if not image_ref:
                raise DatasetValidationError(f"Sample {sample_id!r} does not publish an image reference.")
            image = read_rgb_chw_uint8(image_ref, self._image_root)
            if tuple(image.shape) != (3, sample["height"], sample["width"]):
                raise DatasetValidationError(
                    f"Decoded image for sample {sample_id!r} has shape {tuple(image.shape)}, expected "
                    f"(3, {sample['height']}, {sample['width']})."
                )
            frame = Frame(
                image=image,
                sample_id=sample_id,
                sequence_id=sample["sequence_id"],
                frame_index=sample["frame_index"],
                timestamp_s=sample["timestamp_s"],
                source_uri=image_ref,
            )

        return DatasetSample(
            sample_id=sample_id,
            split=sample["split"],
            sequence_id=sample["sequence_id"],
            frame_index=sample["frame_index"],
            timestamp_s=sample["timestamp_s"],
            image_size=(sample["height"], sample["width"]),
            image_ref=image_ref,
            frame=frame,
            detections=detections,
        )

    def __iter__(self) -> Iterator[DatasetSample]:
        for index in range(len(self)):
            yield self[index]


class _SelectedPayloadStream:
    """Incrementally key-join one selected Parquet artifact.

    Physical row order is deliberately irrelevant. Payloads for later samples
    remain buffered until requested, while payloads already consumed can be
    released. A full consumer reaches :meth:`finish`, which also detects
    duplicate or unexpected rows occurring after the final expected payload.
    """

    def __init__(
        self,
        dataset: CachedVisionDataset,
        *,
        name: str,
        sample_ids: set[str],
        expected_keys: frozenset[tuple[str, str]],
        expected_schema: Any,
        artifact_label: str,
    ) -> None:
        self._batches = iter(
            dataset._iter_selected_batches(
                name,
                sample_ids=sample_ids,
                expected_schema=expected_schema,
            )
        )
        self._expected_keys = expected_keys
        self._artifact_label = artifact_label
        self._seen: set[tuple[str, str]] = set()
        self._buffer: dict[tuple[str, str], Any] = {}
        self._exhausted = False
        self._validate_key_index(
            dataset,
            name=name,
            sample_ids=sample_ids,
            expected_schema=expected_schema,
        )

    def _validate_key_index(
        self,
        dataset: CachedVisionDataset,
        *,
        name: str,
        sample_ids: set[str],
        expected_schema: Any,
    ) -> None:
        """Validate selected keys without decoding large payload columns."""

        seen: set[tuple[str, str]] = set()
        for batch in dataset._iter_selected_batches(
            name,
            sample_ids=sample_ids,
            expected_schema=expected_schema,
            columns=("sample_id", "instance_id"),
            batch_size=_SELECTED_KEY_BATCH_ROWS,
        ):
            keys = zip(
                batch.column("sample_id").to_pylist(),
                batch.column("instance_id").to_pylist(),
                strict=True,
            )
            for key in keys:
                if key not in self._expected_keys:
                    raise DatasetValidationError(
                        f"{self._artifact_label} artifacts contain a foreign sample/instance key."
                    )
                if key in seen:
                    raise DatasetValidationError(
                        f"{self._artifact_label} artifacts contain duplicate sample/instance keys."
                    )
                seen.add(key)
        if seen != self._expected_keys:
            raise DatasetValidationError(f"{self._artifact_label} keys must match selected instance keys exactly once.")

    def _decode_batch(self, batch: Any) -> Iterator[tuple[tuple[str, str], Any]]:
        raise NotImplementedError

    def _advance(self) -> bool:
        if self._exhausted:
            return False
        try:
            batch = next(self._batches)
        except StopIteration:
            self._exhausted = True
            self._validate_complete()
            return False

        for key, payload in self._decode_batch(batch):
            if key not in self._expected_keys:
                raise DatasetValidationError(f"{self._artifact_label} artifacts contain a foreign sample/instance key.")
            if key in self._seen:
                raise DatasetValidationError(
                    f"{self._artifact_label} artifacts contain duplicate sample/instance keys."
                )
            self._seen.add(key)
            self._buffer[key] = payload
        return True

    def take(self, keys: tuple[tuple[str, str], ...]) -> tuple[Any, ...]:
        """Return payloads in detection order, advancing only as far as needed."""

        missing = tuple(key for key in keys if key not in self._buffer)
        while missing:
            if not self._advance():
                break
            missing = tuple(key for key in missing if key not in self._buffer)
        if missing:
            raise DatasetValidationError(f"{self._artifact_label} keys must match selected instance keys exactly once.")
        return tuple(self._buffer.pop(key) for key in keys)

    def finish(self) -> None:
        """Drain remaining rows and prove exact selected-key coverage."""

        # The cheap key-only pass proved there are no later selected rows once
        # every expected payload has been consumed. Avoid decoding the unused
        # tail of a large values row group merely to observe StopIteration.
        if self._seen == self._expected_keys:
            self._exhausted = True
            self._validate_complete()
            return
        while self._advance():
            pass

    def _validate_complete(self) -> None:
        if self._seen != self._expected_keys:
            raise DatasetValidationError(f"{self._artifact_label} keys must match selected instance keys exactly once.")

    def close(self) -> None:
        """Release pending payloads and the underlying Parquet iterator."""

        close = getattr(self._batches, "close", None)
        if callable(close):
            close()
        self._buffer.clear()


class _MaskPayloadStream(_SelectedPayloadStream):
    """Lazy selected-mask reader retaining row metadata for validation."""

    def _decode_batch(self, batch: Any) -> Iterator[tuple[tuple[str, str], dict[str, Any]]]:
        for row in batch.to_pylist():
            yield (row["sample_id"], row["instance_id"]), row


class _SequenceDatasetStream:
    """Replay-only iterable that lazily joins optional sequence payloads."""

    def __init__(
        self,
        dataset: CachedVisionDataset,
        *,
        load_masks: bool,
        load_embeddings: bool,
    ) -> None:
        self._dataset = dataset
        self.manifest = dataset.manifest
        self.load_images = dataset.load_images
        self.load_masks = load_masks
        self.load_embeddings = load_embeddings
        self._source_snapshot = source_snapshot(dataset.build, dataset.manifest)

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self._dataset.sample_ids

    def source_is_current(self) -> bool:
        """Whether retained sequence metadata still describes its immutable source."""

        try:
            return source_snapshot(self._dataset.build, self.manifest) == self._source_snapshot
        except (OSError, ValueError):
            return False

    def __iter__(self) -> Iterator[DatasetSample]:
        with ExitStack() as resources:
            yield from self._iter_samples(resources)

    def _iter_samples(self, resources: ExitStack) -> Iterator[DatasetSample]:
        """Join payloads while registering cleanup before the first frame is read."""

        expected_keys = frozenset(
            (sample_id, row["instance_id"])
            for sample_id, rows in self._dataset._instances_by_sample.items()
            for row in rows
        )
        selected_sample_ids = set(self._dataset.sample_ids)
        masks = None
        if self.load_masks:
            masks = _MaskPayloadStream(
                self._dataset,
                name=MASKS_ARTIFACT,
                sample_ids=selected_sample_ids,
                expected_keys=expected_keys,
                expected_schema=masks_schema(),
                artifact_label="Mask",
            )
            resources.callback(masks.close)
        embeddings = None
        if self.load_embeddings:
            embeddings = resources.enter_context(IndexedEmbeddingReader(self._dataset))

        for sample in self._dataset:
            detections = sample.detections
            instance_ids = detections.instance_ids or ()
            keys = tuple((sample.sample_id, instance_id) for instance_id in instance_ids)
            if masks is not None:
                rows = masks.take(keys)
                for row in rows:
                    if (
                        row["height"] != sample.image_size[0]
                        or row["width"] != sample.image_size[1]
                        or row["codec"] != MASK_CODEC
                    ):
                        raise DatasetValidationError(
                            "Mask rows must use the selected sample dimensions and canonical codec."
                        )
                mask_values = unpack_mask_batch(
                    (row["data"] for row in rows),
                    sample.image_size[0],
                    sample.image_size[1],
                )
                detections = detections.with_masks(MaskBatch(mask_values))
            if embeddings is not None:
                detections = detections.with_embeddings(embeddings.take(keys))
            yield replace(sample, detections=detections)

        if masks is not None:
            masks.finish()
        if embeddings is not None:
            embeddings.finish()


__all__ = ("CachedVisionDataset", "DatasetSample")
