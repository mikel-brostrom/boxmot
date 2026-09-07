from __future__ import annotations

import json
from dataclasses import fields, replace
from inspect import Parameter, signature

import cv2
import numpy as np
import pytest
import torch

import boxmot.datasets as datasets
from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.cached import DatasetSample, _row_group_may_contain_sample
from boxmot.datasets.masks import MASK_CODEC, MaskCodecError, pack_mask, unpack_mask, unpack_mask_batch
from boxmot.datasets.readers import attach_masks, read_detection_batches
from boxmot.datasets.readers.images import ImageDecodeError, read_rgb_chw_uint8
from boxmot.datasets.schema import (
    EMBEDDINGS_ARTIFACT,
    INSTANCES_ARTIFACT,
    MASKS_ARTIFACT,
    SCHEMA_ID,
    instances_schema,
)
from boxmot.datasets.storage import (
    describe_parquet_artifact,
    read_parquet_artifact,
    resolve_embedding_metadata,
    write_compacted_parquet_artifact,
    write_parquet_records,
)
from boxmot.datasets.validation import DatasetValidationError, validate_dataset
from boxmot.engine.eval.replay import ReplayProgressEvent, replay_build
from boxmot.structures import Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec


def test_public_loader_and_sample_contract_is_exact() -> None:
    assert datasets.__all__ == (
        "ArtifactRecord",
        "CachedVisionDataset",
        "DatasetSample",
        "DatasetManifest",
        "ManifestError",
        "PublishedContent",
        "ShardRecord",
        "StageProvenance",
    )
    parameters = signature(CachedVisionDataset).parameters

    assert tuple(parameters) == (
        "build",
        "split",
        "load_images",
        "load_masks",
        "load_embeddings",
    )
    assert all(parameters[name].kind is Parameter.KEYWORD_ONLY for name in tuple(parameters)[1:])
    assert tuple(field.name for field in fields(DatasetSample)) == (
        "sample_id",
        "split",
        "sequence_id",
        "frame_index",
        "timestamp_s",
        "image_size",
        "image_ref",
        "frame",
        "detections",
    )


def test_mask_bitpack_roundtrip_preserves_odd_full_frame_shape() -> None:
    mask = torch.tensor(
        [
            [True, False, True, False, True],
            [False, True, False, True, False],
            [True, True, False, False, True],
        ],
        dtype=torch.bool,
    )

    payload = pack_mask(mask)

    assert len(payload) == 2
    assert torch.equal(unpack_mask(payload, 3, 5), mask)
    assert unpack_mask_batch([], 3, 5).shape == (0, 3, 5)
    assert MASK_CODEC == "bitpack-row-major-v1"


def test_mask_bitpack_rejects_wrong_dtype_and_payload_size() -> None:
    with pytest.raises(MaskCodecError, match="boolean"):
        pack_mask(torch.zeros((2, 2), dtype=torch.uint8))
    with pytest.raises(MaskCodecError, match="expected"):
        unpack_mask(b"", 2, 2)
    with pytest.raises(MaskCodecError, match="outside the row-major image extent"):
        unpack_mask(bytes((0x00, 0x80)), 3, 5)


def test_manifest_rejects_legacy_schema_identifier(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "build_id": "build-000000000000000000000000",
                "created_at": "2026-01-01T00:00:00Z",
                "box_type": "aabb",
                "artifacts": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy positional caches"):
        DatasetManifest.load(tmp_path)


@pytest.mark.parametrize("suffix", (".npy", ".npz", ".txt"))
def test_manifest_rejects_legacy_cache_roots_without_reading_them(tmp_path, suffix) -> None:
    legacy = tmp_path / "dets" / f"sequence{suffix}"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy")

    with pytest.raises(ValueError, match="Unsupported dataset schema.*legacy"):
        DatasetManifest.load(tmp_path)


def test_manifest_rejects_string_boolean(materialized_build) -> None:
    raw = json.loads((materialized_build["root"] / "manifest.json").read_text(encoding="utf-8"))
    raw["complete"] = "false"

    with pytest.raises(ValueError, match="complete must be a boolean"):
        DatasetManifest.from_dict(raw)


def test_manifest_requires_full_sha256_build_id(materialized_build) -> None:
    manifest = DatasetManifest.load(materialized_build["root"])

    with pytest.raises(ValueError, match="Invalid build_id"):
        replace(manifest, build_id="build-short-id")


def test_manifest_requires_stage_and_source_fingerprint_provenance(materialized_build) -> None:
    manifest = DatasetManifest.load(materialized_build["root"])

    with pytest.raises(ValueError, match="at least one materialization stage"):
        replace(manifest, stages=())
    with pytest.raises(ValueError, match="source_fingerprint"):
        replace(manifest, metadata={})


def test_instance_schemas_expose_only_active_geometry() -> None:
    assert instances_schema("aabb").names == [
        "instance_id",
        "sample_id",
        "detection_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "class_id",
    ]
    assert instances_schema("obb").names == [
        "instance_id",
        "sample_id",
        "detection_index",
        "cx",
        "cy",
        "w",
        "h",
        "angle",
        "score",
        "class_id",
    ]


def test_materialized_loader_uses_keys_and_semantic_sample_order(materialized_build) -> None:
    root = materialized_build["root"]
    dataset = CachedVisionDataset(root, load_masks=True, load_embeddings=True)

    assert dataset.sample_ids == ("sample-a", "sample-b")
    assert dataset[0].frame is None
    assert dataset[0].split == "train"
    assert dataset[0].sequence_id == "seq-a"
    assert dataset[0].frame_index == 3
    assert dataset[0].image_size == (5, 7)
    assert dataset[0].detections.instance_ids == (
        f"{dataset.manifest.build_id}:sample-a:0",
        f"{dataset.manifest.build_id}:sample-a:1",
    )
    assert dataset[0].detections.masks is not None
    assert dataset[0].detections.masks.values[:, 0, 0].tolist() == [True, False]
    assert dataset[0].detections.embeddings is not None
    assert dataset[0].detections.embeddings.tolist() == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]

    raw_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert raw_manifest["schema"] == SCHEMA_ID
    assert raw_manifest["complete"] is True
    assert raw_manifest["publish"] == {
        "embeddings": True,
        "image_references": True,
        "masks": True,
    }
    assert raw_manifest["counts"] == {"embeddings": 3, "instances": 3, "masks": 3, "samples": 2}
    assert {artifact["path"] for artifact in raw_manifest["artifacts"]} == {
        "samples",
        "instances",
        "masks",
        "embeddings",
    }
    for artifact in raw_manifest["artifacts"]:
        assert artifact["rows"] == sum(shard["rows"] for shard in artifact["shards"])
        assert all(len(shard["sha256"]) == 64 for shard in artifact["shards"])
    assert (root / "samples" / "part-00000.parquet").is_file()

    instance_columns = read_parquet_artifact(
        root / "instances",
        artifact_name=INSTANCES_ARTIFACT,
        box_type="aabb",
    ).column_names
    assert instance_columns == [
        "instance_id",
        "sample_id",
        "detection_index",
        "x1",
        "y1",
        "x2",
        "y2",
        "score",
        "class_id",
    ]
    assert "codec" in read_parquet_artifact(root / "masks", artifact_name=MASKS_ARTIFACT).column_names
    embedding_table = read_parquet_artifact(root / "embeddings", artifact_name=EMBEDDINGS_ARTIFACT)
    assert {"encoder_fingerprint", "dim", "values"} <= set(embedding_table.column_names)
    assert embedding_table.schema.field("values").type.list_size == 3


def test_sequence_worker_loader_reads_only_the_selected_sequence(materialized_build) -> None:
    dataset = CachedVisionDataset._for_sequence(
        materialized_build["root"],
        sequence_id="seq-b",
        split="validation",
        load_masks=True,
        load_embeddings=True,
    )

    assert dataset.sample_ids == ("sample-b",)
    assert dataset[0].sequence_id == "seq-b"
    assert dataset[0].detections.instance_ids == (
        f"{dataset.manifest.build_id}:sample-b:0",
    )
    assert dataset[0].detections.masks is not None
    assert dataset[0].detections.embeddings is not None
    assert dataset[0].detections.embeddings.tolist() == [[1.0, 0.0, 0.0]]


def test_sequence_worker_streams_large_payloads_instead_of_using_eager_read(
    materialized_build,
    monkeypatch,
) -> None:
    eager_reads: list[str] = []
    original_read = CachedVisionDataset._read

    def recording_read(self, name, *, filters=None):
        eager_reads.append(name)
        return original_read(self, name, filters=filters)

    monkeypatch.setattr(CachedVisionDataset, "_read", recording_read)

    dataset = CachedVisionDataset._for_sequence(
        materialized_build["root"],
        sequence_id="seq-b",
        split="validation",
        load_masks=True,
        load_embeddings=True,
    )

    assert len(dataset) == 1
    assert MASKS_ARTIFACT not in eager_reads
    assert EMBEDDINGS_ARTIFACT not in eager_reads


def test_sequence_stream_defers_optional_payload_reads_until_iteration(
    materialized_build,
    monkeypatch,
) -> None:
    streamed: list[tuple[str, tuple[str, ...] | None]] = []
    original_iter = CachedVisionDataset._iter_selected_batches

    def recording_iter(self, name, *, sample_ids, expected_schema, columns=None, batch_size=128):
        streamed.append((name, columns))
        yield from original_iter(
            self,
            name,
            sample_ids=sample_ids,
            expected_schema=expected_schema,
            columns=columns,
            batch_size=batch_size,
        )

    monkeypatch.setattr(CachedVisionDataset, "_iter_selected_batches", recording_iter)

    dataset = CachedVisionDataset._stream_sequence(
        materialized_build["root"],
        sequence_id="seq-a",
        split="train",
        load_masks=True,
        load_embeddings=True,
    )

    assert len(dataset) == 1
    assert dataset.sample_ids == ("sample-a",)
    assert streamed == []

    sample = next(iter(dataset))

    assert streamed == [
        (MASKS_ARTIFACT, ("sample_id", "instance_id")),
        (EMBEDDINGS_ARTIFACT, ("sample_id", "instance_id")),
        (MASKS_ARTIFACT, None),
        (EMBEDDINGS_ARTIFACT, None),
    ]
    assert sample.detections.masks is not None
    assert sample.detections.embeddings is not None


def test_sequence_stream_key_joins_independently_reordered_payloads(
    materialized_build,
    monkeypatch,
) -> None:
    import pyarrow as pa

    original_iter = CachedVisionDataset._iter_selected_batches

    def reversed_iter(self, name, *, sample_ids, expected_schema, columns=None, batch_size=128):
        batches = tuple(
            original_iter(
                self,
                name,
                sample_ids=sample_ids,
                expected_schema=expected_schema,
                columns=columns,
                batch_size=batch_size,
            )
        )
        if not batches:
            return
        table = pa.Table.from_batches(batches)
        if table.num_rows:
            indices = pa.array(list(reversed(range(table.num_rows))), type=pa.int64())
            table = table.take(indices)
        yield from table.to_batches()

    monkeypatch.setattr(CachedVisionDataset, "_iter_selected_batches", reversed_iter)

    sample = next(
        iter(
            CachedVisionDataset._stream_sequence(
                materialized_build["root"],
                sequence_id="seq-a",
                split="train",
                load_masks=True,
                load_embeddings=True,
            )
        )
    )

    assert sample.detections.masks is not None
    assert sample.detections.masks.values[:, 0, 0].tolist() == [True, False]
    assert sample.detections.embeddings is not None
    assert sample.detections.embeddings.tolist() == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("duplicate", "duplicate sample/instance keys"),
        ("foreign", "foreign sample/instance key"),
        ("missing", "match selected instance keys exactly once"),
    ],
)
def test_sequence_stream_validates_all_embedding_keys_by_eof(
    materialized_build,
    monkeypatch,
    mode,
    message,
) -> None:
    import pyarrow as pa

    original_iter = CachedVisionDataset._iter_selected_batches

    def altered_iter(self, name, *, sample_ids, expected_schema, columns=None, batch_size=128):
        batches = tuple(
            original_iter(
                self,
                name,
                sample_ids=sample_ids,
                expected_schema=expected_schema,
                columns=columns,
                batch_size=batch_size,
            )
        )
        if name != EMBEDDINGS_ARTIFACT:
            yield from batches
            return
        table = pa.Table.from_batches(batches)
        if mode == "missing":
            table = table.slice(0, table.num_rows - 1)
        yield from table.to_batches()
        if mode == "duplicate":
            yield from table.slice(0, 1).to_batches()
        elif mode == "foreign":
            foreign = table.slice(0, 1)
            instance_index = foreign.schema.get_field_index("instance_id")
            foreign = foreign.set_column(
                instance_index,
                "instance_id",
                pa.array(["foreign-instance"], type=pa.string()),
            )
            yield from foreign.to_batches()

    monkeypatch.setattr(CachedVisionDataset, "_iter_selected_batches", altered_iter)
    iterator = iter(
        CachedVisionDataset._stream_sequence(
            materialized_build["root"],
            sequence_id="seq-a",
            split="train",
            load_embeddings=True,
        )
    )

    with pytest.raises(DatasetValidationError, match=message):
        next(iterator)


def test_sequence_stream_skips_heavy_tail_after_prevalidated_keys(
    materialized_build,
    monkeypatch,
) -> None:
    original_iter = CachedVisionDataset._iter_selected_batches

    def guarded_iter(self, name, *, sample_ids, expected_schema, columns=None, batch_size=128):
        batches = iter(
            original_iter(
                self,
                name,
                sample_ids=sample_ids,
                expected_schema=expected_schema,
                columns=columns,
                batch_size=batch_size,
            )
        )
        if columns is not None or name != EMBEDDINGS_ARTIFACT:
            yield from batches
            return
        yield next(batches)
        raise AssertionError("the embedding payload tail should not be requested")

    monkeypatch.setattr(CachedVisionDataset, "_iter_selected_batches", guarded_iter)

    samples = tuple(
        CachedVisionDataset._stream_sequence(
            materialized_build["root"],
            sequence_id="seq-a",
            split="train",
            load_embeddings=True,
        )
    )

    assert len(samples) == 1
    assert samples[0].detections.embeddings is not None


def test_selected_artifact_row_group_pruning_is_conservative(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "row-groups.parquet"
    pq.write_table(
        pa.table({"sample_id": ["a", "b", "m", "n"]}),
        path,
        row_group_size=2,
    )
    parquet = pq.ParquetFile(path)

    assert not _row_group_may_contain_sample(parquet, 0, sample_ids=("m",))
    assert _row_group_may_contain_sample(parquet, 1, sample_ids=("m",))


def test_sequence_worker_rejects_duplicate_instance_keys(materialized_build, monkeypatch) -> None:
    import pyarrow as pa

    original_read = CachedVisionDataset._read

    def duplicated_instances(self, name, *, filters=None):
        table = original_read(self, name, filters=filters)
        if name == INSTANCES_ARTIFACT and table.num_rows:
            return pa.concat_tables((table, table.slice(0, 1)))
        return table

    monkeypatch.setattr(CachedVisionDataset, "_read", duplicated_instances)

    with pytest.raises(DatasetValidationError, match="duplicate sample/instance keys"):
        CachedVisionDataset._for_sequence(
            materialized_build["root"],
            sequence_id="seq-b",
            split="validation",
        )


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("duplicate", "duplicate sample/instance keys"),
        ("missing", "match selected instance keys exactly once"),
    ],
)
def test_sequence_worker_validates_streamed_embedding_keys(materialized_build, monkeypatch, mode, message) -> None:
    original_iter = CachedVisionDataset._iter_selected_batches

    def altered_batches(self, name, *, sample_ids, expected_schema, columns=None, batch_size=128):
        batches = tuple(
            original_iter(
                self,
                name,
                sample_ids=sample_ids,
                expected_schema=expected_schema,
                columns=columns,
                batch_size=batch_size,
            )
        )
        if name != EMBEDDINGS_ARTIFACT:
            yield from batches
        elif mode == "duplicate":
            yield from batches
            yield from batches

    monkeypatch.setattr(CachedVisionDataset, "_iter_selected_batches", altered_batches)

    with pytest.raises(DatasetValidationError, match=message):
        CachedVisionDataset._for_sequence(
            materialized_build["root"],
            sequence_id="seq-b",
            split="validation",
            load_embeddings=True,
        )


def test_artifact_readers_decode_keyed_detection_and_mask_rows(materialized_build) -> None:
    root = materialized_build["root"]

    detections = read_detection_batches(root, "aabb")
    enriched = attach_masks(root, detections)

    assert tuple(enriched) == ("sample-a", "sample-b")
    assert enriched["sample-a"].instance_ids == (
        f"{materialized_build['plan'].build_id}:sample-a:0",
        f"{materialized_build['plan'].build_id}:sample-a:1",
    )
    assert enriched["sample-a"].masks is not None
    assert enriched["sample-a"].masks.values[:, 0, 0].tolist() == [True, False]


def test_compaction_keeps_embeddings_in_canonical_parquet_format(materialized_build, tmp_path) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)
    destination = tmp_path / "compacted-embeddings"

    shards = write_compacted_parquet_artifact(
        root / "embeddings",
        destination,
        artifact_name=EMBEDDINGS_ARTIFACT,
        box_type="aabb",
        target_rows=1,
    )
    table = read_parquet_artifact(destination, artifact_name=EMBEDDINGS_ARTIFACT)
    metadata = resolve_embedding_metadata(
        table,
        declared=manifest.artifact(EMBEDDINGS_ARTIFACT).metadata,
    )

    assert [path.name for path in shards] == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    assert [row["instance_id"] for row in table.to_pylist()] == [
        f"{manifest.build_id}:sample-a:0",
        f"{manifest.build_id}:sample-a:1",
        f"{manifest.build_id}:sample-b:0",
    ]
    assert metadata == {
        "encoder_fingerprint": manifest.artifact(EMBEDDINGS_ARTIFACT).metadata["encoder_fingerprint"],
        "dim": 3,
    }
    assert not tuple(destination.glob("*.npy"))


def test_loader_populates_all_frame_metadata(materialized_build) -> None:
    dataset = CachedVisionDataset(
        materialized_build["root"],
        split="validation",
        load_images=True,
    )

    sample = dataset[0]
    assert sample.sample_id == "sample-b"
    assert sample.frame is not None
    assert sample.frame.sample_id == sample.sample_id
    assert sample.frame.sequence_id == "seq-b"
    assert sample.frame.frame_index == 9
    assert sample.frame.timestamp_s == 1.5
    assert sample.frame.source_uri == "images/b.jpg"


def test_loader_resolves_relative_image_ref_against_manifest_source_root(materialized_build, tmp_path) -> None:
    root = materialized_build["root"]
    source_root = tmp_path / "external-source"
    (source_root / "images").mkdir(parents=True)
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    image[..., 2] = 255
    assert cv2.imwrite(str(source_root / "images" / "b.jpg"), image)
    manifest = DatasetManifest.load(root)
    replace(
        manifest,
        metadata={**dict(manifest.metadata), "source_root_uri": source_root.as_uri()},
    ).write(root)

    sample = CachedVisionDataset(root, split="validation", load_images=True)[0]

    assert sample.frame is not None
    assert sample.frame.image[0].float().mean().item() > 250


def test_loader_decodes_relative_mmot_numpy_image_ref(materialized_build, tmp_path) -> None:
    root = materialized_build["root"]
    source_root = tmp_path / "external-mmot-source"
    image_path = source_root / "test" / "npy" / "SEQ-01" / "000001.npy"
    image_path.parent.mkdir(parents=True)
    image = np.empty((5, 7, 8), dtype=np.uint8)
    for channel_index in range(image.shape[2]):
        image[..., channel_index] = 10 + channel_index
    np.save(image_path, image)

    rows = read_parquet_artifact(root / "samples", artifact_name="samples").to_pylist()
    validation_row = next(row for row in rows if row["sample_id"] == "sample-b")
    validation_row["image_ref"] = "test/npy/SEQ-01/000001.npy"
    write_parquet_records(
        root / "samples" / "part-00000.parquet",
        rows,
        artifact_name="samples",
    )
    manifest = DatasetManifest.load(root)
    samples_artifact = describe_parquet_artifact(root, name="samples", relative_path="samples")
    replace(
        manifest,
        artifacts=tuple(
            samples_artifact if artifact.name == "samples" else artifact for artifact in manifest.artifacts
        ),
        metadata={**dict(manifest.metadata), "source_root_uri": source_root.as_uri()},
    ).write(root)

    sample = CachedVisionDataset(root, split="validation", load_images=True)[0]

    assert sample.frame is not None
    assert sample.frame.image[:, 0, 0].tolist() == [14, 12, 11]
    assert sample.frame.image.dtype is torch.uint8
    assert sample.frame.image.is_contiguous()


def test_image_reader_decodes_selected_local_video_frame(monkeypatch, tmp_path) -> None:
    import boxmot.datasets.readers.images as image_reader

    class FakeCapture:
        def __init__(self) -> None:
            self.seek = None
            self.released = False

        def isOpened(self):
            return True

        def set(self, prop, value):
            self.seek = (prop, value)
            return True

        def read(self):
            return True, np.array([[[1, 2, 3]]], dtype=np.uint8)

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(image_reader.cv2, "VideoCapture", lambda _path: capture)

    image = read_rgb_chw_uint8("clip.mp4#frame=7", tmp_path.as_uri())

    assert capture.seek == (cv2.CAP_PROP_POS_FRAMES, 7)
    assert capture.released is True
    assert image[:, 0, 0].tolist() == [3, 2, 1]
    assert image.dtype is torch.uint8
    assert image.is_contiguous()


@pytest.mark.parametrize(
    ("channel_count", "expected_rgb"),
    (
        pytest.param(8, [14, 12, 11], id="mmot-3ch-checkpoint-model-rgb-bands"),
        pytest.param(5, [12, 11, 10], id="other-multichannel-first-three"),
    ),
)
def test_image_reader_decodes_numpy_channel_conventions(tmp_path, channel_count, expected_rgb) -> None:
    path = tmp_path / "test" / "npy" / "SEQ-01" / f"frame-{channel_count}.npy"
    path.parent.mkdir(parents=True)
    source = np.empty((3, 5, channel_count), dtype=np.uint8)
    for channel_index in range(channel_count):
        source[..., channel_index] = 10 + channel_index
    np.save(path, source)

    image = read_rgb_chw_uint8(path.relative_to(tmp_path).as_posix(), tmp_path.as_uri())

    assert image[:, 0, 0].tolist() == expected_rgb
    assert image.shape == (3, 3, 5)
    assert image.dtype is torch.uint8
    assert image.is_contiguous()


@pytest.mark.parametrize(
    ("reference", "root", "message"),
    [
        ("frame.jpg", "https://example.invalid/data", "nonlocal scheme"),
        ("frame.jpg", "file://remote.invalid/data", "nonlocal host"),
        ("https://example.invalid/frame.jpg", ".", "nonlocal scheme"),
        ("clip.mp4#page=2", ".", "expected '#frame=N'"),
        ("frame.jpg#frame=2", ".", "only valid for a local video"),
        ("clip.mp4", ".", "must select one frame"),
    ],
)
def test_image_reader_rejects_nonlocal_or_ambiguous_references(reference, root, message) -> None:
    with pytest.raises(ImageDecodeError, match=message):
        read_rgb_chw_uint8(reference, root)


def test_loader_rejects_missing_requested_optional_artifact(materialized_boxes_only_build) -> None:
    with pytest.raises(DatasetValidationError, match="does not publish masks"):
        CachedVisionDataset(materialized_boxes_only_build, load_masks=True)


@pytest.mark.parametrize("option", ("load_images", "load_masks", "load_embeddings"))
def test_loader_rejects_non_boolean_load_options(materialized_boxes_only_build, option) -> None:
    kwargs = {option: 1}

    with pytest.raises(TypeError, match=f"{option} must be a boolean"):
        CachedVisionDataset(materialized_boxes_only_build, **kwargs)


def test_loader_rejects_noncanonical_split(materialized_boxes_only_build) -> None:
    with pytest.raises(ValueError, match="split must be"):
        CachedVisionDataset(materialized_boxes_only_build, split=" train ")


def test_validation_rejects_non_zstd_parquet_shard(materialized_build) -> None:
    import pyarrow.parquet as pq

    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)
    samples_path = root / "samples" / "part-00000.parquet"
    table = read_parquet_artifact(root / "samples", artifact_name="samples")
    pq.write_table(table, samples_path, compression="snappy")
    replacement = describe_parquet_artifact(root, name="samples", relative_path="samples")
    modified = replace(
        manifest,
        artifacts=tuple(
            replacement if artifact.name == "samples" else artifact for artifact in manifest.artifacts
        ),
    )

    with pytest.raises(DatasetValidationError, match="zstd compression"):
        validate_dataset(root, manifest=modified)


def test_validation_checks_encoder_fingerprint_on_every_embedding_shard(materialized_build) -> None:
    import pyarrow.parquet as pq

    from boxmot.datasets.storage import ENCODER_FINGERPRINT_METADATA_KEY

    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)
    shard_path = root / manifest.artifact(EMBEDDINGS_ARTIFACT).shards[-1].path
    table = pq.read_table(shard_path).replace_schema_metadata(
        {ENCODER_FINGERPRINT_METADATA_KEY: ("f" * 64).encode("ascii")}
    )
    pq.write_table(table, shard_path, compression="zstd")
    replacement = describe_parquet_artifact(
        root,
        name=EMBEDDINGS_ARTIFACT,
        relative_path=EMBEDDINGS_ARTIFACT,
        metadata=manifest.artifact(EMBEDDINGS_ARTIFACT).metadata,
    )
    modified = replace(
        manifest,
        artifacts=tuple(
            replacement if artifact.name == EMBEDDINGS_ARTIFACT else artifact
            for artifact in manifest.artifacts
        ),
    )

    with pytest.raises(DatasetValidationError, match="shard.*encoder fingerprint"):
        validate_dataset(root, manifest=modified)


def test_validation_rejects_unpublished_optional_artifact_directory(materialized_boxes_only_build) -> None:
    (materialized_boxes_only_build / "masks").mkdir()

    with pytest.raises(DatasetValidationError, match="unpublished 'masks'"):
        validate_dataset(materialized_boxes_only_build)


def test_validation_checks_each_manifest_shard_row_count(materialized_build) -> None:
    manifest = DatasetManifest.load(materialized_build["root"])
    embeddings = manifest.artifact(EMBEDDINGS_ARTIFACT)
    assert [shard.rows for shard in embeddings.shards] == [2, 1]
    swapped = (
        replace(embeddings.shards[0], rows=1),
        replace(embeddings.shards[1], rows=2),
    )
    modified = replace(embeddings, shards=swapped)
    bad_manifest = replace(
        manifest,
        artifacts=tuple(
            modified if artifact.name == EMBEDDINGS_ARTIFACT else artifact
            for artifact in manifest.artifacts
        ),
    )

    with pytest.raises(DatasetValidationError, match="shard.*row count"):
        validate_dataset(materialized_build["root"], manifest=bad_manifest, verify_hashes=False)


def test_validation_checks_publication_marker_contents(materialized_build) -> None:
    (materialized_build["root"] / "_SUCCESS").write_text(
        json.dumps({"schema": SCHEMA_ID, "build_id": "build-wrongwrongwrongwrongwrongwr"}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="marker does not match"):
        validate_dataset(materialized_build["root"])


def test_validation_binds_sample_count_to_source_catalog(materialized_build) -> None:
    manifest = DatasetManifest.load(materialized_build["root"])
    modified = replace(manifest, metadata={**dict(manifest.metadata), "source_count": 3})

    with pytest.raises(DatasetValidationError, match="does not match manifest source_count"):
        validate_dataset(materialized_build["root"], manifest=modified)


def test_validation_rejects_duplicate_sequence_frame_identity(materialized_build) -> None:
    root = materialized_build["root"]
    manifest = DatasetManifest.load(root)
    rows = read_parquet_artifact(root / "samples", artifact_name="samples").to_pylist()
    rows[1]["split"] = rows[0]["split"]
    rows[1]["sequence_id"] = rows[0]["sequence_id"]
    rows[1]["frame_index"] = rows[0]["frame_index"]
    write_parquet_records(
        root / "samples" / "part-00000.parquet",
        rows,
        artifact_name="samples",
    )
    replacement = describe_parquet_artifact(root, name="samples", relative_path="samples")
    modified = replace(
        manifest,
        artifacts=tuple(
            replacement if artifact.name == "samples" else artifact for artifact in manifest.artifacts
        ),
    )

    with pytest.raises(DatasetValidationError, match="unique.*sequence_id.*frame_index"):
        validate_dataset(root, manifest=modified)


def test_keyed_build_replays_precomputed_embeddings_through_live_tracker(materialized_build, tmp_path) -> None:
    class Tracker:
        name = "fixture"
        supports_obb = False
        requirements = TrackerRequirements(embeddings=True)

        def update(self, detections, frame=None):
            del frame
            assert detections.embeddings is not None
            count = len(detections)
            return Tracks(
                geometry=detections.geometry,
                track_ids=torch.arange(1, count + 1, dtype=torch.int64),
                scores=detections.scores,
                class_ids=detections.class_ids,
                detection_indices=torch.arange(count, dtype=torch.int64),
                sample_id=detections.sample_id,
            )

        def reset(self):
            return None

    replay = replay_build(
        materialized_build["root"],
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "tracks",
        tracker=Tracker(),
    )

    assert replay.frames == 2
    assert replay.track_rows == 3
    assert [path.name for path in replay.sequence_files] == ["seq-a.txt", "seq-b.txt"]
    assert (tmp_path / "tracks" / "seq-a.txt").read_text(encoding="utf-8").startswith("4,1,")


def test_keyed_build_replays_sequences_in_spawned_workers(materialized_build, tmp_path) -> None:
    events: list[ReplayProgressEvent] = []

    replay = replay_build(
        materialized_build["root"],
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "spawned-tracks",
        workers=2,
        progress_callback=events.append,
    )

    assert replay.frames == 2
    assert [path.name for path in replay.sequence_files] == ["seq-a.txt", "seq-b.txt"]
    assert {(event.sequence_id, event.status) for event in events if event.status == "queued"} == {
        ("seq-a", "queued"),
        ("seq-b", "queued"),
    }
    assert {(event.sequence_id, event.status) for event in events if event.status == "completed"} == {
        ("seq-a", "completed"),
        ("seq-b", "completed"),
    }
