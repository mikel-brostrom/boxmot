from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

import boxmot.datasets.replay_cache as cache
from boxmot.datasets import CachedVisionDataset
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.datasets.validation import DatasetValidationError
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint


def _sequence_build(tmp_path: Path, box_type: str) -> Path:
    """Publish numeric/lexical frame and detection order, including an empty frame."""
    detect = StagePlan.create("detect", component={"id": "fixture-detector"})
    embed = StagePlan.create(
        "embed",
        component={"id": "fixture-encoder"},
        depends_on=(detect.name,),
        upstream_fingerprints=(detect.fingerprint,),
    )
    finalize = StagePlan.create("finalize", depends_on=(embed.name,), upstream_fingerprints=(embed.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "materializations",
        dataset_name="fixture",
        box_type=box_type,
        source_fingerprint=fingerprint({"fixture": 1}),
        publish=PublishOptions(image_references=False, masks=False, embeddings=True),
        stages=(detect, embed, finalize),
    )
    plan.staging_root.mkdir(parents=True)
    writer = ParquetShardWriter(plan.staging_root, box_type=box_type)
    samples, instances, embeddings = [], [], []
    encoder = fingerprint("fixture-encoder")
    for frame_index, count in ((10, 3), (2, 0), (1, 13)):
        sample_id = f"val:sequence:{frame_index}"
        samples.append(
            dict(
                sample_id=sample_id,
                split="val",
                sequence_id="sequence",
                frame_index=frame_index,
                timestamp_s=frame_index / 10,
                image_ref=None,
                height=50,
                width=70,
            )
        )
        for index in reversed(range(count)):
            instance_id = f"{plan.build_id}:{sample_id}:{index}"
            geometry = (
                dict(x1=float(index), y1=0.0, x2=float(index + 2), y2=3.0)
                if box_type == "aabb"
                else dict(cx=float(index + 1), cy=1.5, w=2.0, h=3.0, angle=0.1)
            )
            instances.append(
                dict(
                    instance_id=instance_id,
                    sample_id=sample_id,
                    detection_index=index,
                    score=0.9,
                    class_id=1,
                    **geometry,
                )
            )
            embeddings.append(
                dict(
                    sample_id=sample_id,
                    instance_id=instance_id,
                    encoder_fingerprint=encoder,
                    dim=3,
                    values=[float(frame_index), float(index), 1.0],
                )
            )
    writer.write(SAMPLES_ARTIFACT, samples, shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances, shard_index=0)
    writer.write(EMBEDDINGS_ARTIFACT, embeddings, shard_index=0, embedding_dim=3)
    return finalize_build(plan, embedding_metadata={"encoder_fingerprint": encoder, "dim": 3}, target_shard_rows=8)


def _assert_sample_equal(actual, expected) -> None:
    for name in ("sample_id", "split", "sequence_id", "frame_index", "timestamp_s", "image_size", "image_ref"):
        assert getattr(actual, name) == getattr(expected, name)
    assert type(actual.detections.geometry) is type(expected.detections.geometry)
    assert actual.detections.instance_ids == expected.detections.instance_ids
    for name in ("scores", "class_ids", "embeddings"):
        left, right = getattr(actual.detections, name), getattr(expected.detections, name)
        if right is None:
            assert left is None
        else:
            assert torch.equal(left, right)
    assert torch.equal(actual.detections.geometry.values, expected.detections.geometry.values)
    if expected.detections.masks is not None:
        assert torch.equal(actual.detections.masks.values, expected.detections.masks.values)
    if expected.frame is not None:
        assert torch.equal(actual.frame.image, expected.frame.image)


@pytest.mark.parametrize("box_type", ["aabb", "obb"])
def test_mapped_replay_preserves_frame_and_detection_order_and_empty_frames(tmp_path, box_type) -> None:
    build = _sequence_build(tmp_path, box_type)
    expected = list(
        CachedVisionDataset._stream_sequence(build, sequence_id="sequence", split="val", load_embeddings=True)
    )
    path = cache.prepare_replay_sequence(build, sequence_id="sequence", split="val")
    replay = cache.open_replay_sequence(path)
    try:
        assert path.parent == tmp_path / "replay_cache"
        assert [sample.frame_index for sample in replay] == [1, 2, 10]
        assert len(replay) == 3
        assert replay.sample_ids == tuple(sample.sample_id for sample in expected)
        for actual, reference in zip(replay, expected, strict=True):
            _assert_sample_equal(actual, reference)
        assert list(replay)[1].detections.embeddings.shape == (0, 3)
        replay.validate()
    finally:
        replay.close()


def test_mapped_replay_is_independently_writable_and_survives_close(materialized_build) -> None:
    path = cache.prepare_replay_sequence(materialized_build["root"], sequence_id="seq-a")
    replay = cache.open_replay_sequence(path)
    first = next(iter(replay))
    expected = next(iter(replay))
    for tensor in (
        first.detections.geometry.values,
        first.detections.scores,
        first.detections.class_ids,
        first.detections.embeddings,
    ):
        tensor.zero_()
    _assert_sample_equal(next(iter(replay)), expected)
    replay.close()
    replay.close()
    assert expected.detections.embeddings.tolist() == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    with pytest.raises(cache.ReplayCacheError, match="closed"):
        next(iter(replay))


def test_warm_replay_reads_detection_and_embedding_arrays_without_parquet_or_payload_hashes(
    materialized_build, monkeypatch
) -> None:
    import pyarrow.parquet as pq

    build = materialized_build["root"]
    path = cache.prepare_replay_sequence(build, sequence_id="seq-a")

    def forbidden(*args, **kwargs):
        pytest.fail("Warm replay must not decode Parquet or hash large source/array payloads.")

    original_hash = cache.sha256_file

    def metadata_hash_only(path):
        assert Path(path).suffix == ".json"
        return original_hash(path)

    monkeypatch.setattr(pq, "ParquetFile", forbidden)
    monkeypatch.setattr(cache, "validate_published_build", forbidden)
    monkeypatch.setattr(cache, "sha256_file", metadata_hash_only)
    assert cache.prepare_replay_sequence(build, sequence_id="seq-a") == path
    replay = cache.open_replay_sequence(path)
    try:
        replay.validate()
        sample = next(iter(replay))
        assert sample.detections.class_ids.tolist() == [1, 1]
        assert sample.detections.embeddings.tolist() == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    finally:
        replay.close()


def test_masks_and_images_reuse_cached_detection_metadata(materialized_build, monkeypatch) -> None:
    import pyarrow.parquet as pq

    build = materialized_build["root"]
    expected = list(
        CachedVisionDataset._stream_sequence(
            build, sequence_id="seq-a", load_images=True, load_masks=True, load_embeddings=True
        )
    )
    path = cache.prepare_replay_sequence(build, sequence_id="seq-a")
    original = pq.ParquetFile
    opened = []

    def masks_only(path, *args, **kwargs):
        opened.append(Path(path).parent.name)
        assert Path(path).parent.name == "masks"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", masks_only)
    replay = cache.open_replay_sequence(path, load_images=True, load_masks=True)
    try:
        for actual, reference in zip(replay, expected, strict=True):
            _assert_sample_equal(actual, reference)
        assert opened
    finally:
        replay.close()


def test_partial_mask_iteration_releases_parquet_resources(materialized_build, monkeypatch) -> None:
    path = cache.prepare_replay_sequence(materialized_build["root"], sequence_id="seq-a")
    closed = []
    original = cache._MaskPayloadStream.close

    def close(stream):
        closed.append(stream)
        original(stream)

    monkeypatch.setattr(cache._MaskPayloadStream, "close", close)
    replay = cache.open_replay_sequence(path, load_masks=True)
    iterator = iter(replay)
    try:
        next(iterator)
        assert not closed
        iterator.close()
        assert len(closed) == 1
        assert closed[0]._buffer == {}
    finally:
        iterator.close()
        replay.close()


def test_detection_only_cache_works_without_embeddings(materialized_boxes_only_build) -> None:
    build = materialized_boxes_only_build
    path = cache.prepare_replay_sequence(build, sequence_id="seq-a", load_embeddings=False)
    assert not (path / "embeddings.npy").exists()
    replay = cache.open_replay_sequence(path, load_embeddings=False)
    try:
        assert next(iter(replay)).detections.embeddings is None
    finally:
        replay.close()
    with pytest.raises(cache.ReplayCacheError, match="without embeddings"):
        cache.open_replay_sequence(path)
    with pytest.raises(DatasetValidationError, match="does not publish embeddings"):
        cache.prepare_replay_sequence(build, sequence_id="seq-a")


@pytest.mark.parametrize(
    "component", ["geometry.npy", "scores.npy", "embeddings.npy", "index.json", "_SUCCESS", "directory"]
)
def test_corrupt_or_incomplete_cache_is_regenerated(materialized_build, component) -> None:
    build = materialized_build["root"]
    path = cache.prepare_replay_sequence(build, sequence_id="seq-a")
    if component == "directory":
        import shutil

        shutil.rmtree(path)
        path.write_bytes(b"interrupted cache publication")
    elif component == "_SUCCESS":
        (path / component).unlink()
    else:
        (path / component).write_bytes(b"invalid")
    assert cache.prepare_replay_sequence(build, sequence_id="seq-a") == path
    replay = cache.open_replay_sequence(path)
    try:
        assert next(iter(replay)).detections.embeddings.tolist() == [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    finally:
        replay.close()


def test_mutated_array_with_restored_mtime_is_rejected(materialized_build) -> None:
    path = cache.prepare_replay_sequence(materialized_build["root"], sequence_id="seq-a")
    file = path / "embeddings.npy"
    stat = file.stat()
    array = np.load(file, mmap_mode="r+", allow_pickle=False)
    array[0, 0] = 99.0
    array.flush()
    array._mmap.close()
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(cache.ReplayCacheError, match="changed after hashing"):
        cache.open_replay_sequence(path)


def test_changed_source_cannot_be_hidden_by_an_existing_cache(materialized_build) -> None:
    build = materialized_build["root"]
    path = cache.prepare_replay_sequence(build, sequence_id="seq-a")
    source = next((build / "embeddings").glob("*.parquet"))
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(cache.ReplayCacheError, match="source files changed"):
        cache.open_replay_sequence(path)
    with pytest.raises(DatasetValidationError):
        cache.prepare_replay_sequence(build, sequence_id="seq-a")


def test_retained_view_detects_republication(materialized_build) -> None:
    build = materialized_build["root"]
    path = cache.prepare_replay_sequence(build, sequence_id="seq-a")
    replay = cache.open_replay_sequence(path)
    try:
        (path / "_SUCCESS").unlink()
        cache.prepare_replay_sequence(build, sequence_id="seq-a")
        with pytest.raises(cache.ReplayCacheError, match="republished"):
            replay.validate()
    finally:
        replay.close()


def test_concurrent_preparation_publishes_once(materialized_build, monkeypatch) -> None:
    build = materialized_build["root"]
    original = cache._write_entry
    calls = []

    def write_once(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(cache, "_write_entry", write_once)
    barrier = threading.Barrier(2)

    def prepare():
        barrier.wait(timeout=10)
        return cache.prepare_replay_sequence(build, sequence_id="seq-a")

    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda _: prepare(), range(2)))
    assert paths[0] == paths[1]
    assert len(calls) == 1
    assert not list(paths[0].parent.glob(".*.tmp-*"))


def test_failed_publication_leaves_no_partial_entry_and_can_retry(materialized_build, monkeypatch) -> None:
    build = materialized_build["root"]
    original = cache._write_entry

    def interrupted(staging, *args):
        (staging / "geometry.npy").write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr(cache, "_write_entry", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        cache.prepare_replay_sequence(build, sequence_id="seq-a")
    root = build.parent.parent / "replay_cache"
    assert not list(root.glob(".*.tmp-*"))
    assert not list(root.glob("*/_SUCCESS"))
    monkeypatch.setattr(cache, "_write_entry", original)
    assert cache.prepare_replay_sequence(build, sequence_id="seq-a").is_dir()


def test_cache_preparation_never_modifies_the_published_build(materialized_build) -> None:
    build = materialized_build["root"]
    before = {path.relative_to(build): sha256_file(path) for path in build.rglob("*") if path.is_file()}
    cache.prepare_replay_sequence(build, sequence_id="seq-a")
    after = {path.relative_to(build): sha256_file(path) for path in build.rglob("*") if path.is_file()}
    assert after == before
    with pytest.raises(ValueError, match="outside the immutable dataset build"):
        cache.prepare_replay_sequence(build, sequence_id="seq-a", cache_root=build / "replay_cache")


def test_source_hash_validation_is_shared_across_sequence_builders(materialized_build, monkeypatch) -> None:
    build = materialized_build["root"]
    original = cache.validate_published_build
    calls = []

    def validate(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "validate_published_build", validate)
    cache.prepare_replay_sequence(build, sequence_id="seq-a")
    cache.prepare_replay_sequence(build, sequence_id="seq-b")
    assert calls == [build]


def test_manifest_changes_invalidate_retained_view(materialized_build) -> None:
    build = materialized_build["root"]
    path = cache.prepare_replay_sequence(build, sequence_id="seq-a")
    replay = cache.open_replay_sequence(path)
    try:
        manifest = build / "manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
        with pytest.raises(cache.ReplayCacheError, match="source files changed"):
            replay.validate()
        cache.prepare_replay_sequence(build, sequence_id="seq-a")
        fresh = cache.open_replay_sequence(path)
        try:
            fresh.validate()
        finally:
            fresh.close()
    finally:
        replay.close()


def test_embedding_encoder_and_detection_only_projections_have_separate_identities(materialized_build) -> None:
    build = materialized_build["root"]
    full = cache.prepare_replay_sequence(build, sequence_id="seq-a")
    boxes = cache.prepare_replay_sequence(build, sequence_id="seq-a", load_embeddings=False)
    assert full != boxes
    metadata = json.loads((full / "index.json").read_text())
    identity = metadata["identity"]
    assert "device" not in identity
    embedding = next(artifact for artifact in identity["artifacts"] if artifact["name"] == "embeddings")
    assert embedding["metadata"]["dim"] == 3
    assert embedding["metadata"]["encoder_fingerprint"]
