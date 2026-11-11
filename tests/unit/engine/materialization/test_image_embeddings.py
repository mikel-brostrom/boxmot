"""ReID checkpoints save each image independently of inference crop batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter, read_parquet_artifact
from boxmot.engine.materialization import (
    BuildPlan,
    DatasetMaterializer,
    DetectStage,
    EmbedStage,
    FinalizeStage,
    PublishOptions,
    SourceSample,
    StagePlan,
    fingerprint,
)
from boxmot.structures import Boxes, Detections, Frame

ENCODER_FINGERPRINT = fingerprint("per-image-test-encoder")


class CountingDetector:
    """Supply a chosen number of detections for each source image."""

    def __init__(self, counts: Sequence[int]) -> None:
        self.counts = counts

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        """Return ordinary detections, including valid empty image results."""
        outputs = []
        for frame in frames:
            assert frame.frame_index is not None
            count = self.counts[frame.frame_index]
            outputs.append(
                Detections(
                    geometry=Boxes(torch.tensor([[1.0, 1.0, 4.0, 5.0]]).repeat(count, 1)),
                    scores=torch.full((count,), 0.9, dtype=torch.float32),
                    class_ids=torch.zeros(count, dtype=torch.int64),
                    sample_id=frame.sample_id,
                )
            )
        return outputs


class RecordingEncoder:
    """Expose image/crop boundaries and optionally interrupt one image."""

    embedding_dim = 2

    def __init__(self, *, fail_on_image: str | None = None) -> None:
        self.fail_on_image = fail_on_image
        self.calls: list[tuple[str, int]] = []

    def encode(self, frames: Sequence[Frame], detections: Sequence[Detections]) -> list[torch.Tensor]:
        """Encode instance position and image index without loading a model."""
        outputs = []
        for frame, batch in zip(frames, detections, strict=True):
            self.calls.append((frame.sample_id, len(batch)))
            if frame.sample_id == self.fail_on_image:
                raise RuntimeError("interrupted image embedding")
            assert frame.frame_index is not None
            assert batch.instance_ids is not None
            outputs.append(
                torch.tensor(
                    [
                        [float(instance_id.rsplit(":", 1)[1]), float(frame.frame_index)]
                        for instance_id in batch.instance_ids
                    ],
                    dtype=torch.float32,
                ).reshape(-1, self.embedding_dim)
            )
        return outputs


@pytest.fixture
def samples(tmp_path: Path) -> list[SourceSample]:
    """Decode real tiny PNG images while keeping all perception synthetic."""
    source_root = tmp_path / "images"
    source_root.mkdir()
    result = []
    for index in range(3):
        path = source_root / f"sample-{index}.png"
        assert cv2.imwrite(str(path), np.full((8, 10, 3), index, dtype=np.uint8))
        result.append(
            SourceSample(
                sample_id=f"sample-{index}",
                split="train",
                sequence_id="sequence",
                frame_index=index,
                timestamp_s=float(index),
                image_size=(8, 10),
                source_uri=path.as_uri(),
                source_sha256=sha256_file(path),
            )
        )
    return result


def _plan(tmp_path: Path, *, batch_size: int = 4) -> BuildPlan:
    """Use different detection and crop batches to expose accidental coupling."""
    detect = StagePlan.create("detect", batch_size=2)
    embed = StagePlan.create(
        "embed",
        batch_size=batch_size,
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    finalize = StagePlan.create(
        "finalize",
        depends_on=("embed",),
        upstream_fingerprints=(embed.fingerprint,),
    )
    return BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="per-image-embeddings",
        box_type="aabb",
        source_fingerprint=fingerprint("per-image-test-source"),
        publish=PublishOptions(image_references=False, embeddings=True),
        stages=(detect, embed, finalize),
    )


def _materializer(
    plan: BuildPlan,
    samples: Sequence[SourceSample],
    counts: Sequence[int],
    encoder: RecordingEncoder,
    *,
    batch_size_resolvers: Mapping[str, Callable[[], int]] | None = None,
) -> DatasetMaterializer:
    """Construct the real detect/embed/finalize workflow with small fake models."""
    return DatasetMaterializer(
        plan,
        [
            DetectStage(CountingDetector(counts), samples),
            EmbedStage(encoder, samples, encoder_fingerprint=ENCODER_FINGERPRINT),
            FinalizeStage(embedding_metadata={"encoder_fingerprint": ENCODER_FINGERPRINT, "dim": 2}),
        ],
        batch_size_resolvers=batch_size_resolvers,
    )


def _assert_published_images(output: Path, counts: Sequence[int]) -> None:
    """Check independent physical files and key-based loader associations."""
    assert not (output / "_materialization_state.json").exists()
    assert not (output / "_materialization_state.jsonl").exists()
    files = sorted((output / "embeddings").glob("*.parquet"))
    assert [path.name for path in files] == [f"part-{index:05d}.parquet" for index in range(len(counts))]
    manifest = DatasetManifest.load(output)
    assert len(manifest.artifact(EMBEDDINGS_ARTIFACT).shards) == len(counts)
    for index, (path, count) in enumerate(zip(files, counts, strict=True)):
        rows = read_parquet_artifact(path, artifact_name=EMBEDDINGS_ARTIFACT).to_pylist()
        assert len(rows) == count
        assert {row["sample_id"] for row in rows} == ({f"sample-{index}"} if count else set())
        assert {row["instance_id"] for row in rows} == {
            f"{manifest.build_id}:sample-{index}:{detection_index}" for detection_index in range(count)
        }

    dataset = CachedVisionDataset(output, load_embeddings=True)
    assert len(dataset) == len(counts)
    for sample, count in zip(dataset, counts, strict=True):
        assert sample.detections.embeddings is not None
        expected = torch.tensor(
            [[float(index), float(sample.frame_index)] for index in range(count)], dtype=torch.float32
        ).reshape(count, 2)
        torch.testing.assert_close(sample.detections.embeddings, expected)


@pytest.mark.parametrize("automatic", [False, True])
def test_interrupted_image_resumes_with_a_different_crop_batch_size(
    tmp_path: Path, samples: list[SourceSample], automatic: bool
) -> None:
    """Finish and checkpoint the first image before starting the second one."""
    plan = _plan(tmp_path, batch_size=4)
    counts = (5, 7, 0)
    interrupted_encoder = RecordingEncoder(fail_on_image="sample-1")
    interrupted = _materializer(plan, samples, counts, interrupted_encoder)

    with pytest.raises(RuntimeError, match="interrupted image embedding"):
        interrupted.run()

    assert interrupted_encoder.calls == [("sample-0", 4), ("sample-0", 1), ("sample-1", 4)]
    assert interrupted.state.state.by_name["embed"].completed_shards == ("00000",)
    first_path = plan.staging_root / "embeddings" / "part-00000.parquet"
    first_hash = sha256_file(first_path)
    assert not (first_path.parent / "part-00001.parquet").exists()

    requested = replace(
        plan,
        stages=tuple(
            replace(stage, batch_size=8 if automatic else 2) if stage.name == "embed" else stage
            for stage in plan.stages
        ),
    )
    profile_calls = []

    def profile_batch_size() -> int:
        """Stand in for GPU profiling while preserving its orchestration contract."""
        profile_calls.append("embed")
        return 2

    resumed_encoder = RecordingEncoder()
    resumed = _materializer(
        requested,
        samples,
        counts,
        resumed_encoder,
        batch_size_resolvers={"embed": profile_batch_size} if automatic else None,
    )
    output = resumed.run()

    assert resumed.plan.stage_by_name["embed"].batch_size == 2
    assert resumed.state.state.by_name["embed"].batch_size == 2
    assert profile_calls == (["embed"] if automatic else [])
    assert resumed_encoder.calls == [("sample-1", 2), ("sample-1", 2), ("sample-1", 2), ("sample-1", 1)]
    assert sha256_file(output / "embeddings" / first_path.name) == first_hash
    _assert_published_images(output, counts)


def test_empty_image_is_checkpointed_before_the_next_image(tmp_path: Path, samples: list[SourceSample]) -> None:
    """An image without detections is saved and skipped on resume too."""
    plan = _plan(tmp_path)
    counts = (0, 5, 1)
    interrupted = _materializer(plan, samples, counts, RecordingEncoder(fail_on_image="sample-1"))

    with pytest.raises(RuntimeError, match="interrupted image embedding"):
        interrupted.run()

    assert interrupted.state.state.by_name["embed"].completed_shards == ("00000",)
    empty_path = plan.staging_root / "embeddings" / "part-00000.parquet"
    assert read_parquet_artifact(empty_path, artifact_name=EMBEDDINGS_ARTIFACT).num_rows == 0
    empty_hash = sha256_file(empty_path)

    resumed_encoder = RecordingEncoder()
    output = _materializer(plan, samples, counts, resumed_encoder).run()

    assert all(sample_id != "sample-0" for sample_id, _ in resumed_encoder.calls)
    assert sha256_file(output / "embeddings" / empty_path.name) == empty_hash
    _assert_published_images(output, counts)


@pytest.mark.parametrize("counts", [(3, 0, 5), (0, 0, 0)])
def test_publication_retains_one_embedding_file_per_image(
    tmp_path: Path, samples: list[SourceSample], counts: tuple[int, ...]
) -> None:
    """Finalization retains image files even when every image has no detections."""
    encoder = RecordingEncoder()
    output = _materializer(_plan(tmp_path, batch_size=2), samples, counts, encoder).run()

    assert all(count <= 2 for _, count in encoder.calls)
    assert sum(count for _, count in encoder.calls) == sum(counts)
    _assert_published_images(output, counts)


@pytest.mark.parametrize("key", ["sample_id", "instance_id"])
def test_resume_recomputes_checkpoint_with_incorrect_image_association(
    tmp_path: Path, samples: list[SourceSample], key: str
) -> None:
    """A matching file hash cannot make incorrectly associated embeddings reusable."""
    plan = _plan(tmp_path)
    counts = (2, 2, 0)
    interrupted = _materializer(plan, samples, counts, RecordingEncoder(fail_on_image="sample-1"))
    with pytest.raises(RuntimeError, match="interrupted image embedding"):
        interrupted.run()

    checkpoint = plan.staging_root / "embeddings" / "part-00000.parquet"
    rows = read_parquet_artifact(checkpoint, artifact_name=EMBEDDINGS_ARTIFACT).to_pylist()
    rows[0][key] = "incorrect-association"
    ParquetShardWriter(plan.staging_root, box_type="aabb").write(
        EMBEDDINGS_ARTIFACT,
        rows,
        shard_index=0,
        embedding_dim=2,
        encoder_fingerprint=ENCODER_FINGERPRINT,
    )
    interrupted.state.begin("embed")
    interrupted.state.record_shard("embed", "00000", hashes={EMBEDDINGS_ARTIFACT: sha256_file(checkpoint)})
    resumed_encoder = RecordingEncoder()

    output = _materializer(plan, samples, counts, resumed_encoder).run()

    assert resumed_encoder.calls == [("sample-0", 2), ("sample-1", 2)]
    _assert_published_images(output, counts)
