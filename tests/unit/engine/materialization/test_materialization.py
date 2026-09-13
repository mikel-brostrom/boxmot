from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.readers import attach_masks, read_detection_batches
from boxmot.datasets.schema import (
    INSTANCES_ARTIFACT,
    MASKS_ARTIFACT,
    SAMPLES_ARTIFACT,
)
from boxmot.datasets.storage import ParquetShardWriter, describe_parquet_artifact, read_parquet_artifact
from boxmot.datasets.validation import DatasetValidationError, validate_dataset
from boxmot.detectors import DetectorSpec
from boxmot.engine.materialization import (
    BuildPlan,
    DatasetMaterializer,
    DetectStage,
    EmbedStage,
    ExecutorSpec,
    FinalizeError,
    FinalizeStage,
    InlineExecutor,
    MaterializationContext,
    MaterializationStateStore,
    PublishOptions,
    SegmentStage,
    SourceSample,
    StageOutcome,
    StagePlan,
    StateError,
    create_executor,
    finalize_build,
    fingerprint,
    make_build_id,
)
from boxmot.structures import Boxes, Detections, MaskBatch, OrientedBoxes


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frames):
        self.calls += 1
        return [
            Detections(
                geometry=Boxes(torch.tensor([[1.0, 1.0, 4.0, 5.0]], dtype=torch.float32)),
                scores=torch.tensor([0.9], dtype=torch.float32),
                class_ids=torch.tensor([2], dtype=torch.int64),
                sample_id=frame.sample_id,
            )
            for frame in frames
        ]


class FakeSegmentor:
    def segment(self, frames, detections):
        batches = []
        for frame, batch in zip(frames, detections, strict=True):
            masks = torch.zeros((len(batch), frame.height, frame.width), dtype=torch.bool)
            masks[:, 1:5, 1:4] = True
            batches.append(MaskBatch(masks))
        return batches


class FakeEncoder:
    embedding_dim = 4

    def encode(self, frames, detections):
        del frames
        return [torch.arange(len(batch) * 4, dtype=torch.float32).reshape(len(batch), 4) for batch in detections]


class NativePayloadDetector:
    embedding_dim = 3

    def __init__(self, *, empty: bool = False, obb: bool = False) -> None:
        self.empty = empty
        self.obb = obb

    def predict(self, frames):
        outputs = []
        for frame in frames:
            count = 0 if self.empty else 1
            if self.obb:
                geometry = OrientedBoxes(
                    torch.tensor([[3.0, 3.0, 2.0, 4.0, 0.2]], dtype=torch.float32)[:count].contiguous()
                )
            else:
                geometry = Boxes(torch.tensor([[1.0, 1.0, 4.0, 5.0]], dtype=torch.float32)[:count].contiguous())
            mask_values = torch.zeros((count, frame.height, frame.width), dtype=torch.bool)
            if count:
                mask_values[:, 1:5, 1:4] = True
            outputs.append(
                Detections(
                    geometry=geometry,
                    scores=torch.full((count,), 0.9, dtype=torch.float32),
                    class_ids=torch.full((count,), 2, dtype=torch.int64),
                    sample_id=frame.sample_id,
                    masks=MaskBatch(mask_values),
                    embeddings=torch.arange(count * self.embedding_dim, dtype=torch.float32).reshape(
                        count, self.embedding_dim
                    ),
                )
            )
        return outputs


class ClassMappingDetector:
    embedding_dim = 2

    def predict(self, frames):
        outputs = []
        for frame in frames:
            masks = torch.zeros((3, frame.height, frame.width), dtype=torch.bool)
            masks[0, 0, 0] = True
            masks[1, 0, 1] = True
            masks[2, 0, 2] = True
            outputs.append(
                Detections(
                    geometry=Boxes(
                        torch.tensor(
                            [[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 4.0, 4.0]],
                            dtype=torch.float32,
                        )
                    ),
                    scores=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
                    class_ids=torch.tensor([5, 99, 7], dtype=torch.int64),
                    sample_id=frame.sample_id,
                    masks=MaskBatch(masks),
                    embeddings=torch.tensor([[5.0, 0.0], [99.0, 0.0], [7.0, 0.0]], dtype=torch.float32),
                )
            )
        return outputs


def _make_plan(tmp_path) -> BuildPlan:
    detect = StagePlan.create("detect", component={"backend": "fake"}, batch_size=1)
    segment = StagePlan.create(
        "segment",
        component={"backend": "fake"},
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    embed = StagePlan.create(
        "embed",
        component={"backend": "fake"},
        depends_on=("segment",),
        upstream_fingerprints=(segment.fingerprint,),
    )
    finalize = StagePlan.create(
        "finalize",
        depends_on=("embed",),
        upstream_fingerprints=(embed.fingerprint,),
    )
    return BuildPlan.create(
        build_root=tmp_path / "external-builds",
        dataset_name="unit-test",
        box_type="aabb",
        source_fingerprint=fingerprint({"source": "fixture"}),
        publish=PublishOptions(image_references=False, masks=True, embeddings=True),
        stages=(detect, segment, embed, finalize),
    )


def _detect_finalize_plan(build_root) -> BuildPlan:
    detect = StagePlan.create("detect", batch_size=1)
    finalize = StagePlan.create(
        "finalize",
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    return BuildPlan.create(
        build_root=build_root,
        dataset_name="detect-finalize",
        box_type="aabb",
        source_fingerprint=fingerprint("detect-finalize-source"),
        publish=PublishOptions(image_references=False),
        stages=(detect, finalize),
    )


def _checkpoint_detect_stage(materializer, stage) -> None:
    plan = materializer.plan
    materializer.state.initialize(plan)
    plan.staging_root.mkdir(parents=True, exist_ok=True)
    materializer.state.begin("detect")
    outcome = stage.run(
        MaterializationContext(
            build_plan=plan,
            stage_plan=plan.stage_by_name["detect"],
            staging_root=plan.staging_root,
            state=materializer.state,
            executor=InlineExecutor(),
        )
    )
    materializer.state.complete("detect", artifacts=outcome.artifacts)


def _source_samples(tmp_path: Path) -> list[SourceSample]:
    source_root = tmp_path / "source-frames"
    source_root.mkdir(exist_ok=True)
    samples = []
    for index in reversed(range(2)):
        path = source_root / f"sample-{index}.png"
        image = np.full((8, 10, 3), index, dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        samples.append(
            SourceSample(
                sample_id=f"sample-{index}",
                split="train",
                sequence_id="sequence",
                frame_index=index,
                timestamp_s=float(index),
                image_size=(8, 10),
                source_uri=path.as_uri(),
                source_sha256=sha256_file(path),
                image_ref=None,
            )
        )
    return samples


def _publish_detect_build(tmp_path: Path) -> tuple[FakeDetector, DatasetMaterializer, Path]:
    """Publish a minimal build and retain its materializer for reuse checks."""

    plan = _detect_finalize_plan(tmp_path / "builds")
    detector = FakeDetector()
    materializer = DatasetMaterializer(
        plan,
        [DetectStage(detector, _source_samples(tmp_path)), FinalizeStage()],
    )
    return detector, materializer, materializer.run()


def test_build_and_stage_fingerprints_are_deterministic() -> None:
    payload = {"model": "weights.pt", "threshold": 0.3}
    build_id = make_build_id(payload)
    assert build_id == make_build_id({"threshold": 0.3, "model": "weights.pt"})
    assert len(build_id) == 64
    assert set(build_id) <= set("0123456789abcdef")
    assert make_build_id(payload) != make_build_id({**payload, "threshold": 0.4})

    first = StagePlan.create("detect", config=payload, component={"sha256": "a"})
    second = StagePlan.create("detect", config=dict(reversed(list(payload.items()))), component={"sha256": "a"})
    assert first.fingerprint == second.fingerprint


def test_build_id_includes_schema_and_code_version(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.plan as plan_module

    stage = StagePlan.create("detect")

    def create_plan():
        return BuildPlan.create(
            build_root=tmp_path,
            dataset_name="versioned",
            box_type="aabb",
            source_fingerprint=fingerprint("source"),
            publish=PublishOptions(),
            stages=(stage,),
        )

    original = create_plan()
    monkeypatch.setattr(plan_module, "__version__", "24.0.1")
    changed_code = create_plan()
    monkeypatch.setattr(plan_module, "SCHEMA_VERSION", 2)
    changed_schema = create_plan()

    assert len({original.build_id, changed_code.build_id, changed_schema.build_id}) == 3


def test_execution_knobs_do_not_change_content_build_id(tmp_path) -> None:
    first_stage = StagePlan.create("detect", batch_size=2, workers=1, max_attempts=1)
    second_stage = StagePlan.create(
        "detect",
        batch_size=2,
        workers=8,
        executor="thread",
        max_attempts=5,
        retry_backoff_s=2.0,
    )
    changed_batch = StagePlan.create("detect", batch_size=4)

    def plan_for(stage):
        return BuildPlan.create(
            build_root=tmp_path,
            dataset_name="fingerprints",
            box_type="aabb",
            source_fingerprint=fingerprint("source"),
            publish=PublishOptions(),
            stages=(stage,),
        )

    assert first_stage.fingerprint == second_stage.fingerprint
    assert plan_for(first_stage).build_id == plan_for(second_stage).build_id
    assert plan_for(first_stage).build_id != plan_for(changed_batch).build_id
    with pytest.raises(ValueError, match="execution-only"):
        StagePlan.create("detect", config={"executor": "thread"})


def test_manifest_provenance_paths_do_not_change_content_build_id(tmp_path) -> None:
    stage = StagePlan.create("detect", config={"threshold": 0.25})

    def plan_for(checkout: str) -> BuildPlan:
        return BuildPlan.create(
            build_root=tmp_path / checkout / "builds",
            dataset_name="portable",
            box_type="aabb",
            source_fingerprint=fingerprint("same-source-catalog"),
            publish=PublishOptions(),
            stages=(stage,),
            metadata={
                "dataset_id": "mot17",
                "split": "val",
                "source_root_uri": f"file:///{checkout}/datasets/MOT17",
                "source": f"file:///{checkout}/datasets/MOT17/val",
                "experiment_config": f"/{checkout}/configs/experiment.yaml",
                "dataset_config": f"/{checkout}/configs/dataset.yaml",
                "annotation_path": f"/{checkout}/datasets/MOT17/val/gt.txt",
            },
        )

    first = plan_for("checkout-a")
    second = plan_for("checkout-b")

    assert first.build_id == second.build_id
    assert first.metadata["source_root_uri"] != second.metadata["source_root_uri"]
    assert first.metadata["experiment_config"] != second.metadata["experiment_config"]


def test_experiment_identity_produces_distinct_builds_without_reuse_conflicts(tmp_path) -> None:
    stages = (
        StagePlan.create("detect", batch_size=1),
        StagePlan.create("finalize", depends_on=("detect",)),
    )
    source_fingerprint = fingerprint("same-experiment-source")
    build_root = tmp_path / "builds"

    def plan_for(experiment_id: str, config_path: Path) -> BuildPlan:
        return BuildPlan.create(
            build_root=build_root,
            dataset_name="experiment-fixture",
            box_type="aabb",
            source_fingerprint=source_fingerprint,
            publish=PublishOptions(image_references=False),
            stages=stages,
            metadata={
                "experiment_id": experiment_id,
                "experiment_config": str(config_path),
            },
        )

    first = plan_for("experiment-one", tmp_path / "checkout-a" / "experiment.yaml")
    relocated = plan_for("experiment-one", tmp_path / "checkout-b" / "experiment.yaml")
    second = plan_for("experiment-two", tmp_path / "checkout-a" / "experiment.yaml")

    assert first.build_id == relocated.build_id
    assert first.build_id != second.build_id
    assert first.output_root != second.output_root

    samples = _source_samples(tmp_path)
    first_materializer = DatasetMaterializer(
        first,
        [DetectStage(FakeDetector(), samples), FinalizeStage()],
    )
    first_output = first_materializer.run()
    second_output = DatasetMaterializer(
        second,
        [DetectStage(FakeDetector(), samples), FinalizeStage()],
    ).run()

    assert first_output != second_output
    assert DatasetManifest.load(first_output).metadata["experiment_id"] == "experiment-one"
    assert DatasetManifest.load(second_output).metadata["experiment_id"] == "experiment-two"

    first_manifest = DatasetManifest.load(first_output)
    replace(
        first_manifest,
        metadata={**dict(first_manifest.metadata), "experiment_id": "experiment-two"},
    ).write(first_output)
    with pytest.raises(RuntimeError, match="provenance does not match"):
        first_materializer.run()


def test_explicit_semantic_inputs_still_change_content_build_id(tmp_path) -> None:
    source = fingerprint("source-a")

    def plan_for(*, source_fingerprint: str = source, annotation_ref: str = "gt/gt.txt") -> BuildPlan:
        stage = StagePlan.create("detect", config={"annotation_ref": annotation_ref})
        return BuildPlan.create(
            build_root=tmp_path,
            dataset_name="portable",
            box_type="aabb",
            source_fingerprint=source_fingerprint,
            publish=PublishOptions(),
            stages=(stage,),
        )

    original = plan_for()

    assert plan_for(source_fingerprint=fingerprint("source-b")).build_id != original.build_id
    assert plan_for(annotation_ref="annotations/sequence.txt").build_id != original.build_id


def test_plan_snapshots_semantic_mappings() -> None:
    config = {"threshold": 0.5, "nested": {"classes": [1, 2]}}
    stage = StagePlan.create("detect", config=config)
    config["threshold"] = 0.1
    config["nested"]["classes"].append(3)

    assert stage.config["threshold"] == 0.5
    assert stage.config["nested"]["classes"] == (1, 2)
    with pytest.raises(TypeError):
        stage.config["threshold"] = 0.2


def test_build_plan_uses_external_staging_and_publication_layout(tmp_path) -> None:
    plan = _make_plan(tmp_path)
    assert plan.staging_root == tmp_path / "external-builds" / ".staging" / plan.build_id
    assert plan.output_root == tmp_path / "external-builds" / plan.build_id
    assert plan.state_path == plan.staging_root / "_materialization_state.json"


def test_state_recovers_running_stage_and_preserves_completed_shards(tmp_path) -> None:
    plan = _make_plan(tmp_path)
    store = MaterializationStateStore(plan.state_path)
    store.initialize(plan)
    store.begin("detect")
    store.record_shard("detect", "00000")

    resumed = MaterializationStateStore(plan.state_path)
    state = resumed.initialize(plan)

    assert state.by_name["detect"].status == "pending"
    assert state.by_name["detect"].completed_shards == ("00000",)
    with pytest.raises(StateError, match="incomplete dependencies"):
        resumed.begin("segment")
    resumed.begin("detect")
    resumed.complete("detect")
    assert resumed.begin("segment").status == "running"


def test_state_rejects_plan_fingerprint_drift(tmp_path) -> None:
    plan = _make_plan(tmp_path)
    store = MaterializationStateStore(plan.state_path)
    store.initialize(plan)
    changed_detect = replace(plan.stages[0], fingerprint="0" * 64)
    drifted = replace(plan, stages=(changed_detect, *plan.stages[1:]))

    with pytest.raises(StateError, match="fingerprints"):
        MaterializationStateStore(plan.state_path).initialize(drifted)


@pytest.mark.parametrize("corruption", ("duplicate-stage", "boolean-attempts"))
def test_state_rejects_malformed_persisted_records(tmp_path, corruption) -> None:
    plan = _make_plan(tmp_path)
    store = MaterializationStateStore(plan.state_path)
    store.initialize(plan)
    raw = json.loads(plan.state_path.read_text(encoding="utf-8"))
    if corruption == "duplicate-stage":
        raw["stages"].append(dict(raw["stages"][0]))
    else:
        raw["stages"][0]["attempts"] = True
    plan.state_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(StateError):
        MaterializationStateStore(plan.state_path).initialize(plan)


def test_local_executors_preserve_input_order() -> None:
    assert InlineExecutor().map(lambda value: value * 2, [3, 1, 2]) == [6, 2, 4]
    executor = create_executor(ExecutorSpec(kind="thread", max_workers=2))
    try:
        assert executor.map(str, [3, 1, 2]) == ["3", "1", "2"]
    finally:
        executor.close()


def test_materializer_runs_all_components_publishes_atomically_and_reuses(tmp_path) -> None:
    plan = _make_plan(tmp_path)
    samples = _source_samples(tmp_path)
    detector = FakeDetector()
    encoder_fingerprint = fingerprint("fake-encoder")
    materializer = DatasetMaterializer(
        plan,
        [
            DetectStage(detector, samples),
            SegmentStage(FakeSegmentor(), samples),
            EmbedStage(
                FakeEncoder(),
                samples,
                encoder_fingerprint=encoder_fingerprint,
                use_masks=True,
            ),
            FinalizeStage(
                embedding_metadata={"encoder_fingerprint": encoder_fingerprint, "dim": 4},
            ),
        ],
    )

    output = materializer.run()

    assert output == plan.output_root
    assert output.is_dir()
    assert not plan.staging_root.exists()
    assert (output / "_materialization_state.json").is_file()
    assert not (plan.build_root / ".state").exists()
    assert validate_dataset(output).samples == 2
    dataset = CachedVisionDataset(output, load_masks=True, load_embeddings=True)
    assert len(dataset) == 2
    assert dataset[0].detections.masks is not None
    assert dataset[0].detections.masks.values.sum().item() == 12
    assert dataset[0].detections.embeddings is not None
    assert dataset[0].detections.embeddings.shape == (1, 4)
    assert detector.calls == 2

    # Batch-sized stage shards are globally sorted and compacted into stable
    # publication shards, independent of reverse source ordering.
    assert len(tuple((output / "samples").glob("part-*.parquet"))) == 1
    assert len(tuple((output / "instances").glob("part-*.parquet"))) == 1
    sample_rows = read_parquet_artifact(output / "samples", artifact_name=SAMPLES_ARTIFACT).to_pylist()
    instance_rows = read_parquet_artifact(
        output / "instances",
        artifact_name=INSTANCES_ARTIFACT,
        box_type="aabb",
    ).to_pylist()
    assert [row["sample_id"] for row in sample_rows] == ["sample-0", "sample-1"]
    assert [row["sample_id"] for row in instance_rows] == ["sample-0", "sample-1"]

    # Published reuse validates provenance and content without executing models.
    assert materializer.run() == output
    assert detector.calls == 2


def test_published_reuse_hashes_each_shard_once_without_decoding_payloads(tmp_path, monkeypatch) -> None:
    import boxmot.datasets.validation as validation_module

    detector, materializer, output = _publish_detect_build(tmp_path)
    manifest = DatasetManifest.load(output)
    expected_shards = {(output / shard.path).resolve() for artifact in manifest.artifacts for shard in artifact.shards}
    hash_calls: list[Path] = []
    real_sha256_file = validation_module.sha256_file

    def recording_sha256_file(path):
        hash_calls.append(Path(path).resolve())
        return real_sha256_file(path)

    def reject_payload_decode(*_args, **_kwargs):
        raise AssertionError("published-build reuse must not decode Parquet payloads")

    monkeypatch.setattr(validation_module, "sha256_file", recording_sha256_file)
    monkeypatch.setattr(validation_module, "read_parquet_artifact", reject_payload_decode)

    assert materializer.run() == output
    assert set(hash_calls) == expected_shards
    assert len(hash_calls) == len(expected_shards)
    assert detector.calls == 2


def test_published_reuse_rejects_same_size_shard_corruption(tmp_path) -> None:
    detector, materializer, output = _publish_detect_build(tmp_path)
    manifest = DatasetManifest.load(output)
    shard_path = output / manifest.artifact(INSTANCES_ARTIFACT).shards[0].path
    payload = bytearray(shard_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    shard_path.write_bytes(payload)

    with pytest.raises(DatasetValidationError, match=r"shard .* failed its SHA-256 check"):
        materializer.run()

    assert detector.calls == 2


def test_published_reuse_requires_matching_success_marker(tmp_path) -> None:
    _, materializer, output = _publish_detect_build(tmp_path)
    (output / "_SUCCESS").write_text(
        json.dumps({"schema": "boxmot.dataset/v1", "build_id": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(DatasetValidationError, match="marker does not match"):
        materializer.run()


def test_published_reuse_derives_artifact_hash_from_verified_shards(tmp_path) -> None:
    _, materializer, output = _publish_detect_build(tmp_path)
    manifest = DatasetManifest.load(output)
    instances = manifest.artifact(INSTANCES_ARTIFACT)
    wrong_digest = "0" * 64 if instances.sha256 != "0" * 64 else "1" * 64
    changed_instances = replace(instances, sha256=wrong_digest)
    changed_manifest = replace(
        manifest,
        artifacts=tuple(
            changed_instances if artifact.name == INSTANCES_ARTIFACT else artifact for artifact in manifest.artifacts
        ),
    )
    changed_manifest.write(output)

    with pytest.raises(DatasetValidationError, match=r"artifact 'instances' failed its SHA-256 check"):
        materializer.run()


def test_published_reuse_rejects_non_zstd_shard_from_footer(tmp_path) -> None:
    import pyarrow.parquet as pq

    _, materializer, output = _publish_detect_build(tmp_path)
    manifest = DatasetManifest.load(output)
    shard_path = output / manifest.artifact(SAMPLES_ARTIFACT).shards[0].path
    table = read_parquet_artifact(output / SAMPLES_ARTIFACT, artifact_name=SAMPLES_ARTIFACT)
    pq.write_table(table, shard_path, compression="snappy")
    changed_samples = describe_parquet_artifact(
        output,
        name=SAMPLES_ARTIFACT,
        relative_path=SAMPLES_ARTIFACT,
    )
    changed_manifest = replace(
        manifest,
        artifacts=tuple(
            changed_samples if artifact.name == SAMPLES_ARTIFACT else artifact for artifact in manifest.artifacts
        ),
    )
    changed_manifest.write(output)

    with pytest.raises(DatasetValidationError, match="zstd compression"):
        materializer.run()


def test_published_reuse_rejects_schema_drift_from_footer(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _, materializer, output = _publish_detect_build(tmp_path)
    manifest = DatasetManifest.load(output)
    shard_path = output / manifest.artifact(SAMPLES_ARTIFACT).shards[0].path
    table = pq.read_table(shard_path)
    height_index = table.schema.get_field_index("height")
    table = table.set_column(height_index, "height", table.column(height_index).cast(pa.int64()))
    pq.write_table(table, shard_path, compression="zstd")
    changed_samples = describe_parquet_artifact(
        output,
        name=SAMPLES_ARTIFACT,
        relative_path=SAMPLES_ARTIFACT,
    )
    changed_manifest = replace(
        manifest,
        artifacts=tuple(
            changed_samples if artifact.name == SAMPLES_ARTIFACT else artifact for artifact in manifest.artifacts
        ),
    )
    changed_manifest.write(output)

    with pytest.raises(DatasetValidationError, match="Parquet schema mismatch"):
        materializer.run()


def test_published_reuse_rejects_manifest_footer_row_count_drift(tmp_path) -> None:
    _, materializer, output = _publish_detect_build(tmp_path)
    manifest = DatasetManifest.load(output)
    instances = manifest.artifact(INSTANCES_ARTIFACT)
    first_shard, *remaining_shards = instances.shards
    changed_shard = replace(first_shard, rows=first_shard.rows + 1)
    changed_instances = replace(
        instances,
        rows=instances.rows + 1,
        shards=(changed_shard, *remaining_shards),
    )
    changed_manifest = replace(
        manifest,
        artifacts=tuple(
            changed_instances if artifact.name == INSTANCES_ARTIFACT else artifact for artifact in manifest.artifacts
        ),
        counts={},
    )
    changed_manifest.write(output)

    with pytest.raises(DatasetValidationError, match=r"shard .* row count differs"):
        materializer.run()


def test_published_reuse_rejects_unpublished_optional_artifact(tmp_path) -> None:
    _, materializer, output = _publish_detect_build(tmp_path)
    (output / MASKS_ARTIFACT).mkdir()

    with pytest.raises(DatasetValidationError, match="unpublished 'masks'"):
        materializer.run()


def test_published_reuse_rejects_plan_provenance_drift(tmp_path) -> None:
    detector, materializer, output = _publish_detect_build(tmp_path)
    manifest = DatasetManifest.load(output)
    changed_manifest = replace(
        manifest,
        metadata={
            **dict(manifest.metadata),
            "source_fingerprint": fingerprint("different-source-catalog"),
        },
    )
    changed_manifest.write(output)

    with pytest.raises(RuntimeError, match="provenance does not match"):
        materializer.run()

    assert detector.calls == 2


def test_embed_stage_bounds_model_rows_and_resume_preserves_keyed_output(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.source as source_module

    source_root = tmp_path / "bounded-embed-frames"
    source_root.mkdir()
    samples = []
    for frame_index in range(5):
        path = source_root / f"sample-{frame_index}.png"
        assert cv2.imwrite(str(path), np.full((8, 10, 3), frame_index, dtype=np.uint8))
        samples.append(
            SourceSample(
                sample_id=f"sample-{frame_index}",
                split="train",
                sequence_id="sequence",
                frame_index=frame_index,
                timestamp_s=float(frame_index),
                image_size=(8, 10),
                source_uri=path.as_uri(),
                source_sha256=sha256_file(path),
            )
        )

    class ThreeRowDetector:
        def predict(self, frames):
            geometry = torch.tensor(
                [[0.0, 0.0, 1.0, 2.0], [1.0, 0.0, 2.0, 2.0], [2.0, 0.0, 3.0, 2.0]],
                dtype=torch.float32,
            )
            return [
                Detections(
                    geometry=Boxes(geometry),
                    scores=torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
                    class_ids=torch.zeros(3, dtype=torch.int64),
                    sample_id=frame.sample_id,
                )
                for frame in frames
            ]

    class RecordingEncoder:
        embedding_dim = 2

        def __init__(self, *, fail_on_call: int | None = None) -> None:
            self.fail_on_call = fail_on_call
            self.calls: list[tuple[str, ...]] = []

        def encode(self, frames, detections):
            row_count = sum(len(batch) for batch in detections)
            assert row_count <= 2
            instance_ids = tuple(instance_id for batch in detections for instance_id in (batch.instance_ids or ()))
            self.calls.append(instance_ids)
            if len(self.calls) == self.fail_on_call:
                raise RuntimeError("interrupted embedding shard")
            return [
                torch.tensor(
                    [
                        [float(instance_id.rsplit(":", 1)[1]), float(frame.frame_index)]
                        for instance_id in batch.instance_ids or ()
                    ],
                    dtype=torch.float32,
                ).reshape(-1, self.embedding_dim)
                for frame, batch in zip(frames, detections, strict=True)
            ]

    detect = StagePlan.create("detect", batch_size=2)
    embed = StagePlan.create(
        "embed",
        batch_size=2,
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    finalize = StagePlan.create(
        "finalize",
        depends_on=("embed",),
        upstream_fingerprints=(embed.fingerprint,),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "bounded-embed-builds",
        dataset_name="bounded-embed",
        box_type="aabb",
        source_fingerprint=fingerprint("bounded-embed-source"),
        publish=PublishOptions(image_references=False, embeddings=True),
        stages=(detect, embed, finalize),
    )
    encoder_fingerprint = fingerprint("bounded-encoder")
    interrupted_encoder = RecordingEncoder(fail_on_call=4)
    interrupted = DatasetMaterializer(
        plan,
        [
            DetectStage(ThreeRowDetector(), samples),
            EmbedStage(
                interrupted_encoder,
                samples,
                encoder_fingerprint=encoder_fingerprint,
            ),
            FinalizeStage(
                embedding_metadata={"encoder_fingerprint": encoder_fingerprint, "dim": 2},
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="interrupted embedding shard"):
        interrupted.run()

    assert interrupted.state.state.by_name["embed"].completed_shards == ("00000",)
    completed_path = plan.staging_root / "embeddings" / "part-00000.parquet"
    completed_hash = sha256_file(completed_path)

    hashed_sources = []
    real_source_sha256 = source_module.sha256_file

    def recording_source_sha256(path):
        hashed_sources.append(Path(path))
        return real_source_sha256(path)

    monkeypatch.setattr(source_module, "sha256_file", recording_source_sha256)
    resumed_encoder = RecordingEncoder()

    class ResumeCheckingFinalize(FinalizeStage):
        def run(self, context):
            assert sha256_file(completed_path) == completed_hash
            return super().run(context)

    output = DatasetMaterializer(
        plan,
        [
            DetectStage(ThreeRowDetector(), samples),
            EmbedStage(
                resumed_encoder,
                samples,
                encoder_fingerprint=encoder_fingerprint,
            ),
            ResumeCheckingFinalize(
                embedding_metadata={"encoder_fingerprint": encoder_fingerprint, "dim": 2},
            ),
        ],
    ).run()

    resumed_ids = {instance_id for call in resumed_encoder.calls for instance_id in call}
    assert all(":sample-0:" not in instance_id and ":sample-1:" not in instance_id for instance_id in resumed_ids)
    assert hashed_sources == [source_root / f"sample-{index}.png" for index in range(2, 5)]
    dataset = CachedVisionDataset(output, load_embeddings=True)
    assert len(dataset) == 5
    for sample in dataset:
        assert sample.detections.embeddings is not None
        torch.testing.assert_close(
            sample.detections.embeddings,
            torch.tensor(
                [[0.0, float(sample.frame_index)], [1.0, float(sample.frame_index)], [2.0, float(sample.frame_index)]],
                dtype=torch.float32,
            ),
        )


@pytest.mark.parametrize(
    ("error", "expected_error", "expected_attempts"),
    [
        (RuntimeError("boom"), "boom", 2),
        (RuntimeError(), "RuntimeError", 2),
        (KeyboardInterrupt(), "KeyboardInterrupt", 1),
        (SystemExit(), "SystemExit", 1),
    ],
)
def test_materializer_persists_failure_for_resume(tmp_path, error, expected_error, expected_attempts) -> None:
    plan = BuildPlan.create(
        build_root=tmp_path,
        dataset_name="failure",
        box_type="aabb",
        source_fingerprint=fingerprint("source"),
        publish=PublishOptions(),
        stages=(StagePlan.create("explode", max_attempts=2),),
    )

    class Explode:
        name = "explode"

        def __init__(self) -> None:
            self.released = False

        def run(self, context: MaterializationContext) -> StageOutcome:
            context.state.record_shard(self.name, "00000")
            raise error

        def release(self) -> None:
            self.released = True

    stage = Explode()
    materializer = DatasetMaterializer(plan, [stage])
    with pytest.raises(type(error)) as raised:
        materializer.run()
    assert raised.value is error
    assert stage.released

    store = MaterializationStateStore(plan.state_path)
    state = store.initialize(plan).by_name["explode"]
    assert state.status == "failed"
    assert state.error == expected_error
    assert state.attempts == expected_attempts
    assert state.completed_shards == ("00000",)

    resumed = store.begin("explode")
    assert resumed.status == "running"
    assert resumed.error is None
    assert resumed.completed_shards == ("00000",)


def test_source_mutation_after_plan_creation_prevents_publication(tmp_path) -> None:
    plan = _detect_finalize_plan(tmp_path / "builds")
    samples = _source_samples(tmp_path)
    mutated_path = tmp_path / "source-frames" / "sample-1.png"
    assert cv2.imwrite(str(mutated_path), np.full((8, 10, 3), 255, dtype=np.uint8))
    materializer = DatasetMaterializer(
        plan,
        [DetectStage(FakeDetector(), samples), FinalizeStage()],
    )

    with pytest.raises(ValueError, match="changed after cataloging"):
        materializer.run()

    assert not plan.output_root.exists()


def test_resume_decodes_only_pending_detection_shards(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.source as source_module

    detect = StagePlan.create("detect", batch_size=1)
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="pending-decode",
        box_type="aabb",
        source_fingerprint=fingerprint("pending-decode-source"),
        publish=PublishOptions(image_references=False),
        stages=(detect,),
    )
    samples = _source_samples(tmp_path)

    class InterruptedDetector(FakeDetector):
        def predict(self, frames):
            if self.calls == 1:
                raise RuntimeError("interrupted")
            return super().predict(frames)

    with pytest.raises(RuntimeError, match="interrupted"):
        DatasetMaterializer(plan, [DetectStage(InterruptedDetector(), samples)]).run()

    decoded_ids = []
    hashed_sources = []
    real_decode = source_module._decode_source_sample
    real_source_sha256 = source_module.sha256_file

    def recording_decode(sample):
        decoded_ids.append(sample.sample_id)
        return real_decode(sample)

    def recording_source_sha256(path):
        hashed_sources.append(Path(path))
        return real_source_sha256(path)

    monkeypatch.setattr(source_module, "_decode_source_sample", recording_decode)
    monkeypatch.setattr(source_module, "sha256_file", recording_source_sha256)
    DatasetMaterializer(plan, [DetectStage(FakeDetector(), samples)]).run()

    assert decoded_ids == ["sample-0"]
    assert hashed_sources == [tmp_path / "source-frames" / "sample-0.png"]


def test_completed_perception_stages_do_not_rehash_sources_on_resume(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.source as source_module

    samples = _source_samples(tmp_path)
    detect = StagePlan.create("detect", batch_size=1)
    segment = StagePlan.create(
        "segment",
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
        batch_size=1,
    )
    embed = StagePlan.create(
        "embed",
        depends_on=("detect", "segment"),
        upstream_fingerprints=(detect.fingerprint, segment.fingerprint),
        batch_size=1,
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="completed-source-verification",
        box_type="aabb",
        source_fingerprint=fingerprint("completed-source-verification"),
        publish=PublishOptions(image_references=False, masks=True, embeddings=True),
        stages=(detect, segment, embed),
    )

    def stages():
        return [
            DetectStage(FakeDetector(), samples),
            SegmentStage(FakeSegmentor(), samples),
            EmbedStage(
                FakeEncoder(),
                samples,
                encoder_fingerprint=fingerprint("completed-source-encoder"),
            ),
        ]

    assert DatasetMaterializer(plan, stages()).run() == plan.staging_root

    hashed_sources = []

    def recording_source_sha256(path):
        hashed_sources.append(Path(path))
        return sha256_file(path)

    monkeypatch.setattr(source_module, "sha256_file", recording_source_sha256)

    assert DatasetMaterializer(plan, stages()).run() == plan.staging_root
    assert hashed_sources == []


def test_resume_skips_completed_detector_spec_but_constructs_pending_encoder_spec(
    tmp_path,
    monkeypatch,
) -> None:
    import boxmot.engine.materialization.stages.detect as detect_module
    import boxmot.engine.materialization.stages.embed as embed_module
    from boxmot.reid import ReIDEncoderSpec

    samples = _source_samples(tmp_path)
    detect = StagePlan.create("detect", batch_size=2)
    embed = StagePlan.create(
        "embed",
        batch_size=2,
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="lazy-resume-models",
        box_type="aabb",
        source_fingerprint=fingerprint("lazy-resume-models-source"),
        publish=PublishOptions(image_references=False, embeddings=True),
        stages=(detect, embed),
    )
    encoder_fingerprint = fingerprint("lazy-resume-models-encoder")

    class InterruptedEncoder(FakeEncoder):
        def encode(self, _frames, _detections):
            raise RuntimeError("leave embedding pending")

    interrupted = DatasetMaterializer(
        plan,
        [
            DetectStage(FakeDetector(), samples),
            EmbedStage(
                InterruptedEncoder(),
                samples,
                encoder_fingerprint=encoder_fingerprint,
            ),
        ],
    )
    with pytest.raises(RuntimeError, match="leave embedding pending"):
        interrupted.run()

    initial_state = interrupted.state.state
    assert initial_state.by_name["detect"].status == "completed"
    assert initial_state.by_name["embed"].status == "failed"

    factory_calls = {"detect": 0, "embed": 0}
    monkeypatch.setattr(detect_module, "_WORKER_DETECTORS", {})
    monkeypatch.setattr(embed_module, "_WORKER_ENCODERS", {})

    def create_detector(_spec):
        factory_calls["detect"] += 1
        return FakeDetector()

    def create_encoder(_spec):
        factory_calls["embed"] += 1
        return FakeEncoder()

    monkeypatch.setattr(detect_module, "create_detector", create_detector)
    monkeypatch.setattr(embed_module, "create_reid_encoder", create_encoder)

    result = DatasetMaterializer(
        plan,
        [
            DetectStage(DetectorSpec("fixture", geometry_mode="aabb"), samples),
            EmbedStage(
                ReIDEncoderSpec("fixture"),
                samples,
                encoder_fingerprint=encoder_fingerprint,
            ),
        ],
    ).run()

    assert result == plan.staging_root
    assert factory_calls == {"detect": 0, "embed": 1}


def test_materializer_retries_a_failed_stage(tmp_path) -> None:
    plan = BuildPlan.create(
        build_root=tmp_path,
        dataset_name="retry",
        box_type="aabb",
        source_fingerprint=fingerprint("source"),
        publish=PublishOptions(),
        stages=(StagePlan.create("flaky", max_attempts=2),),
    )

    class Flaky:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, context):
            del context
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            return StageOutcome(metrics={"recovered": True})

    stage = Flaky()
    materializer = DatasetMaterializer(plan, [stage])

    assert materializer.run() == plan.staging_root
    assert stage.calls == 2
    state = materializer.state.state.by_name["flaky"]
    assert state.status == "completed"
    assert state.attempts == 2


def test_executor_factory_failure_is_included_in_retry_policy(tmp_path) -> None:
    plan = BuildPlan.create(
        build_root=tmp_path,
        dataset_name="factory-retry",
        box_type="aabb",
        source_fingerprint=fingerprint("source"),
        publish=PublishOptions(),
        stages=(StagePlan.create("noop", max_attempts=2),),
    )
    factory_calls = 0

    class Noop:
        name = "noop"

        def run(self, context):
            del context
            return StageOutcome()

    def factory(_stage):
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            raise RuntimeError("executor unavailable")
        return InlineExecutor()

    materializer = DatasetMaterializer(plan, [Noop()], executor_factory=factory)

    assert materializer.run() == plan.staging_root
    assert factory_calls == 2
    assert materializer.state.state.by_name["noop"].attempts == 2


def test_model_inference_is_dispatched_through_stage_executor(tmp_path) -> None:
    plan = _make_plan(tmp_path)
    samples = _source_samples(tmp_path)
    calls: list[str] = []

    class RecordingExecutor(InlineExecutor):
        def map(self, function, items):
            calls.append(function.__name__)
            return super().map(function, items)

    encoder_fingerprint = fingerprint("fake-encoder")
    materializer = DatasetMaterializer(
        plan,
        [
            DetectStage(FakeDetector(), samples),
            SegmentStage(FakeSegmentor(), samples),
            EmbedStage(
                FakeEncoder(),
                samples,
                encoder_fingerprint=encoder_fingerprint,
                use_masks=True,
            ),
            FinalizeStage(
                embedding_metadata={"encoder_fingerprint": encoder_fingerprint, "dim": 4},
            ),
        ],
        executor_factory=lambda _stage: RecordingExecutor(),
    )

    materializer.run()

    assert {"_predict", "_segment", "_encode"} <= set(calls)


def test_inline_spec_runtimes_survive_failed_attempt_and_release_after_success(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.stages.detect as detect_module
    import boxmot.engine.materialization.stages.embed as embed_module
    import boxmot.engine.materialization.stages.segment as segment_module
    from boxmot.reid import ReIDEncoderSpec
    from boxmot.segmentors import SegmentorSpec

    detector_cache = {}
    segmentor_cache = {}
    encoder_cache = {}
    monkeypatch.setattr(detect_module, "_WORKER_DETECTORS", detector_cache)
    monkeypatch.setattr(segment_module, "_WORKER_SEGMENTORS", segmentor_cache)
    monkeypatch.setattr(embed_module, "_WORKER_ENCODERS", encoder_cache)

    class FlakyDetector(FakeDetector):
        def predict(self, frames):
            if self.calls == 0:
                self.calls += 1
                raise RuntimeError("transient detector failure")
            return super().predict(frames)

    class FlakySegmentor(FakeSegmentor):
        def __init__(self) -> None:
            self.calls = 0

        def segment(self, frames, detections):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient segmentor failure")
            return super().segment(frames, detections)

    class FlakyEncoder(FakeEncoder):
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, frames, detections):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient encoder failure")
            return super().encode(frames, detections)

    detector_spec = DetectorSpec("fixture", geometry_mode="aabb")
    segmentor_spec = SegmentorSpec("fixture", geometry_mode="aabb")
    encoder_spec = ReIDEncoderSpec("fixture")
    factory_calls = {"detect": 0, "segment": 0, "embed": 0}

    def create_detector(_spec):
        factory_calls["detect"] += 1
        return FlakyDetector()

    def create_segmentor(_spec):
        factory_calls["segment"] += 1
        return FlakySegmentor()

    def create_encoder(_spec):
        factory_calls["embed"] += 1
        return FlakyEncoder()

    monkeypatch.setattr(detect_module, "create_detector", create_detector)
    monkeypatch.setattr(segment_module, "create_segmentor", create_segmentor)
    monkeypatch.setattr(embed_module, "create_reid_encoder", create_encoder)

    detect = StagePlan.create("detect", batch_size=2, max_attempts=2)
    segment = StagePlan.create(
        "segment",
        batch_size=2,
        max_attempts=2,
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    embed = StagePlan.create(
        "embed",
        batch_size=2,
        max_attempts=2,
        depends_on=("detect", "segment"),
        upstream_fingerprints=(detect.fingerprint, segment.fingerprint),
    )
    finalize = StagePlan.create(
        "finalize",
        depends_on=("detect", "segment", "embed"),
        upstream_fingerprints=(detect.fingerprint, segment.fingerprint, embed.fingerprint),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "inline-spec-cleanup",
        dataset_name="inline-spec-cleanup",
        box_type="aabb",
        source_fingerprint=fingerprint("inline-spec-cleanup-source"),
        publish=PublishOptions(image_references=False, masks=True, embeddings=True),
        stages=(detect, segment, embed, finalize),
    )
    encoder_fingerprint = fingerprint("inline-spec-cleanup-encoder")
    samples = _source_samples(tmp_path)

    materializer = DatasetMaterializer(
        plan,
        [
            DetectStage(detector_spec, samples),
            SegmentStage(segmentor_spec, samples),
            EmbedStage(
                encoder_spec,
                samples,
                encoder_fingerprint=encoder_fingerprint,
                use_masks=True,
            ),
            FinalizeStage(
                embedding_metadata={"encoder_fingerprint": encoder_fingerprint, "dim": 4},
            ),
        ],
    )
    output = materializer.run()

    assert output == plan.output_root
    assert factory_calls == {"detect": 1, "segment": 1, "embed": 1}
    assert {name: stage.attempts for name, stage in materializer.state.state.by_name.items()} == {
        "detect": 2,
        "segment": 2,
        "embed": 2,
        "finalize": 1,
    }
    assert detector_cache == {}
    assert segmentor_cache == {}
    assert encoder_cache == {}


def test_inline_spec_runtime_releases_after_terminal_failure(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.stages.detect as detect_module

    detector_cache = {}
    monkeypatch.setattr(detect_module, "_WORKER_DETECTORS", detector_cache)

    class FailingDetector:
        def predict(self, _frames):
            raise RuntimeError("terminal detector failure")

    monkeypatch.setattr(detect_module, "create_detector", lambda _spec: FailingDetector())
    stage_plan = StagePlan.create("detect", batch_size=2, max_attempts=1)
    plan = BuildPlan.create(
        build_root=tmp_path / "terminal-spec-cleanup",
        dataset_name="terminal-spec-cleanup",
        box_type="aabb",
        source_fingerprint=fingerprint("terminal-spec-cleanup-source"),
        publish=PublishOptions(image_references=False),
        stages=(stage_plan,),
    )
    samples = _source_samples(tmp_path)

    with pytest.raises(RuntimeError, match="terminal detector failure"):
        DatasetMaterializer(
            plan,
            [DetectStage(DetectorSpec("fixture", geometry_mode="aabb"), samples)],
        ).run()

    assert detector_cache == {}


def test_multiworker_detector_uses_only_spec_in_top_level_worker_task(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.stages.detect as detect_module

    stage_plan = StagePlan.create("detect", workers=2, executor="process")
    plan = BuildPlan.create(
        build_root=tmp_path,
        dataset_name="worker-spec",
        box_type="aabb",
        source_fingerprint=fingerprint("worker-spec-source"),
        publish=PublishOptions(image_references=False),
        stages=(stage_plan,),
    )
    spec = DetectorSpec(backend="fake")
    factory_calls = 0
    mapped_functions = []
    mapped_items = []

    def create_fake(received_spec):
        nonlocal factory_calls
        assert received_spec is spec
        factory_calls += 1
        return FakeDetector()

    class InspectingExecutor(InlineExecutor):
        def map(self, function, items):
            batch = list(items)
            mapped_functions.append(function)
            mapped_items.extend(batch)
            return super().map(function, batch)

    detect_module._WORKER_DETECTORS.clear()
    monkeypatch.setattr(detect_module, "create_detector", create_fake)
    DatasetMaterializer(
        plan,
        [DetectStage(spec, _source_samples(tmp_path))],
        executor_factory=lambda _stage: InspectingExecutor(),
    ).run()

    assert {function.__name__ for function in mapped_functions} == {"_predict_from_spec"}
    assert all(item[0] is spec for item in mapped_items)
    assert factory_calls == 1


def test_multiworker_detector_rejects_a_runtime_model_handle(tmp_path) -> None:
    stage_plan = StagePlan.create("detect", workers=2, executor="process")
    plan = BuildPlan.create(
        build_root=tmp_path,
        dataset_name="worker-handle",
        box_type="aabb",
        source_fingerprint=fingerprint("worker-handle-source"),
        publish=PublishOptions(image_references=False),
        stages=(stage_plan,),
    )
    materializer = DatasetMaterializer(
        plan,
        [DetectStage(FakeDetector(), _source_samples(tmp_path))],
        executor_factory=lambda _stage: InlineExecutor(),
    )

    with pytest.raises(TypeError, match="serializable DetectorSpec"):
        materializer.run()


def test_completed_finalize_state_with_no_output_resumes_publication(tmp_path) -> None:
    plan = _detect_finalize_plan(tmp_path / "builds")
    detector = FakeDetector()
    detect_stage = DetectStage(detector, _source_samples(tmp_path))
    materializer = DatasetMaterializer(plan, [detect_stage, FinalizeStage()])
    _checkpoint_detect_stage(materializer, detect_stage)
    materializer.state.begin("finalize")
    materializer.state.complete("finalize", artifacts=("manifest.json", "_SUCCESS"))

    output = materializer.run()

    assert output == plan.output_root
    assert validate_dataset(output).samples == 2
    assert detector.calls == 2


def test_finalize_does_not_publish_stale_atomic_temp_files(tmp_path) -> None:
    plan = _detect_finalize_plan(tmp_path / "builds")
    detector = FakeDetector()
    detect_stage = DetectStage(detector, _source_samples(tmp_path))
    materializer = DatasetMaterializer(plan, [detect_stage, FinalizeStage()])
    _checkpoint_detect_stage(materializer, detect_stage)
    (plan.staging_root / ".manifest-orphan").write_text("partial", encoding="utf-8")
    (plan.staging_root / "samples" / ".part-00000.parquet-orphan").write_text(
        "partial",
        encoding="utf-8",
    )

    output = materializer.run()

    assert not (output / ".manifest-orphan").exists()
    assert not (output / "samples" / ".part-00000.parquet-orphan").exists()


def test_corrupt_recorded_shard_is_recomputed_on_resume(tmp_path) -> None:
    plan = _detect_finalize_plan(tmp_path / "builds")
    detector = FakeDetector()
    detect_stage = DetectStage(detector, _source_samples(tmp_path))
    materializer = DatasetMaterializer(plan, [detect_stage, FinalizeStage()])
    _checkpoint_detect_stage(materializer, detect_stage)
    (plan.staging_root / "samples" / "part-00000.parquet").write_bytes(b"corrupt")

    output = materializer.run()

    assert validate_dataset(output).samples == 2
    assert detector.calls == 3


def test_valid_but_changed_recorded_shard_is_recomputed_from_checkpoint_hash(tmp_path) -> None:
    plan = _detect_finalize_plan(tmp_path / "builds")
    detector = FakeDetector()
    detect_stage = DetectStage(detector, _source_samples(tmp_path))
    materializer = DatasetMaterializer(plan, [detect_stage, FinalizeStage()])
    _checkpoint_detect_stage(materializer, detect_stage)
    shard_path = plan.staging_root / "instances" / "part-00000.parquet"
    rows = read_parquet_artifact(
        shard_path,
        artifact_name=INSTANCES_ARTIFACT,
        box_type="aabb",
    ).to_pylist()
    rows[0]["x1"] = 100.0
    rows[0]["x2"] = 104.0
    ParquetShardWriter(plan.staging_root, box_type="aabb").write(
        INSTANCES_ARTIFACT,
        rows,
        shard_index=0,
    )

    output = materializer.run()

    assert detector.calls == 3
    assert CachedVisionDataset(output)[1].detections.geometry.values[0, 0].item() == 1.0


def test_repaired_detection_shard_invalidates_all_downstream_stages(tmp_path) -> None:
    detect_plan = StagePlan.create("detect", batch_size=1)
    segment_plan = StagePlan.create(
        "segment",
        depends_on=("detect",),
        upstream_fingerprints=(detect_plan.fingerprint,),
        batch_size=1,
    )
    embed_plan = StagePlan.create(
        "embed",
        depends_on=("segment",),
        upstream_fingerprints=(segment_plan.fingerprint,),
        batch_size=1,
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="descendant-invalidation",
        box_type="aabb",
        source_fingerprint=fingerprint("source"),
        publish=PublishOptions(masks=True, embeddings=True),
        stages=(detect_plan, segment_plan, embed_plan),
    )

    class CountingSegmentor(FakeSegmentor):
        def __init__(self) -> None:
            self.calls = 0

        def segment(self, frames, detections):
            self.calls += 1
            return super().segment(frames, detections)

    class CountingEncoder(FakeEncoder):
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, frames, detections):
            self.calls += 1
            return super().encode(frames, detections)

    detector = FakeDetector()
    segmentor = CountingSegmentor()
    encoder = CountingEncoder()
    materializer = DatasetMaterializer(
        plan,
        [
            DetectStage(detector, _source_samples(tmp_path)),
            SegmentStage(segmentor, _source_samples(tmp_path)),
            EmbedStage(
                encoder,
                _source_samples(tmp_path),
                encoder_fingerprint=fingerprint("encoder"),
                use_masks=True,
            ),
        ],
    )
    assert materializer.run() == plan.staging_root
    shard_path = plan.staging_root / "instances" / "part-00000.parquet"
    rows = read_parquet_artifact(
        shard_path,
        artifact_name=INSTANCES_ARTIFACT,
        box_type="aabb",
    ).to_pylist()
    rows[0]["score"] = 0.5
    ParquetShardWriter(plan.staging_root, box_type="aabb").write(
        INSTANCES_ARTIFACT,
        rows,
        shard_index=0,
    )

    assert materializer.run() == plan.staging_root
    assert (detector.calls, segmentor.calls, encoder.calls) == (3, 4, 4)


def test_failed_publication_removes_success_marker_from_staging(tmp_path, monkeypatch) -> None:
    import boxmot.engine.materialization.finalize as finalize_module

    plan = _detect_finalize_plan(tmp_path / "builds")
    detector = FakeDetector()
    detect_stage = DetectStage(detector, _source_samples(tmp_path))
    materializer = DatasetMaterializer(plan, [detect_stage, FinalizeStage()])
    _checkpoint_detect_stage(materializer, detect_stage)
    real_replace = finalize_module.os.replace

    def fail_publication(source, destination):
        if Path(source) == plan.staging_root and Path(destination) == plan.output_root:
            raise OSError("injected directory rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(finalize_module.os, "replace", fail_publication)

    with pytest.raises(FinalizeError, match="atomically publish"):
        finalize_build(plan)

    assert not (plan.staging_root / "_SUCCESS").exists()
    with pytest.raises(ValueError, match="staging directories"):
        validate_dataset(plan.staging_root)

    (plan.staging_root / "_SUCCESS").write_text(
        json.dumps({"schema": "boxmot.dataset/v1", "build_id": plan.build_id}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="staging directories"):
        CachedVisionDataset(plan.staging_root)


def test_relative_build_root_materializes_successfully(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    plan = _detect_finalize_plan("relative-builds")
    output = DatasetMaterializer(
        plan,
        [DetectStage(FakeDetector(), _source_samples(tmp_path)), FinalizeStage()],
    ).run()

    assert output == plan.output_root
    assert validate_dataset(output).samples == 2


def test_loader_rejects_unpublished_image_refs_at_construction(tmp_path) -> None:
    plan = _detect_finalize_plan(tmp_path / "builds")
    output = DatasetMaterializer(
        plan,
        [DetectStage(FakeDetector(), _source_samples(tmp_path)), FinalizeStage()],
    ).run()

    with pytest.raises(ValueError, match="does not publish image references"):
        CachedVisionDataset(output, load_images=True)


@pytest.mark.parametrize(("empty", "box_type"), [(False, "aabb"), (True, "obb")])
def test_detector_native_payloads_publish_without_enrichment_stages(tmp_path, empty, box_type) -> None:
    detect = StagePlan.create("detect", component={"backend": "native"})
    finalize = StagePlan.create(
        "finalize",
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "native-builds",
        dataset_name=f"native-{box_type}-{str(empty).lower()}",
        box_type=box_type,
        source_fingerprint=fingerprint({"native": box_type, "empty": empty}),
        publish=PublishOptions(image_references=False, masks=True, embeddings=True),
        stages=(detect, finalize),
    )
    samples = _source_samples(tmp_path)
    output = DatasetMaterializer(
        plan,
        [
            DetectStage(NativePayloadDetector(empty=empty, obb=box_type == "obb"), samples),
            FinalizeStage(),
        ],
    ).run()

    report = validate_dataset(output)
    expected_instances = 0 if empty else len(samples)
    assert report.instances == expected_instances
    assert report.masks == expected_instances
    assert report.embeddings == expected_instances
    dataset = CachedVisionDataset(output, load_masks=True, load_embeddings=True)
    assert dataset[0].detections.masks is not None
    assert dataset[0].detections.embeddings is not None
    assert dataset[0].detections.embeddings.shape == (0 if empty else 1, 3)
    assert dataset[0].detections.geometry.is_obb is (box_type == "obb")
    assert dataset.manifest.artifact("embeddings").metadata["dim"] == 3


def test_detector_class_mapping_filters_and_remaps_before_stable_ids(tmp_path) -> None:
    detect = StagePlan.create("detect", config={"class_id_map": {"5": 1, "7": 3}})
    finalize = StagePlan.create(
        "finalize",
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "class-map-builds",
        dataset_name="class-map",
        box_type="aabb",
        source_fingerprint=fingerprint("class-map-source"),
        publish=PublishOptions(image_references=False, masks=True, embeddings=True),
        stages=(detect, finalize),
    )
    output = DatasetMaterializer(
        plan,
        [
            DetectStage(ClassMappingDetector(), _source_samples(tmp_path), class_id_map={5: 1, 7: 3}),
            FinalizeStage(),
        ],
    ).run()

    sample = CachedVisionDataset(output, load_masks=True, load_embeddings=True)[0]
    assert sample.detections.class_ids.tolist() == [1, 3]
    assert sample.detections.instance_ids == (
        f"{plan.build_id}:{sample.sample_id}:0",
        f"{plan.build_id}:{sample.sample_id}:1",
    )
    assert sample.detections.masks is not None
    assert sample.detections.masks.values[:, 0].nonzero().tolist() == [[0, 0], [1, 2]]
    assert sample.detections.embeddings is not None
    assert sample.detections.embeddings.tolist() == [[5.0, 0.0], [7.0, 0.0]]


def test_detector_class_mapping_is_immutable_and_validated(tmp_path) -> None:
    mapping = {5: 1}
    stage = DetectStage(ClassMappingDetector(), _source_samples(tmp_path), class_id_map=mapping)
    mapping[5] = 9

    assert dict(stage.class_id_map or {}) == {5: 1}
    with pytest.raises(TypeError):
        stage.class_id_map[5] = 2
    with pytest.raises(ValueError, match="detector IDs"):
        DetectStage(ClassMappingDetector(), _source_samples(tmp_path), class_id_map={True: 1})


def test_native_masks_can_feed_embeddings_without_being_published(tmp_path) -> None:
    detect = StagePlan.create("detect")
    embed = StagePlan.create(
        "embed",
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    finalize = StagePlan.create(
        "finalize",
        depends_on=("embed",),
        upstream_fingerprints=(embed.fingerprint,),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "native-mask-builds",
        dataset_name="native-mask-embedding",
        box_type="aabb",
        source_fingerprint=fingerprint("native-mask-source"),
        publish=PublishOptions(image_references=False, masks=False, embeddings=True),
        stages=(detect, embed, finalize),
    )
    samples = _source_samples(tmp_path)
    encoder_fingerprint = fingerprint("fake-encoder")
    output = DatasetMaterializer(
        plan,
        [
            DetectStage(NativePayloadDetector(), samples),
            EmbedStage(
                FakeEncoder(),
                samples,
                encoder_fingerprint=encoder_fingerprint,
                use_masks=True,
            ),
            FinalizeStage(),
        ],
    ).run()

    assert not (output / "masks").exists()
    assert "masks" not in DatasetManifest.load(output).artifacts_by_name
    assert CachedVisionDataset(output, load_embeddings=True)[0].detections.embeddings is not None


def test_unpublished_intermediate_masks_are_strictly_key_validated(tmp_path) -> None:
    detect = StagePlan.create("detect", batch_size=1)
    embed = StagePlan.create(
        "embed",
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "intermediate-mask-builds",
        dataset_name="intermediate-mask-keys",
        box_type="aabb",
        source_fingerprint=fingerprint("intermediate-mask-keys"),
        publish=PublishOptions(image_references=False, embeddings=True),
        stages=(detect, embed),
    )
    stage = DetectStage(NativePayloadDetector(), _source_samples(tmp_path))
    materializer = DatasetMaterializer(
        plan,
        [
            stage,
            EmbedStage(
                FakeEncoder(),
                _source_samples(tmp_path),
                encoder_fingerprint=fingerprint("unused"),
                use_masks=True,
            ),
        ],
    )
    _checkpoint_detect_stage(materializer, stage)
    mask_table = read_parquet_artifact(plan.staging_root / "masks", artifact_name=MASKS_ARTIFACT)
    first_shard_rows = [row for row in mask_table.to_pylist() if row["sample_id"] == "sample-1"]
    ParquetShardWriter(plan.staging_root, box_type="aabb").write(
        MASKS_ARTIFACT,
        [*first_shard_rows, dict(first_shard_rows[0])],
        shard_index=0,
    )
    detections = read_detection_batches(plan.staging_root, "aabb")

    with pytest.raises(ValueError, match="duplicate sample/instance keys"):
        attach_masks(plan.staging_root, detections)


def test_resume_repairs_corrupt_detector_native_intermediate_mask(tmp_path) -> None:
    detect = StagePlan.create("detect", batch_size=1)
    embed = StagePlan.create(
        "embed",
        depends_on=("detect",),
        upstream_fingerprints=(detect.fingerprint,),
    )
    finalize = StagePlan.create(
        "finalize",
        depends_on=("embed",),
        upstream_fingerprints=(embed.fingerprint,),
    )
    plan = BuildPlan.create(
        build_root=tmp_path / "native-mask-resume-builds",
        dataset_name="native-mask-resume",
        box_type="aabb",
        source_fingerprint=fingerprint("native-mask-resume-source"),
        publish=PublishOptions(image_references=False, masks=False, embeddings=True),
        stages=(detect, embed, finalize),
    )

    class InterruptedNativeDetector(NativePayloadDetector):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def predict(self, frames):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("interrupted")
            return super().predict(frames)

    encoder_fingerprint = fingerprint("resume-encoder")
    interrupted = DatasetMaterializer(
        plan,
        [
            DetectStage(InterruptedNativeDetector(), _source_samples(tmp_path)),
            EmbedStage(
                FakeEncoder(),
                _source_samples(tmp_path),
                encoder_fingerprint=encoder_fingerprint,
                use_masks=True,
            ),
            FinalizeStage(),
        ],
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted.run()
    (plan.staging_root / "masks" / "part-00000.parquet").write_bytes(b"corrupt")

    output = DatasetMaterializer(
        plan,
        [
            DetectStage(NativePayloadDetector(), _source_samples(tmp_path)),
            EmbedStage(
                FakeEncoder(),
                _source_samples(tmp_path),
                encoder_fingerprint=encoder_fingerprint,
                use_masks=True,
            ),
            FinalizeStage(),
        ],
    ).run()

    assert validate_dataset(output).embeddings == 2
