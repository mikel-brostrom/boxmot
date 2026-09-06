from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot import __version__
from boxmot.datasets.manifest import DatasetManifest, sha256_file
from boxmot.datasets.readers import read_detection_batches
from boxmot.datasets.schema import SAMPLES_ARTIFACT
from boxmot.datasets.storage import read_parquet_artifact, write_rekeyed_instance_artifact
from boxmot.datasets.validation import validate_published_build
from boxmot.engine.materialization import (
    BuildPlan,
    DatasetMaterializer,
    DetectionCache,
    DetectStage,
    EmbedStage,
    FinalizeStage,
    InlineExecutor,
    MaterializationContext,
    MaterializationStateStore,
    PublishOptions,
    SourceSample,
    StagePlan,
    fingerprint,
    make_detection_cache_id,
)
from boxmot.engine.materialization import builds as builds_module
from boxmot.engine.materialization import detection_cache as detection_cache_module
from boxmot.structures import Boxes, Detections, MaskBatch


class CountingDetector:
    """Tiny detector whose call count makes cache hits observable."""

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


class CountingEncoder:
    embedding_dim = 2

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0

    def encode(self, frames, detections):
        del frames
        self.calls += 1
        return [torch.full((len(batch), self.embedding_dim), self.value, dtype=torch.float32) for batch in detections]


class NativePayloadDetector(CountingDetector):
    embedding_dim = 2

    def __init__(self, artifact: str) -> None:
        super().__init__()
        self.artifact = artifact

    def predict(self, frames):
        self.calls += 1
        outputs = []
        for frame in frames:
            masks = None
            embeddings = None
            if self.artifact == "masks":
                masks = MaskBatch(torch.ones((1, frame.height, frame.width), dtype=torch.bool))
            else:
                embeddings = torch.ones((1, self.embedding_dim), dtype=torch.float32)
            outputs.append(
                Detections(
                    geometry=Boxes(torch.tensor([[1.0, 1.0, 4.0, 5.0]], dtype=torch.float32)),
                    scores=torch.tensor([0.9], dtype=torch.float32),
                    class_ids=torch.tensor([2], dtype=torch.int64),
                    sample_id=frame.sample_id,
                    masks=masks,
                    embeddings=embeddings,
                )
            )
        return outputs


class UnevenDetector(CountingDetector):
    """Emit zero rows for one sample and several for another."""

    def predict(self, frames):
        self.calls += 1
        outputs = []
        for frame in frames:
            count = 0 if frame.frame_index == 0 else 3
            outputs.append(
                Detections(
                    geometry=Boxes(torch.tensor([[1.0, 1.0, 4.0, 5.0]] * count, dtype=torch.float32).reshape(count, 4)),
                    scores=torch.full((count,), 0.9, dtype=torch.float32),
                    class_ids=torch.full((count,), 2, dtype=torch.int64),
                    sample_id=frame.sample_id,
                )
            )
        return outputs


def _samples(tmp_path: Path) -> tuple[SourceSample, ...]:
    source_root = tmp_path / "frames"
    source_root.mkdir(exist_ok=True)
    samples: list[SourceSample] = []
    for frame_index in range(2):
        path = source_root / f"{frame_index:06d}.png"
        assert cv2.imwrite(str(path), np.full((8, 10, 3), frame_index, dtype=np.uint8))
        samples.append(
            SourceSample(
                sample_id=f"sequence:{frame_index}",
                split="test",
                sequence_id="sequence",
                frame_index=frame_index,
                timestamp_s=float(frame_index),
                image_size=(8, 10),
                source_uri=path.as_uri(),
                source_sha256=sha256_file(path),
                image_ref=path.as_uri(),
            )
        )
    return tuple(samples)


def _derived_plan(
    build_root: Path,
    *,
    experiment_id: str = "cache-experiment-a",
    reid: str = "reid-a",
    dataset_name: str = "cache-fixture",
    source_fingerprint: str | None = None,
    box_type: str = "aabb",
    detector: str = "detector-a",
    class_id_map: dict[str, int] | None = None,
    image_references: bool = True,
) -> tuple[BuildPlan, str]:
    mapping = {"2": 0} if class_id_map is None else class_id_map
    detect = StagePlan.create(
        "detect",
        config={"geometry": box_type, "class_id_map": mapping},
        component={"backend": "fake", "model": detector},
        batch_size=2,
    )
    encoder_fingerprint = fingerprint({"encoder": reid})
    embed = StagePlan.create(
        "embed",
        component={"backend": "fake", "model": reid},
        upstream_fingerprints=(detect.fingerprint,),
        depends_on=("detect",),
        batch_size=2,
    )
    finalize = StagePlan.create(
        "finalize",
        upstream_fingerprints=(detect.fingerprint, embed.fingerprint),
        depends_on=("detect", "embed"),
    )
    plan = BuildPlan.create(
        build_root=build_root,
        dataset_name=dataset_name,
        box_type=box_type,  # type: ignore[arg-type]
        source_fingerprint=source_fingerprint or fingerprint("source-a"),
        publish=PublishOptions(image_references=image_references, embeddings=True),
        stages=(detect, embed, finalize),
        metadata={
            "experiment_id": experiment_id,
            "boxmot_version": __version__,
            "source_count": 2,
        },
    )
    return plan, encoder_fingerprint


def _run_derived_build(
    plan: BuildPlan,
    samples: tuple[SourceSample, ...],
    detector: CountingDetector,
    encoder: CountingEncoder,
    encoder_fingerprint: str,
    *,
    use_cache: bool = True,
) -> Path:
    cache = DetectionCache.from_plan(plan, samples) if use_cache else None
    return DatasetMaterializer(
        plan,
        [
            DetectStage(detector, samples, class_id_map={2: 0}, cache=cache),
            EmbedStage(encoder, samples, encoder_fingerprint=encoder_fingerprint),
            FinalizeStage(),
        ],
    ).run()


def _detect_only_plan(build_root: Path, experiment_id: str, *, batch_size: int = 2) -> BuildPlan:
    detect = StagePlan.create(
        "detect",
        config={"geometry": "aabb", "class_id_map": {"2": 0}},
        component={"backend": "fake", "model": "detector-a"},
        batch_size=batch_size,
    )
    return BuildPlan.create(
        build_root=build_root,
        dataset_name="cache-fixture",
        box_type="aabb",
        source_fingerprint=fingerprint("source-a"),
        publish=PublishOptions(image_references=True),
        stages=(detect,),
        metadata={
            "experiment_id": experiment_id,
            "boxmot_version": __version__,
            "source_count": 2,
        },
    )


def _detect_finalize_plan(
    build_root: Path,
    experiment_id: str = "cache-experiment-a",
    *,
    batch_size: int = 2,
) -> BuildPlan:
    detect = StagePlan.create(
        "detect",
        config={"geometry": "aabb", "class_id_map": {"2": 0}},
        component={"backend": "fake", "model": "detector-a"},
        batch_size=batch_size,
    )
    finalize = StagePlan.create(
        "finalize",
        upstream_fingerprints=(detect.fingerprint,),
        depends_on=("detect",),
    )
    return BuildPlan.create(
        build_root=build_root,
        dataset_name="cache-fixture",
        box_type="aabb",
        source_fingerprint=fingerprint("source-a"),
        publish=PublishOptions(image_references=True),
        stages=(detect, finalize),
        metadata={
            "experiment_id": experiment_id,
            "boxmot_version": __version__,
            "source_count": 2,
        },
    )


def _native_plan(build_root: Path, experiment_id: str, artifact: str) -> BuildPlan:
    detect = StagePlan.create(
        "detect",
        config={"geometry": "aabb", "class_id_map": {"2": 0}},
        component={"backend": "fake", "model": "native-detector"},
        batch_size=2,
    )
    return BuildPlan.create(
        build_root=build_root,
        dataset_name="cache-fixture",
        box_type="aabb",
        source_fingerprint=fingerprint("source-a"),
        publish=PublishOptions(
            image_references=True,
            masks=artifact == "masks",
            embeddings=artifact == "embeddings",
        ),
        stages=(detect,),
        metadata={
            "experiment_id": experiment_id,
            "boxmot_version": __version__,
            "source_count": 2,
        },
    )


def test_detection_cache_identity_excludes_experiment_and_reid() -> None:
    root = Path("runs/materializations")
    base, _ = _derived_plan(root)
    changed_reid, _ = _derived_plan(
        root,
        experiment_id="cache-experiment-b",
        reid="reid-b",
    )

    cache_id = make_detection_cache_id(base, base.stage_by_name["detect"])
    assert base.build_id != changed_reid.build_id
    assert cache_id == make_detection_cache_id(changed_reid, changed_reid.stage_by_name["detect"])


def test_detection_cache_identity_covers_detector_input_contract() -> None:
    root = Path("runs/materializations")
    base, _ = _derived_plan(root)
    cache_id = make_detection_cache_id(base, base.stage_by_name["detect"])
    variants = (
        _derived_plan(root, dataset_name="other-dataset")[0],
        _derived_plan(root, source_fingerprint=fingerprint("source-b"))[0],
        _derived_plan(root, box_type="obb")[0],
        _derived_plan(root, detector="detector-b")[0],
        _derived_plan(root, class_id_map={"2": 1})[0],
    )

    assert all(cache_id != make_detection_cache_id(plan, plan.stage_by_name["detect"]) for plan in variants)


def test_second_reid_build_reuses_detections_and_rekeys_instances(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    first_plan, first_encoder_fingerprint = _derived_plan(build_root)
    second_plan, second_encoder_fingerprint = _derived_plan(
        build_root,
        experiment_id="cache-experiment-b",
        reid="reid-b",
    )
    first_detector = CountingDetector()
    second_detector = CountingDetector()
    first_encoder = CountingEncoder(1.0)
    second_encoder = CountingEncoder(2.0)

    _run_derived_build(
        first_plan,
        samples,
        first_detector,
        first_encoder,
        first_encoder_fingerprint,
    )
    second_output = _run_derived_build(
        second_plan,
        samples,
        second_detector,
        second_encoder,
        second_encoder_fingerprint,
    )

    assert first_detector.calls == 1
    assert second_detector.calls == 0
    assert first_encoder.calls == 1
    assert second_encoder.calls == 1
    restored = read_detection_batches(second_output, "aabb")
    assert {instance_id for detections in restored.values() for instance_id in detections.instance_ids or ()} == {
        f"{second_plan.build_id}:{sample.sample_id}:0" for sample in samples
    }


def test_detection_cache_bootstraps_from_compatible_published_build(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    first_plan, first_encoder_fingerprint = _derived_plan(build_root)
    second_plan, second_encoder_fingerprint = _derived_plan(
        build_root,
        experiment_id="cache-experiment-b",
        reid="reid-b",
    )
    first_detector = CountingDetector()
    second_detector = CountingDetector()

    _run_derived_build(
        first_plan,
        samples,
        first_detector,
        CountingEncoder(1.0),
        first_encoder_fingerprint,
        use_cache=False,
    )
    cache = DetectionCache.from_plan(second_plan, samples)
    assert not cache.root.exists()

    _run_derived_build(
        second_plan,
        samples,
        second_detector,
        CountingEncoder(2.0),
        second_encoder_fingerprint,
    )

    assert first_detector.calls == 1
    assert second_detector.calls == 0
    assert cache.root.is_dir()
    validate_published_build(cache.root, manifest=DatasetManifest.load(cache.root))


def test_detection_cache_bootstraps_from_former_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BOXMOT_BUILDS_DIR", raising=False)
    samples = _samples(tmp_path)
    former_root = tmp_path / "former-platform-cache" / "builds"
    monkeypatch.setattr(detection_cache_module, "former_default_build_root", lambda: former_root)
    first_plan = _detect_finalize_plan(former_root, "cache-experiment-a")
    first_detector = CountingDetector()
    DatasetMaterializer(
        first_plan,
        [DetectStage(first_detector, samples, class_id_map={2: 0}), FinalizeStage()],
    ).run()

    canonical_root = Path("runs") / "materializations"
    second_plan = _detect_finalize_plan(canonical_root, "cache-experiment-b")
    second_detector = CountingDetector()
    cache = DetectionCache.from_plan(second_plan, samples)
    output = DatasetMaterializer(
        second_plan,
        [
            DetectStage(second_detector, samples, class_id_map={2: 0}, cache=cache),
            FinalizeStage(),
        ],
    ).run()

    assert first_detector.calls == 1
    assert second_detector.calls == 0
    assert output == second_plan.output_root
    assert cache.root.parent == second_plan.build_root / ".cache" / "detect"
    validate_published_build(cache.root, manifest=DatasetManifest.load(cache.root))


def test_identical_build_is_imported_from_former_default_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("BOXMOT_BUILDS_DIR", raising=False)
    samples = _samples(tmp_path)
    former_root = tmp_path / "former-platform-cache" / "builds"
    monkeypatch.setattr(builds_module, "former_default_build_root", lambda: former_root)
    former_plan = _detect_finalize_plan(former_root)
    former_output = DatasetMaterializer(
        former_plan,
        [DetectStage(CountingDetector(), samples, class_id_map={2: 0}), FinalizeStage()],
    ).run()
    canonical_plan = _detect_finalize_plan(Path("runs") / "materializations")
    messages: list[str] = []

    imported = builds_module.import_former_default_build(
        canonical_plan,
        status_callback=messages.append,
    )

    assert canonical_plan.build_id == former_plan.build_id
    assert imported is True
    assert canonical_plan.output_root.is_dir()
    assert former_output.is_dir()
    assert any("former build root" in message for message in messages)
    validate_published_build(canonical_plan.output_root)


def test_cache_restore_preserves_detector_checkpoint_shards_for_repair(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    first_plan = _detect_only_plan(build_root, "cache-experiment-a", batch_size=1)
    second_plan = _detect_only_plan(build_root, "cache-experiment-b", batch_size=1)
    cache = DetectionCache.from_plan(first_plan, samples)
    DatasetMaterializer(
        first_plan,
        [DetectStage(CountingDetector(), samples, class_id_map={2: 0}, cache=cache)],
    ).run()
    second_detector = CountingDetector()
    DatasetMaterializer(
        second_plan,
        [
            DetectStage(
                second_detector,
                samples,
                class_id_map={2: 0},
                cache=DetectionCache.from_plan(second_plan, samples),
            )
        ],
    ).run()

    state_store = MaterializationStateStore(second_plan.state_path)
    state_store.initialize(second_plan)
    state = state_store.state.by_name["detect"]
    assert second_detector.calls == 0
    assert state.completed_shards == ("00000", "00001")
    corrupt = second_plan.staging_root / "instances" / "part-00001.parquet"
    payload = bytearray(corrupt.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    corrupt.write_bytes(payload)
    cache.root.rename(tmp_path / "removed-detection-cache")

    repair_detector = CountingDetector()
    DatasetMaterializer(
        second_plan,
        [DetectStage(repair_detector, samples, class_id_map={2: 0}, cache=cache)],
    ).run()

    assert repair_detector.calls == 1
    restored = read_detection_batches(second_plan.staging_root, "aabb")
    assert set(restored) == {sample.sample_id for sample in samples}


def test_streaming_restore_routes_multiple_cache_shards_and_empty_samples(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    source_plan = _detect_only_plan(build_root, "cache-experiment-a")
    DatasetMaterializer(
        source_plan,
        [DetectStage(UnevenDetector(), samples, class_id_map={2: 0})],
    ).run()
    cache = DetectionCache.from_plan(source_plan, samples)
    cache_instances = tmp_path / "compact-cache" / "instances"
    cache_paths = write_rekeyed_instance_artifact(
        source_plan.staging_root / "instances",
        cache_instances,
        box_type="aabb",
        source_build_id=source_plan.build_id,
        target_build_id=cache.cache_id,
        target_rows=1,
    )
    target_plan = _detect_only_plan(build_root, "cache-experiment-b", batch_size=1)
    restored_root = tmp_path / "restored"
    restored_paths = detection_cache_module._write_restored_detection_shards(
        cache_instances,
        restored_root,
        samples,
        box_type="aabb",
        source_build_id=cache.cache_id,
        target_build_id=target_plan.build_id,
        batch_size=1,
        image_references=True,
    )

    assert len(cache_paths) == 3
    assert len(restored_paths) == 2
    restored = read_detection_batches(restored_root, "aabb")
    assert len(restored[samples[0].sample_id]) == 0
    assert len(restored[samples[1].sample_id]) == 3
    assert restored[samples[1].sample_id].instance_ids == tuple(
        f"{target_plan.build_id}:{samples[1].sample_id}:{index}" for index in range(3)
    )


def test_corrupt_detection_cache_is_discarded_and_rebuilt(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    first_plan = _detect_only_plan(build_root, "cache-experiment-a")
    second_plan = _detect_only_plan(build_root, "cache-experiment-b")
    first_cache = DetectionCache.from_plan(first_plan, samples)
    first_detector = CountingDetector()
    DatasetMaterializer(
        first_plan,
        [DetectStage(first_detector, samples, class_id_map={2: 0}, cache=first_cache)],
    ).run()
    cache_manifest = DatasetManifest.load(first_cache.root)
    instance_shard = first_cache.root / cache_manifest.artifact("instances").shards[0].path
    payload = bytearray(instance_shard.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    instance_shard.write_bytes(payload)

    second_detector = CountingDetector()
    second_cache = DetectionCache.from_plan(second_plan, samples)
    DatasetMaterializer(
        second_plan,
        [DetectStage(second_detector, samples, class_id_map={2: 0}, cache=second_cache)],
    ).run()

    assert first_detector.calls == 1
    assert second_detector.calls == 1
    validate_published_build(second_cache.root, manifest=DatasetManifest.load(second_cache.root))


def test_completed_pre_cache_detect_stage_seeds_cache_on_resume(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    plan = _detect_finalize_plan(tmp_path / "runs" / "materializations")
    first_detector = CountingDetector()

    class InterruptedFinalize:
        name = "finalize"

        def run(self, context):
            del context
            raise RuntimeError("interrupted before publication")

    with pytest.raises(RuntimeError, match="interrupted before publication"):
        DatasetMaterializer(
            plan,
            [DetectStage(first_detector, samples, class_id_map={2: 0}), InterruptedFinalize()],
        ).run()
    assert first_detector.calls == 1
    assert not plan.output_root.exists()

    cache = DetectionCache.from_plan(plan, samples)
    resume_detector = CountingDetector()
    output = DatasetMaterializer(
        plan,
        [
            DetectStage(resume_detector, samples, class_id_map={2: 0}, cache=cache),
            FinalizeStage(),
        ],
    ).run()

    assert output == plan.output_root
    assert resume_detector.calls == 0
    assert cache.root.is_dir()
    validate_published_build(cache.root, manifest=DatasetManifest.load(cache.root))


def test_cache_reuse_applies_target_image_reference_policy(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    first_plan, first_encoder_fingerprint = _derived_plan(build_root)
    second_plan, second_encoder_fingerprint = _derived_plan(build_root, image_references=False)
    first_detector = CountingDetector()
    second_detector = CountingDetector()

    _run_derived_build(
        first_plan,
        samples,
        first_detector,
        CountingEncoder(1.0),
        first_encoder_fingerprint,
    )
    second_output = _run_derived_build(
        second_plan,
        samples,
        second_detector,
        CountingEncoder(1.0),
        second_encoder_fingerprint,
    )

    sample_rows = read_parquet_artifact(
        second_output / "samples",
        artifact_name=SAMPLES_ARTIFACT,
    ).to_pylist()
    assert first_detector.calls == 1
    assert second_detector.calls == 0
    assert all(row["image_ref"] is None for row in sample_rows)


@pytest.mark.parametrize("artifact", ("masks", "embeddings"))
def test_required_detector_native_payload_bypasses_core_detection_cache(tmp_path: Path, artifact: str) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    first_plan = _native_plan(build_root, "cache-experiment-a", artifact)
    second_plan = _native_plan(build_root, "cache-experiment-b", artifact)
    cache = DetectionCache.from_plan(first_plan, samples)
    first_detector = NativePayloadDetector(artifact)
    second_detector = NativePayloadDetector(artifact)

    DatasetMaterializer(
        first_plan,
        [DetectStage(first_detector, samples, class_id_map={2: 0}, cache=cache)],
    ).run()
    DatasetMaterializer(
        second_plan,
        [
            DetectStage(
                second_detector,
                samples,
                class_id_map={2: 0},
                cache=DetectionCache.from_plan(second_plan, samples),
            )
        ],
    ).run()

    assert first_detector.calls == 1
    assert second_detector.calls == 1
    assert not cache.root.exists()


def test_custom_mask_requiring_encoder_bypasses_cache_for_unpublished_detector_native_masks(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    seed_plan = _detect_only_plan(build_root, "cache-seed")
    seed_cache = DetectionCache.from_plan(seed_plan, samples)
    seed_detector = CountingDetector()
    DatasetMaterializer(
        seed_plan,
        [DetectStage(seed_detector, samples, class_id_map={2: 0}, cache=seed_cache)],
    ).run()
    first_plan, first_encoder_fingerprint = _derived_plan(build_root)
    second_plan, second_encoder_fingerprint = _derived_plan(
        build_root,
        experiment_id="cache-experiment-b",
        reid="reid-b",
    )
    detectors = (NativePayloadDetector("masks"), NativePayloadDetector("masks"))
    for plan, detector, encoder_fingerprint in zip(
        (first_plan, second_plan),
        detectors,
        (first_encoder_fingerprint, second_encoder_fingerprint),
        strict=True,
    ):
        DatasetMaterializer(
            plan,
            [
                DetectStage(
                    detector,
                    samples,
                    class_id_map={2: 0},
                    cache=DetectionCache.from_plan(plan, samples),
                    requires_native_masks=True,
                ),
                EmbedStage(
                    CountingEncoder(1.0),
                    samples,
                    encoder_fingerprint=encoder_fingerprint,
                    use_masks=True,
                ),
                FinalizeStage(),
            ],
        ).run()

    assert seed_detector.calls == 1
    assert seed_cache.root == DetectionCache.from_plan(first_plan, samples).root
    assert [detector.calls for detector in detectors] == [1, 1]
    validate_published_build(seed_cache.root, manifest=DatasetManifest.load(seed_cache.root))


@pytest.mark.parametrize("operation", ("restore", "publish"))
def test_detection_cache_rejects_a_context_from_another_build_root(tmp_path: Path, operation: str) -> None:
    samples = _samples(tmp_path)
    cache_plan = _detect_only_plan(tmp_path / "first" / "materializations", "cache-experiment-a")
    context_plan = _detect_only_plan(tmp_path / "second" / "materializations", "cache-experiment-a")
    cache = DetectionCache.from_plan(cache_plan, samples)
    assert cache.cache_id == make_detection_cache_id(context_plan, context_plan.stage_by_name["detect"])

    state = MaterializationStateStore(context_plan.state_path)
    state.initialize(context_plan)
    state.begin("detect")
    context = MaterializationContext(
        build_plan=context_plan,
        stage_plan=context_plan.stage_by_name["detect"],
        staging_root=context_plan.staging_root,
        state=state,
        executor=InlineExecutor(),
    )

    with pytest.raises(ValueError, match="does not belong to the active materialization context"):
        getattr(cache, operation)(context)


def test_concurrent_builds_populate_a_shared_detection_cache_once(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    build_root = tmp_path / "runs" / "materializations"
    first_plan = _detect_only_plan(build_root, "cache-experiment-a")
    second_plan = _detect_only_plan(build_root, "cache-experiment-b")
    cache = DetectionCache.from_plan(first_plan, samples)
    second_cache = DetectionCache.from_plan(second_plan, samples)
    assert cache.root == second_cache.root

    inference_started = threading.Event()
    release_inference = threading.Event()

    class BlockingDetector(CountingDetector):
        def predict(self, frames):
            inference_started.set()
            if not release_inference.wait(timeout=5):
                raise TimeoutError("test did not release detector inference")
            return super().predict(frames)

    first_detector = BlockingDetector()
    second_detector = BlockingDetector()
    materializers = (
        DatasetMaterializer(
            first_plan,
            [DetectStage(first_detector, samples, class_id_map={2: 0}, cache=cache)],
        ),
        DatasetMaterializer(
            second_plan,
            [DetectStage(second_detector, samples, class_id_map={2: 0}, cache=second_cache)],
        ),
    )
    start = threading.Barrier(3)

    def run(materializer: DatasetMaterializer) -> Path:
        start.wait(timeout=5)
        return materializer.run()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, materializer) for materializer in materializers]
        start.wait(timeout=5)
        try:
            assert inference_started.wait(timeout=5)
        finally:
            release_inference.set()
        outputs = [future.result(timeout=10) for future in futures]

    assert first_detector.calls + second_detector.calls == 1
    assert set(outputs) == {first_plan.staging_root, second_plan.staging_root}
    validate_published_build(cache.root, manifest=DatasetManifest.load(cache.root))
    for plan, output in zip((first_plan, second_plan), outputs, strict=True):
        detections = read_detection_batches(output, "aabb")
        assert {instance_id for batch in detections.values() for instance_id in batch.instance_ids or ()} == {
            f"{plan.build_id}:{sample.sample_id}:0" for sample in samples
        }


def test_artifact_generation_install_rolls_back_both_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    temporary = tmp_path / "temporary"
    target = tmp_path / "target"
    for root, value in ((temporary, "new"), (target, "old")):
        for name in ("samples", "instances"):
            directory = root / name
            directory.mkdir(parents=True)
            (directory / "marker").write_text(value, encoding="utf-8")
    real_replace = detection_cache_module.os.replace

    def replace(source, destination):
        if Path(source) == temporary / "instances":
            raise OSError("injected second-directory failure")
        return real_replace(source, destination)

    monkeypatch.setattr(detection_cache_module.os, "replace", replace)

    with pytest.raises(OSError, match="injected second-directory failure"):
        detection_cache_module._install_artifact_directories(
            temporary,
            target,
            ("samples", "instances"),
        )

    assert (target / "samples" / "marker").read_text(encoding="utf-8") == "old"
    assert (target / "instances" / "marker").read_text(encoding="utf-8") == "old"


def test_cache_root_symlink_is_unlinked_without_touching_target(tmp_path: Path) -> None:
    samples = _samples(tmp_path)
    plan = _detect_only_plan(tmp_path / "runs" / "materializations", "cache-experiment-a")
    cache = DetectionCache.from_plan(plan, samples)
    external = tmp_path / "external"
    external.mkdir()
    marker = external / "keep"
    marker.write_text("external", encoding="utf-8")
    cache.root.parent.mkdir(parents=True)
    try:
        cache.root.symlink_to(external, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - platform policy
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    assert cache._load_valid_manifest() is None
    assert not detection_cache_module._path_exists(cache.root)
    assert marker.read_text(encoding="utf-8") == "external"


def test_transient_cache_io_error_does_not_delete_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = _samples(tmp_path)
    plan = _detect_only_plan(tmp_path / "runs" / "materializations", "cache-experiment-a")
    cache = DetectionCache.from_plan(plan, samples)
    cache.root.mkdir(parents=True)
    marker = cache.root / "keep"
    marker.write_text("cache", encoding="utf-8")

    def fail_load(_cls, _root):
        raise PermissionError("transient cache read failure")

    monkeypatch.setattr(DatasetManifest, "load", classmethod(fail_load))

    with pytest.raises(PermissionError, match="transient cache read failure"):
        cache._load_valid_manifest()
    assert marker.read_text(encoding="utf-8") == "cache"
