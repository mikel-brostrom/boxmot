"""Cached frame subsets retain keyed perception and canonical provenance."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.datasets.config import load_dataset_config
from boxmot.datasets.manifest import sha256_file
from boxmot.datasets.masks import MASK_CODEC, pack_mask
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, INSTANCES_ARTIFACT, MASKS_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter, read_parquet_artifact
from boxmot.engine.dataset_variants.cache import derive_cached_build, reuse_cached_build
from boxmot.engine.eval.evaluator import eval_setup
from boxmot.engine.materialization import (
    BuildPlan,
    DatasetMaterializer,
    DetectStage,
    EmbedStage,
    FinalizeStage,
    FunctionStage,
    PublishOptions,
    SegmentStage,
    StagePlan,
    finalize_build,
    fingerprint,
)
from boxmot.engine.materialization.catalog import catalog_mot_dataset


def _raw_catalog(tmp_path: Path, name: str, indices: tuple[int, ...], *, timed: bool):
    split = "variable" if timed else "train"
    sequence = tmp_path / name / split / "seq"
    image_root = sequence / "img1"
    image_root.mkdir(parents=True)
    for index, original in enumerate(indices, start=1):
        assert cv2.imwrite(str(image_root / f"{index:06d}.png"), np.full((8, 10, 3), original, dtype=np.uint8))
    (sequence / "seqinfo.ini").write_text("[Sequence]\nframeRate=10\n", encoding="utf-8")
    if timed:
        (sequence / "timestamps.csv").write_text(
            "frame_id,timestamp_s\n"
            + "".join(f"{index},{original / 10}\n" for index, original in enumerate(indices, 1)),
            encoding="utf-8",
        )
    gt = sequence / "gt" / "gt.txt"
    gt.parent.mkdir()
    gt.write_text("".join(f"{index},1,1,1,4,5,1,1,1\n" for index in range(1, len(indices) + 1)), encoding="utf-8")
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "id": name,
                "format": {"layout": "mot", "box_type": "aabb"},
                "storage": {"root": name},
                "default_split": split,
                "splits": {split: {"path": split, "has_ground_truth": True}},
                "classes": {"target": {"person": 1}},
            }
        ),
        encoding="utf-8",
    )
    catalog = catalog_mot_dataset(load_dataset_config(config_path), split=split, data_root=tmp_path)
    return catalog, config_path


def _parent_build(tmp_path: Path, catalog, *, optional: bool) -> Path:
    component = {"id": "fixture-detector"}
    detect = StagePlan.create("detect", component=component)
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "parents",
        dataset_name="parent",
        box_type="aabb",
        source_fingerprint=catalog.fingerprint,
        publish=PublishOptions(image_references=False, embeddings=optional, masks=optional),
        stages=(detect, finalize),
        metadata={
            **catalog.metadata,
            "experiment_id": "parent-experiment",
            "components": {"detector": component},
            "component_fingerprints": {"detector": fingerprint(component)},
            "class_bridge": [{"detector_id": 0, "dataset_id": 1}],
        },
    )
    plan.staging_root.mkdir(parents=True)
    writer = ParquetShardWriter(plan.staging_root, box_type="aabb")
    samples, instances, embeddings, masks = [], [], [], []
    encoder = fingerprint("fixture-encoder")
    for sample in catalog.samples:
        samples.append(
            {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "sequence_id": sample.sequence_id,
                "frame_index": sample.frame_index,
                "timestamp_s": sample.timestamp_s,
                "image_ref": None,
                "height": 8,
                "width": 10,
            }
        )
        if sample.frame_index == 2:
            continue
        instance_id = f"{plan.build_id}:{sample.sample_id}:0"
        keys = {"sample_id": sample.sample_id, "instance_id": instance_id}
        instances.append(
            {**keys, "detection_index": 0, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 6.0, "score": 0.9, "class_id": 1}
        )
        embeddings.append(
            {**keys, "dim": 3, "encoder_fingerprint": encoder, "values": [1.0, -0.5, sample.frame_index + 1.0]}
        )
        mask = torch.zeros((8, 10), dtype=torch.bool)
        mask[1:6, 1:5] = True
        masks.append({**keys, "height": 8, "width": 10, "codec": MASK_CODEC, "data": pack_mask(mask)})
    writer.write(SAMPLES_ARTIFACT, samples, shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances, shard_index=0)
    if optional:
        writer.write(EMBEDDINGS_ARTIFACT, embeddings, shard_index=0, embedding_dim=3, encoder_fingerprint=encoder)
        writer.write(MASKS_ARTIFACT, masks, shard_index=0)
    return finalize_build(plan)


def _fixture(tmp_path: Path, *, optional: bool = False):
    parent_catalog, _ = _raw_catalog(tmp_path, "parent", (0, 1, 2, 3), timed=False)
    catalog, config_path = _raw_catalog(tmp_path, "derived", (0, 2, 3), timed=True)
    parent = _parent_build(tmp_path, parent_catalog, optional=optional)
    sample_map = {
        sample.sample_id: parent_catalog.samples[index].sample_id
        for sample, index in zip(catalog.samples, (0, 2, 3), strict=True)
    }
    return parent, parent_catalog, catalog, sample_map, config_path


@pytest.mark.parametrize("optional", [False, True])
def test_frame_subset_preserves_payloads_and_passes_normal_dataset_eval(tmp_path, monkeypatch, optional):
    parent, parent_catalog, catalog, sample_map, config_path = _fixture(tmp_path, optional=optional)
    parent_hash = sha256_file(parent / "manifest.json")
    kwargs = dict(
        parent_catalog=parent_catalog, catalog=catalog, sample_map=sample_map, build_root=tmp_path / "variants"
    )
    derived = derive_cached_build(parent, **kwargs)
    assert derive_cached_build(parent, **kwargs) == derived
    assert sha256_file(parent / "manifest.json") == parent_hash
    original, manifest = DatasetManifest.load(parent), DatasetManifest.load(derived)
    assert manifest.build_id != original.build_id
    assert manifest.metadata["source_catalog_digest"] == catalog.fingerprint
    assert manifest.metadata["timestamps_digest"] == catalog.metadata["timestamps_digest"]
    assert manifest.metadata["component_fingerprints"] == original.metadata["component_fingerprints"]
    assert manifest.metadata["class_bridge"] == original.metadata["class_bridge"]
    assert "experiment_id" not in manifest.metadata
    assert manifest.metadata["derivation"]["parent_experiment_id"] == "parent-experiment"
    assert manifest.metadata["derivation"]["parent_build_id"] == original.build_id
    assert [stage.name for stage in manifest.stages] == ["reuse-cached-frames", "finalize"]
    assert manifest.publish.image_references is True

    for artifact in original.artifacts:
        if artifact.name == SAMPLES_ARTIFACT:
            continue
        source = read_parquet_artifact(parent / artifact.name, artifact_name=artifact.name, box_type="aabb").to_pylist()
        target = read_parquet_artifact(
            derived / artifact.name, artifact_name=artifact.name, box_type="aabb"
        ).to_pylist()
        source = {row["instance_id"]: row for row in source}
        assert len(target) == 2
        for row in target:
            original_id = f"{original.build_id}:{sample_map[row['sample_id']]}:0"
            assert row["instance_id"] == f"{manifest.build_id}:{row['sample_id']}:0"
            assert {k: v for k, v in row.items() if k not in {"sample_id", "instance_id"}} == {
                k: v for k, v in source[original_id].items() if k not in {"sample_id", "instance_id"}
            }
    if optional:
        assert manifest.artifact(EMBEDDINGS_ARTIFACT).metadata == original.artifact(EMBEDDINGS_ARTIFACT).metadata
    frames = list(CachedVisionDataset(derived, load_images=True, load_embeddings=optional, load_masks=optional))
    assert [frame.timestamp_s for frame in frames] == [0.0, 0.2, 0.3]
    assert [len(frame.detections) for frame in frames] == [1, 0, 1]
    assert [int(frame.frame.image[0, 0, 0]) for frame in frames] == [0, 2, 3]

    monkeypatch.setenv("BOXMOT_CACHE_DIR", str(tmp_path / "cache"))
    args = Namespace(dataset=config_path, experiment=None, build=derived, data_root=tmp_path, split="variable")
    eval_setup(args)
    assert args._build_validated is True
    assert args.seq_info == {"seq": 3}


def test_subset_with_no_detections_preserves_empty_optional_schemas(tmp_path):
    parent, parent_catalog, _, _, _ = _fixture(tmp_path, optional=True)
    catalog, _ = _raw_catalog(tmp_path, "empty", (2,), timed=True)
    derived = derive_cached_build(
        parent,
        parent_catalog=parent_catalog,
        catalog=catalog,
        sample_map={catalog.samples[0].sample_id: parent_catalog.samples[2].sample_id},
        build_root=tmp_path / "variants",
    )
    manifest = DatasetManifest.load(derived)
    assert (
        manifest.artifact(EMBEDDINGS_ARTIFACT).metadata
        == DatasetManifest.load(parent).artifact(EMBEDDINGS_ARTIFACT).metadata
    )
    assert all(manifest.artifact(name).rows == 0 for name in (INSTANCES_ARTIFACT, EMBEDDINGS_ARTIFACT, MASKS_ARTIFACT))
    frames = list(CachedVisionDataset(derived, load_embeddings=True, load_masks=True))
    assert len(frames) == 1
    assert frames[0].timestamp_s == 0.2
    assert len(frames[0].detections) == 0


@pytest.mark.parametrize("mismatch", ["image", "dimensions", "timestamp", "taxonomy", "mapping", "parent"])
def test_incompatible_subsets_are_rejected_before_writing(tmp_path, mismatch):
    parent, parent_catalog, catalog, sample_map, _ = _fixture(tmp_path)
    if mismatch == "taxonomy":
        catalog = replace(catalog, metadata={**catalog.metadata, "class_taxonomy_digest": "f" * 64})
    elif mismatch == "mapping":
        sample_map.pop(next(iter(sample_map)))
    elif mismatch == "parent":
        parent_catalog = replace(parent_catalog, fingerprint="f" * 64)
    else:
        changed = {
            "image": {"source_sha256": "f" * 64},
            "dimensions": {"image_size": (7, 10)},
            "timestamp": {"timestamp_s": 0.01},
        }[mismatch]
        catalog = replace(catalog, samples=(replace(catalog.samples[0], **changed), *catalog.samples[1:]))
    destination = tmp_path / "variants"
    with pytest.raises(ValueError):
        derive_cached_build(
            parent, parent_catalog=parent_catalog, catalog=catalog, sample_map=sample_map, build_root=destination
        )
    assert not destination.exists()


def _requested_plan(
    tmp_path: Path,
    catalog,
    *,
    masks: bool = False,
    embeddings: bool = False,
    image_references: bool = True,
    native: bool = False,
) -> BuildPlan:
    """Create a canonical perception plan independent of the available parent."""
    detect = StagePlan.create("detect", component={"id": "fixture-detector"}, batch_size=2)
    stages = [detect]
    if not native:
        for name, requested in (("segment", masks), ("embed", embeddings)):
            if requested:
                stages.append(
                    StagePlan.create(name, depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
                )
    stages.append(
        StagePlan.create(
            "finalize",
            depends_on=tuple(stage.name for stage in stages),
            upstream_fingerprints=tuple(stage.fingerprint for stage in stages),
        )
    )
    return BuildPlan.create(
        build_root=tmp_path / "requested",
        dataset_name="derived",
        box_type="aabb",
        source_fingerprint=catalog.fingerprint,
        publish=PublishOptions(image_references=image_references, masks=masks, embeddings=embeddings),
        stages=tuple(stages),
        metadata={**catalog.metadata, "experiment_id": "requested-experiment"},
    )


@pytest.mark.parametrize("masks,embeddings", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("native", [False, True])
def test_reused_build_keeps_canonical_identity_and_requested_payloads(tmp_path, masks, embeddings, native):
    parent, parent_catalog, catalog, sample_map, _ = _fixture(tmp_path, optional=True)
    plan = _requested_plan(tmp_path, catalog, masks=masks, embeddings=embeddings, native=native)
    parent_hash = sha256_file(parent / "manifest.json")
    result = reuse_cached_build(
        parent, parent_catalog=parent_catalog, catalog=catalog, sample_map=sample_map, plan=plan
    )
    assert result == plan.output_root
    assert sha256_file(parent / "manifest.json") == parent_hash
    manifest = DatasetManifest.load(result)
    assert manifest.build_id == plan.build_id
    assert manifest.metadata["experiment_id"] == "requested-experiment"
    assert manifest.metadata["source_catalog_digest"] == catalog.fingerprint
    assert "derivation" not in manifest.metadata
    assert [(stage.name, stage.fingerprint) for stage in manifest.stages] == [
        (stage.name, stage.fingerprint) for stage in plan.ordered_stages()
    ]
    assert manifest.publish.masks == masks
    assert manifest.publish.embeddings == embeddings
    assert (result / MASKS_ARTIFACT).exists() == masks
    assert (result / EMBEDDINGS_ARTIFACT).exists() == embeddings
    frames = list(CachedVisionDataset(result, load_images=True, load_masks=masks, load_embeddings=embeddings))
    assert [frame.timestamp_s for frame in frames] == [0.0, 0.2, 0.3]
    assert [len(frame.detections) for frame in frames] == [1, 0, 1]
    assert [int(frame.frame.image[0, 0, 0]) for frame in frames] == [0, 2, 3]
    original_id = DatasetManifest.load(parent).build_id
    for artifact in manifest.artifacts:
        if artifact.name == SAMPLES_ARTIFACT:
            continue
        original = {
            row["instance_id"]: row
            for row in read_parquet_artifact(
                parent / artifact.name, artifact_name=artifact.name, box_type="aabb"
            ).to_pylist()
        }
        for row in read_parquet_artifact(
            result / artifact.name, artifact_name=artifact.name, box_type="aabb"
        ).to_pylist():
            assert row["instance_id"] == f"{plan.build_id}:{row['sample_id']}:0"
            source = original[f"{original_id}:{sample_map[row['sample_id']]}:0"]
            assert {key: value for key, value in row.items() if key not in {"sample_id", "instance_id"}} == {
                key: value for key, value in source.items() if key not in {"sample_id", "instance_id"}
            }

    def unexpected_inference(context):
        pytest.fail("A canonical build restored from cached frames must be reused without inference.")

    stages = [FunctionStage(stage.name, unexpected_inference) for stage in plan.stages]
    assert DatasetMaterializer(plan, stages).run() == result


def test_canonical_reuse_can_omit_image_references(tmp_path):
    parent, parent_catalog, catalog, sample_map, _ = _fixture(tmp_path)
    plan = _requested_plan(tmp_path, catalog, image_references=False)
    result = reuse_cached_build(
        parent, parent_catalog=parent_catalog, catalog=catalog, sample_map=sample_map, plan=plan
    )
    rows = read_parquet_artifact(result / SAMPLES_ARTIFACT, artifact_name=SAMPLES_ARTIFACT).to_pylist()
    assert len(rows) == 3
    assert all(row["image_ref"] is None for row in rows)


def test_canonical_reuse_recovers_completed_stages_and_removes_partial_shards(tmp_path, monkeypatch):
    from boxmot.engine.dataset_variants import cache

    parent, parent_catalog, catalog, sample_map, _ = _fixture(tmp_path, optional=True)
    plan = _requested_plan(tmp_path, catalog, masks=True, embeddings=True)
    finalize = cache.FinalizeStage.run

    def fail_finalize(self, context):
        raise RuntimeError("simulated publication interruption")

    monkeypatch.setattr(cache.FinalizeStage, "run", fail_finalize)
    kwargs = dict(parent_catalog=parent_catalog, catalog=catalog, sample_map=sample_map, plan=plan)
    with pytest.raises(RuntimeError, match="simulated publication interruption"):
        reuse_cached_build(parent, **kwargs)
    for name in (SAMPLES_ARTIFACT, INSTANCES_ARTIFACT, EMBEDDINGS_ARTIFACT, MASKS_ARTIFACT):
        (plan.staging_root / name / "part-99999.parquet").write_bytes(b"interrupted inference shard")
    monkeypatch.setattr(cache.FinalizeStage, "run", finalize)
    result = reuse_cached_build(parent, **kwargs)
    assert not list(result.rglob("part-99999.parquet"))
    frames = list(CachedVisionDataset(result, load_masks=True, load_embeddings=True))
    assert [frame.timestamp_s for frame in frames] == [0.0, 0.2, 0.3]
    assert [len(frame.detections) for frame in frames] == [1, 0, 1]


@pytest.mark.parametrize("missing", ["masks", "embeddings"])
def test_canonical_reuse_rejects_missing_requested_payload_before_writing(tmp_path, missing):
    parent, parent_catalog, catalog, sample_map, _ = _fixture(tmp_path)
    plan = _requested_plan(tmp_path, catalog, **{missing: True})
    with pytest.raises(ValueError, match=f"does not contain the requested {missing}"):
        reuse_cached_build(
            parent, parent_catalog=parent_catalog, catalog=catalog, sample_map=sample_map, plan=plan
        )
    assert not plan.build_root.exists()


@pytest.mark.parametrize("native", [False, True])
def test_normal_perception_stages_resume_restored_checkpoints_without_parent(tmp_path, monkeypatch, native):
    """A published parent is unnecessary after copied stages finish checkpointing."""
    parent, parent_catalog, catalog, sample_map, _ = _fixture(tmp_path, optional=True)
    plan = _requested_plan(tmp_path, catalog, masks=True, embeddings=True, native=native)
    finalize = FinalizeStage.run

    def fail_finalize(self, context):
        raise RuntimeError("simulated publication interruption")

    monkeypatch.setattr(FinalizeStage, "run", fail_finalize)
    with pytest.raises(RuntimeError, match="simulated publication interruption"):
        reuse_cached_build(
            parent, parent_catalog=parent_catalog, catalog=catalog, sample_map=sample_map, plan=plan
        )
    parent.rename(parent.with_name("unavailable-parent"))
    monkeypatch.setattr(FinalizeStage, "run", finalize)

    class UnusedPerception:
        embedding_dim = 3

        def predict(self, frames):
            pytest.fail("Restored detection checkpoints must skip detector inference.")

        def segment(self, frames, detections):
            pytest.fail("Restored mask checkpoints must skip segmentor inference.")

        def encode(self, frames, detections):
            pytest.fail("Restored embedding checkpoints must skip encoder inference.")

    runtime = UnusedPerception()
    stages = [DetectStage(runtime, catalog.samples)]
    if not native:
        stages.extend(
            [
                SegmentStage(runtime, catalog.samples),
                EmbedStage(runtime, catalog.samples, encoder_fingerprint=fingerprint("fixture-encoder")),
            ]
        )
    stages.append(FinalizeStage())
    result = DatasetMaterializer(plan, stages).run()
    frames = list(CachedVisionDataset(result, load_masks=True, load_embeddings=True))
    assert [frame.timestamp_s for frame in frames] == [0.0, 0.2, 0.3]
    assert [len(frame.detections) for frame in frames] == [1, 0, 1]
