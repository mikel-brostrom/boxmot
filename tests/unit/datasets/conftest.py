from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch

from boxmot.datasets.masks import MASK_CODEC, pack_mask
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, INSTANCES_ARTIFACT, MASKS_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint


def _plan(tmp_path, *, masks: bool, embeddings: bool) -> BuildPlan:
    detect = StagePlan.create("detect", component={"id": "fake-detector"})
    stages = [detect]
    dependency = detect
    if masks:
        segment = StagePlan.create(
            "segment",
            component={"id": "fake-segmentor"},
            depends_on=(dependency.name,),
            upstream_fingerprints=(dependency.fingerprint,),
        )
        stages.append(segment)
        dependency = segment
    if embeddings:
        embed = StagePlan.create(
            "embed",
            component={"id": "fake-encoder"},
            depends_on=(dependency.name,),
            upstream_fingerprints=(dependency.fingerprint,),
        )
        stages.append(embed)
        dependency = embed
    finalize = StagePlan.create(
        "finalize",
        depends_on=(dependency.name,),
        upstream_fingerprints=(dependency.fingerprint,),
    )
    stages.append(finalize)
    return BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="fixture",
        box_type="aabb",
        source_fingerprint=fingerprint({"fixture": 1}),
        publish=PublishOptions(image_references=True, masks=masks, embeddings=embeddings),
        stages=tuple(stages),
    )


def _write_required_tables(plan: BuildPlan) -> ParquetShardWriter:
    plan.staging_root.mkdir(parents=True)
    image_dir = plan.staging_root / "images"
    image_dir.mkdir()
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_dir / "a.jpg"), image)
    assert cv2.imwrite(str(image_dir / "b.jpg"), image)
    writer = ParquetShardWriter(plan.staging_root, box_type=plan.box_type)
    # Physical order is intentionally opposite the loader's semantic order.
    writer.write(
        SAMPLES_ARTIFACT,
        [
            {
                "sample_id": "sample-b",
                "split": "validation",
                "sequence_id": "seq-b",
                "frame_index": 9,
                "timestamp_s": 1.5,
                "image_ref": "images/b.jpg",
                "height": 5,
                "width": 7,
            },
            {
                "sample_id": "sample-a",
                "split": "train",
                "sequence_id": "seq-a",
                "frame_index": 3,
                "timestamp_s": 0.5,
                "image_ref": "images/a.jpg",
                "height": 5,
                "width": 7,
            },
        ],
        shard_index=0,
    )
    instance_rows = []
    for sample_id, count in (("sample-b", 1), ("sample-a", 2)):
        for index in range(count):
            instance_rows.append(
                {
                    "instance_id": f"{plan.build_id}:{sample_id}:{index}",
                    "sample_id": sample_id,
                    "detection_index": index,
                    "x1": float(index),
                    "y1": 0.0,
                    "x2": float(index + 2),
                    "y2": 3.0,
                    "score": 0.9,
                    "class_id": 1,
                }
            )
    writer.write(INSTANCES_ARTIFACT, instance_rows, shard_index=0)
    return writer


@pytest.fixture
def materialized_build(tmp_path):
    plan = _plan(tmp_path, masks=True, embeddings=True)
    writer = _write_required_tables(plan)
    encoder_fingerprint = fingerprint("fixture-encoder")
    masks = []
    embeddings = []
    # Reverse instance order independently in both optional tables. Correct
    # results therefore require key joins rather than row alignment.
    for sample_id, count in (("sample-a", 2), ("sample-b", 1)):
        for index in reversed(range(count)):
            instance_id = f"{plan.build_id}:{sample_id}:{index}"
            mask = torch.zeros((5, 7), dtype=torch.bool)
            mask[0, 0] = index == 0
            masks.append(
                {
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "height": 5,
                    "width": 7,
                    "codec": MASK_CODEC,
                    "data": pack_mask(mask),
                }
            )
            embeddings.append(
                {
                    "sample_id": sample_id,
                    "instance_id": instance_id,
                    "encoder_fingerprint": encoder_fingerprint,
                    "dim": 3,
                    "values": [float(index + 1), 0.0, 0.0],
                }
            )
    writer.write(MASKS_ARTIFACT, masks, shard_index=0)
    writer.write(EMBEDDINGS_ARTIFACT, embeddings, shard_index=0, embedding_dim=3)
    root = finalize_build(
        plan,
        embedding_metadata={"encoder_fingerprint": encoder_fingerprint, "dim": 3},
        target_shard_rows=2,
    )
    return {"root": root, "plan": plan}


@pytest.fixture
def materialized_boxes_only_build(tmp_path):
    plan = _plan(tmp_path, masks=False, embeddings=False)
    _write_required_tables(plan)
    return finalize_build(plan)
