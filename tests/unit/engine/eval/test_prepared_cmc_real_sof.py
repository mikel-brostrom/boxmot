"""Real optical-flow preparation must preserve tracker replay byte for byte."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxmot.datasets import cached
from boxmot.datasets.schema import INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.eval import cmc, replay
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint
from boxmot.trackers import TrackerSpec
from boxmot.trackers.common.motion.cmc.sof import SOF

_FRAME_COUNT = 8


def _textured_build(directory: Path, geometry: str) -> Path:
    """Publish keyed two-class detections over a reproducibly translating scene."""
    detect = StagePlan.create("detect", component={"id": "real-sof-fixture"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=directory / "builds",
        dataset_name="real-sof-fixture",
        box_type=geometry,
        source_fingerprint=fingerprint("real-sof-fixture"),
        publish=PublishOptions(image_references=True),
        stages=(detect, finalize),
        metadata={"split": "validation"},
    )
    image_dir = plan.staging_root / "images"
    image_dir.mkdir(parents=True)
    height, width = 192, 256
    texture = np.random.default_rng(19).integers(20, 235, (height, width, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0.6)
    samples, instances = [], []
    for index in range(_FRAME_COUNT):
        sample_id = f"moving-{index}"
        translation = np.array([[1, 0, 3 * index], [0, 1, -index]], dtype=np.float32)
        rgb = cv2.warpAffine(texture, translation, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        assert cv2.imwrite(str(image_dir / f"{sample_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        samples.append(
            dict(
                sample_id=sample_id,
                split="validation",
                sequence_id="moving",
                frame_index=index,
                timestamp_s=index / 30.0,
                image_ref=f"images/{sample_id}.png",
                height=height,
                width=width,
            )
        )
        for class_id, (x, y) in enumerate(((45, 65), (160, 130))):
            cx, cy = float(x + 3 * index), float(y - index)
            box = (
                dict(x1=cx - 12, y1=cy - 15, x2=cx + 12, y2=cy + 15)
                if geometry == "aabb"
                else dict(cx=cx, cy=cy, w=24.0, h=30.0, angle=0.3 if class_id == 0 else -0.2)
            )
            instances.append(
                dict(
                    sample_id=sample_id,
                    instance_id=f"{plan.build_id}:{sample_id}:{class_id}",
                    detection_index=class_id,
                    score=0.95 if class_id == 0 or index % 3 else 0.05,
                    class_id=class_id,
                    **box,
                )
            )
    writer = ParquetShardWriter(plan.staging_root, box_type=geometry)
    writer.write(SAMPLES_ARTIFACT, list(reversed(samples)), shard_index=0)
    writer.write(INSTANCES_ARTIFACT, list(reversed(instances)), shard_index=0)
    return finalize_build(plan)


def _inline(tasks, *, workers, progress_callback):
    """Use production sequence replay with observable decoding and estimator calls."""
    return tuple(replay._replay_sequence_task(task, progress_callback=progress_callback) for task in tasks)


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("per_class", [False, True])
@pytest.mark.parametrize("cache_inputs", [False, True])
@pytest.mark.parametrize("tracker", ["bytetrack", "botsort", "ocsort"])
def test_prepared_cmc_matches_real_sof_replay(
    tmp_path, monkeypatch, geometry, per_class, cache_inputs, tracker
) -> None:
    """Actual SOF estimates must stay identical through prepared tracker consumption."""
    build = _textured_build(tmp_path, geometry)
    spec = TrackerSpec(
        tracker,
        geometry=geometry,
        per_class=per_class,
        class_ids=(0, 1),
        options=(("use_embeddings", False),) if tracker == "botsort" else (),
    )
    args = SimpleNamespace(
        build_path=build,
        split="validation",
        seq_info={"moving": _FRAME_COUNT},
        sequence_names=("moving",),
        tracker=spec.name,
        tracker_backend=spec.backend,
        geometry=geometry,
        per_class=per_class,
        tracker_class_ids=spec.class_ids,
        tracker_class_names=spec.class_names,
        cmc_policy="auto",
    )
    setup = cmc.prepare_eval_cmc(args, spec, tmp_path / "analysis")
    assert setup.policy.sequences == (("moving", True),)
    assert len(setup.replay_sequences) == 1
    warps = setup.replay_sequences[0].warps
    assert np.median(warps[1:, 0, 2]) == pytest.approx(3.0, abs=0.2)
    assert np.median(warps[1:, 1, 2]) == pytest.approx(-1.0, abs=0.2)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _inline)
    previous_threads = cv2.getNumThreads()
    try:
        # Match production worker initialization and automatic-analysis settings.
        cv2.setNumThreads(1)
        cv2.setRNGSeed(0)
        live = replay.replay_build(
            build,
            spec,
            split="validation",
            output_dir=tmp_path / "live",
            cmc_policy=setup.policy,
        )

        def unexpected(*args, **kwargs):
            pytest.fail("Prepared CMC decoded source pixels or reran the SOF estimator.")

        monkeypatch.setattr(SOF, "apply", unexpected)
        monkeypatch.setattr(cached, "read_rgb_chw_uint8", unexpected)
        monkeypatch.setattr(cv2, "imread", unexpected)
        prepared = replay.replay_build(
            build,
            spec,
            split="validation",
            output_dir=tmp_path / "prepared",
            cmc_policy=setup.policy,
            cmc_transforms=setup.replay_sequences,
            cache_inputs=cache_inputs,
        )
    finally:
        cv2.setNumThreads(previous_threads)
    assert prepared.frames == live.frames == _FRAME_COUNT
    assert prepared.track_rows == live.track_rows > 0
    assert [path.read_bytes() for path in prepared.sequence_files] == [
        path.read_bytes() for path in live.sequence_files
    ]
