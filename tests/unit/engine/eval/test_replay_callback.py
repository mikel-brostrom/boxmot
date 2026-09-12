"""Cached visualization keeps source timing, payloads, and replay publication."""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from boxmot import create_tracker
from boxmot.datasets.masks import MASK_CODEC, pack_mask
from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, INSTANCES_ARTIFACT, MASKS_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.eval import replay
from boxmot.engine.materialization import BuildPlan, PublishOptions, StagePlan, finalize_build, fingerprint
from boxmot.structures import Detections, Frame, Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec


@pytest.fixture
def timestamped_build(tmp_path: Path) -> Path:
    """Publish two sequences with real pixels, irregular times, and keyed payloads."""
    detect = StagePlan.create("detect", component={"id": "fixture"})
    finalize = StagePlan.create("finalize", depends_on=("detect",), upstream_fingerprints=(detect.fingerprint,))
    plan = BuildPlan.create(
        build_root=tmp_path / "builds",
        dataset_name="callback-fixture",
        box_type="aabb",
        source_fingerprint=fingerprint("callback-fixture"),
        publish=PublishOptions(image_references=True, masks=True, embeddings=True),
        stages=(detect, finalize),
        metadata={"split": "variable"},
    )
    image_dir = plan.staging_root / "images"
    image_dir.mkdir(parents=True)
    samples, instances, masks, embeddings = [], [], [], []
    encoder = fingerprint("fixture-encoder")
    for sequence, timestamps in (("a", (1.0, 1.0 + 1.0 / 30.0, 1.2)), ("b", (4.0,))):
        for index, timestamp in enumerate(timestamps):
            sample_id = f"{sequence}-{index}"
            instance_id = f"{plan.build_id}:{sample_id}:0"
            image = np.full((20, 30, 3), 30 + index, dtype=np.uint8)
            assert cv2.imwrite(str(image_dir / f"{sample_id}.png"), image)
            samples.append(
                dict(
                    sample_id=sample_id,
                    split="variable",
                    sequence_id=sequence,
                    frame_index=index,
                    timestamp_s=timestamp,
                    image_ref=f"images/{sample_id}.png",
                    height=20,
                    width=30,
                )
            )
            instances.append(
                dict(
                    sample_id=sample_id,
                    instance_id=instance_id,
                    detection_index=0,
                    x1=1.0 + index,
                    y1=2.0,
                    x2=11.0 + index,
                    y2=12.0,
                    score=0.95,
                    class_id=1,
                )
            )
            masks.append(
                dict(
                    sample_id=sample_id,
                    instance_id=instance_id,
                    height=20,
                    width=30,
                    codec=MASK_CODEC,
                    data=pack_mask(torch.ones((20, 30), dtype=torch.bool)),
                )
            )
            embeddings.append(
                dict(
                    sample_id=sample_id,
                    instance_id=instance_id,
                    encoder_fingerprint=encoder,
                    dim=2,
                    values=[1.0, 0.0],
                )
            )
    writer = ParquetShardWriter(plan.staging_root, box_type=plan.box_type)
    writer.write(SAMPLES_ARTIFACT, list(reversed(samples)), shard_index=0)
    writer.write(INSTANCES_ARTIFACT, instances, shard_index=0)
    writer.write(MASKS_ARTIFACT, list(reversed(masks)), shard_index=0)
    writer.write(EMBEDDINGS_ARTIFACT, list(reversed(embeddings)), shard_index=0, embedding_dim=2)
    return finalize_build(plan, embedding_metadata={"encoder_fingerprint": encoder, "dim": 2})


class PayloadTracker:
    """Require cached payloads and timestamps without requiring image pixels."""

    name = "fixture"
    supports_obb = False
    requirements = TrackerRequirements(masks=True, embeddings=True, frame=True, frame_dimensions_only=True)

    def __init__(self) -> None:
        self.closed = False
        self.timestamps: list[float] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        self.closed = True

    def update(self, detections: Detections, frame: Frame) -> Tracks:
        assert detections.masks.values.all()
        torch.testing.assert_close(detections.embeddings, torch.tensor([[1.0, 0.0]]))
        self.timestamps.append(frame.timestamp_s)
        return Tracks(
            geometry=detections.geometry,
            track_ids=torch.tensor([1]),
            scores=detections.scores,
            class_ids=detections.class_ids,
            detection_indices=torch.tensor([0]),
            sample_id=detections.sample_id,
        )


def test_callback_preserves_selected_pixels_timestamps_payloads_and_progress(timestamped_build, tmp_path, monkeypatch):
    tracker = PayloadTracker()
    monkeypatch.setattr(replay, "create_tracker", lambda _spec: tracker)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", lambda *a, **kw: pytest.fail("unexpected spawn"))
    frames, events = [], []
    owner_thread = threading.get_ident()

    def capture(item: replay.ReplayFrame) -> None:
        assert threading.get_ident() == owner_thread
        frame = item.sample.frame
        assert frame is not None
        assert frame.sample_id == item.sample.sample_id
        assert frame.timestamp_s == item.sample.timestamp_s
        assert torch.all(frame.image == 30 + item.sample.frame_index)
        frames.append(item)

    result = replay.replay_build(
        timestamped_build,
        TrackerSpec(name="bytetrack"),
        split="variable",
        sequence_ids=("a",),
        sequence_frame_counts={"a": 3, "b": 1},
        output_dir=tmp_path / "shown",
        workers=8,
        frame_callback=capture,
        progress_callback=events.append,
    )

    assert tracker.closed
    assert tracker.resets == 1
    assert tracker.timestamps == pytest.approx([1.0, 1.0 + 1.0 / 30.0, 1.2])
    assert [item.sample.sample_id for item in frames] == ["a-0", "a-1", "a-2"]
    assert result.frames == result.track_rows == 3
    assert result.sequence_files == (tmp_path / "shown" / "a.txt",)
    assert not (tmp_path / "shown" / "b.txt").exists()
    assert {event.sequence_id for event in events} == {"a"}
    assert events[0].status == "queued"
    assert (events[-1].status, events[-1].completed, events[-1].total) == ("completed", 3, 3)


def test_callback_keeps_real_kalman_results_identical(timestamped_build, tmp_path, monkeypatch):
    spec = TrackerSpec(name="bytetrack", options=(("variable_dt", True),))
    ordinary = replay.replay_build(
        timestamped_build,
        spec,
        split="variable",
        output_dir=tmp_path / "ordinary",
        tracker=create_tracker(spec),
    )
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", lambda *a, **kw: pytest.fail("unexpected spawn"))
    shown = replay.replay_build(
        timestamped_build,
        spec,
        split="variable",
        output_dir=tmp_path / "shown",
        frame_callback=lambda _item: None,
    )

    assert shown.frames == ordinary.frames == 4
    assert shown.track_rows == ordinary.track_rows
    assert [path.read_bytes() for path in shown.sequence_files] == [
        path.read_bytes() for path in ordinary.sequence_files
    ]
    assert not list(shown.output_dir.glob(".replay-*"))


@pytest.mark.parametrize("render", [False, True])
def test_input_cache_is_used_by_injected_and_rendered_replay(timestamped_build, tmp_path, monkeypatch, render):
    from boxmot.datasets import replay_cache

    baseline = replay.replay_build(
        timestamped_build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "baseline",
        tracker=PayloadTracker(),
    )
    opened = []
    original_open = replay_cache.open_replay_sequence

    def open_cached(path, **kwargs):
        opened.append((path, kwargs))
        return original_open(path, **kwargs)

    monkeypatch.setattr(replay_cache, "open_replay_sequence", open_cached)
    frames = []
    cached = replay.replay_build(
        timestamped_build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "cached",
        tracker=PayloadTracker(),
        frame_callback=frames.append if render else None,
        cache_inputs=True,
    )
    assert len(opened) == 2
    assert all(options["load_images"] is render and options["load_embeddings"] for _, options in opened)
    assert [path.read_bytes() for path in baseline.sequence_files] == [
        path.read_bytes() for path in cached.sequence_files
    ]
    if render:
        assert len(frames) == 4
        assert all(item.sample.frame is not None for item in frames)


@pytest.mark.parametrize("injected", [False, True])
def test_callback_failure_preserves_old_results_and_closes_only_owned_tracker(
    timestamped_build, tmp_path, monkeypatch, injected
):
    tracker = PayloadTracker()
    monkeypatch.setattr(replay, "create_tracker", lambda _spec: tracker)
    output = tmp_path / "tracks"
    output.mkdir()
    (output / "a.txt").write_text("original result")
    events = []

    def fail(_item: replay.ReplayFrame) -> None:
        raise RuntimeError("renderer failed")

    with pytest.raises(RuntimeError, match="renderer failed"):
        replay.replay_build(
            timestamped_build,
            TrackerSpec(name="bytetrack"),
            output_dir=output,
            tracker=tracker if injected else None,
            frame_callback=fail,
            progress_callback=events.append,
        )

    assert tracker.closed is not injected
    assert (output / "a.txt").read_text() == "original result"
    assert not (output / "b.txt").exists()
    assert not list(output.glob(".replay-*"))
    assert events[-1].status == "failed"
    assert "renderer failed" in events[-1].detail


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_ids": ("missing",)}, "does not contain requested sequence"),
        ({"sequence_frame_counts": {"a": 4, "b": 1}}, "has 3 cached frames"),
        ({"sequence_frame_counts": {"b": 1}}, "specifies None"),
    ],
)
def test_callback_validates_selection_before_processing(timestamped_build, tmp_path, monkeypatch, kwargs, message):
    monkeypatch.setattr(replay, "create_tracker", lambda _spec: pytest.fail("tracker must not start"))
    output = tmp_path / "tracks"
    with pytest.raises(ValueError, match=message):
        replay.replay_build(
            timestamped_build,
            TrackerSpec(name="bytetrack"),
            output_dir=output,
            frame_callback=lambda _item: pytest.fail("callback must not run"),
            **kwargs,
        )
    assert not list(output.iterdir())


def test_callback_requires_callable_before_build_resolution(tmp_path):
    with pytest.raises(TypeError, match="frame_callback must be callable"):
        replay.replay_build("missing", TrackerSpec(name="bytetrack"), output_dir=tmp_path, frame_callback=42)
