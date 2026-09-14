"""Guided evaluation must align source pixels, track state, and metric indices."""

from __future__ import annotations

import json
import pickle
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from boxmot.datasets.schema import EMBEDDINGS_ARTIFACT, INSTANCES_ARTIFACT, SAMPLES_ARTIFACT
from boxmot.datasets.storage import ParquetShardWriter
from boxmot.engine.eval import evaluator, replay
from boxmot.engine.materialization import finalize_build
from boxmot.trackers import TrackerSpec
from tests.unit.datasets.conftest import _plan
from tests.unit.engine.eval.test_visualization import capture_rendered_images

BOX_TRACKERS = (
    "boosttrack",
    "botsort",
    "bytetrack",
    "deepocsort",
    "hybridsort",
    "occluboost",
    "ocsort",
    "sfsort",
    "strongsort",
)


@pytest.fixture
def guided_build(tmp_path, request):
    suffix = getattr(request, "param", "jpg")
    embeddings = suffix == "embeddings"
    suffix = "jpg" if embeddings else suffix
    plan = replace(_plan(tmp_path, masks=False, embeddings=embeddings), metadata={"split": "train"})
    plan.staging_root.mkdir(parents=True)
    rows, detections = [], []
    for sequence, offset, color in (("seq-a", 10, 50), ("seq-b", 30, 150)):
        source = plan.staging_root / "images" / sequence
        source.mkdir(parents=True)
        # Unselected neighboring frames must never enter the temporal model.
        for number in range(100, 105):
            assert cv2.imwrite(str(source / f"{number:06d}.{suffix}"), np.full((16, 24, 3), color, np.uint8))
        for index in range(3):
            sample_id = f"{sequence}-{index}"
            rows.append(
                dict(
                    sample_id=sample_id,
                    split="train",
                    sequence_id=sequence,
                    frame_index=offset + index,
                    timestamp_s=index / 30,
                    image_ref=f"images/{sequence}/{101 + index:06d}.{suffix}",
                    height=16,
                    width=24,
                )
            )
            if index != 1:  # No detections still requires one propagation step.
                detections.append(
                    dict(
                        instance_id=f"{plan.build_id}:{sample_id}:0",
                        sample_id=sample_id,
                        detection_index=0,
                        x1=4.0,
                        y1=2.0,
                        x2=12.0,
                        y2=14.0,
                        score=0.95,
                        class_id=1,
                    )
                )
    writer = ParquetShardWriter(plan.staging_root, box_type="aabb")
    writer.write(SAMPLES_ARTIFACT, list(reversed(rows)), shard_index=0)
    writer.write(INSTANCES_ARTIFACT, list(reversed(detections)), shard_index=0)
    embedding_metadata = None
    if embeddings:
        encoder_fingerprint = "a" * 64
        writer.write(
            EMBEDDINGS_ARTIFACT,
            [
                dict(
                    sample_id=row["sample_id"],
                    instance_id=row["instance_id"],
                    encoder_fingerprint=encoder_fingerprint,
                    dim=3,
                    values=[1.0, 0.0, 0.0],
                )
                for row in detections
            ],
            shard_index=0,
            embedding_dim=3,
        )
        embedding_metadata = {"encoder_fingerprint": encoder_fingerprint, "dim": 3}
    build = finalize_build(plan, embedding_metadata=embedding_metadata)
    checkpoint = tmp_path / "edgetam.pt"
    checkpoint.write_bytes(b"fake backend only")
    return build, checkpoint


@pytest.fixture
def fake_propagators(monkeypatch):
    from boxmot.segmentors.propagation import edgetam

    # These tests run worker tasks in the parent process. Mirror the fresh
    # worker's progress state instead of inheriting another test's fake queue.
    monkeypatch.setattr(replay, "_WORKER_PROGRESS_QUEUE", None)
    instances = []

    class Propagator:
        frame_shape = (16, 24)

        def __init__(self, checkpoint, *, device, max_objects=32, prompt_overlap=0.10):
            self.checkpoint = Path(checkpoint)
            self.pixels = []
            self.calls = []
            self.device = device
            self.max_objects = max_objects
            self.prompt_overlap = prompt_overlap
            self.mask_refs = []
            instances.append(self)

        def propagate(self, frame_index, frame, active_tracks, new_tracks):
            self.pixels.append(frame.copy())
            self.calls.append((frame_index, set(active_tracks), set(new_tracks)))
            if frame_index == 0:
                return {}
            mask = np.zeros(self.frame_shape, dtype=bool)
            mask[2:14, 4:12] = True
            mask.setflags(write=False)
            self.mask_refs.append(weakref.ref(mask))
            return {0: mask}

        def retain_tracks(self, track_ids):
            pass

        def reset(self):
            pass

    monkeypatch.setattr(edgetam, "EdgeTAMMaskPropagator", Propagator)
    return instances


def _run_tasks_locally(tasks, **kwargs):
    # Check the config sent to spawned workers contains only pickle-safe values.
    return tuple(replay._replay_sequence_task(pickle.loads(pickle.dumps(task))) for task in tasks)


@pytest.mark.parametrize("cap_override", [None, 16])
def test_replay_workers_receive_tuned_guidance_settings(
    guided_build, fake_propagators, monkeypatch, tmp_path, cap_override
) -> None:
    """All settings survive worker serialization and configure the actual runtime."""
    build, checkpoint = guided_build
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_tasks_locally)
    created_configs = []
    create = replay.create_tracker

    def capture(spec, *, mask_guidance=None):
        tracker = create(spec, mask_guidance=mask_guidance)
        if mask_guidance is not None:
            created_configs.append(tracker._mask_guidance.config)
        return tracker

    monkeypatch.setattr(replay, "create_tracker", capture)
    result = replay.replay_build(
        build,
        TrackerSpec("bytetrack", options=tuple(sorted((
            ("edgetam.min_coverage", 0.75),
            ("edgetam.min_fill", 0.12),
            ("edgetam.prompt_overlap", 0.25),
            ("edgetam.max_objects", 8),
        )))),
        output_dir=tmp_path / "tracks", split="train",
        mask_guidance_weights=checkpoint, mask_guidance_device="cpu",
        mask_guidance_max_objects=cap_override,
    )

    assert result.frames == 6
    assert len(created_configs) == len(fake_propagators) == 2
    for config, backend in zip(created_configs, fake_propagators):
        assert config.min_coverage == 0.75
        assert config.min_fill == 0.12
        assert config.prompt_overlap == backend.prompt_overlap == 0.25
        assert config.max_objects == backend.max_objects == (8 if cap_override is None else cap_override)


@pytest.mark.parametrize("cache_inputs", [False, True])
def test_guided_replay_aligns_split_frames_and_resets_each_sequence(
    guided_build, fake_propagators, monkeypatch, tmp_path, cache_inputs
):
    build, checkpoint = guided_build
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_tasks_locally)
    result = replay.replay_build(
        build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "tracks",
        split="train",
        mask_guidance_weights=checkpoint,
        mask_guidance_device="cpu",
        cache_inputs=cache_inputs,
        mask_guidance_max_objects=4,
    )
    assert result.frames == 6
    assert result.track_rows == 4
    assert len(fake_propagators) == 2
    for backend, color in zip(fake_propagators, (50, 150)):
        assert backend.max_objects == 4
        assert len(backend.pixels) == 3
        assert all(np.all(pixels == color) for pixels in backend.pixels)
        assert [call[0] for call in backend.calls] == [0, 1, 2]
        assert backend.calls[0][1] == set()
        assert backend.calls[1][1] == {0}
    for path, offset in zip(result.sequence_files, (10, 30)):
        rows = np.loadtxt(path, delimiter=",", ndmin=2)
        assert rows[:, 0].tolist() == [offset + 1, offset + 3]
        assert rows[:, 1].tolist() == [0, 0]


@pytest.mark.parametrize("tracker_name", BOX_TRACKERS)
@pytest.mark.parametrize("cache_inputs", [False, True])
@pytest.mark.parametrize("guided_build", ["embeddings"], indirect=True)
def test_each_box_tracker_replays_guidance_with_cached_appearance(
    guided_build, fake_propagators, monkeypatch, tmp_path, tracker_name, cache_inputs
):
    """Use actual tracker factories and cached features without loading either model."""
    build, checkpoint = guided_build
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_tasks_locally)

    result = replay.replay_build(
        build,
        TrackerSpec(name=tracker_name, options=(("asso_func", "iou"),)),
        output_dir=tmp_path / "tracks",
        split="train",
        mask_guidance_weights=checkpoint,
        mask_guidance_device="cpu",
        mask_guidance_max_objects=4,
        cache_inputs=cache_inputs,
    )

    assert result.frames == 6
    assert len(result.sequence_files) == len(fake_propagators) == 2
    for backend, color in zip(fake_propagators, (50, 150), strict=True):
        assert backend.max_objects == 4
        assert [call[0] for call in backend.calls] == [0, 1, 2]
        assert backend.calls[0][1:] == (set(), set())
        assert all(np.all(pixels == color) for pixels in backend.pixels)


def test_replay_resolves_checkpoint_before_spawning_sequence_workers(
    guided_build, fake_propagators, monkeypatch, tmp_path
):
    """Workers receive the absolute model path and never select a download target."""
    from boxmot.segmentors.propagation import weights

    build, original_checkpoint = guided_build
    original_checkpoint.unlink()
    monkeypatch.chdir(tmp_path)
    checkpoint = tmp_path / "models" / "edgetam.pt"
    requests = []

    def resolve(selected):
        requests.append(selected)
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"downloaded model")
        return checkpoint

    def run_tasks(tasks, **kwargs):
        tasks = tuple(tasks)
        assert requests == ["edgetam.pt"]
        assert all(task.mask_guidance_weights == str(checkpoint) for task in tasks)
        return _run_tasks_locally(tasks, **kwargs)

    monkeypatch.setattr(weights, "resolve_edgetam_checkpoint", resolve)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", run_tasks)

    result = replay.replay_build(
        build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "tracks",
        split="train",
        mask_guidance_weights="edgetam.pt",
        mask_guidance_device="cpu",
    )

    assert result.frames == 6
    assert requests == ["edgetam.pt"]
    assert len(fake_propagators) == 2
    assert all(backend.checkpoint == checkpoint for backend in fake_propagators)


def test_guided_callback_preserves_original_frame_identity(guided_build, fake_propagators, tmp_path):
    build, checkpoint = guided_build
    frames = []
    events = []
    result = replay.replay_build(
        build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "tracks",
        split="train",
        sequence_ids=("seq-b",),
        mask_guidance_weights=checkpoint,
        frame_callback=frames.append,
        progress_callback=events.append,
    )
    assert len(fake_propagators) == 1
    assert result.frames == 3
    assert [item.sample.frame.frame_index for item in frames] == [30, 31, 32]
    assert [item.result.tracks.sample_id for item in frames] == [f"seq-b-{i}" for i in range(3)]
    assert events[-1].status == "completed"
    assert events[-1].completed == 3


@pytest.mark.parametrize("enabled", [False, True])
def test_public_cached_iterator_exposes_borrowed_masks_and_resets_them_between_sequences(
    guided_build, fake_propagators, enabled
) -> None:
    """Injected tracker guidance stays auxiliary while real cached pixels stream."""
    from boxmot.datasets import CachedVisionDataset
    from boxmot.trackers import MaskGuidance, MaskGuidanceConfig, create_tracker

    build, checkpoint = guided_build
    guidance = MaskGuidance(MaskGuidanceConfig(checkpoint, device="cpu")) if enabled else None
    tracker = create_tracker("bytetrack", mask_guidance=guidance)
    counts = []
    mot_rows = []
    dataset = CachedVisionDataset(build, split="train", load_images=True)
    for item in replay.iter_cached_tracks(dataset, tracker):
        assert item.sample.frame is not None
        assert item.result.tracks.masks is None
        assert item.result.tracks.sample_id == item.sample.sample_id
        mot_rows.extend(replay.tracks_to_mot_rows(item.result.tracks, item.sample.frame_index))
        if enabled:
            masks = item.guidance_masks
            assert masks is not None
            counts.append(len(masks))
            if masks:
                assert masks[0] is tracker.guidance_masks[0]
                assert masks[0] is fake_propagators[0].mask_refs[-1]()
                with pytest.raises(TypeError):
                    masks[3] = masks[0]
            del masks
        else:
            assert item.guidance_masks is None
        del item
    assert [row[:2] for row in mot_rows] == [(11, 0), (13, 0), (31, 0), (33, 0)]
    if enabled:
        assert counts == [0, 1, 1, 0, 1, 1]
        backend = fake_propagators[0]
        assert [call[0] for call in backend.calls] == [0, 1, 2, 0, 1, 2]
        for index, pixels in enumerate(backend.pixels):
            assert np.all(pixels == (50 if index < 3 else 150))
        tracker.reset()
        assert all(reference() is None for reference in backend.mask_refs)
    else:
        assert fake_propagators == []


@pytest.mark.parametrize("show, save", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("cache_inputs", [False, True])
def test_guided_replay_draws_current_and_lost_masks_without_changing_metrics_or_propagating_twice(
    guided_build, fake_propagators, monkeypatch, tmp_path, show, save, cache_inputs
) -> None:
    """Exercise the real renderer through serial replay's callback boundary."""
    from boxmot.engine.eval.visualization import ReplayVisualization

    build, checkpoint = guided_build
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_tasks_locally)
    captured = capture_rendered_images(monkeypatch)
    options = dict(
        split="train",
        sequence_ids=("seq-b",),
        mask_guidance_weights=checkpoint,
        mask_guidance_device="cpu",
        cache_inputs=cache_inputs,
    )
    baseline = replay.replay_build(build, TrackerSpec("bytetrack"), output_dir=tmp_path / "plain", **options)
    mask_counts = []
    track_counts = []

    with ReplayVisualization(tmp_path / "rendered", show=show, save=save) as visualizer:
        def consume(replayed):
            masks = replayed.guidance_masks
            assert masks is not None
            mask_counts.append(len(masks))
            track_counts.append(len(replayed.result.tracks))
            assert replayed.result.tracks.masks is None
            if masks:
                assert masks[0] is fake_propagators[-1].mask_refs[-1]()
                assert not masks[0].flags.writeable
                with pytest.raises(TypeError):
                    masks[99] = masks[0]
            if len(mask_counts) == 3:
                assert fake_propagators[-1].mask_refs[0]() is None, "Replay retained an obsolete mask array"
            visualizer(replayed)

        rendered = replay.replay_build(
            build,
            TrackerSpec("bytetrack"),
            output_dir=tmp_path / "rendered",
            frame_callback=consume,
            **options,
        )

    assert mask_counts == [0, 1, 1]
    assert track_counts == [1, 0, 1]
    assert len(fake_propagators[-1].calls) == 3
    assert baseline.sequence_files[0].read_bytes() == rendered.sequence_files[0].read_bytes()
    images = captured.shown if show else captured.writers[0].images
    assert len(images) == 3
    np.testing.assert_array_equal(images[0][10, 8], [150, 150, 150])
    assert np.any(images[1][10, 8] != images[0][10, 8])
    np.testing.assert_array_equal(images[1][10, 8], images[2][10, 8])
    if save:
        assert captured.writers[0].closed
    if show and save:
        assert len(captured.writers[0].images) == len(captured.shown)
        for preview, encoded in zip(captured.shown, captured.writers[0].images, strict=True):
            np.testing.assert_array_equal(preview, encoded)


def test_failed_guided_callback_does_not_publish_partial_sequences(guided_build, fake_propagators, tmp_path):
    build, checkpoint = guided_build
    destination = tmp_path / "tracks"
    destination.mkdir()
    previous = destination / "seq-a.txt"
    previous.write_text("previous result\n")

    def fail(_frame):
        raise RuntimeError("display failed")

    with pytest.raises(RuntimeError, match="Cached replay failed"):
        replay.replay_build(
            build,
            TrackerSpec(name="bytetrack"),
            output_dir=destination,
            split="train",
            mask_guidance_weights=checkpoint,
            frame_callback=fail,
        )
    assert previous.read_text() == "previous result\n"
    assert not (destination / "seq-b.txt").exists()


@pytest.mark.parametrize("guided_build", ["png"], indirect=True)
def test_guidance_consumes_non_jpeg_frames_without_staging_a_video(
    guided_build, fake_propagators, monkeypatch, tmp_path
):
    build, checkpoint = guided_build
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_tasks_locally)

    def no_source_links(*args, **kwargs):
        pytest.fail("Streamed guidance must not stage another source directory")

    monkeypatch.setattr(Path, "symlink_to", no_source_links)
    result = replay.replay_build(
        build,
        TrackerSpec(name="bytetrack"),
        output_dir=tmp_path / "tracks",
        split="train",
        mask_guidance_weights=checkpoint,
    )
    assert result.frames == 6
    assert len(fake_propagators) == 2


@pytest.mark.parametrize(
    "spec",
    [
        TrackerSpec(name="maf_hda"),
        TrackerSpec(name="eagermot"),
        TrackerSpec(name="bytetrack", backend="cpp"),
        TrackerSpec(name="bytetrack", geometry="obb"),
        TrackerSpec(name="bytetrack", per_class=True),
        TrackerSpec(name="bytetrack", options=(("asso_func", "giou"),)),
    ],
)
def test_replay_rejects_unsupported_guidance_before_loading_build(tmp_path, spec):
    with pytest.raises(ValueError, match="Mask guidance requires"):
        replay.replay_build(
            "missing-build",
            spec,
            output_dir=tmp_path,
            mask_guidance_weights="missing.pt",
        )


@pytest.mark.parametrize("checkpoint_selection", ("custom", "edgetam.pt", "models/edgetam.pt"))
@pytest.mark.parametrize("enabled", (False, True))
def test_evaluator_runs_guidance_before_metrics_and_records_model(
    guided_build, fake_propagators, monkeypatch, tmp_path, checkpoint_selection, enabled
):
    build, checkpoint = guided_build
    if checkpoint_selection != "custom":
        from boxmot.segmentors.propagation import weights

        monkeypatch.chdir(tmp_path)
        checkpoint.unlink()
        checkpoint = tmp_path / "models" / "edgetam.pt"

        def resolve(selected):
            if Path(selected) != checkpoint:
                assert Path(selected) == Path(checkpoint_selection)
            if not checkpoint.is_file():
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(b"downloaded model")
            return checkpoint

        monkeypatch.setattr(weights, "resolve_edgetam_checkpoint", resolve)
    monkeypatch.setattr(replay, "_run_spawned_sequence_tasks", _run_tasks_locally)
    metrics_inputs = []

    def metrics(args, **kwargs):
        metrics_inputs.append(args.exp_dir)
        assert len(fake_propagators) == (2 if enabled else 0)
        assert len(list(args.exp_dir.glob("*.txt"))) == 2
        return {}

    monkeypatch.setattr(evaluator, "run_motmetrics", metrics)
    args = SimpleNamespace(
        _build_validated=True,
        asso_func=None,
        build_path=build,
        dataset_id="fixture",
        exist_ok=False,
        experiment_id=None,
        geometry="aabb",
        sequence_workers=1,
        name="eval",
        per_class=False,
        project=tmp_path / "runs",
        seq_info={"seq-a": 13, "seq-b": 33},
        sequence_frame_counts={"seq-a": 3, "seq-b": 3},
        sequence_names=None,
        split="train",
        tracker="bytetrack",
        tracker_backend="python",
        tracker_class_ids=(1,),
        tracker_class_names=((1, "person"),),
        edgetam=enabled,
        mask_guidance_weights=checkpoint if checkpoint_selection == "custom" else Path(checkpoint_selection),
        device="cpu",
    )
    result = evaluator.run_eval(args, setup=False)
    assert result.timings["frames"] == 6
    assert metrics_inputs == [result.exp_dir]
    if not enabled:
        assert "edgetam" not in result.exp_dir.name
        assert not (result.exp_dir / "mask-guidance.json").exists()
        assert not fake_propagators
        return
    assert "bytetrack-edgetam" in result.exp_dir.name
    provenance = json.loads((result.exp_dir / "mask-guidance.json").read_text())
    assert provenance["checkpoint"] == str(checkpoint.resolve())
    assert all(backend.checkpoint == checkpoint.resolve() for backend in fake_propagators)
    assert provenance["device"] == "cpu"
    assert provenance["tracker"]["name"] == "bytetrack"
