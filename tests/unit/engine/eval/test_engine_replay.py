from __future__ import annotations

import concurrent.futures
import math
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from boxmot.datasets import DatasetSample
from boxmot.engine.eval import evaluator as evaluator_module
from boxmot.engine.eval import replay as replay_module
from boxmot.engine.eval.replay import (
    ReplayProgressEvent,
    ReplayResult,
    iter_cached_tracks,
    tracks_to_mot_rows,
)
from boxmot.structures import Boxes, Detections, OrientedBoxes, Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec


def _sample(sample_id: str, sequence_id: str, frame_index: int) -> DatasetSample:
    detections = Detections(
        geometry=Boxes(torch.tensor([[1.0, 2.0, 5.0, 8.0]], dtype=torch.float32)),
        scores=torch.tensor([0.9], dtype=torch.float32),
        class_ids=torch.tensor([3], dtype=torch.int64),
        sample_id=sample_id,
    )
    return DatasetSample(
        sample_id=sample_id,
        split="validation",
        sequence_id=sequence_id,
        frame_index=frame_index,
        timestamp_s=None,
        image_size=(10, 12),
        image_ref=None,
        frame=None,
        detections=detections,
    )


class _Tracker:
    name = "fake"
    supports_obb = True
    requirements = TrackerRequirements()

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def update(self, detections: Detections, frame=None) -> Tracks:
        assert frame is None
        count = len(detections)
        return Tracks(
            geometry=detections.geometry,
            track_ids=torch.arange(10, 10 + count, dtype=torch.int64),
            scores=detections.scores,
            class_ids=detections.class_ids,
            detection_indices=torch.arange(count, dtype=torch.int64),
            sample_id=detections.sample_id,
        )


def test_iter_cached_tracks_uses_one_sequence_state_and_filters_by_key() -> None:
    tracker = _Tracker()
    samples = [
        _sample("a-0", "a", 0),
        _sample("a-1", "a", 1),
        _sample("b-0", "b", 0),
    ]

    replayed = list(iter_cached_tracks(samples, tracker, sequence_ids=frozenset({"a"})))

    assert [item.sample.sample_id for item in replayed] == ["a-0", "a-1"]
    assert [item.result.tracks.track_ids.tolist() for item in replayed] == [[10], [10]]
    assert tracker.resets == 1


def test_replay_reuses_placeholder_image_storage_for_equal_frame_sizes() -> None:
    placeholders: dict[tuple[int, int], torch.Tensor] = {}

    first = replay_module._frame_for_sample(_sample("a-0", "a", 0), placeholders)
    second = replay_module._frame_for_sample(_sample("a-1", "a", 1), placeholders)

    assert first.image is second.image
    assert tuple(first.image.shape) == (3, 10, 12)
    assert list(placeholders) == [(10, 12)]


def test_tracks_to_mot_rows_serializes_aabb_and_obb_without_positional_cache_state() -> None:
    aabb = Tracks(
        geometry=Boxes(torch.tensor([[1.0, 2.0, 5.0, 8.0]], dtype=torch.float32)),
        track_ids=torch.tensor([2], dtype=torch.int64),
        scores=torch.tensor([0.75], dtype=torch.float32),
        class_ids=torch.tensor([3], dtype=torch.int64),
        detection_indices=torch.tensor([0], dtype=torch.int64),
        sample_id="a",
    )
    assert tracks_to_mot_rows(aabb, 4) == [(5, 2, 1.0, 2.0, 4.0, 6.0, 0.75, 3, 0)]

    obb = Tracks(
        geometry=OrientedBoxes(torch.tensor([[10.0, 20.0, 4.0, 2.0, math.pi / 2]], dtype=torch.float32)),
        track_ids=torch.tensor([7], dtype=torch.int64),
        scores=torch.tensor([0.8], dtype=torch.float32),
        class_ids=torch.tensor([1], dtype=torch.int64),
        detection_indices=torch.tensor([-1], dtype=torch.int64),
        sample_id="b",
    )
    row = tracks_to_mot_rows(obb, 0)[0]
    assert len(row) == 13
    assert row[:2] == (1, 7)
    assert row[-2:] == (1, -1)


def test_mot_serialization_preserves_large_ids_and_python_float_subtraction() -> None:
    """Bulk conversion must not round IDs or subtract geometry in float32."""
    values = torch.tensor([[0.1, 0.3, 1.0, 2.0]], dtype=torch.float32)
    tracks = Tracks(
        geometry=Boxes(values),
        track_ids=torch.tensor([2**60 + 7], dtype=torch.int64),
        scores=torch.tensor([0.75], dtype=torch.float32),
        class_ids=torch.tensor([2**60 + 1], dtype=torch.int64),
        detection_indices=torch.tensor([-1], dtype=torch.int64),
        sample_id="a",
    )
    x1, y1, x2, y2 = map(float, values[0])
    assert tracks_to_mot_rows(tracks, 4) == [
        (5, 2**60 + 7, x1, y1, x2 - x1, y2 - y1, 0.75, 2**60 + 1, -1)
    ]


def test_sequence_replay_task_constructs_isolated_tracker_per_sequence(tmp_path, monkeypatch) -> None:
    created_trackers: list[_Tracker] = []
    tracked_by_sequence: dict[str, list[_Tracker]] = {"a": [], "b": []}
    emitted: list[ReplayProgressEvent] = []

    class _ProgressQueue:
        def put(self, event: ReplayProgressEvent) -> None:
            emitted.append(event)

    class _SequenceDataset(list[DatasetSample]):
        manifest = SimpleNamespace()

    @contextmanager
    def _owned_tracker(_spec):
        tracker = _Tracker()
        original_update = tracker.update

        def update(detections, frame=None):
            sequence_id = detections.sample_id.split("-", maxsplit=1)[0]
            tracked_by_sequence[sequence_id].append(tracker)
            return original_update(detections, frame)

        tracker.update = update
        created_trackers.append(tracker)
        yield tracker

    def _stream_sequence(
        _cls,
        _build,
        *,
        sequence_id,
        split,
        load_images,
        load_masks,
        load_embeddings,
    ):
        assert split == "validation"
        assert not load_images
        assert not load_masks
        assert not load_embeddings
        return _SequenceDataset([_sample(f"{sequence_id}-0", sequence_id, 0)])

    monkeypatch.setattr(replay_module, "_owned_tracker", _owned_tracker)
    monkeypatch.setattr(
        replay_module.CachedVisionDataset,
        "_stream_sequence",
        classmethod(_stream_sequence),
    )
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_module, "_WORKER_PROGRESS_QUEUE", _ProgressQueue())

    results = [
        replay_module._replay_sequence_task(
            replay_module._SequenceReplayTask(
                build=str(tmp_path / "build"),
                tracker_spec=TrackerSpec(name="bytetrack"),
                split="validation",
                sequence_id=sequence_id,
                frame_total=1,
                output_path=str(tmp_path / f"{sequence_id}.txt"),
                ordinal=ordinal,
            )
        )
        for ordinal, sequence_id in enumerate(("a", "b"))
    ]

    assert [result.sequence_id for result in results] == ["a", "b"]
    assert len(created_trackers) == 2
    assert tracked_by_sequence == {"a": [created_trackers[0]], "b": [created_trackers[1]]}
    assert [(event.sequence_id, event.status, event.completed, event.total) for event in emitted] == [
        ("a", "running", 0, 1),
        ("a", "running", 0, 1),
        ("a", "running", 1, 1),
        ("a", "completed", 1, 1),
        ("b", "running", 0, 1),
        ("b", "running", 0, 1),
        ("b", "running", 1, 1),
        ("b", "completed", 1, 1),
    ]
    assert emitted[0].detail == "loading cached inputs"
    assert emitted[4].detail == "loading cached inputs"


def test_sequence_replay_uses_sample_dimensions_without_loading_image_pixels(tmp_path, monkeypatch) -> None:
    received_frames = []

    class _DimensionsTracker(_Tracker):
        requirements = TrackerRequirements(frame=True, frame_dimensions_only=True)

        def update(self, detections: Detections, frame=None) -> Tracks:
            assert frame is not None
            received_frames.append(frame)
            count = len(detections)
            return Tracks(
                geometry=detections.geometry,
                track_ids=torch.arange(10, 10 + count, dtype=torch.int64),
                scores=detections.scores,
                class_ids=detections.class_ids,
                detection_indices=torch.arange(count, dtype=torch.int64),
                sample_id=detections.sample_id,
            )

    class _SequenceDataset(list[DatasetSample]):
        manifest = SimpleNamespace()

    @contextmanager
    def _owned_tracker(_spec):
        yield _DimensionsTracker()

    def _stream_sequence(
        _cls,
        _build,
        *,
        sequence_id,
        split,
        load_images,
        load_masks,
        load_embeddings,
    ):
        assert split == "validation"
        assert load_images is False
        assert load_masks is False
        assert load_embeddings is False
        return _SequenceDataset([_sample(f"{sequence_id}-{index}", sequence_id, index) for index in range(2)])

    monkeypatch.setattr(replay_module, "_owned_tracker", _owned_tracker)
    monkeypatch.setattr(replay_module.CachedVisionDataset, "_stream_sequence", classmethod(_stream_sequence))
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_module, "_WORKER_PROGRESS_QUEUE", None)

    result = replay_module._replay_sequence_task(
        replay_module._SequenceReplayTask(
            build=str(tmp_path / "build"),
            tracker_spec=TrackerSpec(name="sfsort"),
            split="validation",
            sequence_id="a",
            frame_total=2,
            output_path=str(tmp_path / "a.txt"),
            ordinal=0,
        )
    )

    assert result.frames == 2
    assert [frame.image_size for frame in received_frames] == [(10, 12), (10, 12)]
    assert received_frames[0].image is received_frames[1].image


def test_replay_worker_does_not_join_observational_progress_feeder(monkeypatch) -> None:
    class _ProgressQueue:
        cancelled = False

        def cancel_join_thread(self) -> None:
            self.cancelled = True

    progress_queue = _ProgressQueue()
    logger = replay_module.logging.getLogger("boxmot")
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", True)
    monkeypatch.setattr(logger, "disabled", False)

    replay_module._initialize_replay_worker(progress_queue)

    assert progress_queue.cancelled is True
    assert replay_module._WORKER_PROGRESS_QUEUE is progress_queue
    assert logger.propagate is False
    assert logger.disabled is True


def test_spawned_sequence_tasks_use_spawn_context_and_return_ordinal_order(monkeypatch, tmp_path) -> None:
    observed: dict[str, object] = {}
    progress: list[ReplayProgressEvent] = []

    class _ProgressQueue:
        def get_nowait(self):
            raise replay_module.queue.Empty

        def close(self):
            return None

        def join_thread(self):
            return None

    class _Context:
        def Queue(self):
            return _ProgressQueue()

    context = _Context()

    class _Executor:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            observed.update(
                max_workers=max_workers,
                mp_context=mp_context,
                initializer=initializer,
                initargs=initargs,
            )

        def submit(self, _function, task):
            future = concurrent.futures.Future()
            future.set_result(
                replay_module._SequenceReplayResult(
                    sequence_id=task.sequence_id,
                    output_path=task.output_path,
                    frames=task.ordinal + 1,
                    track_rows=task.ordinal + 2,
                    ordinal=task.ordinal,
                )
            )
            return future

        def shutdown(self, *, wait, cancel_futures):
            observed["shutdown"] = (wait, cancel_futures)

    def _get_context(method):
        observed["start_method"] = method
        return context

    monkeypatch.setattr(replay_module, "get_context", _get_context)
    monkeypatch.setattr(replay_module.concurrent.futures, "ProcessPoolExecutor", _Executor)

    spec = TrackerSpec(name="bytetrack")
    tasks = (
        replay_module._SequenceReplayTask("build", spec, None, "b", 2, str(tmp_path / "b.tmp"), 1),
        replay_module._SequenceReplayTask("build", spec, None, "a", 1, str(tmp_path / "a.tmp"), 0),
    )
    results = replay_module._run_spawned_sequence_tasks(tasks, workers=2, progress_callback=progress.append)

    assert observed["start_method"] == "spawn"
    assert observed["max_workers"] == 2
    assert observed["mp_context"] is context
    assert observed["initializer"] is replay_module._initialize_replay_worker
    assert observed["shutdown"] == (True, False)
    assert [result.sequence_id for result in results] == ["a", "b"]
    assert [(event.sequence_id, event.status) for event in progress[:2]] == [
        ("b", "queued"),
        ("a", "queued"),
    ]
    assert [event.total for event in progress[:2]] == [2, 1]
    assert {(event.sequence_id, event.status) for event in progress[2:]} == {
        ("a", "completed"),
        ("b", "completed"),
    }


def test_spawned_sequence_failure_is_attributed_after_pool_completion(monkeypatch, tmp_path) -> None:
    progress: list[ReplayProgressEvent] = []
    observed: dict[str, object] = {}

    class _Executor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, _function, _task):
            future = concurrent.futures.Future()
            future.set_exception(ValueError("broken sequence"))
            return future

        def shutdown(self, *, wait, cancel_futures):
            observed["shutdown"] = (wait, cancel_futures)

    class _ProgressQueue:
        def get_nowait(self):
            raise replay_module.queue.Empty

        def close(self):
            return None

        def join_thread(self):
            return None

    monkeypatch.setattr(
        replay_module,
        "get_context",
        lambda _method: SimpleNamespace(Queue=lambda: _ProgressQueue()),
    )
    monkeypatch.setattr(replay_module.concurrent.futures, "ProcessPoolExecutor", _Executor)
    task = replay_module._SequenceReplayTask(
        "build",
        TrackerSpec(name="bytetrack"),
        None,
        "failed-sequence",
        1,
        str(tmp_path / "failed.tmp"),
        0,
    )

    with pytest.raises(RuntimeError, match="failed-sequence"):
        replay_module._run_spawned_sequence_tasks((task,), workers=1, progress_callback=progress.append)

    assert observed["shutdown"] == (True, False)
    assert [(event.sequence_id, event.status) for event in progress] == [
        ("failed-sequence", "queued"),
        ("failed-sequence", "failed"),
    ]
    assert "broken sequence" in (progress[-1].detail or "")


def test_replay_build_schedules_sorted_sequence_tasks_with_only_specs_crossing_workers(monkeypatch, tmp_path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    destination = tmp_path / "tracks"
    manifest = SimpleNamespace()
    callback = lambda _event: None
    observed: dict[str, object] = {}

    monkeypatch.setattr(replay_module, "resolve_build_path", lambda *_args, **_kwargs: build)
    monkeypatch.setattr(replay_module.DatasetManifest, "load", lambda _path: manifest)
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        replay_module,
        "_sequence_frame_counts",
        lambda *_args, **_kwargs: (("a", 1), ("z", 3)),
    )
    probe_trackers: list[_Tracker] = []

    @contextmanager
    def _requirements_probe(_spec):
        tracker = _Tracker()
        probe_trackers.append(tracker)
        yield tracker

    monkeypatch.setattr(replay_module, "_owned_tracker", _requirements_probe)

    def _run_tasks(tasks, *, workers, progress_callback):
        observed.update(tasks=tasks, workers=workers, progress_callback=progress_callback)
        results = []
        for task in tasks:
            Path(task.output_path).write_text(task.sequence_id, encoding="utf-8")
            results.append(
                replay_module._SequenceReplayResult(
                    sequence_id=task.sequence_id,
                    output_path=task.output_path,
                    frames=task.frame_total,
                    track_rows=task.ordinal + 1,
                    ordinal=task.ordinal,
                )
            )
        return tuple(results)

    monkeypatch.setattr(replay_module, "_run_spawned_sequence_tasks", _run_tasks)
    result = replay_module.replay_build(
        build,
        TrackerSpec(name="bytetrack"),
        split="validation",
        output_dir=destination,
        sequence_ids=("z", "a"),
        workers=8,
        progress_callback=callback,
    )

    tasks = observed["tasks"]
    assert [task.sequence_id for task in tasks] == ["a", "z"]
    assert [task.frame_total for task in tasks] == [1, 3]
    assert all(task.tracker_spec == TrackerSpec(name="bytetrack") for task in tasks)
    assert len(probe_trackers) == 1
    assert observed["workers"] == 2
    assert observed["progress_callback"] is callback
    assert result.sequence_files == (destination / "a.txt", destination / "z.txt")
    assert [path.read_text(encoding="utf-8") for path in result.sequence_files] == ["a", "z"]
    assert result.frames == 4
    assert result.track_rows == 3


def test_replay_build_reuses_prevalidated_catalog_frame_counts(monkeypatch, tmp_path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    destination = tmp_path / "tracks"
    manifest = SimpleNamespace()
    observed: dict[str, object] = {}

    monkeypatch.setattr(replay_module, "resolve_build_path", lambda *_args, **_kwargs: build)
    monkeypatch.setattr(replay_module.DatasetManifest, "load", lambda _path: manifest)
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        replay_module,
        "_sequence_frame_counts",
        lambda *_args, **_kwargs: pytest.fail("coordinator must reuse the validated catalog counts"),
    )

    @contextmanager
    def _requirements_probe(_spec):
        yield _Tracker()

    monkeypatch.setattr(replay_module, "_owned_tracker", _requirements_probe)

    def _run_tasks(tasks, *, workers, progress_callback):
        observed["tasks"] = tasks
        results = []
        for task in tasks:
            Path(task.output_path).write_text(task.sequence_id, encoding="utf-8")
            results.append(
                replay_module._SequenceReplayResult(
                    sequence_id=task.sequence_id,
                    output_path=task.output_path,
                    frames=task.frame_total,
                    track_rows=0,
                    ordinal=task.ordinal,
                )
            )
        return tuple(results)

    monkeypatch.setattr(replay_module, "_run_spawned_sequence_tasks", _run_tasks)

    result = replay_module.replay_build(
        build,
        TrackerSpec(name="bytetrack"),
        split="validation",
        output_dir=destination,
        sequence_ids=("z", "a"),
        sequence_frame_counts={"z": 3, "a": 1},
        workers=8,
    )

    tasks = observed["tasks"]
    assert [(task.sequence_id, task.frame_total) for task in tasks] == [("a", 1), ("z", 3)]
    assert result.frames == 4


def test_replay_build_rejects_missing_images_for_pixel_required_tracker_before_spawning(
    monkeypatch,
    tmp_path,
) -> None:
    build = tmp_path / "build"
    build.mkdir()
    manifest = SimpleNamespace(
        build_id="0" * 64,
        publish=SimpleNamespace(image_references=False),
    )

    class _FrameTracker(_Tracker):
        requirements = TrackerRequirements(frame=True)

    @contextmanager
    def _requirements_probe(_spec):
        yield _FrameTracker()

    monkeypatch.setattr(replay_module, "resolve_build_path", lambda *_args, **_kwargs: build)
    monkeypatch.setattr(replay_module.DatasetManifest, "load", lambda _path: manifest)
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_module, "_owned_tracker", _requirements_probe)
    monkeypatch.setattr(
        replay_module,
        "_sequence_frame_counts",
        lambda *_args, **_kwargs: pytest.fail("sequence workers must not be scheduled"),
    )

    with pytest.raises(replay_module.BuildCompatibilityError, match="--publish-image-refs"):
        replay_module.replay_build(
            build,
            TrackerSpec(name="sfsort"),
            output_dir=tmp_path / "tracks",
            workers=2,
        )


def test_replay_build_accepts_missing_image_references_for_dimensions_only_tracker(
    monkeypatch,
    tmp_path,
) -> None:
    build = tmp_path / "build"
    build.mkdir()
    manifest = SimpleNamespace(
        build_id="0" * 64,
        publish=SimpleNamespace(image_references=False),
    )

    class _DimensionsTracker(_Tracker):
        requirements = TrackerRequirements(frame=True, frame_dimensions_only=True)

    @contextmanager
    def _requirements_probe(_spec):
        yield _DimensionsTracker()

    monkeypatch.setattr(replay_module, "resolve_build_path", lambda *_args, **_kwargs: build)
    monkeypatch.setattr(replay_module.DatasetManifest, "load", lambda _path: manifest)
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_module, "_owned_tracker", _requirements_probe)

    result = replay_module.replay_build(
        build,
        TrackerSpec(name="sfsort"),
        output_dir=tmp_path / "tracks",
        sequence_frame_counts={},
        workers=2,
    )

    assert result.frames == 0


def test_run_eval_wires_sequence_processes_and_structured_progress(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _Pipeline:
        workflow = None

        def __init__(self) -> None:
            self.advances: list[str] = []
            self.stored: list[tuple[object, int]] = []

        def advance(self, detail: str) -> None:
            self.advances.append(detail)

        def callback(self):
            return object()

        def store_step_info(self, renderable, *, step: int) -> None:
            self.stored.append((renderable, step))

    class _Presenter:
        renderable = object()

        def __init__(self, callback, sequence_totals) -> None:
            captured["presenter_callback"] = callback
            captured["sequence_totals"] = sequence_totals

        def __enter__(self):
            return self

        def __exit__(self, *_exc) -> None:
            return None

        def __call__(self, _event) -> None:
            return None

    def _replay_build(build, tracker_spec, **kwargs):
        captured.update(build=build, tracker_spec=tracker_spec, replay_kwargs=kwargs)
        return ReplayResult(
            build=Path(build),
            output_dir=Path(kwargs["output_dir"]),
            sequence_files=(),
            frames=2,
            track_rows=0,
        )

    monkeypatch.setattr(evaluator_module, "EvalSequenceProgressPresenter", _Presenter)
    monkeypatch.setattr(evaluator_module, "_refresh_eval_pipeline_intro", lambda *_args: None)
    monkeypatch.setattr(evaluator_module, "replay_build", _replay_build)
    monkeypatch.setattr(evaluator_module, "run_motmetrics", lambda *_args, **_kwargs: {})

    args = SimpleNamespace(
        _build_validated=True,
        asso_func=None,
        build_path=tmp_path / "build",
        dataset_id="fixture",
        exist_ok=True,
        experiment_id=None,
        geometry="aabb",
        sequence_workers=3,
        name="eval",
        per_class=False,
        project=tmp_path / "runs",
        seq_info={"seq-a": 1, "seq-b": 1},
        sequence_names=None,
        split="validation",
        tracker="bytetrack",
        tracker_backend="python",
        tracker_class_ids=(1,),
        tracker_class_names=((1, "person"),),
    )
    pipeline = _Pipeline()

    result = evaluator_module.run_eval(
        args,
        setup=False,
        verbose=False,
        pipeline=pipeline,
    )

    replay_kwargs = captured["replay_kwargs"]
    assert replay_kwargs["workers"] == 3
    assert replay_kwargs["sequence_frame_counts"] == args.seq_info
    assert replay_kwargs["progress_callback"].__class__ is _Presenter
    assert "tracker" not in replay_kwargs
    assert captured["sequence_totals"] == args.seq_info
    assert pipeline.advances == [
        "Replaying materialized detections through the tracker…",
        "Computing evaluation metrics…",
    ]
    assert pipeline.stored == [(_Presenter.renderable, evaluator_module.EvalWorkflowReporter.TRACK)]
    assert result.timings["frames"] == 2
