from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

from boxmot.datasets import DatasetSample
from boxmot.datasets.schema import SAMPLES_ARTIFACT
from boxmot.engine.eval import replay as replay_module
from boxmot.engine.eval.replay import ReplayProgressEvent
from boxmot.structures import Boxes, Detections, Tracks
from boxmot.trackers import TrackerRequirements, TrackerSpec


class _EmbeddingTracker:
    name = "embedding-tracker"
    supports_obb = False
    requirements = TrackerRequirements(embeddings=True)

    def reset(self) -> None:
        return None

    def update(self, detections: Detections, frame=None) -> Tracks:
        del frame
        assert detections.embeddings is not None
        count = len(detections)
        return Tracks(
            geometry=detections.geometry,
            track_ids=torch.arange(1, count + 1, dtype=torch.int64),
            scores=detections.scores,
            class_ids=detections.class_ids,
            detection_indices=torch.arange(count, dtype=torch.int64),
            sample_id=detections.sample_id,
        )


def _embedding_sample() -> DatasetSample:
    sample_id = "validation:sequence-a:0"
    detections = Detections(
        geometry=Boxes(torch.tensor([[0.0, 0.0, 2.0, 3.0]], dtype=torch.float32)),
        scores=torch.tensor([0.9], dtype=torch.float32),
        class_ids=torch.tensor([0], dtype=torch.int64),
        embeddings=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        sample_id=sample_id,
    )
    return DatasetSample(
        sample_id=sample_id,
        split="validation",
        sequence_id="sequence-a",
        frame_index=0,
        timestamp_s=None,
        image_size=(4, 5),
        image_ref=None,
        frame=None,
        detections=detections,
    )


def test_worker_reports_loading_before_lazy_embedding_iteration(monkeypatch, tmp_path) -> None:
    timeline: list[str] = []
    sample = _embedding_sample()

    class _ProgressQueue:
        def put(self, event: ReplayProgressEvent) -> None:
            timeline.append(f"progress:{event.detail or event.status}")

    class _LazySequence:
        manifest = SimpleNamespace()

        def __len__(self) -> int:
            return 1

        def __iter__(self):
            timeline.append("read:embedding payload")
            yield sample

    @contextmanager
    def _owned_tracker(_spec):
        yield _EmbeddingTracker()

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
        assert sequence_id == "sequence-a"
        assert split == "validation"
        assert not load_images
        assert not load_masks
        assert load_embeddings
        timeline.append("open:sequence metadata")
        return _LazySequence()

    monkeypatch.setattr(replay_module, "_owned_tracker", _owned_tracker)
    monkeypatch.setattr(
        replay_module.CachedVisionDataset,
        "_stream_sequence",
        classmethod(_stream_sequence),
    )
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_module, "_WORKER_PROGRESS_QUEUE", _ProgressQueue())

    replay_module._replay_sequence_task(
        replay_module._SequenceReplayTask(
            build=str(tmp_path / "build"),
            tracker_spec=TrackerSpec(name="bytetrack"),
            split="validation",
            sequence_id="sequence-a",
            frame_total=1,
            output_path=str(tmp_path / "sequence-a.txt"),
            ordinal=0,
        )
    )

    assert timeline.index("progress:loading cached inputs") < timeline.index("open:sequence metadata")
    assert timeline.index("progress:streaming cached inputs") < timeline.index("read:embedding payload")


def test_parent_reads_only_sequence_ids_before_workers_report_progress(monkeypatch, tmp_path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    output = tmp_path / "tracks"
    manifest = SimpleNamespace(
        build_id="a" * 64,
        publish=SimpleNamespace(image_references=True),
        artifact=lambda name: SimpleNamespace(path=name),
    )
    reads: list[tuple[str, list[str] | None]] = []
    events: list[ReplayProgressEvent] = []

    class _Column:
        def to_pylist(self) -> list[str]:
            return ["sequence-a"]

    class _Table:
        def column(self, name: str) -> _Column:
            assert name == "sequence_id"
            return _Column()

    def _read_parquet_artifact(_path, *, artifact_name, box_type=None, columns=None, filters=None):
        del box_type, filters
        reads.append((artifact_name, columns))
        assert artifact_name == SAMPLES_ARTIFACT
        assert columns == ["sequence_id"]
        return _Table()

    @contextmanager
    def _owned_tracker(_spec):
        yield _EmbeddingTracker()

    def _run_tasks(tasks, *, workers, progress_callback):
        assert reads == [(SAMPLES_ARTIFACT, ["sequence_id"])]
        assert workers == 1
        task = tasks[0]
        progress_callback(
            ReplayProgressEvent(
                sequence_id=task.sequence_id,
                status="queued",
                completed=0,
                total=task.frame_total,
                track_rows=0,
                detail=None,
                ordinal=task.ordinal,
            )
        )
        Path(task.output_path).write_text("", encoding="utf-8")
        return (
            replay_module._SequenceReplayResult(
                sequence_id=task.sequence_id,
                output_path=task.output_path,
                frames=1,
                track_rows=0,
                ordinal=task.ordinal,
            ),
        )

    monkeypatch.setattr(replay_module, "resolve_build_path", lambda *_args, **_kwargs: build)
    monkeypatch.setattr(replay_module.DatasetManifest, "load", lambda _path: manifest)
    monkeypatch.setattr(replay_module, "read_parquet_artifact", _read_parquet_artifact)
    monkeypatch.setattr(replay_module, "validate_build_compatibility", lambda *args, **kwargs: None)
    monkeypatch.setattr(replay_module, "_owned_tracker", _owned_tracker)
    monkeypatch.setattr(replay_module, "_run_spawned_sequence_tasks", _run_tasks)
    monkeypatch.setattr(
        replay_module,
        "load_cached_build",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parent eagerly opened the build")),
    )

    result = replay_module.replay_build(
        build,
        TrackerSpec(name="bytetrack"),
        split="validation",
        output_dir=output,
        workers=1,
        progress_callback=events.append,
    )

    assert reads == [(SAMPLES_ARTIFACT, ["sequence_id"])]
    assert [(event.sequence_id, event.status) for event in events] == [("sequence-a", "queued")]
    assert result.sequence_files == (output / "sequence-a.txt",)
