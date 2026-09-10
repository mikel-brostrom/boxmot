"""Keyed Parquet replay through the canonical live tracker API."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Iterator, Literal, TextIO

import torch

from boxmot import create_tracker
from boxmot.datasets import CachedVisionDataset, DatasetManifest, DatasetSample
from boxmot.datasets.schema import SAMPLES_ARTIFACT
from boxmot.datasets.storage import read_parquet_artifact, resolve_artifact_path
from boxmot.engine.config.trackers import validate_image_tracker
from boxmot.engine.eval.mots_io import prepare_mots_tracks, tracks_to_mots_rows, write_mots_rows
from boxmot.engine.materialization.builds import (
    BuildCompatibilityError,
    load_cached_build,
    resolve_build_path,
    validate_build_compatibility,
)
from boxmot.pipelines import PipelineOutputs, PipelineResult, TrackingPipeline
from boxmot.structures import Boxes, Frame, OrientedBoxes, Tracks
from boxmot.trackers import Tracker, TrackerSpec

ReplayProgressStatus = Literal["queued", "running", "completed", "failed"]
ReplayProgressCallback = Callable[["ReplayProgressEvent"], None]

_WORKER_PROGRESS_QUEUE: Any | None = None


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """One cached sample and its canonical tracking result."""

    sample: DatasetSample
    result: PipelineResult


ReplayFrameCallback = Callable[[ReplayFrame], None]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Published tracker text files produced by a replay run."""

    build: Path
    output_dir: Path
    sequence_files: tuple[Path, ...]
    frames: int
    track_rows: int


@dataclass(frozen=True, slots=True)
class ReplayProgressEvent:
    """Pickle-safe progress emitted by one sequence replay task."""

    sequence_id: str
    status: ReplayProgressStatus
    completed: int
    total: int
    track_rows: int
    detail: str | None
    ordinal: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise ValueError("ReplayProgressEvent.sequence_id must be a non-empty string.")
        if self.status not in {"queued", "running", "completed", "failed"}:
            raise ValueError(f"Unknown replay progress status {self.status!r}.")
        for name in ("completed", "total", "track_rows", "ordinal"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"ReplayProgressEvent.{name} must be a non-negative integer.")
        if self.completed > self.total:
            raise ValueError("Replay progress cannot exceed its total frame count.")
        if self.detail is not None and not isinstance(self.detail, str):
            raise TypeError("ReplayProgressEvent.detail must be a string or None.")


@dataclass(frozen=True, slots=True)
class _SequenceReplayTask:
    """Only immutable, pickle-safe data sent to a spawned worker."""

    build: str
    tracker_spec: TrackerSpec
    split: str | None
    sequence_id: str
    frame_total: int
    output_path: str
    ordinal: int
    output_format: str = "mot"


@dataclass(frozen=True, slots=True)
class _SequenceReplayResult:
    """Small worker result; tracker and dataset objects never cross processes."""

    sequence_id: str
    output_path: str
    frames: int
    track_rows: int
    ordinal: int


def _frame_for_sample(
    sample: DatasetSample,
    placeholder_images: dict[tuple[int, int], torch.Tensor] | None = None,
) -> Frame:
    if sample.frame is not None:
        return sample.frame
    height, width = sample.image_size
    image = None if placeholder_images is None else placeholder_images.get(sample.image_size)
    if image is None:
        # Detector-free replay still needs frame identity and dimensions for
        # pipeline ordering, but a tracker without a pixel requirement never
        # observes these pixels. Reuse one backing tensor per resolution.
        image = torch.empty((3, height, width), dtype=torch.uint8)
        if placeholder_images is not None:
            placeholder_images[sample.image_size] = image
    return Frame(
        image=image,
        sample_id=sample.sample_id,
        sequence_id=sample.sequence_id,
        frame_index=sample.frame_index,
        timestamp_s=sample.timestamp_s,
        source_uri=sample.image_ref,
    )


def iter_cached_tracks(
    dataset: Iterable[DatasetSample],
    tracker: Tracker,
    *,
    sequence_ids: frozenset[str] | None = None,
    output_format: str = "mot",
) -> Iterator[ReplayFrame]:
    """Replay an ordered cached dataset with one sequence-local tracker."""

    _validate_output_format(output_format)
    pipeline = TrackingPipeline(
        detector=None,
        tracker=tracker,
        outputs=PipelineOutputs(
            masks=tracker.requirements.masks or output_format == "mots",
            embeddings=tracker.requirements.embeddings,
        ),
    )
    active_sequence: str | None = None
    placeholder_images: dict[tuple[int, int], torch.Tensor] = {}
    for sample in dataset:
        if sequence_ids is not None and sample.sequence_id not in sequence_ids:
            continue
        if sample.sequence_id != active_sequence:
            pipeline.reset()
            active_sequence = sample.sequence_id
        frame = _frame_for_sample(sample, placeholder_images)
        result = pipeline.step_detections(frame, sample.detections)
        if output_format == "mots":
            result = PipelineResult(result.detections, prepare_mots_tracks(result, sample.image_size))
        yield ReplayFrame(sample=sample, result=result)


def _obb_corners(geometry: torch.Tensor) -> torch.Tensor:
    centers = geometry[:, :2]
    half = geometry[:, 2:4] * 0.5
    angles = geometry[:, 4]
    template = geometry.new_tensor(((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)))
    local = template.unsqueeze(0) * half.unsqueeze(1)
    cosine = angles.cos()
    sine = angles.sin()
    rotation = torch.stack((cosine, -sine, sine, cosine), dim=1).reshape(-1, 2, 2)
    return torch.bmm(local, rotation.transpose(1, 2)) + centers.unsqueeze(1)


def tracks_to_mot_rows(tracks: Tracks, frame_index: int) -> list[tuple[float | int, ...]]:
    """Serialize tracks to MOT AABB9 or MMOT corner13 rows."""

    frame_number = frame_index + 1
    rows: list[tuple[float | int, ...]] = []
    track_ids = tracks.track_ids.tolist()
    scores = tracks.scores.tolist()
    class_ids = tracks.class_ids.tolist()
    detection_indices = tracks.detection_indices.tolist()
    if isinstance(tracks.geometry, Boxes):
        for index, (x1, y1, x2, y2) in enumerate(tracks.geometry.values.tolist()):
            # Subtract Python floats, as before, to retain double-precision
            # widths/heights and byte-identical MOT formatting.
            rows.append(
                (
                    frame_number,
                    track_ids[index],
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    scores[index],
                    class_ids[index],
                    detection_indices[index],
                )
            )
        return rows

    if not isinstance(tracks.geometry, OrientedBoxes):
        raise TypeError(f"Unsupported track geometry {type(tracks.geometry).__name__}.")
    corners = _obb_corners(tracks.geometry.values).reshape(-1, 8)
    for index, values in enumerate(corners.tolist()):
        rows.append(
            (
                frame_number,
                track_ids[index],
                *values,
                scores[index],
                class_ids[index],
                detection_indices[index],
            )
        )
    return rows


def _write_rows(handle: TextIO, rows: list[tuple[float | int, ...]]) -> None:
    for row in rows:
        values = [str(value) if isinstance(value, int) else f"{value:.8g}" for value in row]
        handle.write(",".join(values) + "\n")


def _validate_output_format(output_format: str) -> None:
    """Reject unknown tracker output schemas before opening result files."""
    if output_format not in ("mot", "mots"):
        raise ValueError("output_format must be 'mot' or 'mots'.")


def _write_tracks(handle: TextIO, tracks: Tracks, frame_index: int, output_format: str) -> int:
    """Serialize one prepared frame and return the number of emitted objects."""
    if output_format == "mots":
        mots_rows = tracks_to_mots_rows(tracks, frame_index)
        write_mots_rows(handle, mots_rows)
        return len(mots_rows)
    mot_rows = tracks_to_mot_rows(tracks, frame_index)
    _write_rows(handle, mot_rows)
    return len(mot_rows)


def _validate_sequence_id(sequence_id: str) -> str:
    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("Sequence identifiers must be non-empty strings.")
    if any(token in sequence_id for token in ("/", "\\", "..")):
        raise ValueError(f"Unsafe sequence identifier {sequence_id!r}.")
    return sequence_id


def _sequence_frame_counts(
    build: Path,
    manifest: DatasetManifest,
    *,
    split: str | None,
) -> tuple[tuple[str, int], ...]:
    """Read only sample keys needed to form deterministic replay tasks."""

    samples = manifest.artifact(SAMPLES_ARTIFACT)
    sample_path = resolve_artifact_path(build, samples.path)
    filters = None if split is None else [("split", "=", split)]
    table = read_parquet_artifact(
        sample_path,
        artifact_name=SAMPLES_ARTIFACT,
        columns=["sequence_id"],
        filters=filters,
    )
    counts: Counter[str] = Counter()
    for value in table.column("sequence_id").to_pylist():
        sequence_id = _validate_sequence_id(value)
        counts[sequence_id] += 1
    return tuple(sorted(counts.items()))


def _select_sequence_ids(
    available: tuple[str, ...],
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if requested is None:
        return available
    if not isinstance(requested, tuple):
        raise TypeError("sequence_ids must be a tuple of sequence identifiers or None")
    if not requested:
        raise ValueError("sequence_ids must not be empty when supplied")
    normalized = tuple(sorted({_validate_sequence_id(value) for value in requested}))
    missing = sorted(set(normalized).difference(available))
    if missing:
        raise ValueError(f"Build does not contain requested sequence(s): {', '.join(missing)}")
    return normalized


def _initialize_replay_worker(progress_queue: Any | None) -> None:
    """Bind worker-only state and silence logs that would corrupt parent Rich output."""

    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue
    if progress_queue is not None:
        # Progress is lossy/observational and terminal states are also
        # synthesized from Future results. Do not let a child wait for its
        # Queue feeder thread while the parent is joining the process pool.
        progress_queue.cancel_join_thread()
    boxmot_logger = logging.getLogger("boxmot")
    boxmot_logger.handlers.clear()
    boxmot_logger.propagate = False
    boxmot_logger.disabled = True


def _emit_worker_progress(event: ReplayProgressEvent) -> None:
    progress_queue = _WORKER_PROGRESS_QUEUE
    if progress_queue is None:
        return
    try:
        progress_queue.put(event)
    except (BrokenPipeError, EOFError, OSError):
        # Progress is observational. A closed parent queue must not invalidate
        # an otherwise deterministic sequence result.
        return


@contextmanager
def _owned_tracker(spec: TrackerSpec) -> Iterator[Tracker]:
    """Create and deterministically release a worker-local tracker."""

    tracker = create_tracker(spec)
    primary_error: BaseException | None = None
    try:
        yield tracker
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        close = getattr(tracker, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                add_note = getattr(primary_error, "add_note", None)
                if callable(add_note):
                    add_note(f"Tracker cleanup also failed: {cleanup_error}")


@contextmanager
def _prefetch_samples(dataset: Iterable[DatasetSample]) -> Iterator[Iterator[DatasetSample]]:
    """Overlap one cached sample read with tracking without sharing tracker state.

    Only the producer thread advances the dataset iterator. At most one next
    sample is outstanding, so decoded images and embeddings remain bounded to
    the current frame and one prefetched frame. Wait for an active read before
    closing its generator, including when tracking or serialization raises.
    """

    iterator = iter(dataset)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-input")
    pending: concurrent.futures.Future[DatasetSample | None] | None = None
    primary_error: BaseException | None = None

    def samples() -> Iterator[DatasetSample]:
        nonlocal pending
        while pending is not None:
            sample = pending.result()
            if sample is None:
                return
            pending = executor.submit(next, iterator, None)
            yield sample

    prefetched = samples()
    try:
        pending = executor.submit(next, iterator, None)
        yield prefetched
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        prefetched.close()
        close = getattr(iterator, "close", None)
        if callable(close):
            try:
                close()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                add_note = getattr(primary_error, "add_note", None)
                if callable(add_note):
                    add_note(f"Cached input cleanup also failed: {cleanup_error}")


def _replay_sequence_task(task: _SequenceReplayTask) -> _SequenceReplayResult:
    """Replay exactly one sequence in a spawned process."""

    completed = 0
    total = 0
    track_rows = 0
    _emit_worker_progress(
        ReplayProgressEvent(
            sequence_id=task.sequence_id,
            status="running",
            completed=0,
            total=task.frame_total,
            track_rows=0,
            detail="loading cached inputs",
            ordinal=task.ordinal,
        )
    )
    try:
        with _owned_tracker(task.tracker_spec) as tracker:
            requirements = tracker.requirements
            needs_masks = requirements.masks or task.output_format == "mots"
            dataset = CachedVisionDataset._stream_sequence(
                task.build,
                sequence_id=task.sequence_id,
                split=task.split,
                load_images=requirements.frame_pixels,
                load_masks=needs_masks,
                load_embeddings=requirements.embeddings,
            )
            validate_build_compatibility(
                dataset.manifest,
                split=task.split,
                geometry=task.tracker_spec.geometry,
                require_masks=needs_masks,
                require_embeddings=requirements.embeddings,
            )
            total = len(dataset)
            if total != task.frame_total:
                raise RuntimeError(
                    f"Sequence {task.sequence_id!r} changed while replay was starting: "
                    f"expected {task.frame_total} frames, loaded {total}."
                )
            _emit_worker_progress(
                ReplayProgressEvent(
                    sequence_id=task.sequence_id,
                    status="running",
                    completed=0,
                    total=total,
                    track_rows=0,
                    detail="streaming cached inputs",
                    ordinal=task.ordinal,
                )
            )

            output_path = Path(task.output_path)
            pipeline = TrackingPipeline(
                detector=None,
                tracker=tracker,
                outputs=PipelineOutputs(
                    masks=needs_masks,
                    embeddings=requirements.embeddings,
                ),
            )
            placeholder_images: dict[tuple[int, int], torch.Tensor] = {}
            # Prefetch overlaps pixel/payload decoding with tracking. Small
            # detection-only samples are cheaper to consume on this thread.
            sample_context = (
                _prefetch_samples(dataset)
                if requirements.frame_pixels or needs_masks or requirements.embeddings
                else nullcontext(iter(dataset))
            )
            with output_path.open("x", encoding="utf-8") as handle, sample_context as samples:
                for sample in samples:
                    if sample.sequence_id != task.sequence_id:
                        raise RuntimeError(
                            f"Sequence-scoped loader returned {sample.sequence_id!r} for task {task.sequence_id!r}."
                        )
                    frame = _frame_for_sample(sample, placeholder_images)
                    result = pipeline.step_detections(frame, sample.detections)
                    tracks = (
                        prepare_mots_tracks(result, sample.image_size)
                        if task.output_format == "mots"
                        else result.tracks
                    )
                    row_count = _write_tracks(handle, tracks, sample.frame_index, task.output_format)
                    completed += 1
                    track_rows += row_count
                    _emit_worker_progress(
                        ReplayProgressEvent(
                            sequence_id=task.sequence_id,
                            status="running",
                            completed=completed,
                            total=total,
                            track_rows=track_rows,
                            detail=sample.sample_id,
                            ordinal=task.ordinal,
                        )
                    )
                handle.flush()
                os.fsync(handle.fileno())
    except Exception as exc:
        _emit_worker_progress(
            ReplayProgressEvent(
                sequence_id=task.sequence_id,
                status="failed",
                completed=completed,
                total=max(task.frame_total, total, completed),
                track_rows=track_rows,
                detail=f"{type(exc).__name__}: {exc}",
                ordinal=task.ordinal,
            )
        )
        raise RuntimeError(f"Cached replay failed for sequence {task.sequence_id!r}.") from exc

    _emit_worker_progress(
        ReplayProgressEvent(
            sequence_id=task.sequence_id,
            status="completed",
            completed=completed,
            total=total,
            track_rows=track_rows,
            detail=None,
            ordinal=task.ordinal,
        )
    )
    return _SequenceReplayResult(
        sequence_id=task.sequence_id,
        output_path=task.output_path,
        frames=completed,
        track_rows=track_rows,
        ordinal=task.ordinal,
    )


def _replay_worker_count(sequence_count: int, cpu_count: int | None = None) -> int:
    """Bound automatic parallelism by both sequences and logical CPUs."""

    if sequence_count <= 0:
        return 0
    available = os.cpu_count() if cpu_count is None else cpu_count
    return min(sequence_count, 8, max(1, available or 1))


def _publish_progress(
    event: ReplayProgressEvent,
    callback: ReplayProgressCallback | None,
    latest: dict[int, ReplayProgressEvent],
) -> None:
    previous = latest.get(event.ordinal)
    if previous == event:
        return
    if previous is not None and previous.status in {"completed", "failed"} and event.status == "running":
        return
    latest[event.ordinal] = event
    if callback is not None:
        try:
            callback(event)
        except Exception:
            # Progress reporting is observational and cannot invalidate a
            # successfully tracked sequence.
            return


def _drain_progress_queue(
    progress_queue: Any | None,
    callback: ReplayProgressCallback | None,
    latest: dict[int, ReplayProgressEvent],
) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            event = progress_queue.get_nowait()
        except (queue.Empty, EOFError, OSError):
            return
        if not isinstance(event, ReplayProgressEvent):
            raise TypeError(f"Replay worker emitted unsupported progress {type(event).__name__}.")
        _publish_progress(event, callback, latest)


def _run_spawned_sequence_tasks(
    tasks: tuple[_SequenceReplayTask, ...],
    *,
    workers: int,
    progress_callback: ReplayProgressCallback | None,
) -> tuple[_SequenceReplayResult, ...]:
    """Run sequence tasks in spawn workers and return stable ordinal order."""

    context = get_context("spawn")
    progress_queue = context.Queue() if progress_callback is not None else None
    latest: dict[int, ReplayProgressEvent] = {}
    for task in tasks:
        _publish_progress(
            ReplayProgressEvent(
                sequence_id=task.sequence_id,
                status="queued",
                completed=0,
                total=task.frame_total,
                track_rows=0,
                detail=None,
                ordinal=task.ordinal,
            ),
            progress_callback,
            latest,
        )

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_initialize_replay_worker,
        initargs=(progress_queue,),
    )
    futures: dict[concurrent.futures.Future[_SequenceReplayResult], _SequenceReplayTask] = {}
    results: dict[int, _SequenceReplayResult] = {}
    failures: list[tuple[_SequenceReplayTask, BaseException]] = []
    try:
        futures = {executor.submit(_replay_sequence_task, task): task for task in tasks}
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=0.1,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            _drain_progress_queue(progress_queue, progress_callback, latest)
            for future in done:
                task = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    _drain_progress_queue(progress_queue, progress_callback, latest)
                    previous = latest.get(task.ordinal)
                    if previous is None or previous.status != "failed":
                        _publish_progress(
                            ReplayProgressEvent(
                                sequence_id=task.sequence_id,
                                status="failed",
                                completed=0 if previous is None else previous.completed,
                                total=0 if previous is None else previous.total,
                                track_rows=0 if previous is None else previous.track_rows,
                                detail=f"{type(exc).__name__}: {exc}",
                                ordinal=task.ordinal,
                            ),
                            progress_callback,
                            latest,
                        )
                    failures.append((task, exc))
                    continue
                if (
                    result.sequence_id != task.sequence_id
                    or result.output_path != task.output_path
                    or result.ordinal != task.ordinal
                ):
                    failures.append(
                        (
                            task,
                            RuntimeError(f"Replay worker returned mismatched metadata for {task.sequence_id!r}."),
                        )
                    )
                    _publish_progress(
                        ReplayProgressEvent(
                            sequence_id=task.sequence_id,
                            status="failed",
                            completed=result.frames,
                            total=result.frames,
                            track_rows=result.track_rows,
                            detail="Replay worker returned mismatched task metadata.",
                            ordinal=task.ordinal,
                        ),
                        progress_callback,
                        latest,
                    )
                    continue
                results[result.ordinal] = result
                previous = latest.get(task.ordinal)
                if previous is None or previous.status != "completed":
                    _publish_progress(
                        ReplayProgressEvent(
                            sequence_id=result.sequence_id,
                            status="completed",
                            completed=result.frames,
                            total=result.frames,
                            track_rows=result.track_rows,
                            detail=None,
                            ordinal=result.ordinal,
                        ),
                        progress_callback,
                        latest,
                    )
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True, cancel_futures=False)
    finally:
        _drain_progress_queue(progress_queue, progress_callback, latest)
        if progress_queue is not None:
            progress_queue.close()
            progress_queue.join_thread()

    if failures:
        failures.sort(key=lambda item: item[0].ordinal)
        failed_names = ", ".join(task.sequence_id for task, _ in failures)
        raise RuntimeError(f"Tracking failed for {len(failures)} sequence(s): {failed_names}.") from failures[0][1]

    return tuple(results[index] for index in range(len(tasks)))


def _replay_with_injected_tracker(
    build_path: Path,
    tracker_spec: TrackerSpec,
    *,
    split: str | None,
    destination: Path,
    tracker: Tracker,
    sequence_ids: tuple[str, ...] | None,
    frame_callback: ReplayFrameCallback | None = None,
    progress_callback: ReplayProgressCallback | None = None,
    sequence_frame_counts: Mapping[str, int] | None = None,
    output_format: str = "mot",
) -> ReplayResult:
    """Preserve the caller-owned, in-process tracker path used by tests and embedding clients."""

    requirements = tracker.requirements
    needs_masks = requirements.masks or output_format == "mots"
    if frame_callback is None:
        dataset = load_cached_build(
            build_path,
            split=split,
            load_images=requirements.frame_pixels,
            load_masks=needs_masks,
            load_embeddings=requirements.embeddings,
        )
        validate_build_compatibility(
            dataset.manifest,
            split=split,
            geometry=tracker_spec.geometry,
            require_masks=needs_masks,
            require_embeddings=requirements.embeddings,
        )
    else:
        assert sequence_frame_counts is not None
        dataset = _callback_samples(
            build_path, tracker, split=split, frame_counts=sequence_frame_counts, output_format=output_format
        )

    handles: dict[str, TextIO] = {}
    sequence_paths: dict[str, Path] = {}
    frames = 0
    track_rows = 0
    selected_sequences = None
    if sequence_ids is not None:
        if not isinstance(sequence_ids, tuple):
            raise TypeError("sequence_ids must be a tuple of sequence identifiers or None")
        if not sequence_ids:
            raise ValueError("sequence_ids must not be empty when supplied")
        selected_sequences = frozenset(_validate_sequence_id(value) for value in sequence_ids)
    frame_counts = dict(sequence_frame_counts or {})
    ordinals = {sequence: ordinal for ordinal, sequence in enumerate(frame_counts)}
    completed: Counter[str] = Counter()
    sequence_rows: Counter[str] = Counter()
    latest: dict[int, ReplayProgressEvent] = {}

    def progress(sequence: str, status: ReplayProgressStatus, detail: str | None = None) -> None:
        if sequence not in frame_counts:
            return
        _publish_progress(
            ReplayProgressEvent(
                sequence_id=sequence,
                status=status,
                completed=completed[sequence],
                total=frame_counts[sequence],
                track_rows=sequence_rows[sequence],
                detail=detail,
                ordinal=ordinals[sequence],
            ),
            progress_callback,
            latest,
        )

    for sequence in frame_counts:
        progress(sequence, "queued")
    replayed_frames = iter_cached_tracks(dataset, tracker, sequence_ids=selected_sequences, output_format=output_format)
    active_sequence: str | None = None
    try:
        for replayed in replayed_frames:
            sequence = _validate_sequence_id(replayed.sample.sequence_id)
            active_sequence = sequence
            if sequence not in handles:
                progress(sequence, "running", "streaming cached inputs")
                path = destination / f"{sequence}.txt"
                path.unlink(missing_ok=True)
                handles[sequence] = path.open("x", encoding="utf-8")
                sequence_paths[sequence] = path
            row_count = _write_tracks(
                handles[sequence], replayed.result.tracks, replayed.sample.frame_index, output_format
            )
            frames += 1
            track_rows += row_count
            completed[sequence] += 1
            sequence_rows[sequence] += row_count
            if frame_callback is not None:
                frame_callback(replayed)
            progress(sequence, "running", replayed.sample.sample_id)
            if completed[sequence] == frame_counts.get(sequence):
                progress(sequence, "completed")
        if frame_callback is not None:
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
    except Exception as exc:
        if active_sequence is not None:
            progress(active_sequence, "failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        replayed_frames.close()
        close_samples = getattr(dataset, "close", None)
        if callable(close_samples):
            close_samples()
        for handle in handles.values():
            handle.close()

    if selected_sequences is not None:
        missing = sorted(selected_sequences.difference(sequence_paths))
        if missing:
            raise ValueError(f"Build does not contain requested sequence(s): {', '.join(missing)}")

    return ReplayResult(
        build=build_path,
        output_dir=destination,
        sequence_files=tuple(sequence_paths[name] for name in sorted(sequence_paths)),
        frames=frames,
        track_rows=track_rows,
    )


def _callback_samples(
    build_path: Path,
    tracker: Tracker,
    *,
    split: str | None,
    frame_counts: Mapping[str, int],
    output_format: str = "mot",
) -> Iterator[DatasetSample]:
    """Stream selected sequences with real pixels and bounded optional payloads."""

    requirements = tracker.requirements
    for sequence, expected in frame_counts.items():
        dataset = CachedVisionDataset._stream_sequence(
            build_path,
            sequence_id=sequence,
            split=split,
            load_images=True,
            load_masks=requirements.masks or output_format == "mots",
            load_embeddings=requirements.embeddings,
        )
        if len(dataset) != expected:
            raise RuntimeError(
                f"Sequence {sequence!r} changed while replay was starting: "
                f"expected {expected} frames, loaded {len(dataset)}."
            )
        with _prefetch_samples(dataset) as samples:
            yield from samples


def _replay_with_frame_callback(
    build_path: Path,
    tracker_spec: TrackerSpec,
    *,
    split: str | None,
    destination: Path,
    tracker: Tracker | None,
    sequence_ids: tuple[str, ...] | None,
    sequence_frame_counts: Mapping[str, int] | None,
    workers: int | None,
    frame_callback: ReplayFrameCallback,
    progress_callback: ReplayProgressCallback | None,
    output_format: str = "mot",
) -> ReplayResult:
    """Keep rendering on the caller's thread and publish only a complete replay."""

    manifest = DatasetManifest.load(build_path)
    actual_counts = dict(_sequence_frame_counts(build_path, manifest, split=split))
    selected = _select_sequence_ids(tuple(actual_counts), sequence_ids)
    _validated_worker_count(workers, len(selected))
    if sequence_frame_counts is not None:
        supplied_counts = _validated_sequence_frame_counts(sequence_frame_counts)
        for sequence in selected:
            if supplied_counts.get(sequence) != actual_counts[sequence]:
                raise ValueError(
                    f"Sequence {sequence!r} has {actual_counts[sequence]} cached frames, "
                    f"but sequence_frame_counts specifies {supplied_counts.get(sequence)!r}."
                )
    frame_counts = {sequence: actual_counts[sequence] for sequence in selected}
    with tempfile.TemporaryDirectory(prefix=".replay-", dir=destination) as staging_dir:
        ownership = _owned_tracker(tracker_spec) if tracker is None else nullcontext(tracker)
        with ownership as active_tracker:
            requirements = active_tracker.requirements
            validate_build_compatibility(
                manifest,
                split=split,
                geometry=tracker_spec.geometry,
                require_masks=requirements.masks or output_format == "mots",
                require_embeddings=requirements.embeddings,
            )
            if not manifest.publish.image_references:
                raise BuildCompatibilityError(
                    f"Build {manifest.build_id!r} is missing image references required for rendering. "
                    "Run `boxmot materialize ... --publish-image-refs` and pass the resulting --build."
                )
            if not selected:
                return ReplayResult(build_path, destination, (), 0, 0)
            replayed = _replay_with_injected_tracker(
                build_path,
                tracker_spec,
                split=split,
                destination=Path(staging_dir),
                tracker=active_tracker,
                sequence_ids=selected,
                sequence_frame_counts=frame_counts,
                frame_callback=frame_callback,
                progress_callback=progress_callback,
                output_format=output_format,
            )
        sequence_paths: list[Path] = []
        for staged in replayed.sequence_files:
            path = destination / staged.name
            os.replace(staged, path)
            sequence_paths.append(path)
    return ReplayResult(
        build=build_path,
        output_dir=destination,
        sequence_files=tuple(sequence_paths),
        frames=replayed.frames,
        track_rows=replayed.track_rows,
    )


def _validated_worker_count(workers: int | None, sequence_count: int) -> int:
    if workers is None:
        return _replay_worker_count(sequence_count)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer or None")
    return min(workers, sequence_count) if sequence_count else 0


def _validated_sequence_frame_counts(
    values: Mapping[str, int],
) -> dict[str, int]:
    """Normalize trusted catalog counts supplied by an engine workflow."""

    if not isinstance(values, Mapping):
        raise TypeError("sequence_frame_counts must be a mapping of sequence IDs to frame counts")
    normalized: dict[str, int] = {}
    for raw_sequence_id, frame_count in values.items():
        sequence_id = _validate_sequence_id(raw_sequence_id)
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError("sequence frame counts must be positive integers")
        normalized[sequence_id] = frame_count
    return dict(sorted(normalized.items()))


def replay_build(
    build: str | Path,
    tracker_spec: TrackerSpec,
    *,
    build_root: str | Path | None = None,
    split: str | None = None,
    output_dir: str | Path,
    tracker: Tracker | None = None,
    sequence_ids: tuple[str, ...] | None = None,
    sequence_frame_counts: Mapping[str, int] | None = None,
    workers: int | None = None,
    progress_callback: ReplayProgressCallback | None = None,
    frame_callback: ReplayFrameCallback | None = None,
    output_format: str = "mot",
) -> ReplayResult:
    """Replay keyed detections, isolating each sequence in a spawned process.

    Supplying ``tracker`` deliberately selects the serial caller-owned path;
    this keeps dependency-injected trackers usable without attempting to
    pickle their state. The default path sends only immutable specs and paths
    to workers, which construct and release their own trackers.

    A ``frame_callback`` also selects serial replay on the caller's thread and
    receives each result with its decoded source frame and original timestamp.
    Callback failures propagate without publishing partial MOT result files.
    ``output_format='mots'`` requires published detection masks and writes
    zero-based KITTI MOTS segmentation rows with deterministic, disjoint masks.
    """

    _validate_output_format(output_format)
    if not isinstance(tracker_spec, TrackerSpec):
        raise TypeError("tracker_spec must be a TrackerSpec")
    validate_image_tracker(tracker_spec.name)
    if split is not None and (not isinstance(split, str) or not split or split != split.strip()):
        raise ValueError("split must be a non-empty canonical string or None")
    if frame_callback is not None and not callable(frame_callback):
        raise TypeError("frame_callback must be callable or None")
    build_path = resolve_build_path(build, build_root=build_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if frame_callback is not None:
        return _replay_with_frame_callback(
            build_path,
            tracker_spec,
            split=split,
            destination=destination,
            tracker=tracker,
            sequence_ids=sequence_ids,
            sequence_frame_counts=sequence_frame_counts,
            workers=workers,
            frame_callback=frame_callback,
            progress_callback=progress_callback,
            output_format=output_format,
        )
    if tracker is not None:
        staging_context = (
            tempfile.TemporaryDirectory(prefix=".replay-", dir=destination)
            if output_format == "mots"
            else nullcontext(str(destination))
        )
        with staging_context as staged_dir:
            replayed = _replay_with_injected_tracker(
                build_path,
                tracker_spec,
                split=split,
                destination=Path(staged_dir),
                tracker=tracker,
                sequence_ids=sequence_ids,
                output_format=output_format,
            )
            if output_format == "mot":
                return replayed
            sequence_paths = []
            for staged in replayed.sequence_files:
                path = destination / staged.name
                os.replace(staged, path)
                sequence_paths.append(path)
        return ReplayResult(build_path, destination, tuple(sequence_paths), replayed.frames, replayed.track_rows)

    manifest = DatasetManifest.load(build_path)
    with _owned_tracker(tracker_spec) as probe:
        requirements = probe.requirements
        validate_build_compatibility(
            manifest,
            split=split,
            geometry=tracker_spec.geometry,
            require_masks=requirements.masks or output_format == "mots",
            require_embeddings=requirements.embeddings,
        )
        if requirements.frame_pixels and not manifest.publish.image_references:
            raise BuildCompatibilityError(
                f"Build {manifest.build_id!r} is missing required image references. "
                "Run `boxmot materialize ... --publish-image-refs` and pass the resulting --build."
            )
    # Evaluation already derived these counts while validating the raw source
    # catalog. Reusing them avoids opening the samples Parquet a second time in
    # the coordinator. Each worker still verifies its sequence count against
    # the immutable build before processing its first frame.
    frame_counts = (
        dict(_sequence_frame_counts(build_path, manifest, split=split))
        if sequence_frame_counts is None
        else _validated_sequence_frame_counts(sequence_frame_counts)
    )
    selected = _select_sequence_ids(tuple(frame_counts), sequence_ids)
    worker_count = _validated_worker_count(workers, len(selected))
    if not selected:
        return ReplayResult(
            build=build_path,
            output_dir=destination,
            sequence_files=(),
            frames=0,
            track_rows=0,
        )

    with tempfile.TemporaryDirectory(prefix=".replay-", dir=destination) as staging_dir:
        staging = Path(staging_dir)
        tasks = tuple(
            _SequenceReplayTask(
                build=str(build_path),
                tracker_spec=tracker_spec,
                split=split,
                sequence_id=sequence_id,
                frame_total=frame_counts[sequence_id],
                output_path=str(staging / f"part-{ordinal:05d}.txt"),
                ordinal=ordinal,
                output_format=output_format,
            )
            for ordinal, sequence_id in enumerate(selected)
        )
        replayed = _run_spawned_sequence_tasks(
            tasks,
            workers=worker_count,
            progress_callback=progress_callback,
        )

        sequence_paths: list[Path] = []
        for result in replayed:
            path = destination / f"{result.sequence_id}.txt"
            os.replace(result.output_path, path)
            sequence_paths.append(path)

    return ReplayResult(
        build=build_path,
        output_dir=destination,
        sequence_files=tuple(sequence_paths),
        frames=sum(result.frames for result in replayed),
        track_rows=sum(result.track_rows for result in replayed),
    )


__all__ = (
    "ReplayFrame",
    "ReplayFrameCallback",
    "ReplayProgressEvent",
    "ReplayResult",
    "iter_cached_tracks",
    "replay_build",
    "tracks_to_mot_rows",
)
