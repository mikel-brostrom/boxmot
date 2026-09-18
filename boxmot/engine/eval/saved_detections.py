"""Evaluate image trackers on configured saved boxes without detector inference."""

from __future__ import annotations

import concurrent.futures
import json
import os
import time
from contextlib import ExitStack, closing
from dataclasses import asdict, dataclass, replace
from multiprocessing import get_context
from multiprocessing.util import Finalize
from pathlib import Path
from typing import Any

import torch

from boxmot import __version__, create_tracker
from boxmot.datasets.cached import DatasetSample
from boxmot.datasets.inputs import DatasetInputs, SequenceInputs
from boxmot.datasets.readers.boxes2d import read_kitti_tracking_labels_2d
from boxmot.datasets.readers.images import read_rgb_chw_uint8
from boxmot.datasets.sequence import DetectionSequence
from boxmot.engine.config.datasets import load_saved_2d_evaluation_inputs
from boxmot.engine.config.runtime import resolve_sequence_workers
from boxmot.engine.config.trackers import resolve_tracker_options
from boxmot.engine.eval.kitti_tracking import run_kitti_tracking_metrics
from boxmot.engine.eval.output import increment_path
from boxmot.engine.eval.replay import (
    ReplayFrame,
    ReplayProgressCallback,
    ReplayProgressEvent,
    _drain_progress_queue,
    _emit_worker_progress,
    _initialize_replay_worker,
    _publish_progress,
    _write_rows,
    tracks_to_mot_rows,
)
from boxmot.engine.eval.results import ValidationResult, _select_plot_metrics_data
from boxmot.engine.eval.visualization import ReplayVisualization
from boxmot.engine.eval.workers import shutdown_sequence_pool
from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter, EvalWorkflowReporter
from boxmot.pipelines import TrackingPipeline
from boxmot.reid import create_reid_encoder
from boxmot.reid.config import resolve_reid_spec
from boxmot.reid.specs import ReIDEncoderSpec
from boxmot.structures import Frame
from boxmot.trackers import TrackerSpec
from boxmot.utils import logger


@dataclass(frozen=True)
class _SavedSequenceTask:
    """Pass source paths and cache indexes, never mapped arrays or live models."""

    source: SequenceInputs
    dataset: DatasetInputs
    output: Path
    cache_path: Path | None
    load_images: bool
    save: bool
    class_names: dict[int, str]
    ordinal: int
    frame_count: int


@dataclass(frozen=True)
class _SavedSequenceResult:
    """Collect worker results in the authored sequence order."""

    name: str
    ordinal: int
    frames: int
    videos: tuple[Path, ...]
    process_id: int


_WORKER_TRACKING: TrackingPipeline | None = None


def _create_saved_encoder(spec: ReIDEncoderSpec | None, cache_root: Path | None) -> Any:
    """Build one private encoder per process, optionally backed by saved embeddings."""
    if spec is None:
        return None
    encoder = create_reid_encoder(spec)
    if encoder.requirements.masks:
        raise ValueError("The selected ReID encoder requires masks; this dataset declares only 2D boxes.")
    if cache_root is not None:
        from boxmot.engine.eval.saved_input_cache import CachedAppearanceEncoder

        encoder = CachedAppearanceEncoder(encoder, spec, cache_root)
    return encoder


def _initialize_saved_worker(
    progress_queue: Any | None,
    spec: TrackerSpec,
    encoder_spec: ReIDEncoderSpec | None,
    cache_root: Path | None,
) -> None:
    """Initialize isolated tracking state once per spawned process."""
    global _WORKER_TRACKING
    _initialize_replay_worker(progress_queue)
    torch.set_num_threads(1)
    resources = ExitStack()
    Finalize(None, resources.close, exitpriority=10)
    tracker = create_tracker(spec)
    if callable(getattr(tracker, "close", None)):
        resources.callback(tracker.close)
    encoder = _create_saved_encoder(encoder_spec, cache_root)
    if callable(getattr(encoder, "close", None)):
        resources.callback(encoder.close)
    _WORKER_TRACKING = TrackingPipeline(detector=None, tracker=tracker, reid=encoder)


def _replay_saved_sequence(
    task: _SavedSequenceTask,
    sequence: Any,
    tracking: TrackingPipeline,
    visualizer: ReplayVisualization,
    progress_callback: ReplayProgressCallback | None,
) -> _SavedSequenceResult:
    """Track one sequence identically in serial and spawned execution."""
    name = task.source.sequence_id
    completed = 0

    def emit(status: str, detail: str | None = None) -> None:
        if progress_callback is not None:
            progress_callback(ReplayProgressEvent(name, status, completed, len(sequence), 0, detail, task.ordinal))

    try:
        tracking.reset()
        emit("running")
        with (task.output / f"{name}.txt").open("w", encoding="utf-8") as handle:
            for index, sample in enumerate(sequence):
                path = sequence.frame_paths[index]
                if task.load_images:
                    image = (
                        sequence.read_image(index)
                        if task.cache_path is not None and sequence.load_images
                        else read_rgb_chw_uint8(path.as_uri(), task.dataset.root)
                    )
                else:
                    image = torch.zeros((3, *sample.image_size), dtype=torch.uint8)
                frame = Frame(
                    image=image,
                    sample_id=sample.detections.sample_id,
                    sequence_id=name,
                    frame_index=sample.frame_index,
                    timestamp_s=sample.timestamp_s,
                    source_uri=path.as_uri(),
                )
                result = tracking.step_detections(frame, sample.detections)
                _write_rows(handle, tracks_to_mot_rows(result.tracks, sample.frame_index))
                completed += 1
                if visualizer.show or task.save:
                    visualizer(
                        ReplayFrame(
                            DatasetSample(
                                frame.sample_id,
                                task.dataset.split,
                                name,
                                sample.frame_index,
                                sample.timestamp_s,
                                sample.image_size,
                                path.as_uri(),
                                frame,
                                result.detections,
                            ),
                            result,
                        )
                    )
                emit("running")
    except BaseException as exc:
        emit("failed", str(exc) or type(exc).__name__)
        raise
    emit("completed")
    return _SavedSequenceResult(name, task.ordinal, completed, visualizer.video_paths, os.getpid())


def _replay_saved_sequence_task(task: _SavedSequenceTask) -> _SavedSequenceResult:
    """Open caches in the owning process and close them before accepting another task."""
    if _WORKER_TRACKING is None:
        raise RuntimeError("Saved-detection worker has not been initialized.")
    with ExitStack() as contexts, torch.inference_mode():
        if task.cache_path is not None:
            from boxmot.datasets.sensor_cache import open_sensor_sequence

            sequence = contexts.enter_context(closing(open_sensor_sequence(task.cache_path)))
        else:
            sequence = DetectionSequence(
                task.source, classes=task.dataset.classes, fps=task.dataset.fps, split=task.dataset.split
            )
        visualizer = contexts.enter_context(
            ReplayVisualization(
                task.output, show=False, save=task.save, class_names=task.class_names, video_fps=task.dataset.fps
            )
        )
        return _replay_saved_sequence(task, sequence, _WORKER_TRACKING, visualizer, _emit_worker_progress)


def _run_saved_sequence_tasks(
    tasks: tuple[_SavedSequenceTask, ...],
    *,
    workers: int,
    spec: TrackerSpec,
    encoder_spec: ReIDEncoderSpec | None,
    cache_root: Path | None,
    progress_callback: ReplayProgressCallback | None,
) -> tuple[_SavedSequenceResult, ...]:
    """Run isolated sequences concurrently while the parent owns progress and scoring."""
    context = get_context("spawn")
    progress_queue = context.Queue() if progress_callback is not None else None
    executor = None
    latest: dict[int, ReplayProgressEvent] = {}
    results = {}
    interrupted = True
    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_saved_worker,
            initargs=(progress_queue, spec, encoder_spec, cache_root),
        )
        futures = {executor.submit(_replay_saved_sequence_task, task): task for task in tasks}
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
            )
            _drain_progress_queue(progress_queue, progress_callback, latest)
            for future in done:
                task = futures[future]
                name = task.source.sequence_id
                try:
                    result = future.result()
                    if (result.name, result.ordinal, result.frames) != (name, task.ordinal, task.frame_count):
                        raise RuntimeError(f"Saved-detection worker returned mismatched results for {name!r}.")
                except BaseException as exc:
                    previous = latest.get(task.ordinal)
                    _publish_progress(
                        ReplayProgressEvent(
                            name,
                            "failed",
                            0 if previous is None else previous.completed,
                            task.frame_count,
                            0,
                            str(exc),
                            task.ordinal,
                        ),
                        progress_callback,
                        latest,
                    )
                    raise RuntimeError(f"Saved 2D tracking failed for sequence {name!r}.") from exc
                results[task.ordinal] = result
                _publish_progress(
                    ReplayProgressEvent(name, "completed", result.frames, result.frames, 0, None, task.ordinal),
                    progress_callback,
                    latest,
                )
        interrupted = False
    finally:
        try:
            if executor is not None:
                shutdown_sequence_pool(executor, interrupted=interrupted)
        finally:
            if progress_queue is not None:
                if not interrupted:
                    _drain_progress_queue(progress_queue, progress_callback, latest)
                progress_queue.close()
                progress_queue.join_thread()
    return tuple(results[task.ordinal] for task in tasks)


def _tracking_inputs(dataset: DatasetInputs) -> DatasetInputs:
    """Keep annotations in scoring and cache only the configured tracking inputs."""
    return replace(
        dataset,
        sequences=tuple(
            replace(
                sequence,
                modalities={
                    role: modality
                    for role, modality in sequence.modalities.items()
                    if role in {"images", "detections_2d"}
                },
            )
            for sequence in dataset.sequences
        ),
    )


def run_saved_detections(args: Any, *, pipeline: Any | None = None) -> ValidationResult:
    """Replay native 2D predictions, enrich their appearance, and score KITTI tracks."""
    if getattr(args, "postprocessing", None):
        raise ValueError("--postprocessing requires an AABB perception build; saved 2D evaluation is unsupported.")
    if pipeline is not None:
        pipeline.update("Loading saved 2D predictions and KITTI ground truth…")
    dataset = load_saved_2d_evaluation_inputs(
        args.dataset,
        split=getattr(args, "split", None) or None,
        sequence_names=tuple(getattr(args, "sequence_names", ())),
        data_root=getattr(args, "data_root", None),
        experiment=getattr(args, "experiment", None),
    )
    workers = resolve_sequence_workers(len(dataset.sequences), getattr(args, "sequence_workers", None))
    if getattr(args, "show", False) and workers > 1:
        logger.warning("--show forces one sequence worker for live preview (was %s).", workers)
        workers = 1
    options = resolve_tracker_options(args, include_defaults=True, factory_options=True)
    spec = TrackerSpec(
        args.tracker,
        backend=getattr(args, "tracker_backend", "python"),
        per_class=bool(getattr(args, "per_class", False)),
        options=tuple(sorted(options.items())),
    )
    with ExitStack() as resources:
        tracker = create_tracker(spec)
        if callable(getattr(tracker, "close", None)):
            resources.callback(tracker.close)
        cache_inputs = bool(getattr(args, "cache_inputs", False))
        reference = getattr(args, "reid", None)
        if tracker.requirements.embeddings and not reference:
            raise ValueError(f"'{args.tracker}' requires appearance embeddings. Select --reid <profile>.")
        if reference and not tracker.requirements.embeddings:
            raise ValueError(f"'{args.tracker}' does not use ReID in this configuration; omit --reid.")
        encoder_spec = None
        reid_provenance = None
        if reference:
            encoder_spec, reid_provenance = resolve_reid_spec(reference)
            if getattr(args, "device", None):
                encoder_spec = replace(encoder_spec, device=str(args.device))
            reid_provenance = {**reid_provenance, "spec": asdict(encoder_spec)}
        cache_root = dataset.root / ".boxmot" / "replay_cache" / "embeddings" if cache_inputs else None
        encoder = _create_saved_encoder(encoder_spec, cache_root) if workers == 1 else None

        tracking_dataset = _tracking_inputs(dataset)
        show, save = bool(getattr(args, "show", False)), bool(getattr(args, "save", False))
        load_images = tracker.requirements.frame_pixels or encoder_spec is not None or show or save
        output = increment_path(Path(args.project).expanduser().resolve() / dataset.split, mkdir=True)
        args.exp_dir = output
        args.dataset_id = dataset.id
        args.benchmark = dataset.id
        args.split = dataset.split
        args.per_class = spec.per_class
        args.sequence_workers = workers
        args.tracker_class_names = {
            metadata["id"]: name for name, metadata in dataset.classes.items() if metadata["evaluation"] == "target"
        }
        args.remapped_class_names = list(args.tracker_class_names.values())
        args.remapped_class_ids = list(args.tracker_class_names)
        manifest = {
            "status": "running",
            "boxmot_version": __version__,
            "experiment_id": getattr(args, "experiment_id", None),
            "experiment_config": str(args.experiment) if getattr(args, "experiment", None) else None,
            "dataset_id": dataset.id,
            "dataset_config": str(dataset.config_path),
            "split": dataset.split,
            "tracker": args.tracker,
            "tracker_backend": spec.backend,
            "tracker_options": options,
            "sequence_workers": workers,
            "per_class": spec.per_class,
            "reid": reid_provenance,
            "cache_inputs": cache_inputs,
            "image_cache": {},
            "evaluation": "KITTI 2D tracking; BoxMOT built-in image-box IoU HOTA, CLEAR, and Identity metrics",
            "fps": dataset.fps,
            "inputs": {
                sequence.sequence_id: {
                    role: {
                        "format": modality.format,
                        "paths": list(map(str, modality.paths)),
                        "options": modality.options,
                    }
                    for role, modality in sequence.modalities.items()
                }
                for sequence in dataset.sequences
            },
        }
        manifest_path = output / "run.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        frames = 0
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with ExitStack() as contexts, torch.inference_mode():
                sequences = {}
                selections = {}
                tasks = []
                for ordinal, source in enumerate(dataset.sequences):
                    name = source.sequence_id
                    prepared = None
                    if cache_inputs:
                        from boxmot.datasets.sensor_cache import (
                            SensorReplayCacheStorageError,
                            open_sensor_sequence,
                            prepare_sensor_sequence,
                        )

                        try:
                            prepared = prepare_sensor_sequence(tracking_dataset, name, load_images=load_images)
                        except SensorReplayCacheStorageError:
                            if not load_images:
                                raise
                            logger.warning(
                                "Input cache: insufficient disk space for sequence %s images; "
                                "reading original images and retaining detection/ReID caching.",
                                name,
                            )
                            prepared = prepare_sensor_sequence(tracking_dataset, name, load_images=False)
                        sequence = contexts.enter_context(closing(open_sensor_sequence(prepared)))
                    else:
                        sequence = DetectionSequence(
                            source, classes=dataset.classes, fps=dataset.fps, split=dataset.split
                        )
                    sequences[name] = sequence
                    tasks.append(
                        _SavedSequenceTask(
                            source,
                            tracking_dataset,
                            output,
                            prepared,
                            load_images,
                            save,
                            args.tracker_class_names,
                            ordinal,
                            len(sequence),
                        )
                    )
                    manifest["image_cache"][name] = (
                        "cached" if cache_inputs and sequence.load_images else "source" if load_images else "unused"
                    )
                    ground_truth = source.modalities["ground_truth"].paths[0]
                    read_kitti_tracking_labels_2d(ground_truth, frame_count=len(sequence), cache_inputs=cache_inputs)
                    selections[name] = {
                        "path": str(ground_truth),
                        "frame_count": len(sequence),
                        "frames": [(index, index) for index in range(len(sequence))],
                    }
                args.seq_info = {name: len(sequence) for name, sequence in sequences.items()}
                args.evaluation_config = {"classes": dataset.classes, "kitti_gt_sequences": selections}
                manifest["sequences"] = args.seq_info
                visualizer = contexts.enter_context(
                    ReplayVisualization(
                        output, show=show, save=save, class_names=args.tracker_class_names, video_fps=dataset.fps
                    )
                )
                presenter = None
                if pipeline is not None:
                    pipeline.advance("Tracking saved 2D predictions…")
                    presenter = contexts.enter_context(
                        EvalSequenceProgressPresenter(pipeline.callback(), args.seq_info)
                    )
                if workers > 1:
                    results = _run_saved_sequence_tasks(
                        tuple(tasks),
                        workers=workers,
                        spec=spec,
                        encoder_spec=encoder_spec,
                        cache_root=cache_root,
                        progress_callback=presenter,
                    )
                    args.video_paths = tuple(path for result in results for path in result.videos)
                else:
                    tracking = TrackingPipeline(detector=None, tracker=tracker, reid=encoder)
                    results = tuple(
                        _replay_saved_sequence(
                            task, sequences[task.source.sequence_id], tracking, visualizer, presenter
                        )
                        for task in tasks
                    )
                    args.video_paths = visualizer.video_paths
                frames = sum(result.frames for result in results)
                manifest["sequence_processes"] = {result.name: result.process_id for result in results}
                if presenter is not None:
                    presenter.flush()
                    pipeline.store_step_info(presenter.renderable)
            tracking_finished = time.perf_counter()
            if pipeline is not None:
                pipeline.advance("Computing KITTI 2D tracking metrics…")
            metrics = run_kitti_tracking_metrics(args, (), output, dataset.root, seq_info=args.seq_info)
            summary_label, summary = _select_plot_metrics_data(metrics)
            total_ms = (time.perf_counter() - started) * 1000
            tracking_ms = (tracking_finished - started) * 1000
            manifest.update(status="complete", videos=[str(path.relative_to(output)) for path in args.video_paths])
            return ValidationResult(
                benchmark=dataset.id,
                raw=metrics,
                summary_label=summary_label,
                summary=dict(summary),
                exp_dir=output,
                timings={
                    "frames": frames,
                    "totals_ms": {"track": tracking_ms, "eval": total_ms - tracking_ms, "total": total_ms},
                    "avg_ms": {"track": tracking_ms / frames, "total": total_ms / frames},
                    "fps": 1000 * frames / total_ms if total_ms else 0.0,
                },
                args=args,
                workflow_rendered=pipeline is not None,
            )
        except (Exception, KeyboardInterrupt) as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            raise
        finally:
            torch.set_num_threads(previous_threads)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(args: Any) -> ValidationResult:
    """Present saved-detection evaluation through the standard evaluation panel."""
    from rich.console import Group
    from rich.text import Text

    from boxmot.engine.ui.logging import suppress_boxmot_logs

    pipeline = EvalWorkflowReporter(args).pipeline()
    with pipeline:
        with suppress_boxmot_logs(True, level="WARNING"):
            result = run_saved_detections(args, pipeline=pipeline)
        details = [
            result.renderable(include_timings=bool(getattr(args, "show_timing", False))),
            Text(f"Results: {result.exp_dir}"),
        ]
        if getattr(args, "video_paths", ()):
            details.append(Text("Saved tracking videos:\n" + "\n".join(map(str, args.video_paths))))
        pipeline.finish(Group(*details), exp_dir=result.exp_dir)
        return result


__all__ = ("main", "run_saved_detections")
