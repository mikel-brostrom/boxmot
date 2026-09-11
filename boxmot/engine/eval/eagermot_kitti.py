"""KITTI sensor replay and mask or 3D box evaluation of the EagerMOT Python tracker."""

from __future__ import annotations

import concurrent.futures
import json
import pickle
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from boxmot import EagerMot, __version__
from boxmot.datasets.readers.boxes3d import TrackingLabels3D, read_kitti_tracking_labels
from boxmot.datasets.sequence import MultimodalSequence, SensorFrame
from boxmot.engine.config.datasets import load_sensor_evaluation_inputs
from boxmot.engine.eval.kitti_3d import evaluate_kitti_3d, write_kitti_3d_rows
from boxmot.engine.eval.kitti_mots_replay import (
    GroundTruthEntry,
    evaluate_kitti_mots,
    kitti_mots_annotations,
)
from boxmot.engine.eval.mots_io import prepare_mots_tracks, tracks_to_mots_rows, write_mots_rows
from boxmot.engine.eval.output import increment_path
from boxmot.engine.eval.replay import (
    ReplayProgressCallback,
    ReplayProgressEvent,
    ReplayProgressStatus,
    _drain_progress_queue,
    _emit_worker_progress,
    _initialize_replay_worker,
    _publish_progress,
)
from boxmot.engine.eval.results import ValidationResult
from boxmot.pipelines import PipelineResult
from boxmot.structures import Boxes, Boxes3D, Frame, MaskBatch, MultimodalTracks, Tracks, Tracks3D
from boxmot.trackers.common.motion.kalman_filters.noise import KALMAN_NOISE_OPTIONS
from boxmot.utils import logger as LOGGER

if TYPE_CHECKING:
    from boxmot.datasets.sensor_cache import SensorReplaySequence
    from boxmot.engine.eval.session import ReplaySession
    from boxmot.engine.eval.visualization import ReplayVisualization

_KITTI_SHARED = {
    **dict.fromkeys(KALMAN_NOISE_OPTIONS, 1.0),
    "det_thresh_3d": 0.0,
    "max_age": 3,
    "max_age_2d": 3,
    "fusion_iou_threshold": 0.01,
    "iou_threshold": 0.3,
    "first_matching_method": "dist_2d_full",
    "iou_3d_threshold": 0.01,
    "is_angular": False,
    "per_class": False,
    "asso_func": "iou",
}
KITTI_PROFILES = {
    1: {**_KITTI_SHARED, "det_thresh": 0.0, "min_hits": 1, "distance_threshold": 3.5},
    2: {**_KITTI_SHARED, "det_thresh": 0.9, "min_hits": 2, "distance_threshold": 0.3},
}
KITTI_CLASSES = {1: "car", 2: "pedestrian"}


def load_kitti_profiles(path: Path | None = None) -> dict[int, dict[str, Any]]:
    """Load car/pedestrian overrides, validating every option before replay."""
    profiles = {class_id: dict(profile) for class_id, profile in KITTI_PROFILES.items()}
    if path is None:
        return profiles
    try:
        values = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid EagerMOT class configuration in {path}: {exc}") from exc
    if not isinstance(values, dict) or set(values) != set(KITTI_CLASSES.values()):
        raise ValueError("EagerMOT class configuration must contain exactly 'car' and 'pedestrian' mappings.")
    for class_id, name in KITTI_CLASSES.items():
        overrides = values[name]
        if not isinstance(overrides, dict):
            raise ValueError(f"EagerMOT {name} configuration must be a mapping of tracker options.")
        unknown = set(overrides) - set(profiles[class_id])
        if unknown:
            raise ValueError(f"Unknown EagerMOT {name} options: {', '.join(sorted(map(str, unknown)))}")
        profile = {**profiles[class_id], **overrides}
        if profile["per_class"] is not False:
            raise ValueError("EagerMOT class profiles require per_class=false; replay already separates classes.")
        try:
            EagerMot(**profile)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid EagerMOT {name} configuration: {exc}") from exc
        profiles[class_id] = profile
    return profiles


def write_kitti_profiles(path: Path, profiles: dict[int, dict[str, Any]]) -> None:
    """Persist complete class profiles in the format accepted by evaluation."""
    path.write_text(
        yaml.safe_dump({name: profiles[class_id] for class_id, name in KITTI_CLASSES.items()}, sort_keys=False),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class KittiReplayInputs:
    """Own indexed or mapped sensor inputs reused across independent trials."""

    sequences: dict[str, MultimodalSequence | SensorReplaySequence]
    annotations: dict[str, list[GroundTruthEntry]]
    dataset_root: Path
    manifest: dict[str, Any]
    fps: float = 10.0
    cache_inputs: bool = False
    ground_truth_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    annotations_3d: dict[str, TrackingLabels3D] = field(default_factory=dict)

    def close(self) -> None:
        """Release mapped inputs after the evaluation or entire tuning study."""
        for sequence in self.sequences.values():
            close = getattr(sequence, "close", None)
            if close is not None:
                close()


def _track_frame(frame: SensorFrame, trackers: dict[int, EagerMot]) -> MultimodalTracks:
    """Merge class trackers while preserving independent image and spatial rows."""
    outputs: list[Tracks] = []
    spatial_outputs: list[Tracks3D] = []
    for class_id, tracker in trackers.items():
        indices = torch.nonzero(
            (frame.detections.class_ids == class_id) & (frame.detections.scores > tracker.det_thresh),
            as_tuple=False,
        ).flatten()
        # The released KITTI loader visits image observations in ascending score order.
        order = torch.argsort(frame.detections.scores[indices], stable=True)
        indices = indices[order]
        spatial_indices = torch.nonzero(
            (frame.detections_3d.class_ids == class_id) & (frame.detections_3d.scores > tracker.det_thresh_3d),
            as_tuple=False,
        ).flatten()
        result = tracker.update(
            frame.detections.select(indices),
            detections_3d=frame.detections_3d.select(spatial_indices),
            camera=frame.camera,
        )
        image = result.image_tracks
        outputs.append(
            Tracks(
                geometry=image.geometry,
                track_ids=2 * image.track_ids + class_id - 1,
                scores=image.scores,
                class_ids=image.class_ids,
                detection_indices=indices[image.detection_indices],
                sample_id=image.sample_id,
                masks=image.masks,
            )
        )
        spatial = result.spatial_tracks
        original_indices = spatial.detection_indices.clone()
        observed = original_indices >= 0
        original_indices[observed] = spatial_indices[original_indices[observed]]
        spatial_outputs.append(
            Tracks3D(
                geometry=spatial.geometry,
                track_ids=2 * spatial.track_ids + class_id - 1,
                scores=spatial.scores,
                class_ids=spatial.class_ids,
                detection_indices=original_indices,
                sample_id=spatial.sample_id,
            )
        )
    if any(output.masks is None for output in outputs):
        raise ValueError("KITTI segmentation evaluation requires TrackR-CNN masks on image tracks.")
    image_tracks = Tracks(
        geometry=Boxes(torch.cat([output.geometry.values for output in outputs])),
        track_ids=torch.cat([output.track_ids for output in outputs]),
        scores=torch.cat([output.scores for output in outputs]),
        class_ids=torch.cat([output.class_ids for output in outputs]),
        detection_indices=torch.cat([output.detection_indices for output in outputs]),
        sample_id=frame.detections.sample_id,
        masks=MaskBatch(torch.cat([output.masks.values for output in outputs])),
    )
    spatial_tracks = Tracks3D(
        geometry=Boxes3D(torch.cat([output.geometry.values for output in spatial_outputs])),
        track_ids=torch.cat([output.track_ids for output in spatial_outputs]),
        scores=torch.cat([output.scores for output in spatial_outputs]),
        class_ids=torch.cat([output.class_ids for output in spatial_outputs]),
        detection_indices=torch.cat([output.detection_indices for output in spatial_outputs]),
        sample_id=frame.detections.sample_id,
    )
    return MultimodalTracks(image_tracks, spatial_tracks)


def prepare_eagermot_kitti(args: Any) -> KittiReplayInputs:
    """Validate alignment and index sensor inputs once for evaluation or tuning."""
    eval_3d = bool(getattr(args, "eval_3d", False))
    dataset = load_sensor_evaluation_inputs(
        args.dataset,
        split=args.split,
        sequence_names=tuple(args.sequence_names),
        eval_3d=eval_3d,
        calibrate_kf=bool(getattr(args, "calibrate_kf", False)),
    )
    sequences: dict[str, MultimodalSequence | SensorReplaySequence] = {}
    annotations: dict[str, list[GroundTruthEntry]] = {}
    annotations_3d: dict[str, TrackingLabels3D] = {}
    cache_inputs = bool(getattr(args, "cache_inputs", False))
    try:
        for paths in dataset.sequences:
            name = paths.sequence_id
            if cache_inputs:
                from boxmot.datasets.sensor_cache import open_sensor_sequence, prepare_sensor_sequence

                cache_path = prepare_sensor_sequence(
                    dataset, name, load_images=bool(getattr(args, "show", False) or getattr(args, "save", False))
                )
                sequence = open_sensor_sequence(cache_path)
            else:
                sequence = MultimodalSequence(paths, classes=dataset.classes, fps=dataset.fps, split=dataset.split)
            sequences[name] = sequence
            if eval_3d:
                ground_truth = paths.modalities["ground_truth_3d"]
                labels = (
                    sequence.ground_truth_3d()
                    if cache_inputs
                    else read_kitti_tracking_labels(
                        ground_truth.paths[0],
                        frame_count=len(sequence),
                        classes=dataset.classes,
                        options=ground_truth.options,
                    )
                )
                if labels is None:
                    raise ValueError(f"3D evaluation requires ground_truth_3d for sequence {name!r}.")
                annotations_3d[name] = labels
            else:
                ground_truth = paths.modalities["ground_truth"].paths[0]
                annotations[name] = kitti_mots_annotations(
                    name, sequence.frame_paths, sequence.image_size, ground_truth
                )
    except BaseException:
        for sequence in sequences.values():
            close = getattr(sequence, "close", None)
            if close is not None:
                close()
        raise

    manifest = {
        "boxmot_version": __version__,
        "tracker": "eagermot",
        "evaluation": (
            "3D volumetric IoU HOTA, CLEAR, and Identity metrics"
            if eval_3d
            else "KITTI MOTS; mask IoU HOTA, CLEAR, and Identity metrics"
        ),
        "eval_3d": eval_3d,
        "split": dataset.split,
        "fps": dataset.fps,
        "cache_inputs": cache_inputs,
        "sequences": {name: len(sequence) for name, sequence in sequences.items()},
        "missing_3d_frames": {name: sequence.missing_3d_frames for name, sequence in sequences.items()},
        "dataset_config": str(dataset.config_path),
        "dataset_id": dataset.id,
        "sequence_inputs": {
            paths.sequence_id: {
                role: {"format": value.format, "paths": [str(path) for path in value.paths], "options": value.options}
                for role, value in paths.modalities.items()
            }
            for paths in dataset.sequences
        },
        "score_transforms": {
            item.sequence_id: item.modalities["detections_3d"].options.get("score_transform", "identity")
            for item in dataset.sequences
        },
        "limitations": [
            "Detector checkpoint training provenance is not independently verified.",
            (
                "Custom 3D protocol; no official KITTI difficulty or DontCare suppression."
                if eval_3d
                else "This evaluates segmentation tracking, not 3D boxes or published EagerMOT benchmark parity."
            ),
            "Full rigid poses transform centers; box orientation remains yaw-only.",
        ],
    }
    ground_truth_options = {
        paths.sequence_id: {
            "ignore_ids": tuple(paths.modalities["ground_truth"].options.get("ignore_ids", ())),
            "ignore_class_ids": tuple(
                value["id"] for value in dataset.classes.values() if value["evaluation"] == "ignore"
            ),
        }
        for paths in dataset.sequences
        if not eval_3d
    }
    return KittiReplayInputs(
        sequences, annotations, dataset.root, manifest, dataset.fps, cache_inputs, ground_truth_options, annotations_3d
    )


def _visualize_frame(
    visualization: ReplayVisualization,
    frame: SensorFrame,
    tracks: Tracks,
    sequence: MultimodalSequence | SensorReplaySequence,
    split: str,
    *,
    spatial_tracks: Tracks3D | None = None,
) -> None:
    """Decode RGB only for requested visualization of the evaluated track masks."""
    from boxmot.datasets import DatasetSample
    from boxmot.datasets.readers.images import read_rgb_chw_uint8
    from boxmot.engine.eval.replay import ReplayFrame

    image_path = sequence.frame_paths[frame.frame_index]
    image = Frame(
        image=(
            sequence.read_image(frame.frame_index)
            if callable(getattr(sequence, "read_image", None))
            else read_rgb_chw_uint8(image_path.as_uri(), image_path.parent)
        ),
        sample_id=frame.detections.sample_id,
        sequence_id=sequence.sequence_id,
        frame_index=frame.frame_index,
        timestamp_s=frame.timestamp_s,
        source_uri=image_path.as_uri(),
    )
    if image.image_size != frame.image_size:
        raise ValueError(f"KITTI visualization image dimensions changed: {image_path}")
    sample = DatasetSample(
        sample_id=image.sample_id,
        split=split,
        sequence_id=sequence.sequence_id,
        frame_index=frame.frame_index,
        timestamp_s=image.timestamp_s,
        image_size=frame.image_size,
        image_ref=image.source_uri,
        frame=image,
        detections=frame.detections,
    )
    replayed = ReplayFrame(sample, PipelineResult(frame.detections, tracks))
    if spatial_tracks is None:
        visualization(replayed)
    else:
        visualization(replayed, spatial_tracks=spatial_tracks, camera=frame.camera)


@dataclass(frozen=True)
class _KittiSequenceTask:
    """Send indexed inputs and plain replay options to one sequence worker."""

    name: str
    sequence: MultimodalSequence | SensorReplaySequence
    profiles: dict[int, dict[str, Any]]
    output: Path
    split: str
    ordinal: int
    save: bool = False
    show_3d: bool = False
    run_id: str | None = None
    report_progress: bool = True
    eval_3d: bool = False


@dataclass(frozen=True)
class _KittiSequenceResult:
    """Return output counts and video artifacts without transferring track tensors."""

    name: str
    ordinal: int
    frames: int
    track_rows: int
    videos: tuple[Path, ...]


def _replay_kitti_sequence(
    task: _KittiSequenceTask,
    *,
    visualization: ReplayVisualization | None = None,
    progress_callback: ReplayProgressCallback | None = None,
) -> _KittiSequenceResult:
    """Replay one sequence with independent trackers and owned output resources."""
    completed = track_rows = 0
    videos: tuple[Path, ...] = ()
    owns_visualization = visualization is None

    def emit(status: ReplayProgressStatus, detail: str | None = None) -> None:
        """Use a stable sequence ordinal for both worker and in-process progress."""
        if progress_callback is not None:
            progress_callback(
                ReplayProgressEvent(task.name, status, completed, len(task.sequence), track_rows, detail, task.ordinal)
            )

    emit("running")
    try:
        with torch.inference_mode(), ExitStack() as stack:
            if visualization is None and task.save:
                from boxmot.engine.eval.visualization import ReplayVisualization

                visualization = stack.enter_context(
                    ReplayVisualization(
                        task.output, show=False, save=True, class_names=KITTI_CLASSES, video_fps=task.sequence.fps
                    )
                )
            trackers = {class_id: EagerMot(**profile) for class_id, profile in task.profiles.items()}
            LOGGER.info("EagerMOT %s: tracking %s frames", task.name, len(task.sequence))
            prediction_dir = task.output / ("kitti_3d" if task.eval_3d else "mots")
            with (prediction_dir / f"{task.name}.txt").open("x", encoding="utf-8") as handle:
                for frame in task.sequence:
                    tracks = _track_frame(frame, trackers)
                    if not task.eval_3d or visualization is not None:
                        prepared = prepare_mots_tracks(
                            PipelineResult(frame.detections, tracks.image_tracks), frame.image_size
                        )
                    if task.eval_3d:
                        track_rows += write_kitti_3d_rows(
                            handle, tracks.spatial_tracks, frame.frame_index, frame.camera
                        )
                    else:
                        rows = tracks_to_mots_rows(prepared, frame.frame_index)
                        write_mots_rows(handle, rows)
                        track_rows += len(rows)
                    if visualization is not None:
                        _visualize_frame(
                            visualization,
                            frame,
                            prepared,
                            task.sequence,
                            task.split,
                            spatial_tracks=tracks.spatial_tracks if task.show_3d else None,
                        )
                    completed += 1
                    emit("running")
                    if completed % 200 == 0:
                        LOGGER.info("EagerMOT %s: %s/%s frames", task.name, completed, len(task.sequence))
            if owns_visualization and visualization is not None:
                videos = visualization.video_paths
    except BaseException as exc:
        emit("failed", str(exc) or type(exc).__name__)
        raise
    emit("completed")
    return _KittiSequenceResult(task.name, task.ordinal, completed, track_rows, videos)


def _initialize_kitti_worker(progress_queue: Any | None, log_level: int, logs_disabled: bool) -> None:
    """Use one CPU thread per child and let the parent own logging and progress."""
    from boxmot.utils import configure_logging

    _initialize_replay_worker(progress_queue)
    configure_logging(main_only=False)
    LOGGER.setLevel(log_level)
    LOGGER.disabled = logs_disabled
    torch.set_num_threads(1)


def _replay_kitti_sequence_task(payload: bytes) -> _KittiSequenceResult:
    """Restore a private copy of parent-owned indexes in the sequence worker."""
    from boxmot.engine.eval import replay

    task = pickle.loads(payload)
    replay._WORKER_RUN_ID = task.run_id
    replay._WORKER_REPORT_PROGRESS = task.report_progress
    torch.set_num_threads(1)
    try:
        return _replay_kitti_sequence(task, progress_callback=_emit_worker_progress)
    finally:
        close = getattr(task.sequence, "close", None)
        if close is not None:
            close()


def _shutdown_kitti_pool(executor: concurrent.futures.ProcessPoolExecutor, *, interrupted: bool) -> None:
    """Stop interrupted CPU work before joining the pool and its queue threads."""
    if interrupted:
        # Python 3.11 has no public ProcessPoolExecutor.terminate_workers().
        # Snapshot its child handles before shutdown clears them, and bound
        # both graceful termination and a final kill of unresponsive children.
        processes = tuple(executor._processes.values())
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 2.0
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        deadline = time.monotonic() + 2.0
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
    executor.shutdown(wait=True, cancel_futures=True)


def _run_parallel_sequences(
    tasks: tuple[_KittiSequenceTask, ...],
    workers: int,
    progress_callback: ReplayProgressCallback | None,
    *,
    _pool: tuple[concurrent.futures.ProcessPoolExecutor, Any, str] | None = None,
) -> tuple[_KittiSequenceResult, ...]:
    """Run real spawned processes, delivering progress only in the caller process."""
    context = get_context("spawn")
    progress_queue = _pool[1] if _pool is not None else context.Queue() if progress_callback is not None else None
    run_id = None if _pool is None else _pool[2]
    latest: dict[int, ReplayProgressEvent] = {}
    results: dict[int, _KittiSequenceResult] = {}
    failures: list[tuple[_KittiSequenceTask, BaseException]] = []
    executor = None if _pool is None else _pool[0]
    interrupted = True
    futures: dict[concurrent.futures.Future[_KittiSequenceResult], _KittiSequenceTask] = {}
    try:
        for task in tasks:
            _publish_progress(
                ReplayProgressEvent(task.name, "queued", 0, len(task.sequence), 0, None, task.ordinal, run_id),
                progress_callback,
                latest,
            )
        if executor is None:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_kitti_worker,
                initargs=(progress_queue, LOGGER.getEffectiveLevel(), LOGGER.disabled),
            )
        for task in tasks:
            # Ordinary pickle keeps calibration tensors private to each worker.
            # Passing tensors directly through multiprocessing invokes PyTorch
            # shared-memory reducers and adds an unnecessary manager process.
            payload = pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)
            futures[executor.submit(_replay_kitti_sequence_task, payload)] = task
        pending = set(futures)
        while pending:
            done, pending = concurrent.futures.wait(
                pending, timeout=0.1, return_when=concurrent.futures.FIRST_COMPLETED
            )
            _drain_progress_queue(progress_queue, progress_callback, latest, run_id)
            for future in done:
                task = futures[future]
                try:
                    result = future.result()
                    if (
                        result.name != task.name
                        or result.ordinal != task.ordinal
                        or result.frames != len(task.sequence)
                    ):
                        raise RuntimeError(f"EagerMOT worker returned mismatched results for {task.name!r}.")
                except BaseException as exc:
                    failures.append((task, exc))
                    previous = latest.get(task.ordinal)
                    if previous is None or previous.status != "failed":
                        _publish_progress(
                            ReplayProgressEvent(
                                task.name,
                                "failed",
                                0 if previous is None else previous.completed,
                                len(task.sequence),
                                0 if previous is None else previous.track_rows,
                                str(exc) or type(exc).__name__,
                                task.ordinal,
                                run_id,
                            ),
                            progress_callback,
                            latest,
                        )
                    continue
                results[task.ordinal] = result
                _publish_progress(
                    ReplayProgressEvent(
                        task.name,
                        "completed",
                        result.frames,
                        result.frames,
                        result.track_rows,
                        None,
                        task.ordinal,
                        run_id,
                    ),
                    progress_callback,
                    latest,
                )
            if failures:
                break
        interrupted = bool(failures)
    finally:
        for future in futures:
            future.cancel()
        try:
            if executor is not None and _pool is None:
                _shutdown_kitti_pool(executor, interrupted=interrupted)
        finally:
            try:
                # Termination can leave a partial message or a held queue lock.
                # Never read this observational channel after killing writers.
                if not interrupted:
                    _drain_progress_queue(progress_queue, progress_callback, latest, run_id)
            finally:
                if progress_queue is not None and _pool is None:
                    progress_queue.close()
                    progress_queue.join_thread()
    if failures:
        failures.sort(key=lambda item: item[0].ordinal)
        names = ", ".join(task.name for task, _ in failures)
        raise RuntimeError(f"EagerMOT tracking failed for sequence(s): {names}.") from failures[0][1]
    return tuple(results[task.ordinal] for task in tasks)


def _replay(
    inputs: KittiReplayInputs,
    profiles: dict[int, dict[str, Any]],
    output: Path,
    *,
    sequence_workers: int,
    show: bool = False,
    save: bool = False,
    show_3d: bool = False,
    progress_callback: ReplayProgressCallback | None = None,
    on_evaluate: Callable[[], None] | None = None,
    replay_session: ReplaySession | None = None,
) -> tuple[dict[str, dict[str, Any]], tuple[Path, ...]]:
    """Replay isolated sequences before scoring the selected tracking geometry."""
    eval_3d = bool(inputs.manifest.get("eval_3d", False))
    prediction_dir = output / ("kitti_3d" if eval_3d else "mots")
    prediction_dir.mkdir()
    tasks = tuple(
        _KittiSequenceTask(
            name, sequence, profiles, output, inputs.manifest["split"], ordinal, save, show_3d, eval_3d=eval_3d
        )
        for ordinal, (name, sequence) in enumerate(inputs.sequences.items())
    )
    if replay_session is not None and sequence_workers > 1 and not show:
        results = replay_session.run_sensor(tasks, progress_callback=progress_callback)
        videos = tuple(path for result in results for path in result.videos)
    elif sequence_workers > 1:
        results = _run_parallel_sequences(tasks, sequence_workers, progress_callback)
        videos = tuple(path for result in results for path in result.videos)
    else:
        latest: dict[int, ReplayProgressEvent] = {}

        def publish(event: ReplayProgressEvent) -> None:
            _publish_progress(event, progress_callback, latest)

        with ExitStack() as stack:
            visualization = None
            if show or save:
                from boxmot.engine.eval.visualization import ReplayVisualization

                # Keep preview dismissal (q/Esc) across sequence boundaries.
                visualization = stack.enter_context(
                    ReplayVisualization(output, show=show, save=save, class_names=KITTI_CLASSES, video_fps=inputs.fps)
                )
            for task in tasks:
                publish(ReplayProgressEvent(task.name, "queued", 0, len(task.sequence), 0, None, task.ordinal))
                _replay_kitti_sequence(task, visualization=visualization, progress_callback=publish)
            videos = () if visualization is None else visualization.video_paths
    if on_evaluate is not None:
        on_evaluate()
    if eval_3d:
        return evaluate_kitti_3d(prediction_dir, output, inputs.annotations_3d, inputs.manifest["sequences"]), videos
    cache_options = {"cached_ground_truth": inputs.sequences} if inputs.cache_inputs else {}
    if inputs.ground_truth_options:
        cache_options["ground_truth_options"] = inputs.ground_truth_options
    metrics = evaluate_kitti_mots(prediction_dir, output, inputs.dataset_root, inputs.annotations, **cache_options)
    return metrics, videos


def evaluate_eagermot_kitti(
    inputs: KittiReplayInputs,
    profiles: dict[int, dict[str, Any]],
    output: Path,
    *,
    sequence_workers: int | None = None,
    show: bool = False,
    save: bool = False,
    show_3d: bool = False,
    progress_callback: ReplayProgressCallback | None = None,
    on_evaluate: Callable[[], None] | None = None,
    replay_session: ReplaySession | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate profiles on CPU into a new directory, restoring thread settings."""
    from boxmot.engine.config.runtime import resolve_sequence_workers

    workers = resolve_sequence_workers(len(inputs.sequences), sequence_workers)
    if show:
        workers = min(1, len(inputs.sequences))
    if replay_session is not None and replay_session.workers != workers:
        raise ValueError("ReplaySession workers must match the resolved sensor sequence workers.")
    if set(profiles) != set(KITTI_CLASSES):
        raise ValueError("EagerMOT KITTI replay requires profiles for both car and pedestrian.")
    if show_3d and not (show or save):
        raise ValueError("--show-3d requires --show or --save.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **inputs.manifest,
        "status": "running",
        "tracker_profiles": profiles,
        "fps": inputs.fps,
        "sequence_workers": workers,
        "visualization": {"show": show, "save": save, "show_3d": show_3d, "video_fps": inputs.fps},
    }
    # Calibration may already have created kf-tuning; reserve the replay itself
    # exclusively so an existing run and its predictions cannot be overwritten.
    with (output / "run.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode():
            results, videos = _replay(
                inputs,
                profiles,
                output,
                sequence_workers=workers,
                show=show,
                save=save,
                show_3d=show_3d,
                progress_callback=progress_callback,
                on_evaluate=on_evaluate,
                replay_session=replay_session,
            )
            if show or save:
                manifest["videos"] = [str(path.relative_to(output)) for path in videos]
        manifest["status"] = "complete"
        return results
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        raise
    finally:
        torch.set_num_threads(previous_threads)
        (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def run_eagermot_kitti(
    args: Any, *, pipeline: Any | None = None, show_progress: bool | None = None
) -> ValidationResult:
    """Replay saved sensor predictions and return class-average tracking metrics."""
    from boxmot.engine.config.runtime import resolve_sequence_workers

    if pipeline is not None:
        pipeline.update("Loading KITTI sensor predictions and ground truth…")
    profiles = load_kitti_profiles(getattr(args, "class_config", None))
    inputs = prepare_eagermot_kitti(args)
    try:
        args.sequence_workers = resolve_sequence_workers(len(inputs.sequences), getattr(args, "sequence_workers", None))
        if getattr(args, "show", False):
            args.sequence_workers = min(1, len(inputs.sequences))
        output = increment_path(Path(args.project).expanduser().resolve() / inputs.manifest["split"])
        calibration = None
        if getattr(args, "calibrate_kf", False):
            from boxmot.engine.calibration.kalman_sensor import calibrate_sensor_kalman

            dataset = load_sensor_evaluation_inputs(
                args.dataset,
                split=args.split,
                sequence_names=tuple(args.sequence_names),
                eval_3d=bool(getattr(args, "eval_3d", False)),
                calibrate_kf=True,
            )
            calibration = calibrate_sensor_kalman(
                dataset,
                profiles,
                output_dir=output,
                progress=pipeline.update if pipeline is not None else None,
                **({"cached_sequences": inputs.sequences} if inputs.cache_inputs else {}),
            )
            profiles = load_kitti_profiles(calibration.config_path)
            inputs.manifest["kf_calibration"] = {
                "config_path": str(calibration.config_path),
                "report_path": str(calibration.report_path),
            }
            LOGGER.info(calibration.description)
        if getattr(args, "class_config", None) is not None:
            inputs.manifest["class_config"] = str(Path(args.class_config).expanduser().resolve())
        args.dataset_id = inputs.manifest["dataset_id"]
        args.seq_info = args.sequence_frame_counts = inputs.manifest["sequences"]
        args.tracker_class_names = tuple(KITTI_CLASSES.items())
        if pipeline is not None:
            from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter, _refresh_eval_pipeline_intro

            _refresh_eval_pipeline_intro(pipeline.workflow, args)
            pipeline.advance("Replaying saved sensor detections through the tracker…")
        started = time.perf_counter()
        tracking_finished = started
        presenter = None
        with ExitStack() as contexts:
            if pipeline is not None and show_progress is not False:
                presenter = contexts.enter_context(EvalSequenceProgressPresenter(pipeline.callback(), args.seq_info))

            def computing_metrics() -> None:
                """Finish sequence progress before entering the evaluation stage."""
                nonlocal tracking_finished
                tracking_finished = time.perf_counter()
                if pipeline is not None:
                    if presenter is not None:
                        presenter.flush()
                        pipeline.store_step_info(presenter.renderable)
                        contexts.close()
                    geometry = "3D box" if inputs.manifest.get("eval_3d", False) else "mask"
                    pipeline.advance(f"Computing KITTI {geometry} evaluation metrics…")

            metrics = evaluate_eagermot_kitti(
                inputs,
                profiles,
                output,
                sequence_workers=args.sequence_workers,
                show=bool(getattr(args, "show", False)),
                save=bool(getattr(args, "save", False)),
                show_3d=bool(getattr(args, "show_3d", False)),
                progress_callback=presenter,
                on_evaluate=computing_metrics,
            )
        total_ms = (time.perf_counter() - started) * 1000
        track_ms = (tracking_finished - started) * 1000
        frames = sum(args.seq_info.values())
        timings = {
            "frames": frames,
            "totals_ms": {"track": track_ms, "eval": total_ms - track_ms, "total": total_ms},
            "avg_ms": {"track": track_ms / frames if frames else 0.0, "total": total_ms / frames if frames else 0.0},
            "fps": 1000 * frames / total_ms if total_ms else 0.0,
        }
        args.exp_dir = output
        manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
        args.video_paths = tuple(output / path for path in manifest.get("videos", ()))
        result = ValidationResult(
            benchmark=str(inputs.manifest["dataset_id"]),
            raw=metrics,
            summary_label="cls_comb_cls_av",
            summary=dict(metrics["cls_comb_cls_av"]),
            exp_dir=output,
            timings=timings,
            args=args,
            workflow_rendered=pipeline is not None,
        )
        if calibration is not None:
            calibration.record_final(result)
        return result
    finally:
        close = getattr(inputs, "close", None)
        if close is not None:
            close()


__all__ = (
    "evaluate_eagermot_kitti",
    "load_kitti_profiles",
    "prepare_eagermot_kitti",
    "run_eagermot_kitti",
    "write_kitti_profiles",
)
