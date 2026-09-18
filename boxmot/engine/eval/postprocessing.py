"""Apply offline association and smoothing to evaluation-owned MOT results."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import shutil
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from boxmot.datasets import CachedVisionDataset, DatasetManifest
from boxmot.engine.config.postprocessing import normalize_postprocessing
from boxmot.engine.config.runtime import resolve_sequence_workers
from boxmot.engine.materialization.builds import validate_build_compatibility

if TYPE_CHECKING:
    from boxmot.engine.eval.replay import ReplayResult

_SMOOTHING_PARAMETERS = {
    "gsi": {"interval": 20, "tau": 10},
    "gbrc": {"interval": 20, "n_estimators": 115, "learning_rate": 0.065, "min_samples_split": 6},
}
_WORKER_PROGRESS_QUEUE: Any | None = None


@dataclass(frozen=True, slots=True)
class PostprocessingProgressEvent:
    """Small, pickle-safe sequence updates; counts refer to the current phase."""

    sequence_id: str
    status: Literal["queued", "running", "completed", "failed"]
    completed: int
    total: int | None
    detail: str | None
    ordinal: int
    phase_index: int


PostprocessingProgressCallback = Callable[[PostprocessingProgressEvent], None]


def _initialize_postprocessing_worker(progress_queue: Any | None) -> None:
    """Keep rendering in the parent and avoid waiting for observational events."""
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue
    if progress_queue is not None:
        progress_queue.cancel_join_thread()
    logger = logging.getLogger("boxmot")
    logger.handlers.clear()
    logger.propagate = False
    logger.disabled = True


def _emit_worker_progress(event: PostprocessingProgressEvent) -> None:
    """Send a bounded, lossy update without blocking numerical work."""
    if _WORKER_PROGRESS_QUEUE is not None:
        try:
            _WORKER_PROGRESS_QUEUE.put_nowait(event)
        except (queue.Full, BrokenPipeError, EOFError, OSError):
            pass


class _SequenceProgress:
    """Throttle within phases while preserving transitions and final counts."""

    def __init__(self, task: _PostprocessTask, callback: PostprocessingProgressCallback | None) -> None:
        self.task = task
        self.callback = callback
        self.phase: str | None = None
        self.phase_index = -1
        self.last_update = 0.0

    def __call__(self, phase: str, completed: int, total: int | None) -> None:
        if self.callback is None:
            return
        changed = phase != self.phase
        if changed:
            self.phase = phase
            self.phase_index += 1
        now = time.monotonic()
        if not changed and completed != total and now - self.last_update < 0.1:
            return
        self.last_update = now
        self.callback(
            PostprocessingProgressEvent(
                self.task.source.stem, "running", completed, total, phase, self.task.ordinal, self.phase_index
            )
        )


def _publish_progress(
    event: PostprocessingProgressEvent,
    callback: PostprocessingProgressCallback | None,
    latest: dict[int, PostprocessingProgressEvent],
) -> None:
    """Ignore stale worker events and isolate observational callback failures."""
    previous = latest.get(event.ordinal)
    if previous is not None:
        if previous.status in {"completed", "failed"} or event.phase_index < previous.phase_index:
            return
        if event == previous:
            return
    latest[event.ordinal] = event
    if callback is not None:
        try:
            callback(event)
        except Exception:
            pass


def _drain_progress(
    progress_queue: Any,
    callback: PostprocessingProgressCallback | None,
    latest: dict[int, PostprocessingProgressEvent],
) -> None:
    """Refresh running sequence bars while the process futures are unfinished."""
    if progress_queue is None:
        return
    while True:
        try:
            event = progress_queue.get_nowait()
        except (queue.Empty, EOFError, OSError):
            return
        _publish_progress(event, callback, latest)


def _terminal_progress(
    task: _PostprocessTask,
    latest: dict[int, PostprocessingProgressEvent],
    error: BaseException | None = None,
) -> PostprocessingProgressEvent:
    """Future results guarantee a terminal state even when queue events are lost."""
    previous = latest.get(task.ordinal)
    if error is not None:
        return PostprocessingProgressEvent(
            task.source.stem,
            "failed",
            previous.completed if previous else 0,
            previous.total if previous else None,
            f"{type(error).__name__}: {error}",
            task.ordinal,
            previous.phase_index if previous else 0,
        )
    return PostprocessingProgressEvent(
        task.source.stem,
        "completed",
        1,
        1,
        " → ".join(step.upper() for step in task.steps),
        task.ordinal,
        previous.phase_index + 1 if previous else 0,
    )


def validate_postprocessing_inputs(
    steps: tuple[str, ...], *, geometry: str, eval_masks: bool = False, manifest: DatasetManifest | None = None
) -> None:
    """Reject unsupported output representations or missing GTA embeddings early."""
    steps = normalize_postprocessing(steps)
    if not steps:
        return
    if geometry != "aabb" or eval_masks:
        raise ValueError("--postprocessing requires AABB box evaluation; OBB, mask, and 3D results are unsupported.")
    if manifest is not None:
        validate_build_compatibility(manifest, geometry="aabb", require_embeddings="gta" in steps)


def _validate_rows(rows: np.ndarray, *, sequence: str) -> None:
    """Require the canonical nine-column AABB replay layout before and after processing."""
    if rows.ndim != 2 or rows.shape[1] != 9 or not np.isfinite(rows).all():
        raise ValueError(f"Postprocessing {sequence!r} requires finite, nine-column AABB MOT rows.")
    if not len(rows):
        return
    identifiers = rows[:, (0, 1, 7, 8)]
    if not np.equal(identifiers, np.floor(identifiers)).all():
        raise ValueError(f"Postprocessing {sequence!r} requires integer frame, track, class, and detection IDs.")
    if (rows[:, 0] < 1).any() or (rows[:, 8] < -1).any() or (rows[:, 4:6] < 0).any():
        raise ValueError(f"Postprocessing {sequence!r} encountered invalid frame, detection, or box values.")
    identities = rows[:, (0, 1, 7)]
    if len(np.unique(identities, axis=0)) != len(rows):
        raise ValueError(f"Postprocessing {sequence!r} encountered duplicate track observations in one frame.")


def discard_previous_postprocessing(output_dir: Path) -> None:
    """Remove prior derived artifacts after a new replay replaces their inputs."""
    metadata = output_dir / "postprocessing.json"
    if not metadata.is_file():
        return
    raw_directory = output_dir / "raw"
    if raw_directory.is_dir():
        shutil.rmtree(raw_directory)
    metadata.unlink()


@dataclass(frozen=True, slots=True)
class _PostprocessTask:
    """Sequence-local work containing only paths and immutable selections."""

    source: Path
    destination: Path
    build: Path
    split: str | None
    steps: tuple[str, ...]
    ordinal: int = 0


def _process_sequence(
    task: _PostprocessTask, progress_callback: PostprocessingProgressCallback | None = None
) -> dict[str, Any]:
    """Join sequence embeddings lazily and apply the requested algorithms in order."""
    sequence = task.source.stem
    callback = progress_callback
    if callback is None and _WORKER_PROGRESS_QUEUE is not None:
        callback = _emit_worker_progress
    progress = _SequenceProgress(task, callback)
    progress("Read tracks", 0, None)
    rows = (
        np.loadtxt(task.source, delimiter=",", ndmin=2)
        if task.source.stat().st_size
        else np.empty((0, 9), dtype=np.float64)
    )
    _validate_rows(rows, sequence=sequence)
    progress("Read tracks", 1, 1)
    input_count = len(rows)
    elapsed: dict[str, float] = {}
    for step in task.steps:
        started = time.perf_counter()
        progress(f"{step.upper()} · Prepare", 0, None)
        if len(rows):
            if step == "gta":
                from boxmot.engine.eval.gta import associate_track_rows

                with closing(
                    iter(
                        CachedVisionDataset._stream_sequence(
                            task.build, sequence_id=sequence, split=task.split, load_embeddings=True
                        )
                    )
                ) as dataset:
                    rows = associate_track_rows(
                        rows,
                        dataset,
                        progress_fn=lambda phase, current, total: progress(f"GTA · {phase}", current, total),
                    )
            elif step == "gsi":
                from boxmot.postprocessing.gsi import gaussian_smooth, linear_interpolation

                parameters = _SMOOTHING_PARAMETERS[step]
                progress("GSI · Interpolate", 0, 1)
                rows = linear_interpolation(rows, parameters["interval"])
                progress("GSI · Interpolate", 1, 1)
                progress("GSI · Smooth tracks", 0, None)
                rows = gaussian_smooth(
                    rows,
                    parameters["tau"],
                    progress_fn=lambda current, total: progress("GSI · Smooth tracks", current, total),
                )
            else:
                from boxmot.postprocessing.gbrc import gradient_boosting_smooth, linear_interpolation

                parameters = _SMOOTHING_PARAMETERS[step]
                progress("GBRC · Interpolate", 0, 1)
                rows = linear_interpolation(rows, parameters["interval"])
                progress("GBRC · Interpolate", 1, 1)
                progress("GBRC · Smooth tracks", 0, None)
                rows = gradient_boosting_smooth(
                    rows,
                    progress_fn=lambda current, total: progress("GBRC · Smooth tracks", current, total),
                    **{key: value for key, value in parameters.items() if key != "interval"},
                )
        elapsed[step] = (time.perf_counter() - started) * 1000
        _validate_rows(rows, sequence=sequence)
    progress("Write results", 0, 1)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    np.savetxt(task.destination, rows, delimiter=",", fmt=["%d", "%d", *(["%.17g"] * 5), "%d", "%d"])
    result = {
        "sequence": sequence,
        "input_rows": input_count,
        "output_rows": len(rows),
        "timings_ms": elapsed,
        "raw_sha256": hashlib.sha256(task.source.read_bytes()).hexdigest(),
        "processed_sha256": hashlib.sha256(task.destination.read_bytes()).hexdigest(),
    }
    progress("Write results", 1, 1)
    return result


def _process_parallel(
    tasks: list[_PostprocessTask],
    *,
    workers: int,
    callback: PostprocessingProgressCallback | None,
    latest: dict[int, PostprocessingProgressEvent],
) -> list[dict[str, Any]]:
    """Drain progress while waiting; future results own reliable terminal states."""
    context = get_context("spawn")
    progress_queue = context.Queue(maxsize=max(32, workers * 8)) if callback is not None else None
    executor = None
    futures = {}
    results = []
    failures = []
    try:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_postprocessing_worker,
            initargs=(progress_queue,),
        )
        futures = {executor.submit(_process_sequence, task): task for task in tasks}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            _drain_progress(progress_queue, callback, latest)
            for future in done:
                task = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failures.append((task, exc))
                    _publish_progress(_terminal_progress(task, latest, exc), callback, latest)
                else:
                    _publish_progress(_terminal_progress(task, latest), callback, latest)
    finally:
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _drain_progress(progress_queue, callback, latest)
        if progress_queue is not None:
            progress_queue.close()
            progress_queue.join_thread()
    if failures:
        failures.sort(key=lambda item: item[0].ordinal)
        names = ", ".join(task.source.stem for task, _error in failures)
        raise RuntimeError(f"Postprocessing failed for sequence(s): {names}.") from failures[0][1]
    return results


def postprocess_replay(
    replay: ReplayResult,
    steps: tuple[str, ...],
    *,
    split: str | None = None,
    workers: int | None = None,
    progress_callback: PostprocessingProgressCallback | None = None,
) -> float:
    """Stage all results before publishing; keep original tracks in ``raw/``.

    Returns elapsed wall time in milliseconds. An algorithm failure leaves every
    original replay file intact and prevents scoring partially processed results.
    Perception artifacts are read only, including GTA's keyed embedding joins.
    Each worker handles one sequence's ordered methods at a time. Phase progress
    is delivered on the caller's thread while those workers are still running.
    """
    steps = normalize_postprocessing(steps)
    if not steps:
        return 0.0
    started = time.perf_counter()
    manifest = DatasetManifest.load(replay.build)
    validate_postprocessing_inputs(steps, geometry=manifest.box_type, manifest=manifest)
    parameters: dict[str, Any] = {name: _SMOOTHING_PARAMETERS[name] for name in steps if name != "gta"}
    if "gta" in steps:
        from boxmot.engine.eval.gta import GTA_PARAMETERS

        parameters["gta"] = GTA_PARAMETERS
    destination = replay.output_dir.resolve()
    with tempfile.TemporaryDirectory(prefix=".postprocess-", dir=destination) as temporary:
        staging = Path(temporary)
        originals, processed = staging / "raw", staging / "processed"
        originals.mkdir()
        processed.mkdir()
        tasks = []
        for ordinal, path in enumerate(replay.sequence_files):
            if path.resolve().parent != destination:
                raise ValueError("Postprocessing result files must belong to the replay output directory.")
            shutil.copy2(path, originals / path.name)
            tasks.append(
                _PostprocessTask(originals / path.name, processed / path.name, replay.build, split, steps, ordinal)
            )
        results = []
        latest: dict[int, PostprocessingProgressEvent] = {}
        for task in tasks:
            _publish_progress(
                PostprocessingProgressEvent(task.source.stem, "queued", 0, None, None, task.ordinal, 0),
                progress_callback,
                latest,
            )

        count = resolve_sequence_workers(len(tasks), workers)
        if count <= 1:
            callback = (
                (lambda event: _publish_progress(event, progress_callback, latest))
                if progress_callback is not None
                else None
            )
            for task in tasks:
                try:
                    results.append(_process_sequence(task, callback))
                except BaseException as exc:
                    _publish_progress(_terminal_progress(task, latest, exc), progress_callback, latest)
                    raise
                _publish_progress(_terminal_progress(task, latest), progress_callback, latest)
        else:
            results = _process_parallel(tasks, workers=count, callback=progress_callback, latest=latest)
        record = {
            "schema_version": 1,
            "build_id": manifest.build_id,
            "split": split,
            "methods": list(steps),
            "parameters": parameters,
            "raw_directory": "raw",
            "sequences": sorted(results, key=lambda result: result["sequence"]),
        }
        metadata = staging / "postprocessing.json"
        metadata.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raw_directory = destination / "raw"
        raw_directory.mkdir(exist_ok=True)
        for task in tasks:
            os.replace(task.source, raw_directory / task.source.name)
            os.replace(task.destination, destination / task.destination.name)
        os.replace(metadata, destination / metadata.name)
    return (time.perf_counter() - started) * 1000
