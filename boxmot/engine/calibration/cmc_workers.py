"""Sequence-owned camera analysis in isolated, bounded CPU worker processes."""

from __future__ import annotations

import argparse
import json
import logging
import queue
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any

import numpy as np

from boxmot.engine.eval.workers import initialize_worker_threads, shutdown_sequence_pool
from boxmot.trackers import create_tracker
from boxmot.trackers.common.config import get_tracker_config_class
from boxmot.utils.processes import guard_parent_lifetime

_WORKER_PROGRESS_QUEUE: Any | None = None
CMCSequenceResult = tuple[dict[str, Any], dict[str, np.ndarray]]


@dataclass(frozen=True, slots=True)
class CMCSequenceTask:
    """Only resolved camera inputs cross the process boundary, never workflow state."""

    build_path: str
    split: str | None
    sequence_id: str
    ordinal: int
    total_sequences: int
    frame_count: int
    tracker: str
    geometry: str
    per_class: bool
    class_ids: tuple[int, ...] | None
    class_names: tuple[tuple[int, str], ...]
    tracker_config_json: str
    mask_min_conf: float
    estimator_options: tuple[tuple[str, int | float | None], ...]


def _initialize_cmc_worker(progress_queue: Any | None) -> None:
    """Constrain native CPU pools and keep worker lifetime and UI with the parent."""
    global _WORKER_PROGRESS_QUEUE
    guard_parent_lifetime()
    initialize_worker_threads()
    _WORKER_PROGRESS_QUEUE = progress_queue
    if progress_queue is not None:
        # The parent owns reliable completion through Future results. A worker
        # must never wait on the feeder for these disposable progress messages.
        progress_queue.cancel_join_thread()
    logger = logging.getLogger("boxmot")
    logger.handlers.clear()
    logger.propagate = False
    logger.disabled = True


def _emit_worker_progress(ordinal: int, message: str) -> None:
    """Drop updates when the bounded observational channel is full or closed."""
    if _WORKER_PROGRESS_QUEUE is not None:
        try:
            _WORKER_PROGRESS_QUEUE.put_nowait((ordinal, message))
        except (queue.Full, BrokenPipeError, EOFError, OSError, ValueError):
            pass


def _analyze_worker(task: CMCSequenceTask) -> CMCSequenceResult:
    """Own one sequence's masks and estimator history until every frame finishes."""
    from boxmot.engine.calibration.cmc import _analyze_sequence

    args = argparse.Namespace(
        build_path=task.build_path,
        split=task.split,
        geometry=task.geometry,
        seq_info={task.sequence_id: task.frame_count},
        tracker_class_ids=task.class_ids,
        tracker_class_names=task.class_names,
    )
    config = get_tracker_config_class(task.tracker).from_mapping(json.loads(task.tracker_config_json))
    tracker = create_tracker(task.tracker, config=config, geometry=task.geometry, per_class=task.per_class)

    def progress(message: str) -> None:
        _emit_worker_progress(task.ordinal, message)

    try:
        progress(
            f"CMC analysis: estimating camera motion for {task.sequence_id} "
            f"({task.ordinal + 1}/{task.total_sequences})…"
        )
        return _analyze_sequence(
            args,
            task.sequence_id,
            mask_min_conf=task.mask_min_conf,
            estimator_options=dict(task.estimator_options),
            runtime_masker=tracker.cmc_detection_boxes,
            progress=progress if _WORKER_PROGRESS_QUEUE is not None else None,
        )
    finally:
        close = getattr(tracker, "close", None)
        if callable(close):
            close()


def _publish_progress(progress: Callable[[str], None] | None, message: str) -> None:
    """An observational callback cannot invalidate completed numerical work."""
    if progress is not None:
        try:
            progress(message)
        except Exception:
            pass


def _drain_progress(
    progress_queue: Any | None,
    progress: Callable[[str], None] | None,
    completed: set[int],
) -> None:
    """Render live updates in the parent and discard stale completed-sequence events."""
    if progress_queue is None:
        return
    while True:
        try:
            ordinal, message = progress_queue.get_nowait()
        except (queue.Empty, EOFError, OSError):
            return
        if ordinal not in completed:
            _publish_progress(progress, message)


def analyze_parallel(
    tasks: tuple[CMCSequenceTask, ...],
    *,
    workers: int,
    progress: Callable[[str], None] | None,
) -> list[CMCSequenceResult]:
    """Fail fast, publish no artifacts, and return results in selected sequence order."""
    context = get_context("spawn")
    progress_queue = context.Queue(maxsize=max(32, workers * 8)) if progress is not None else None
    executor = None
    futures = {}
    results: dict[int, CMCSequenceResult] = {}
    completed: set[int] = set()
    interrupted = True
    try:
        _publish_progress(progress, f"CMC analysis: analyzing {len(tasks)} sequences with {workers} worker processes…")
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_initialize_cmc_worker,
            initargs=(progress_queue,),
        )
        futures = {executor.submit(_analyze_worker, task): task for task in tasks}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            ordered_done = sorted(done, key=lambda future: futures[future].ordinal)
            for future in ordered_done:
                # A dead writer can leave a partial queue message. Propagate
                # failures before reading progress, then terminate all peers.
                if future.exception() is not None:
                    future.result()
            _drain_progress(progress_queue, progress, completed)
            for future in ordered_done:
                task = futures[future]
                results[task.ordinal] = future.result()
                completed.add(task.ordinal)
                _publish_progress(
                    progress,
                    f"CMC analysis: completed {task.sequence_id} ({len(completed)}/{len(tasks)})…",
                )
        interrupted = False
    finally:
        for future in futures:
            future.cancel()
        try:
            if executor is not None:
                shutdown_sequence_pool(executor, interrupted=interrupted)
        finally:
            try:
                # Never drain a queue after terminating one of its writers.
                if not interrupted:
                    _drain_progress(progress_queue, progress, completed)
            finally:
                if progress_queue is not None:
                    progress_queue.close()
                    progress_queue.join_thread()
    return [results[task.ordinal] for task in tasks]
