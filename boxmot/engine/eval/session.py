"""Explicit ownership of replay processes retained across tuning trials."""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from multiprocessing import get_context
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

if TYPE_CHECKING:
    from boxmot.engine.eval.eagermot_kitti import _KittiSequenceResult, _KittiSequenceTask
    from boxmot.engine.eval.replay import ReplayProgressCallback, _SequenceReplayResult, _SequenceReplayTask

_T = TypeVar("_T")
_R = TypeVar("_R")


class ReplaySession:
    """Reuse spawned processes and immutable inputs while owning no trackers.

    Each call still creates and closes a tracker for every sequence. A session
    accepts one replay or metrics operation at a time; concurrent tuning slots
    own separate sessions. Failed operations discard the pool so the next
    trial starts with healthy processes and clean worker-local caches.
    """

    def __init__(self, workers: int, *, cache_inputs: bool = False) -> None:
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("ReplaySession workers must be a positive integer.")
        if not isinstance(cache_inputs, bool):
            raise TypeError("ReplaySession cache_inputs must be bool.")
        self.workers = workers
        self.cache_inputs = cache_inputs
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._progress_queue: Any | None = None
        self._lock = threading.Lock()
        self._closed = False

    def __enter__(self) -> ReplaySession:
        if self._closed:
            raise RuntimeError("ReplaySession is closed.")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _start(self) -> concurrent.futures.ProcessPoolExecutor:
        """Create worker resources lazily inside the owning process."""
        if self._closed:
            raise RuntimeError("ReplaySession is closed.")
        if self._executor is None:
            from boxmot.engine.eval.replay import _initialize_replay_worker

            context = get_context("spawn")
            progress_queue = context.Queue()
            try:
                executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=context,
                    initializer=_initialize_replay_worker,
                    initargs=(progress_queue,),
                )
            except BaseException:
                progress_queue.close()
                progress_queue.join_thread()
                raise
            self._progress_queue = progress_queue
            self._executor = executor
        return self._executor

    def _shutdown(self, *, interrupted: bool) -> None:
        """Release the pool and its queue, including broken or cancelled runs."""
        executor, self._executor = self._executor, None
        progress_queue, self._progress_queue = self._progress_queue, None
        try:
            if executor is not None:
                if interrupted:
                    # Python 3.11/3.12 provide no public terminate_workers API.
                    # Stop active sequence work before staging directories vanish.
                    processes = tuple((getattr(executor, "_processes", None) or {}).values())
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                executor.shutdown(wait=True, cancel_futures=interrupted)
        finally:
            if progress_queue is not None:
                progress_queue.close()
                progress_queue.join_thread()

    @contextmanager
    def _operation(self) -> Iterator[concurrent.futures.ProcessPoolExecutor]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("ReplaySession already has an active operation.")
        try:
            executor = self._start()
            try:
                yield executor
            except BaseException:
                self._shutdown(interrupted=True)
                raise
        finally:
            self._lock.release()

    def run(
        self,
        tasks: tuple[_SequenceReplayTask, ...],
        *,
        progress_callback: ReplayProgressCallback | None = None,
    ) -> tuple[_SequenceReplayResult, ...]:
        """Run a fresh sequence batch and route only its progress messages."""
        from boxmot.engine.eval.replay import _run_spawned_sequence_tasks

        run_id = uuid4().hex
        tasks = tuple(replace(task, run_id=run_id, report_progress=progress_callback is not None) for task in tasks)
        with self._operation() as executor:
            return _run_spawned_sequence_tasks(
                tasks,
                workers=self.workers,
                progress_callback=progress_callback,
                _pool=(executor, self._progress_queue, run_id),
            )

    def map(self, function: Callable[[_T], _R], values: Sequence[_T]) -> list[_R]:
        """Execute ordered metric tasks in the existing spawn pool."""
        with self._operation() as executor:
            return list(executor.map(function, values, chunksize=1))

    def run_sensor(
        self,
        tasks: tuple[_KittiSequenceTask, ...],
        *,
        progress_callback: ReplayProgressCallback | None = None,
    ) -> tuple[_KittiSequenceResult, ...]:
        """Replay sensor trials in retained workers with fresh trackers and run IDs."""
        from boxmot.engine.eval.eagermot_kitti import _run_parallel_sequences

        run_id = uuid4().hex
        tasks = tuple(replace(task, run_id=run_id, report_progress=progress_callback is not None) for task in tasks)
        with self._operation() as executor:
            return _run_parallel_sequences(
                tasks,
                self.workers,
                progress_callback,
                _pool=(executor, self._progress_queue, run_id),
            )

    @contextmanager
    def metric_execution(self) -> Iterator[None]:
        """Keep metrics out of fork pools while replay manager threads live."""
        from boxmot.engine.eval.motmetrics import use_metric_executor

        with use_metric_executor(self.map):
            yield

    def close(self) -> None:
        """Close a session once its current operation has completed."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._shutdown(interrupted=False)
