"""Native thread limits and shutdown for spawned sequence replay workers."""

from __future__ import annotations

import concurrent.futures
import os
import time


def initialize_worker_threads() -> None:
    """Use one native compute thread in each spawned sequence worker.

    Sequence processes already provide CPU parallelism. In particular, Ray's
    actor-level OMP_NUM_THREADS allocation must not multiply across its children.
    Call only in a worker initializer so the parent's settings remain untouched.
    """
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    import cv2
    import torch
    from threadpoolctl import threadpool_limits

    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    # Inter-op configuration is one-shot; do not reset it on repeated setup.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    # Spawn imports can load NumPy/SciPy before this initializer. Environment
    # changes only cover later imports, so cap existing BLAS/OpenMP pools too.
    # Used as a callable, this persists for the worker's entire lifetime.
    threadpool_limits(limits=1)


def shutdown_sequence_pool(executor: concurrent.futures.ProcessPoolExecutor, *, interrupted: bool) -> None:
    """Stop interrupted CPU work before joining the pool and its queue threads."""
    if interrupted:
        # Python 3.11 has no public ProcessPoolExecutor.terminate_workers().
        # Snapshot its child handles before shutdown clears them, and bound
        # both graceful termination and a final kill of unresponsive children.
        processes = tuple((executor._processes or {}).values())
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
    executor.shutdown(wait=True, cancel_futures=interrupted)
