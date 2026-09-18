"""Shared shutdown for spawned sequence replay workers."""

from __future__ import annotations

import concurrent.futures
import time


def shutdown_sequence_pool(executor: concurrent.futures.ProcessPoolExecutor, *, interrupted: bool) -> None:
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
