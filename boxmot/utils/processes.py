"""Tie multiprocessing workers to the lifetime of their owning process."""

from __future__ import annotations

import multiprocessing
import os
import threading
from multiprocessing.connection import wait

_parent_watcher: threading.Thread | None = None


def _exit_when_parent_stops(sentinel: int) -> None:
    """Exit even when normal pool cleanup cannot run in a dead parent."""
    wait([sentinel])
    # The owner cannot consume results or progress anymore. Running Python
    # finalizers here can deadlock on queues or locks left behind by its death.
    os._exit(1)


def guard_parent_lifetime() -> None:
    """Stop a multiprocessing child when its original parent exits.

    Call first in pool initializers, before opening inputs or allocating model
    state. The parent sentinel stays ready if the owner died during worker
    startup, so this also covers death before initialization. Waiting on the
    sentinel avoids PID reuse and works with both spawn and fork contexts.

    Direct calls in the main process are harmless. Repeated initialization in
    a retained worker reuses its watcher; a forked child starts its own thread.
    """
    global _parent_watcher

    parent = multiprocessing.parent_process()
    if parent is None:
        return
    if _parent_watcher is not None and _parent_watcher.is_alive():
        return
    if not parent.is_alive():
        os._exit(1)
    _parent_watcher = threading.Thread(
        target=_exit_when_parent_stops,
        args=(parent.sentinel,),
        name="boxmot-parent-watcher",
        daemon=True,
    )
    _parent_watcher.start()
