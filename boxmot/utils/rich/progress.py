"""Standalone Rich-based progress bar with a ``tqdm``-compatible API.

Use this when no workflow ``WorkflowDetailCallback`` is active (i.e. the
caller is not running inside a Rich Live panel). When a workflow IS
active, prefer ``WorkflowDetailCallback.bar()`` so the progress is hosted
by the same Live region.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from boxmot.utils.rich.ui import get_console


class RichTqdm:
    """Drop-in replacement for ``tqdm.tqdm`` backed by ``rich.progress.Progress``.

    Supports the subset of the ``tqdm`` API used in this codebase:
    constructor kwargs (``total``, ``desc``, ``unit``, ``unit_scale``,
    ``dynamic_ncols``, ``disable``, ``initial``, ``leave``), ``update(n)``,
    ``close()``, iteration over an input iterable, and the context manager
    protocol. Unknown kwargs are silently ignored for compatibility.
    """

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        *,
        total: int | float | None = None,
        desc: str | None = None,
        unit: str | None = None,
        unit_scale: bool = False,
        initial: int = 0,
        disable: bool = False,
        leave: bool = True,
        **_: Any,
    ) -> None:
        self._iterable = iterable
        self._desc = desc or ""
        self._disable = disable

        if total is None and iterable is not None:
            try:
                total = len(iterable)  # type: ignore[arg-type]
            except TypeError:
                total = None

        self._total = total

        if disable:
            self._progress: Progress | None = None
            self._task_id: int | None = None
            return

        is_bytes = unit == "B" or unit_scale
        columns: list[Any] = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ]
        if is_bytes:
            columns.extend([DownloadColumn(), TransferSpeedColumn()])
        else:
            columns.append(MofNCompleteColumn())
            if unit:
                columns.append(TextColumn(unit))
        columns.append(TimeRemainingColumn())

        self._progress = Progress(
            *columns,
            console=get_console(stderr=True),
            transient=not leave,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            self._desc,
            total=total,
            completed=initial,
        )

    def update(self, n: int = 1) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=n)

    def set_description(self, desc: str | None = None, **_: Any) -> None:
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=desc or "")
            self._desc = desc or ""

    def set_postfix(self, *_: Any, **__: Any) -> None:  # tqdm-compat no-op
        return None

    def set_postfix_str(self, *_: Any, **__: Any) -> None:  # tqdm-compat no-op
        return None

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def refresh(self) -> None:
        if self._progress is not None:
            self._progress.refresh()

    def __iter__(self) -> Iterator[Any]:
        if self._iterable is None:
            return
        try:
            for item in self._iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __enter__(self) -> "RichTqdm":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = ["RichTqdm"]
