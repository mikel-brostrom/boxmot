"""Small pluggable local executors for materialization stages."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Generic, Literal, Protocol, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
ExecutorKind = Literal["inline", "thread", "process"]


class StageExecutor(Protocol):
    """Ordered local map interface used by materialization stages."""

    def map(self, function: Callable[[InputT], OutputT], items: Iterable[InputT]) -> list[OutputT]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ExecutorSpec:
    """Serializable local executor selection."""

    kind: ExecutorKind = "inline"
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.kind not in {"inline", "thread", "process"}:
            raise ValueError(f"Unknown executor kind {self.kind!r}.")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive.")
        if self.kind == "inline" and self.max_workers != 1:
            raise ValueError("The inline executor always has exactly one worker.")


class InlineExecutor:
    """Deterministic no-pool executor useful for GPU stages and tests."""

    def map(self, function: Callable[[InputT], OutputT], items: Iterable[InputT]) -> list[OutputT]:
        return [function(item) for item in items]

    def close(self) -> None:
        return None

    def __enter__(self) -> "InlineExecutor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class PoolStageExecutor(Generic[InputT, OutputT]):
    """Ordered adapter over a local thread or process pool."""

    def __init__(self, *, kind: Literal["thread", "process"], max_workers: int) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive.")
        self.kind = kind
        if kind == "thread":
            self._pool = ThreadPoolExecutor(max_workers=max_workers)
        else:
            self._pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context("spawn"))
        self._closed = False

    def map(self, function: Callable[[InputT], OutputT], items: Iterable[InputT]) -> list[OutputT]:
        if self._closed:
            raise RuntimeError("Executor is closed.")
        return list(self._pool.map(function, items))

    def close(self) -> None:
        if not self._closed:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._closed = True

    def __enter__(self) -> "PoolStageExecutor[InputT, OutputT]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def create_executor(spec: ExecutorSpec) -> StageExecutor:
    """Create a fresh executor; model/native handles must be created per worker."""

    if spec.kind == "inline":
        return InlineExecutor()
    return PoolStageExecutor(kind=spec.kind, max_workers=spec.max_workers)


__all__ = (
    "ExecutorKind",
    "ExecutorSpec",
    "InlineExecutor",
    "PoolStageExecutor",
    "StageExecutor",
    "create_executor",
)
