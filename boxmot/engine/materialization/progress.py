"""Concise logging for long-running local materialization builds."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from boxmot.utils import logger as LOGGER

from .plan import BuildPlan, StagePlan
from .stages.base import StageOutcome


class _ProgressLogger(Protocol):
    def info(self, message: str) -> None: ...

    def warning(self, message: str) -> None: ...

    def error(self, message: str) -> None: ...


class MaterializationProgressReporter(Protocol):
    """Event interface shared by interactive and non-interactive reporters."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def setup_status(self, message: str) -> None: ...

    def unhandled_failure(self, error: BaseException) -> None: ...

    def build_started(self, plan: BuildPlan) -> None: ...

    def build_lock_waiting(self, lock_path: Path) -> None: ...

    def build_lock_acquired(self) -> None: ...

    def build_reused(self, output_root: Path) -> None: ...

    def build_completed(self, output_root: Path) -> None: ...

    def stage_started(self, plan: BuildPlan, stage: StagePlan, *, completed_shards: int) -> None: ...

    def shard_completed(
        self,
        stage_name: str,
        shard_id: str,
        *,
        completed_shards: int,
        items: int | None = None,
        rows: int | None = None,
    ) -> None: ...

    def stage_completed(self, stage: StagePlan, outcome: StageOutcome, *, completed_shards: int) -> None: ...

    def stage_skipped(self, stage_name: str) -> None: ...

    def stage_retry(
        self,
        stage_name: str,
        error: Exception,
        *,
        attempt: int,
        max_attempts: int,
        delay_s: float,
    ) -> None: ...

    def stage_failed(
        self,
        stage_name: str,
        error: BaseException,
        *,
        attempt: int,
        max_attempts: int,
        staging_root: Path,
    ) -> None: ...


@dataclass(slots=True)
class _StageTiming:
    started_s: float
    total_shards: int | None
    initial_completed_shards: int


def _source_count(plan: BuildPlan) -> int | None:
    value = plan.metadata.get("source_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _total_shards(plan: BuildPlan, stage: StagePlan) -> int | None:
    if stage.name == "finalize":
        return 1
    if stage.name not in {"detect", "segment", "embed"}:
        return None
    count = _source_count(plan)
    if count is None:
        return None
    return (count + stage.batch_size - 1) // stage.batch_size


def _error_summary(error: BaseException) -> str:
    detail = str(error).strip()
    return type(error).__name__ if not detail else f"{type(error).__name__}: {detail}"


class MaterializationProgress:
    """Render deterministic stage and shard events through the BoxMOT logger."""

    def __init__(
        self,
        *,
        logger: _ProgressLogger = LOGGER,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._logger = logger
        self._clock = clock
        self._timings: dict[str, _StageTiming] = {}

    def start(self) -> None:
        """Plain logging needs no retained display lifecycle."""

    def stop(self) -> None:
        """Plain logging needs no retained display lifecycle."""

    def setup_status(self, message: str) -> None:
        """Report a coarse setup milestone before the build plan exists."""

        self._logger.info(message)

    def unhandled_failure(self, error: BaseException) -> None:
        """Report failures raised before a materialization stage starts."""

        self._logger.error(f"Materialization setup failed ({_error_summary(error)}).")

    def build_started(self, plan: BuildPlan) -> None:
        """Describe immutable build identity and filesystem destinations."""

        self._logger.info(f"Materialization build: {plan.build_id}")
        count = _source_count(plan)
        self._logger.info("Source samples: unknown" if count is None else f"Source samples: {count:,}")
        self._logger.info(f"Output directory: {plan.output_root}")
        self._logger.info(f"Staging directory: {plan.staging_root}")

    def build_reused(self, output_root: Path) -> None:
        """Report validation and reuse of an already complete build."""

        self._logger.info(f"Reusing complete materialization build: {output_root}")

    def build_lock_waiting(self, lock_path: Path) -> None:
        """Make contention with another identical build visible."""

        self._logger.info(f"Acquiring materialization build lock: {lock_path}")

    def build_lock_acquired(self) -> None:
        """Confirm that stage execution may proceed."""

        self._logger.info("Materialization build lock acquired.")

    def build_completed(self, output_root: Path) -> None:
        """Report successful atomic publication."""

        self._logger.info(f"Published immutable dataset build: {output_root}")

    def stage_started(self, plan: BuildPlan, stage: StagePlan, *, completed_shards: int) -> None:
        """Start elapsed-time accounting and report resume state."""

        total = _total_shards(plan, stage)
        self._timings[stage.name] = _StageTiming(
            started_s=self._clock(),
            total_shards=total,
            initial_completed_shards=completed_shards,
        )
        if total is None:
            progress = f"{completed_shards} shards already complete"
        else:
            pending = max(total - completed_shards, 0)
            progress = f"{completed_shards}/{total} shards complete, {pending} pending"
        self._logger.info(
            f"Stage {stage.name} starting: {progress}; batch={stage.batch_size}, "
            f"workers={stage.workers}, executor={stage.executor}."
        )

    def shard_completed(
        self,
        stage_name: str,
        shard_id: str,
        *,
        completed_shards: int,
        items: int | None = None,
        rows: int | None = None,
    ) -> None:
        """Report one durable shard checkpoint, never individual frames."""

        timing = self._timings.get(stage_name)
        if timing is None:
            return
        elapsed = max(self._clock() - timing.started_s, 0.0)
        processed = max(completed_shards - timing.initial_completed_shards, 1)
        rate = processed / elapsed if elapsed > 0.0 else None
        details: list[str] = []
        if timing.total_shards is None:
            details.append(f"{completed_shards} complete")
        else:
            percent = min(100.0, 100.0 * completed_shards / max(timing.total_shards, 1))
            details.append(f"{completed_shards}/{timing.total_shards} ({percent:.1f}%)")
        if items is not None:
            details.append(f"items={items}")
        if rows is not None:
            details.append(f"rows={rows}")
        details.append(f"elapsed={elapsed:.1f}s")
        if rate is None:
            details.extend(("rate=n/a", "ETA=n/a"))
        else:
            details.append(f"rate={rate:.2f} shards/s")
            if timing.total_shards is None:
                details.append("ETA=n/a")
            else:
                remaining = max(timing.total_shards - completed_shards, 0)
                details.append(f"ETA={remaining / rate:.1f}s")
        self._logger.info(f"Stage {stage_name} shard {shard_id}: {', '.join(details)}.")

    def stage_completed(self, stage: StagePlan, outcome: StageOutcome, *, completed_shards: int) -> None:
        """Report stage completion and small outcome counters."""

        timing = self._timings.get(stage.name)
        elapsed = 0.0 if timing is None else max(self._clock() - timing.started_s, 0.0)
        if timing is None or timing.total_shards is None:
            progress = f"{completed_shards} shards"
        else:
            completed = timing.total_shards if stage.name == "finalize" else completed_shards
            progress = f"{completed}/{timing.total_shards} shards"
        metrics = self._format_metrics(outcome.metrics)
        suffix = "" if not metrics else f"; {metrics}"
        self._logger.info(f"Stage {stage.name} complete: {progress}; elapsed={elapsed:.1f}s{suffix}.")

    def stage_skipped(self, stage_name: str) -> None:
        """Report a persisted completed stage that needs no validation pass."""

        self._logger.info(f"Stage {stage_name} already complete; skipping.")

    def stage_retry(
        self,
        stage_name: str,
        error: Exception,
        *,
        attempt: int,
        max_attempts: int,
        delay_s: float,
    ) -> None:
        """Report a failed attempt before its configured retry."""

        self._logger.warning(
            f"Stage {stage_name} attempt {attempt}/{max_attempts} failed "
            f"({_error_summary(error)}); retrying attempt {attempt + 1}/{max_attempts} "
            f"in {delay_s:.1f}s."
        )

    def stage_failed(
        self,
        stage_name: str,
        error: BaseException,
        *,
        attempt: int,
        max_attempts: int,
        staging_root: Path,
    ) -> None:
        """Report terminal failure and the retained resume location."""

        self._logger.error(
            f"Stage {stage_name} failed after attempt {attempt}/{max_attempts} "
            f"({_error_summary(error)}); resumable state remains at {staging_root}."
        )

    @staticmethod
    def _format_metrics(metrics: Mapping[str, Any]) -> str:
        values = []
        for name in sorted(metrics):
            value = metrics[name]
            if value is None or isinstance(value, str | int | float | bool):
                values.append(f"{name}={value}")
        return ", ".join(values)


__all__ = ("MaterializationProgress", "MaterializationProgressReporter")
