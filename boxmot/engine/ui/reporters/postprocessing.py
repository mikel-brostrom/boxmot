"""Driver-local Rich sequence progress for offline evaluation processing."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol

from boxmot.engine.ui.reporters.eval import EvalSequenceProgressPresenter, SequenceProgressStatus
from boxmot.engine.ui.workflow.reporting import WorkflowDetailCallback


class _PostprocessingProgressEvent(Protocol):
    """Structural worker event contract; no engine imports enter the UI."""

    sequence_id: str
    status: SequenceProgressStatus
    completed: int
    total: int | None
    detail: str | None
    phase_index: int


class EvalPostprocessingProgressPresenter(EvalSequenceProgressPresenter):
    """Embed one bar per sequence, counting work within its current method phase.

    Phases can have different units of work or unknown totals. A monotonic
    ``phase_index`` lets each row reset at phase boundaries while rejecting
    delayed events from earlier phases and preserving terminal sequence states.
    """

    def __init__(
        self,
        callback: WorkflowDetailCallback,
        sequence_ids: Sequence[str],
        *,
        clock: Callable[[], float] = time.monotonic,
        refresh_interval_s: float | None = 0.08,
    ) -> None:
        super().__init__(
            callback,
            dict.fromkeys(sequence_ids),
            label="Postprocessing",
            unit="items",
            clock=clock,
            refresh_interval_s=refresh_interval_s,
        )
        self._phase_indices = dict.fromkeys(sequence_ids, -1)

    def __call__(self, event: _PostprocessingProgressEvent) -> None:
        """Consume a worker event without letting late phases regress a row."""
        sequence_id = self._validate_sequence_id(event.sequence_id)
        try:
            state = self._states[sequence_id]
        except KeyError as exc:
            raise KeyError(f"Unknown evaluation sequence {sequence_id!r}.") from exc
        phase_index = self._validate_count(event.phase_index, name="phase_index")
        status = self._normalize_status(event.status)
        if state.status in {"completed", "failed"}:
            return
        if phase_index < self._phase_indices[sequence_id]:
            return
        if status == "queued" and state.status != "queued":
            return
        completed = self._validate_count(event.completed, name="completed")
        total = self._validate_count(event.total, name="total", optional=True)
        if total is not None and completed > total:
            raise ValueError(f"Sequence {sequence_id!r} completed count {completed} exceeds total {total}.")
        if phase_index > self._phase_indices[sequence_id]:
            self._reset_phase(sequence_id, total)
            self._phase_indices[sequence_id] = phase_index
        self.update(sequence_id, completed, total, status=status, detail=event.detail)
