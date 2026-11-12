"""Engine ownership and durable JSON Lines output for background attributes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from boxmot.utils import logger as LOGGER

if TYPE_CHECKING:
    from boxmot.attributes import AttributeEnricher, AttributeResult


class AttributeRun:
    """Own one enrichment session and persist even results arriving after EOF."""

    def __init__(
        self,
        reference: Any,
        output_dir: Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.enricher: AttributeEnricher
        if callable(getattr(reference, "update", None)):
            self.enricher = reference
        else:
            from boxmot.attributes import create_attribute_enricher

            self.enricher = create_attribute_enricher(reference)
        self.path = output_dir / "attributes.jsonl"
        self.progress = progress
        self._closed = False
        self._last_error: str | None = None
        self._completed_count = 0
        self._failed_count = 0
        try:
            self._handle = self.path.open("w", encoding="utf-8")
        except BaseException:
            try:
                self.enricher.cancel()
            finally:
                self.enricher.close()
            raise

    def _write(self, event: AttributeResult) -> None:
        """Flush terminal events and report distinct failures without flooding output."""
        self._handle.write(json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True) + "\n")
        self._handle.flush()
        if event.status == "completed":
            self._completed_count += 1
            self._last_error = None
        elif event.status == "failed":
            self._failed_count += 1
            error = event.error or "Attribute backend did not provide an error message."
            if error != self._last_error:
                LOGGER.warning(
                    "Attribute analysis failed for track %s: %s See %s for all outcomes.",
                    event.track_id, error, self.path,
                )
                self._last_error = error

    @property
    def status_summary(self) -> str:
        """Describe journaled outcomes separately from unfinished inference batches."""
        return (
            f"Attributes • completed analyses: {self._completed_count}"
            f" • pending batches: {self.enricher.pending_count}"
            f" • failed analyses: {self._failed_count}"
        )

    def poll(self, *, all_events: bool = False) -> None:
        """Persist a bounded event page during tracking, or drain on cancellation."""
        while not self._closed:
            events = self.enricher.poll(limit=100)
            for event in events:
                self._write(event)
            if not all_events or not events:
                break

    def finish(self, *, cancel: bool = False) -> bool:
        """Drain all accepted work, or journal cancellation without waiting for inference."""
        if self._closed:
            return cancel
        interrupted = cancel
        try:
            if cancel:
                self.enricher.cancel()
                self.poll(all_events=True)
            else:
                if self.progress is not None:
                    self.progress(f"Finishing attributes for {self.enricher.pending_count} pending batch(es)…")
                try:
                    for event in self.enricher.finish():
                        self._write(event)
                        if self.progress is not None:
                            self.progress(f"Finishing attributes • {self.enricher.pending_count} batch(es) remaining")
                except KeyboardInterrupt:
                    interrupted = True
                    self.enricher.cancel()
                    self.poll(all_events=True)
        except BaseException:
            self.enricher.cancel()
            raise
        finally:
            self._closed = True
            try:
                self.enricher.close()
            finally:
                self._handle.close()
        return interrupted

    def __enter__(self) -> AttributeRun:
        return self

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        self.finish(cancel=exc_type is not None)


__all__ = ("AttributeRun",)
