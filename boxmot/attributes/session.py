"""Background, sequence-aware attribute extraction with a PostgreSQL backlog."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from boxmot.attributes.config import AttributeConfig, resolve_attribute_config
from boxmot.attributes.crops import encode_track_crop
from boxmot.attributes.errors import TransientAttributeError
from boxmot.attributes.models import AttributeResult, AttributeSample
from boxmot.attributes.spool import AttributeSpool
from boxmot.components.registry import LazyComponentRegistry
from boxmot.structures import Frame, Tracks

_BACKENDS = LazyComponentRegistry(
    "attribute",
    {
        "ollama": "boxmot.attributes.backends.ollama:OllamaBackend",
        "vllm": "boxmot.attributes.backends.vllm:VLLMBackend",
        "openai": "boxmot.attributes.backends.openai:OpenAIBackend",
        "anthropic": "boxmot.attributes.backends.anthropic:AnthropicBackend",
    },
)


class AttributeBackend(Protocol):
    """One cancellable extraction backend; scheduling belongs to the session."""

    async def extract(
        self, track_id: int, samples: Sequence[AttributeSample], images: Sequence[bytes]
    ) -> dict[str, Any]: ...

    async def aclose(self) -> None: ...


class AttributeEnricher:
    """Collect crops without waiting for inference and expose completed attributes.

    Use ``poll()`` during tracking and iterate ``finish()`` at normal EOF to
    consume all outcomes, including results for departed tracks. ``reset()``
    isolates a new sequence without losing the old sequence's queued work.
    The context manager drains on normal exit and cancels on exceptional exit.
    Backend ownership transfers when the worker starts successfully.
    """

    def __init__(
        self,
        config: AttributeConfig,
        *,
        backend: AttributeBackend | None = None,
    ) -> None:
        if not isinstance(config, AttributeConfig):
            raise TypeError("config must be an AttributeConfig.")
        self.config = config
        self.run_id = uuid4().hex
        # The worker creates owned backends only after its thread has started.
        # Database or thread startup failure therefore cannot leak an HTTP client.
        self._spool = AttributeSpool(config, self.run_id)
        self._backend = backend
        self._generation = 0
        self._sequence_id: str | None = None
        self._sequence_set = False
        self._last_frame_index: int | None = None
        self._step = 0
        self._accepting = True
        self._closed = False
        self._finished = False
        self._stats: dict[str, int] = {}
        self._stop = threading.Event()
        self._cancel_requested = False
        self._cancel_delivered = False
        self._changed = threading.Event()
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._wake: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._run_worker, name="boxmot-attributes", daemon=True)
        try:
            self._thread.start()
            self._started.wait()
            self._check_worker()
        except BaseException:
            with suppress(Exception):
                if self._thread.ident is None:
                    self._spool.close()
                    self._closed = True
                else:
                    self.close()
            raise

    def _check_worker(self) -> None:
        if self._error is not None:
            raise RuntimeError("Attribute worker failed; pending work was not silently discarded.") from self._error

    def _notify(self) -> None:
        loop, wake = self._loop, self._wake
        if loop is not None and wake is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(wake.set)
            except RuntimeError:
                # The worker may have completed shutdown between the check and
                # notification. Its failure, if any, is propagated separately.
                if not self._stop.is_set():
                    self._check_worker()

    def _run_worker(self) -> None:
        try:
            asyncio.run(self._worker())
        except BaseException as exc:  # Surface database/backend cleanup failures to the caller.
            self._error = exc
        finally:
            self._started.set()
            self._changed.set()

    async def _worker(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        self._wake = asyncio.Event()
        watchdog = asyncio.create_task(self._watch_cancellation())
        backend = self._backend
        try:
            if backend is None:
                backend = _BACKENDS.resolve(self.config.backend)(self.config)
                self._backend = backend
            while not self._stop.is_set():
                self._wake.clear()
                job = self._spool.claim()
                # Constructor readiness includes the first successful DB query.
                self._started.set()
                if job is None:
                    try:
                        # Retry deadlines must also wake the scheduler. Normal
                        # submissions notify it immediately instead of polling.
                        await asyncio.wait_for(self._wake.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue
                try:
                    values = await asyncio.wait_for(
                        backend.extract(job.track_id, job.samples, job.images),
                        timeout=self.config.request_timeout_s,
                    )
                    if not isinstance(values, dict):
                        raise TypeError("Attribute backends must return a JSON object.")
                except (TransientAttributeError, asyncio.TimeoutError) as exc:
                    if self._stop.is_set():
                        break
                    if job.attempts <= self.config.retries:
                        delay: float = min(2 ** (job.attempts - 1), 30)
                        if isinstance(exc, TransientAttributeError) and exc.retry_after_s is not None:
                            delay = max(delay, exc.retry_after_s)
                        self._spool.retry(job, delay=delay)
                    else:
                        self._spool.complete(job, error=str(exc) or "Attribute request timed out.")
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    self._spool.complete(job, error=f"{type(exc).__name__}: {exc}")
                else:
                    if self._stop.is_set():
                        break
                    # Spool errors must escape the worker, not masquerade as
                    # model failures or silently drop coverage on storage failure.
                    self._spool.complete(job, values=values)
                finally:
                    self._changed.set()
        except asyncio.CancelledError:
            if not self._stop.is_set():
                raise
        finally:
            watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog
            if backend is not None:
                await asyncio.wait_for(backend.aclose(), timeout=5.0)

    async def _watch_cancellation(self) -> None:
        """Bound cancellation latency even if cross-thread socket wakeups fail.

        Some restricted hosts disable the event loop's self-pipe wakeup. A
        lightweight timer keeps a long HTTP read interruptible there as well.
        """
        while not self._cancel_requested:
            await asyncio.sleep(0.1)
        self._deliver_cancellation()

    def _deliver_cancellation(self) -> None:
        """Deliver once on the worker loop, preserving asynchronous client cleanup."""
        if not self._cancel_delivered and self._task is not None:
            self._cancel_delivered = True
            self._task.cancel()

    def update(self, frame: Frame, tracks: Tracks) -> dict[int, AttributeResult]:
        """Persist eligible sampled crops and return detached current-track results."""
        if self._closed or not self._accepting:
            raise RuntimeError("Cannot submit frames after finishing or closing attribute extraction.")
        self._check_worker()
        if not isinstance(frame, Frame) or not isinstance(tracks, Tracks):
            raise TypeError("Attribute extraction requires canonical Frame and Tracks objects.")
        frame.validate()
        tracks.validate()
        if tracks.sample_id != frame.sample_id:
            raise ValueError("Attribute track and frame sample IDs must match.")
        if self._sequence_set and frame.sequence_id != self._sequence_id:
            raise ValueError("Reset the attribute enricher before processing another sequence.")
        if (
            frame.frame_index is not None
            and self._last_frame_index is not None
            and frame.frame_index <= self._last_frame_index
        ):
            raise ValueError("Attribute frame_index must increase within a sequence.")
        self._sequence_id, self._sequence_set = frame.sequence_id, True
        self._last_frame_index = frame.frame_index
        self._step += 1
        sample = AttributeSample(frame.sample_id, frame.frame_index, frame.timestamp_s)
        current_ids = []
        # One transaction per frame avoids a durable commit for every visible
        # track, while crop bytes are still copied before returning to callers.
        with self._spool.transaction():
            for index, track_id in enumerate(tracks.track_ids.tolist()):
                class_id = int(tracks.class_ids[index])
                if self.config.class_ids is not None and class_id not in self.config.class_ids:
                    continue
                current_ids.append(track_id)
                last_sample = self._spool.observe(
                    self._generation, frame.sequence_id, track_id, class_id, self._step, sample
                )
                if last_sample is None or self._step - last_sample >= self.config.sample_every:
                    image = encode_track_crop(frame, tracks, index, self.config)
                    if image is not None:
                        self._spool.append(self._generation, track_id, self._step, sample, image)
            # Current rows now have last_seen == self._step. Only the previous
            # step's remaining rows have just disappeared; do not scan all old
            # identities again on every frame of a long-running camera stream.
            self._spool.flush(self._generation, departed_at=self._step - 1)
        self._notify()
        return self._spool.latest(self._generation, current_ids)

    def reset(self) -> None:
        """Flush the current sequence and namespace subsequent track identities."""
        if self._closed or not self._accepting:
            raise RuntimeError("Cannot reset a finished attribute session.")
        self._check_worker()
        self._spool.flush(self._generation)
        self._generation += 1
        self._sequence_id, self._sequence_set = None, False
        self._last_frame_index = None
        self._step = 0
        self._notify()

    def poll(self, limit: int = 100) -> list[AttributeResult]:
        """Consume up to ``limit`` outcomes, including departed/old-sequence tracks."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("poll limit must be a positive integer.")
        if self._closed:
            return []
        self._check_worker()
        return self._spool.poll(limit)

    def finish(self) -> Iterator[AttributeResult]:
        """Stop submissions and yield remaining outcomes until every job finishes.

        There is no global drain deadline. Individual requests retain their
        configured timeout and retry limit. Interrupting iteration allows the
        owner to call ``cancel()`` and consume cancellation events with ``poll``.
        """
        if self._closed:
            return
        self._accepting = False
        self._spool.flush()
        self._notify()
        while True:
            self._changed.clear()
            self._check_worker()
            yield from self.poll()
            if self._spool.pending_count == 0:
                while events := self.poll():
                    yield from events
                self._finished = True
                return
            self._changed.wait(timeout=0.1)

    def cancel(self) -> None:
        """Cancel in-flight/queued requests and preserve explicit terminal events."""
        if self._closed:
            return
        self._accepting = False
        # Cancellation must remain effective even when journaling fails (for
        # example an unavailable database). Never leave a network request running
        # because the cancellation record could not be persisted.
        self._request_stop(cancel=True)
        try:
            self._spool.flush()
            self._spool.cancel()
        finally:
            self._changed.set()

    def _request_stop(self, *, cancel: bool) -> None:
        """Stop worker activity independently of the spool's write availability."""
        self._stop.set()
        loop, task = self._loop, self._task
        if cancel and not self._cancel_requested and loop is not None and task is not None:
            self._cancel_requested = True
            try:
                loop.call_soon_threadsafe(self._deliver_cancellation)
            except RuntimeError:
                if not loop.is_closed():
                    raise
        self._notify()

    @property
    def pending_count(self) -> int:
        """Number of batches still awaiting a terminal outcome."""
        return 0 if self._closed else self._spool.pending_count

    @property
    def stats(self) -> dict[str, int]:
        """Analysis outcome counts plus distinct observed/analyzed track counts."""
        return dict(self._stats) if self._closed else self._spool.stats

    def close(self) -> None:
        """Release resources; consume ``finish()`` first to retain all outcomes."""
        if self._closed:
            return
        error: BaseException | None = None
        # Stop the worker before waiting on database locks or a dead connection.
        self._request_stop(cancel=True)
        try:
            if self._spool.pending_count:
                self.cancel()
        except BaseException as exc:
            error = exc
        self._thread.join(timeout=6.0)
        if self._thread.is_alive():
            # Keep the database connection alive until the worker relinquishes
            # it; callers may retry close after it exits.
            raise RuntimeError("Attribute backend did not stop after cancellation.") from error
        try:
            self._stats = self._spool.stats
        except BaseException as exc:
            error = error or exc
        try:
            self._spool.close()
        finally:
            self._closed = True
        if error is not None:
            raise error
        self._check_worker()

    def __enter__(self) -> AttributeEnricher:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        try:
            if exc_type is None:
                for _event in self.finish():
                    pass
            else:
                self.cancel()
        finally:
            self.close()


def create_attribute_enricher(
    reference: AttributeConfig | str | Path | Mapping[str, Any],
    *,
    backend: AttributeBackend | None = None,
) -> AttributeEnricher:
    """Create a session from a profile/config, optionally injecting a test backend."""
    return AttributeEnricher(resolve_attribute_config(reference), backend=backend)


__all__ = ("AttributeBackend", "AttributeEnricher", "create_attribute_enricher")
