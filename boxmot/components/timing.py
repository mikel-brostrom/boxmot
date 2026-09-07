"""Domain-neutral, opt-in component timing events.

Component packages use this module to describe their internal runtime phases
without depending on the engine that consumes those measurements.  Timing is
inactive unless an engine (or another caller) installs an event sink for the
current context, so independently used components do not pay accelerator
synchronization costs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentTimingEvent:
    """Elapsed time for one internal component phase."""

    component: str
    phase: str
    elapsed_ms: float


TimingEventSink = Callable[[ComponentTimingEvent], None]

_TIMING_EVENT_SINK: ContextVar[TimingEventSink | None] = ContextVar(
    "boxmot_component_timing_event_sink",
    default=None,
)


def synchronize_torch_device(device: object) -> None:
    """Synchronize a CUDA or MPS device; CPU and unknown selectors are no-ops."""

    import torch

    if isinstance(device, torch.device):
        resolved = device
    else:
        value = str(device or "").strip().lower()
        if value.isdecimal():
            value = f"cuda:{value}"
        try:
            resolved = torch.device(value)
        except (RuntimeError, TypeError):
            return

    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)
    elif (
        resolved.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.synchronize()


@contextmanager
def timing_event_sink(sink: TimingEventSink) -> Iterator[None]:
    """Route component timing events to ``sink`` in the current context."""

    if not callable(sink):
        raise TypeError("sink must be callable")
    token = _TIMING_EVENT_SINK.set(sink)
    try:
        yield
    finally:
        _TIMING_EVENT_SINK.reset(token)


@contextmanager
def timed_component_phase(
    component: str,
    phase: str,
    *,
    device: object = None,
) -> Iterator[None]:
    """Measure and emit one component phase when a timing sink is active."""

    sink = _TIMING_EVENT_SINK.get()
    if sink is None:
        yield
        return

    synchronize_torch_device(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        synchronize_torch_device(device)
        sink(
            ComponentTimingEvent(
                component=component,
                phase=phase,
                elapsed_ms=max((time.perf_counter() - started) * 1000.0, 0.0),
            )
        )


__all__ = (
    "ComponentTimingEvent",
    "TimingEventSink",
    "synchronize_torch_device",
    "timed_component_phase",
    "timing_event_sink",
)
