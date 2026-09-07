from __future__ import annotations

import dataclasses

import pytest

import boxmot.components.timing as timing


def test_component_timing_is_inactive_without_a_sink(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(timing, "synchronize_torch_device", calls.append)
    monkeypatch.setattr(
        timing.time,
        "perf_counter",
        lambda: (_ for _ in ()).throw(AssertionError("inactive timing must not read the clock")),
    )

    with timing.timed_component_phase("detector", "process", device="cuda:0"):
        calls.append("work")

    assert calls == ["work"]


def test_component_timing_emits_on_interrupt_and_restores_nested_sinks(monkeypatch) -> None:
    clock = iter((1.0, 1.125, 2.0, 2.25, 3.0, 3.5))
    synchronizations: list[object] = []
    outer: list[timing.ComponentTimingEvent] = []
    inner: list[timing.ComponentTimingEvent] = []
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(timing, "synchronize_torch_device", synchronizations.append)

    with timing.timing_event_sink(outer.append):
        with pytest.raises(KeyboardInterrupt):
            with timing.timed_component_phase("detector", "preprocess", device="mps"):
                raise KeyboardInterrupt
        with timing.timing_event_sink(inner.append):
            with timing.timed_component_phase("reid", "process", device="cpu"):
                pass
        with timing.timed_component_phase("detector", "postprocess", device="cpu"):
            pass

    assert outer == [
        timing.ComponentTimingEvent("detector", "preprocess", 125.0),
        timing.ComponentTimingEvent("detector", "postprocess", 500.0),
    ]
    assert inner == [timing.ComponentTimingEvent("reid", "process", 250.0)]
    assert synchronizations == ["mps", "mps", "cpu", "cpu", "cpu", "cpu"]
    assert dataclasses.fields(outer[0])[0].name == "component"
    with pytest.raises(dataclasses.FrozenInstanceError):
        outer[0].elapsed_ms = 1.0  # type: ignore[misc]


def test_timing_event_sink_requires_a_callable() -> None:
    with pytest.raises(TypeError, match="sink must be callable"):
        with timing.timing_event_sink(None):  # type: ignore[arg-type]
            pass
