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


def test_component_call_is_inactive_without_a_sink(monkeypatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(timing, "synchronize_torch_device", calls.append)
    monkeypatch.setattr(
        timing.time,
        "perf_counter",
        lambda: (_ for _ in ()).throw(AssertionError("inactive timing must not read the clock")),
    )
    with timing.timed_component_call("reid", device="cuda:0"):
        calls.append("work")
    assert calls == ["work"]


def test_component_call_supplies_process_fallback_and_keeps_other_component_events(monkeypatch) -> None:
    clock = iter((1.0, 1.125, 1.250, 1.500))
    events: list[timing.ComponentTimingEvent] = []
    synchronizations: list[object] = []
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(timing, "synchronize_torch_device", synchronizations.append)
    with timing.timing_event_sink(events.append):
        with timing.timed_component_call("reid", device="cpu"):
            with timing.timed_component_phase("detector", "process", device="cpu"):
                pass
    assert events == [
        timing.ComponentTimingEvent("detector", "process", 125.0),
        timing.ComponentTimingEvent("reid", "process", 500.0),
    ]
    assert synchronizations == ["cpu"] * 4


def test_component_call_forwards_internal_events_without_double_counting(monkeypatch) -> None:
    clock = iter((1.0, 1.125, 1.250, 1.375, 1.500))
    events: list[timing.ComponentTimingEvent] = []
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(timing, "synchronize_torch_device", lambda _: None)
    with timing.timing_event_sink(events.append):
        with timing.timed_component_call("ReID"):
            with timing.timed_component_phase("reid", "preprocess"):
                pass
            with timing.timed_component_phase("reid", "process"):
                pass
    assert events == [
        timing.ComponentTimingEvent("reid", "preprocess", 125.0),
        timing.ComponentTimingEvent("reid", "process", 125.0),
    ]


def test_nested_component_calls_emit_one_fallback_and_restore_sink_after_interrupt(monkeypatch) -> None:
    clock = iter((1.0, 1.125, 1.250, 2.0, 2.500))
    events: list[timing.ComponentTimingEvent] = []
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(timing, "synchronize_torch_device", lambda _: None)
    with timing.timing_event_sink(events.append):
        with pytest.raises(KeyboardInterrupt):
            with timing.timed_component_call("reid"):
                with timing.timed_component_call("reid"):
                    raise KeyboardInterrupt
        with timing.timed_component_phase("tracker", "process"):
            pass
    assert events == [
        timing.ComponentTimingEvent("reid", "process", 125.0),
        timing.ComponentTimingEvent("tracker", "process", 500.0),
    ]
