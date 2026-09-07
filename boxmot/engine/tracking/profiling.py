"""Engine-owned profiling for live tracking orchestration.

The domain component contracts intentionally contain no timing methods and
``PipelineResult`` contains only detections and tracks.  This module keeps that
boundary intact by decorating arbitrary protocol implementations at the
engine boundary and collecting optional timing events emitted while their
ordinary public methods run.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, overload

import numpy as np

from boxmot.components.timing import ComponentTimingEvent, timing_event_sink
from boxmot.structures import Detections, Frame, Tracks

RUNTIME_STAGE_KEYS = (
    "source_acquisition",
    "detector_preprocess",
    "detector_inference",
    "detector_postprocess",
    "detector_total",
    "segmentor_preprocess",
    "segmentor_inference",
    "segmentor_postprocess",
    "segmentor_total",
    "reid_preprocess",
    "reid_inference",
    "reid_postprocess",
    "reid_total",
    "enrichment",
    "validation",
    "tracker_association",
    "tracker_update",
    "tracker_total",
    "rendering",
    "sink_io",
    "other_overhead",
    "overall",
)

STARTUP_STAGE_KEYS = (
    "detector_load",
    "tracker_load",
    "reid_load",
    "segmentor_load",
    "source_open",
    "output_prepare",
    "pipeline_prepare",
    "source_first_frame",
)

_COMPONENT_PHASE_STAGES = {
    "detector": {
        "preprocess": "detector_preprocess",
        "process": "detector_inference",
        "inference": "detector_inference",
        "postprocess": "detector_postprocess",
    },
    "segmentor": {
        "preprocess": "segmentor_preprocess",
        "process": "segmentor_inference",
        "inference": "segmentor_inference",
        "postprocess": "segmentor_postprocess",
    },
    "reid": {
        "preprocess": "reid_preprocess",
        "process": "reid_inference",
        "inference": "reid_inference",
        "postprocess": "reid_postprocess",
    },
    "tracker": {
        "association": "tracker_association",
        "process": "tracker_update",
        "update": "tracker_update",
    },
}

_COMPONENT_TOTAL_STAGES = {
    "detector": "detector_total",
    "segmentor": "segmentor_total",
    "reid": "reid_total",
    "tracker": "tracker_total",
}

_COMPONENT_FALLBACK_STAGES = {
    "detector": "detector_inference",
    "segmentor": "segmentor_inference",
    "reid": "reid_inference",
    "tracker": "tracker_update",
}


@dataclass(frozen=True, slots=True)
class RuntimeStageEvent:
    """One engine stage duration emitted by an engine-owned boundary helper."""

    stage: str
    elapsed_ms: float


_RUNTIME_EVENT_SINK: ContextVar[Callable[[RuntimeStageEvent], None] | None] = ContextVar(
    "boxmot_runtime_timing_sink",
    default=None,
)


@contextmanager
def runtime_timing_event_sink(
    sink: Callable[[RuntimeStageEvent], None],
) -> Iterator[None]:
    """Route engine timing events to ``sink`` for the dynamic call scope."""

    token = _RUNTIME_EVENT_SINK.set(sink)
    try:
        yield
    finally:
        _RUNTIME_EVENT_SINK.reset(token)


@contextmanager
def timed_runtime_stage(stage: str) -> Iterator[None]:
    """Measure an engine stage only when a runtime profiler is active."""

    sink = _RUNTIME_EVENT_SINK.get()
    if sink is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        sink(RuntimeStageEvent(stage=stage, elapsed_ms=(time.perf_counter() - started) * 1000.0))


@dataclass(slots=True)
class _ComponentSpan:
    component: str
    started_s: float
    ended_s: float


@dataclass(slots=True)
class _SampleProfile:
    values: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in RUNTIME_STAGE_KEYS})
    pipeline_started_s: float | None = None
    pipeline_ended_s: float | None = None
    spans: list[_ComponentSpan] = field(default_factory=list)


def _sample_ids(frames: Sequence[Frame]) -> tuple[str, ...]:
    identifiers = tuple(frame.sample_id for frame in frames)
    if not identifiers:
        return ()
    return identifiers


class RuntimeProfiler:
    """Collect per-sample and aggregate live-tracking timings."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._samples: dict[str, _SampleProfile] = {}
        self._active_sample_ids: ContextVar[tuple[str, ...]] = ContextVar(
            f"boxmot_runtime_samples_{id(self)}",
            default=(),
        )
        self._active_events: ContextVar[list[ComponentTimingEvent] | None] = ContextVar(
            f"boxmot_component_events_{id(self)}",
            default=None,
        )

    def _sample(self, sample_id: str) -> _SampleProfile:
        return self._samples.setdefault(sample_id, _SampleProfile())

    def add(self, sample_ids: Sequence[str], stage: str, elapsed_ms: float) -> None:
        """Distribute one batch duration over its ordered sample identities."""

        if stage not in RUNTIME_STAGE_KEYS:
            raise ValueError(f"Unknown runtime timing stage {stage!r}.")
        identifiers = tuple(sample_ids)
        if not identifiers:
            return
        value = max(float(elapsed_ms), 0.0) / len(identifiers)
        for sample_id in identifiers:
            self._sample(sample_id).values[stage] += value

    def value(self, sample_id: str, stage: str) -> float:
        """Return one recorded per-sample stage duration."""

        sample = self._samples.get(sample_id)
        return 0.0 if sample is None else float(sample.values.get(stage, 0.0))

    def record_component_event(self, event: ComponentTimingEvent) -> None:
        """Consume one optional domain timing event in the current sample scope."""

        component = str(event.component).strip().lower()
        phase = str(event.phase).strip().lower()
        stage = _COMPONENT_PHASE_STAGES.get(component, {}).get(phase)
        if stage is None:
            return
        identifiers = self._active_sample_ids.get()
        self.add(identifiers, stage, event.elapsed_ms)
        events = self._active_events.get()
        if events is not None:
            events.append(event)

    def record_runtime_event(self, event: RuntimeStageEvent) -> None:
        """Consume one engine stage event in the current sample scope."""

        self.add(self._active_sample_ids.get(), event.stage, event.elapsed_ms)

    @contextmanager
    def activate(self, sample_ids: Sequence[str]) -> Iterator[None]:
        """Activate domain and engine event collection for ``sample_ids``."""

        identifiers = tuple(sample_ids)
        token = self._active_sample_ids.set(identifiers)
        try:
            with timing_event_sink(self.record_component_event), runtime_timing_event_sink(self.record_runtime_event):
                yield
        finally:
            self._active_sample_ids.reset(token)

    @contextmanager
    def component_call(self, component: str, sample_ids: Sequence[str]) -> Iterator[None]:
        """Measure exclusive public component work and synthesize nested totals."""

        canonical = str(component).strip().lower()
        if canonical not in _COMPONENT_TOTAL_STAGES:
            raise ValueError(f"Unknown profiled component {component!r}.")
        identifiers = tuple(sample_ids)
        events: list[ComponentTimingEvent] = []
        event_token = self._active_events.set(events)
        started = self._clock()
        try:
            with self.activate(identifiers):
                yield
        finally:
            ended = self._clock()
            self._active_events.reset(event_token)
            elapsed_ms = max((ended - started) * 1000.0, 0.0)
            # A component may own another timed component. In particular, a
            # Python tracker can lazily run its ReID encoder inside update().
            # Promote those child phase events to a child total and remove
            # their duration from the enclosing total so runtime buckets stay
            # exclusive instead of counting the same wall time twice.
            nested_totals: dict[str, float] = {}
            for event in events:
                event_component = str(event.component).strip().lower()
                if event_component == canonical or event_component not in _COMPONENT_TOTAL_STAGES:
                    continue
                nested_totals[event_component] = nested_totals.get(event_component, 0.0) + max(
                    float(event.elapsed_ms),
                    0.0,
                )
            for event_component, nested_elapsed_ms in nested_totals.items():
                self.add(identifiers, _COMPONENT_TOTAL_STAGES[event_component], nested_elapsed_ms)

            exclusive_elapsed_ms = max(elapsed_ms - sum(nested_totals.values()), 0.0)
            self.add(identifiers, _COMPONENT_TOTAL_STAGES[canonical], exclusive_elapsed_ms)
            if not any(str(event.component).strip().lower() == canonical for event in events):
                self.add(identifiers, _COMPONENT_FALLBACK_STAGES[canonical], exclusive_elapsed_ms)
            for sample_id in identifiers:
                self._sample(sample_id).spans.append(_ComponentSpan(canonical, started, ended))

    def begin_pipeline(self, sample_id: str) -> None:
        """Mark entry into ``TrackingPipeline.step`` for one sample."""

        sample = self._sample(sample_id)
        sample.pipeline_started_s = self._clock()
        sample.pipeline_ended_s = None
        sample.spans.clear()

    def end_pipeline(self, sample_id: str) -> None:
        """Derive exclusive enrichment and validation orchestration residuals."""

        sample = self._sample(sample_id)
        if sample.pipeline_started_s is None:
            return
        ended = self._clock()
        sample.pipeline_ended_s = ended
        tracker_spans = [span for span in sample.spans if span.component == "tracker"]
        perception_spans = [span for span in sample.spans if span.component != "tracker"]
        if tracker_spans:
            tracker_start = min(span.started_s for span in tracker_spans)
            tracker_end = max(span.ended_s for span in tracker_spans)
            perception_elapsed = sum(
                max(min(span.ended_s, tracker_start) - max(span.started_s, sample.pipeline_started_s), 0.0)
                for span in perception_spans
            )
            enrichment_ms = max(
                (tracker_start - sample.pipeline_started_s - perception_elapsed) * 1000.0,
                0.0,
            )
            validation_ms = max((ended - tracker_end) * 1000.0, 0.0)
        else:
            component_elapsed = sum(max(span.ended_s - span.started_s, 0.0) for span in perception_spans)
            enrichment_ms = 0.0
            validation_ms = max(
                ((ended - sample.pipeline_started_s) - component_elapsed) * 1000.0,
                0.0,
            )
        sample.values["enrichment"] += enrichment_ms
        sample.values["validation"] += validation_ms

    def finish_sample(self, sample_id: str, elapsed_ms: float) -> None:
        """Finalize overall and exclusive unclassified overhead for a sample."""

        sample = self._sample(sample_id)
        sample.values["overall"] += max(float(elapsed_ms), 0.0)
        accounted = sum(
            sample.values[key]
            for key in (
                "source_acquisition",
                "detector_total",
                "segmentor_total",
                "reid_total",
                "enrichment",
                "validation",
                "tracker_total",
                "rendering",
                "sink_io",
            )
        )
        sample.values["other_overhead"] = max(sample.values["overall"] - accounted, 0.0)

    def sample_timings_ms(self, sample_id: str) -> dict[str, float]:
        """Return a detached timing snapshot for one sample."""

        sample = self._samples.get(sample_id)
        if sample is None:
            return {key: 0.0 for key in RUNTIME_STAGE_KEYS}
        return {key: float(sample.values[key]) for key in RUNTIME_STAGE_KEYS}

    def totals_ms(self) -> dict[str, float]:
        """Aggregate every attempted sample, including an interrupted sample."""

        return {key: sum(sample.values[key] for sample in self._samples.values()) for key in RUNTIME_STAGE_KEYS}


class ProfiledDetector:
    """Transparent timing decorator for a detector protocol implementation."""

    def __init__(self, component: Any, profiler: RuntimeProfiler) -> None:
        self._component = component
        self._profiler = profiler

    @property
    def capabilities(self) -> Any:
        return self._component.capabilities

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        with self._profiler.component_call("detector", _sample_ids(frames)):
            return self._component.predict(frames)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component, name)


class ProfiledSegmentor:
    """Transparent timing decorator for a segmentor protocol implementation."""

    def __init__(self, component: Any, profiler: RuntimeProfiler) -> None:
        self._component = component
        self._profiler = profiler

    def segment(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
    ) -> list[Any]:
        with self._profiler.component_call("segmentor", _sample_ids(frames)):
            return self._component.segment(frames, detections)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component, name)


class ProfiledAppearanceEncoder:
    """Transparent timing decorator for an appearance encoder implementation."""

    def __init__(self, component: Any, profiler: RuntimeProfiler) -> None:
        self._component = component
        self._profiler = profiler

    @property
    def embedding_dim(self) -> int:
        return self._component.embedding_dim

    @property
    def requirements(self) -> Any:
        return self._component.requirements

    def encode(
        self,
        frames: Sequence[Frame],
        detections: Sequence[Detections],
    ) -> list[Any]:
        with self._profiler.component_call("reid", _sample_ids(frames)):
            return self._component.encode(frames, detections)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component, name)


class ProfiledTracker:
    """Transparent timing decorator for a tracker protocol implementation."""

    def __init__(self, component: Any, profiler: RuntimeProfiler) -> None:
        self._component = component
        self._profiler = profiler
        self._numpy_sample_index = 0

    @property
    def name(self) -> str:
        return self._component.name

    @property
    def capabilities(self) -> Any:
        return self._component.capabilities

    @property
    def requirements(self) -> Any:
        return self._component.requirements

    @property
    def supports_obb(self) -> bool:
        return self._component.supports_obb

    @property
    def generates_embeddings(self) -> bool:
        return getattr(self._component, "generates_embeddings", False)

    @overload
    def update(self, detections: Detections, frame: Frame | None = None) -> Tracks: ...

    @overload
    def update(self, detections: np.ndarray, frame: Frame | None = None) -> np.ndarray: ...

    def update(self, detections: Detections | np.ndarray, frame: Frame | None = None) -> Tracks | np.ndarray:
        if isinstance(detections, Detections):
            sample_id = detections.sample_id
        elif isinstance(frame, Frame):
            sample_id = frame.sample_id
        else:
            sample_id = f"numpy:{self._numpy_sample_index:06d}"
        with self._profiler.component_call("tracker", (sample_id,)):
            result = self._component.update(detections, frame)
        if type(detections) is np.ndarray and frame is None:
            self._numpy_sample_index += 1
        return result

    def reset(self) -> None:
        self._component.reset()
        self._numpy_sample_index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._component, name)


def profile_components(
    profiler: RuntimeProfiler,
    *,
    detector: Any | None,
    segmentor: Any | None,
    reid: Any | None,
    tracker: Any,
) -> tuple[Any | None, Any | None, Any | None, Any]:
    """Decorate resolved components without changing their public contracts."""

    return (
        None if detector is None else ProfiledDetector(detector, profiler),
        None if segmentor is None else ProfiledSegmentor(segmentor, profiler),
        None if reid is None else ProfiledAppearanceEncoder(reid, profiler),
        ProfiledTracker(tracker, profiler),
    )


@contextmanager
def startup_stage(
    timings_ms: dict[str, float],
    stage: str,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> Iterator[None]:
    """Accumulate one workflow setup duration in milliseconds."""

    if stage not in STARTUP_STAGE_KEYS:
        raise ValueError(f"Unknown startup timing stage {stage!r}.")
    started = clock()
    try:
        yield
    finally:
        timings_ms[stage] = timings_ms.get(stage, 0.0) + max(
            (clock() - started) * 1000.0,
            0.0,
        )


def normalize_startup_timings(values: Mapping[str, float] | None = None) -> dict[str, float]:
    """Return stable startup timing keys plus their non-overlapping total."""

    source = values or {}
    normalized = {key: max(float(source.get(key, 0.0)), 0.0) for key in STARTUP_STAGE_KEYS}
    normalized["total"] = sum(normalized.values())
    return normalized


__all__ = (
    "RUNTIME_STAGE_KEYS",
    "STARTUP_STAGE_KEYS",
    "ProfiledAppearanceEncoder",
    "ProfiledDetector",
    "ProfiledSegmentor",
    "ProfiledTracker",
    "RuntimeProfiler",
    "RuntimeStageEvent",
    "normalize_startup_timings",
    "profile_components",
    "runtime_timing_event_sink",
    "startup_stage",
    "timed_runtime_stage",
)
