"""Engine orchestration for stateful tracking pipelines."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from boxmot.engine.tracking.profiling import RuntimeProfiler, normalize_startup_timings
from boxmot.engine.tracking.sinks import TrackSink, _StopTrackingRequested
from boxmot.engine.tracking.sources import FrameSource
from boxmot.pipelines import PipelineResult, TrackingPipeline
from boxmot.structures import Frame


@dataclass(frozen=True, slots=True)
class FrameTiming:
    """Engine timing metadata for one sample."""

    sample_id: str
    elapsed_ms: float
    stage_timings_ms: Mapping[str, float] = field(default_factory=dict)
    completed: bool = True


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Run counters, termination state, and per-sample timings."""

    frames: int
    sequences: int
    elapsed_ms: float
    timings: tuple[FrameTiming, ...] = field(default_factory=tuple)
    interrupted: bool = False
    detections: int = 0
    track_rows: int = 0
    unique_track_ids: int = 0
    startup_timings_ms: Mapping[str, float] = field(default_factory=dict)
    stage_timings_ms: Mapping[str, float] = field(default_factory=dict)


class TrackingRunner:
    """Own source iteration, sequence lifecycle, timing, sinks, and cleanup."""

    def __init__(
        self,
        source: FrameSource,
        pipeline: TrackingPipeline,
        *,
        sinks: Sequence[TrackSink] = (),
        progress: Callable[[int, Frame, PipelineResult], None] | None = None,
        profiler: RuntimeProfiler | None = None,
        startup_timings_ms: Mapping[str, float] | None = None,
    ) -> None:
        self.source = source
        self.pipeline = pipeline
        self.sinks = tuple(sinks)
        self.progress = progress
        self.profiler = profiler or RuntimeProfiler()
        self.startup_timings_ms = dict(startup_timings_ms or {})
        self.summary: RunSummary | None = None

    def run(self) -> Iterator[tuple[Frame, PipelineResult]]:
        """Yield results while guaranteeing source/sink cleanup."""

        started = time.perf_counter()
        timings: list[FrameTiming] = []
        frame_count = 0
        sequence_count = 0
        detection_count = 0
        track_row_count = 0
        unique_track_ids: set[tuple[str, int]] = set()
        active_sequence: str | None = None
        interrupted = False
        first_source_result = True
        try:
            source_iterator = None
            while True:
                acquisition_started = time.perf_counter()
                try:
                    if source_iterator is None:
                        source_iterator = iter(self.source)
                    frame = next(source_iterator)
                except StopIteration:
                    break
                except KeyboardInterrupt:
                    interrupted = True
                    if first_source_result:
                        self.startup_timings_ms["source_first_frame"] = (
                            time.perf_counter() - acquisition_started
                        ) * 1000.0
                        first_source_result = False
                    break
                acquired_at = time.perf_counter()
                acquisition_ms = (acquired_at - acquisition_started) * 1000.0
                if first_source_result:
                    self.startup_timings_ms["source_first_frame"] = acquisition_ms
                    first_source_result = False
                self.profiler.add((frame.sample_id,), "source_acquisition", acquisition_ms)

                sequence = frame.sequence_id or frame.sample_id
                overall_started = acquisition_started
                result: PipelineResult | None = None
                stop_requested = False
                frame_completed = False
                try:
                    with self.profiler.activate((frame.sample_id,)):
                        if active_sequence != sequence:
                            self.pipeline.reset()
                            active_sequence = sequence
                            sequence_count += 1

                        self.profiler.begin_pipeline(frame.sample_id)
                        try:
                            result = self.pipeline.step(frame)
                        finally:
                            self.profiler.end_pipeline(frame.sample_id)

                        for sink in self.sinks:
                            rendering_before = self.profiler.value(frame.sample_id, "rendering")
                            sink_started = time.perf_counter()
                            try:
                                sink.write(frame, result)
                            except _StopTrackingRequested:
                                stop_requested = True
                                interrupted = True
                            finally:
                                sink_elapsed_ms = (time.perf_counter() - sink_started) * 1000.0
                                rendering_ms = max(
                                    self.profiler.value(frame.sample_id, "rendering") - rendering_before,
                                    0.0,
                                )
                                self.profiler.add(
                                    (frame.sample_id,),
                                    "sink_io",
                                    max(sink_elapsed_ms - rendering_ms, 0.0),
                                )

                        frame_count += 1
                        detection_count += len(result.detections)
                        track_row_count += len(result.tracks)
                        unique_track_ids.update((sequence, int(value)) for value in result.tracks.track_ids.tolist())
                        frame_completed = True
                        if self.progress is not None:
                            self.progress(frame_count, frame, result)
                except KeyboardInterrupt:
                    interrupted = True
                    stop_requested = True
                finally:
                    self.profiler.finish_sample(
                        frame.sample_id,
                        (time.perf_counter() - overall_started) * 1000.0,
                    )

                sample_timings = self.profiler.sample_timings_ms(frame.sample_id)
                timings.append(
                    FrameTiming(
                        sample_id=frame.sample_id,
                        elapsed_ms=sample_timings["overall"],
                        stage_timings_ms=sample_timings,
                        completed=frame_completed,
                    )
                )
                if result is not None and frame_completed:
                    yield frame, result
                if stop_requested or not frame_completed:
                    break
        except KeyboardInterrupt:
            # Interrupts raised while constructing/advancing a source iterator
            # are orderly shutdown requests too.  Per-frame interrupts are
            # handled above so their partial component timings are retained.
            interrupted = True
        finally:
            close_error: BaseException | None = None
            for sink in reversed(self.sinks):
                try:
                    sink.close()
                except BaseException as exc:  # noqa: BLE001 - finish cleanup before propagating
                    close_error = close_error or exc
            try:
                self.source.close()
            except BaseException as exc:  # noqa: BLE001
                close_error = close_error or exc
            self.summary = RunSummary(
                frames=frame_count,
                sequences=sequence_count,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                timings=tuple(timings),
                interrupted=interrupted,
                detections=detection_count,
                track_rows=track_row_count,
                unique_track_ids=len(unique_track_ids),
                startup_timings_ms=normalize_startup_timings(self.startup_timings_ms),
                stage_timings_ms=self.profiler.totals_ms(),
            )
            if close_error is not None:
                raise close_error


__all__ = ("FrameTiming", "RunSummary", "TrackingRunner")
