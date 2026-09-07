from __future__ import annotations

import time

import numpy as np
import pytest
import torch

import boxmot.components.timing as component_timing
from boxmot.components.timing import timed_component_phase
from boxmot.detectors import DetectorCapabilities
from boxmot.engine.tracking.profiling import RUNTIME_STAGE_KEYS, ProfiledTracker, RuntimeProfiler
from boxmot.engine.tracking.sinks import NullSink, RenderingSink
from boxmot.engine.tracking.workflow import run_track
from boxmot.reid import EncoderRequirements
from boxmot.structures import Boxes, Detections, Frame, Tracks
from boxmot.trackers import TrackerRequirements


def _frame(sample_id: str = "sample") -> Frame:
    return Frame(
        torch.zeros((3, 12, 10), dtype=torch.uint8),
        sample_id,
        sequence_id="sequence",
        frame_index=0,
    )


def _detections(frame: Frame) -> Detections:
    return Detections(
        Boxes(torch.tensor([[1.0, 1.0, 6.0, 9.0]], dtype=torch.float32)),
        torch.tensor([0.8], dtype=torch.float32),
        torch.tensor([0], dtype=torch.int64),
        frame.sample_id,
    )


class _Source:
    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self) -> None:
        self.closed = True


class _TimedDetector:
    capabilities = DetectorCapabilities()

    def predict(self, frames):
        with timed_component_phase("detector", "preprocess", device="cpu"):
            time.sleep(0.001)
        with timed_component_phase("detector", "process", device="cpu"):
            time.sleep(0.001)
        with timed_component_phase("detector", "postprocess", device="cpu"):
            time.sleep(0.001)
            return [_detections(frame) for frame in frames]


class _TimedEncoder:
    embedding_dim = 4
    requirements = EncoderRequirements()

    def encode(self, frames, detections):
        with timed_component_phase("reid", "preprocess", device="cpu"):
            time.sleep(0.001)
        with timed_component_phase("reid", "process", device="cpu"):
            time.sleep(0.001)
        with timed_component_phase("reid", "postprocess", device="cpu"):
            time.sleep(0.001)
            return [torch.ones((len(item), 4), dtype=torch.float32) for item in detections]


class _Tracker:
    name = "fixture"
    supports_obb = False
    requirements = TrackerRequirements(embeddings=True)

    def update(self, detections, frame=None):
        del frame
        return Tracks(
            detections.geometry,
            torch.tensor([7], dtype=torch.int64),
            detections.scores,
            detections.class_ids,
            torch.tensor([0], dtype=torch.int64),
            detections.sample_id,
        )

    def reset(self):
        return None


def test_live_workflow_reports_component_phases_engine_stages_and_counts() -> None:
    source = _Source([_frame()])
    rendered: list[tuple[int, ...]] = []

    run = run_track(
        type("Args", (), {"geometry": "aabb"})(),
        detector=_TimedDetector(),
        encoder=_TimedEncoder(),
        tracker=_Tracker(),
        source=source,
        sinks=(RenderingSink(lambda _frame, _result, image: rendered.append(image.shape)),),
    )

    summary = run.summary
    stages = summary.stage_timings_ms
    assert source.closed
    assert rendered == [(12, 10, 3)]
    assert summary.frames == 1
    assert summary.detections == 1
    assert summary.track_rows == 1
    assert summary.unique_track_ids == 1
    assert summary.interrupted is False
    assert tuple(stages) == RUNTIME_STAGE_KEYS
    for stage in (
        "source_acquisition",
        "detector_preprocess",
        "detector_inference",
        "detector_postprocess",
        "detector_total",
        "reid_preprocess",
        "reid_inference",
        "reid_postprocess",
        "reid_total",
        "enrichment",
        "validation",
        "tracker_update",
        "tracker_total",
        "rendering",
        "overall",
    ):
        assert stages[stage] > 0.0, stage
    assert stages["tracker_update"] == pytest.approx(stages["tracker_total"])
    assert summary.startup_timings_ms["pipeline_prepare"] > 0.0
    assert summary.startup_timings_ms["source_first_frame"] > 0.0
    assert summary.startup_timings_ms["total"] > 0.0
    assert len(summary.timings) == 1
    assert summary.timings[0].sample_id == "sample"
    assert summary.timings[0].stage_timings_ms["detector_preprocess"] > 0.0


def test_uninstrumented_components_fall_back_to_public_call_duration() -> None:
    class Detector:
        capabilities = DetectorCapabilities()

        def predict(self, frames):
            time.sleep(0.001)
            return [_detections(frame) for frame in frames]

    tracker = _Tracker()
    tracker.requirements = TrackerRequirements()
    run = run_track(
        type("Args", (), {"geometry": "aabb"})(),
        detector=Detector(),
        tracker=tracker,
        source=_Source([_frame()]),
        sinks=(NullSink(),),
    )

    stages = run.summary.stage_timings_ms
    assert stages["detector_inference"] == pytest.approx(stages["detector_total"])
    assert stages["detector_preprocess"] == 0.0
    assert stages["detector_postprocess"] == 0.0
    assert stages["tracker_update"] == pytest.approx(stages["tracker_total"])


def test_first_frame_keyboard_interrupt_keeps_partial_component_timing() -> None:
    class InterruptingDetector:
        capabilities = DetectorCapabilities()

        def predict(self, frames):
            del frames
            with timed_component_phase("detector", "preprocess", device="cpu"):
                time.sleep(0.001)
                raise KeyboardInterrupt

    source = _Source([_frame()])
    run = run_track(
        type("Args", (), {"geometry": "aabb"})(),
        detector=InterruptingDetector(),
        tracker=_TrackerWithoutEmbeddings(),
        source=source,
        sinks=(NullSink(),),
    )

    summary = run.summary
    assert source.closed
    assert summary.interrupted is True
    assert summary.frames == 0
    assert len(summary.timings) == 1
    assert summary.timings[0].sample_id == "sample"
    assert summary.timings[0].completed is False
    assert summary.timings[0].stage_timings_ms["detector_preprocess"] > 0.0
    assert summary.stage_timings_ms["detector_preprocess"] > 0.0
    assert summary.stage_timings_ms["detector_total"] > 0.0
    assert summary.stage_timings_ms["overall"] > 0.0
    assert summary.startup_timings_ms["source_first_frame"] > 0.0


def test_eager_source_iterator_setup_is_counted_as_source_acquisition() -> None:
    class EagerSource(_Source):
        def __iter__(self):
            time.sleep(0.01)
            return iter(self.frames)

    run = run_track(
        type("Args", (), {"geometry": "aabb"})(),
        detector=_TimedDetector(),
        encoder=_TimedEncoder(),
        tracker=_Tracker(),
        source=EagerSource([_frame()]),
        sinks=(NullSink(),),
    )

    assert run.summary.stage_timings_ms["source_acquisition"] >= 8.0
    assert run.summary.startup_timings_ms["source_first_frame"] >= 8.0


class _TrackerWithoutEmbeddings(_Tracker):
    requirements = TrackerRequirements()


def test_profiled_tracker_forwards_numpy_rows_and_uses_available_sample_identity() -> None:
    class NumPyTracker:
        name = "numpy-fixture"
        supports_obb = False
        requirements = TrackerRequirements()

        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, Frame | None]] = []
            self.reset_calls = 0
            self.output = np.empty((0, 8), dtype=np.float64)

        def update(self, detections, frame=None):
            self.calls.append((detections, frame))
            return self.output

        def reset(self) -> None:
            self.reset_calls += 1

    clock_values = iter((0.0, 0.001, 0.002, 0.004, 0.005, 0.008, 0.009, 0.013))
    profiler = RuntimeProfiler(clock=lambda: next(clock_values))
    component = NumPyTracker()
    tracker = ProfiledTracker(component, profiler)
    rows = np.array([[1, 1, 6, 9, 0.8, 0]], dtype=np.float32)
    frame = _frame("camera-1:000042")

    first = tracker.update(rows)
    framed = tracker.update(rows, frame)
    tracker.update(rows)
    tracker.reset()
    tracker.update(rows)

    assert first is component.output
    assert framed is component.output
    assert component.calls[0][0] is rows
    assert component.calls[0][1] is None
    assert component.calls[1][0] is rows
    assert component.calls[1][1] is frame
    assert component.reset_calls == 1
    assert profiler.value("numpy:000000", "tracker_total") == pytest.approx(5.0)
    assert profiler.value("numpy:000000", "tracker_update") == pytest.approx(5.0)
    assert profiler.value("numpy:000001", "tracker_total") == pytest.approx(3.0)
    assert profiler.value("numpy:000001", "tracker_update") == pytest.approx(3.0)
    assert profiler.value(frame.sample_id, "tracker_total") == pytest.approx(2.0)
    assert profiler.value(frame.sample_id, "tracker_update") == pytest.approx(2.0)


def test_profiled_tracker_accounts_owned_reid_as_exclusive_child_time(monkeypatch) -> None:
    class TrackerWithOwnedReid(_Tracker):
        def update(self, detections, frame=None):
            with timed_component_phase("reid", "preprocess", device="cpu"):
                pass
            with timed_component_phase("reid", "process", device="cpu"):
                pass
            with timed_component_phase("reid", "postprocess", device="cpu"):
                pass
            return super().update(detections, frame)

    phase_clock = iter((1.0, 1.002, 2.0, 2.003, 3.0, 3.005))
    monkeypatch.setattr(component_timing, "synchronize_torch_device", lambda _device: None)
    monkeypatch.setattr(component_timing.time, "perf_counter", lambda: next(phase_clock))
    public_call_clock = iter((10.0, 10.02))
    profiler = RuntimeProfiler(clock=lambda: next(public_call_clock))
    frame = _frame()

    ProfiledTracker(TrackerWithOwnedReid(), profiler).update(_detections(frame), frame)
    profiler.finish_sample(frame.sample_id, 20.0)

    timings = profiler.sample_timings_ms(frame.sample_id)
    assert timings["reid_preprocess"] == pytest.approx(2.0)
    assert timings["reid_inference"] == pytest.approx(3.0)
    assert timings["reid_postprocess"] == pytest.approx(5.0)
    assert timings["reid_total"] == pytest.approx(10.0)
    assert timings["tracker_update"] == pytest.approx(10.0)
    assert timings["tracker_total"] == pytest.approx(10.0)
    assert timings["reid_total"] + timings["tracker_total"] == pytest.approx(timings["overall"])
    assert timings["other_overhead"] == pytest.approx(0.0, abs=1e-9)
