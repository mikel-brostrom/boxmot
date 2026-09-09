from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import boxmot.engine.tracking.sinks as sink_module
from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.engine.tracking import workflow
from boxmot.engine.tracking.runner import RunSummary, TrackingRunner
from boxmot.engine.tracking.sinks import DisplaySink, JsonLinesSink, MotSink, NullSink, RenderingSink
from boxmot.engine.tracking.workflow import run_track
from boxmot.engine.ui.core.ui import capture_renderable
from boxmot.engine.ui.reporters.track import TrackWorkflowReporter
from boxmot.pipelines import PipelineResult
from boxmot.reid import ReIDEncoderSpec
from boxmot.structures import Boxes, Boxes3D, CameraModel, Detections, Frame, Tracks, Tracks3D
from boxmot.trackers import TrackerRequirements


def _frame(sample_id: str, sequence_id: str, frame_index: int) -> Frame:
    return Frame(
        torch.zeros((3, 8, 10), dtype=torch.uint8),
        sample_id,
        sequence_id=sequence_id,
        frame_index=frame_index,
    )


def _result(frame: Frame) -> PipelineResult:
    boxes = Boxes(torch.tensor([[1.0, 2.0, 5.0, 7.0]], dtype=torch.float32))
    detections = Detections(
        boxes,
        torch.tensor([0.75], dtype=torch.float32),
        torch.tensor([2], dtype=torch.int64),
        frame.sample_id,
    )
    tracks = Tracks(
        boxes,
        torch.tensor([3], dtype=torch.int64),
        torch.tensor([0.75], dtype=torch.float32),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
        frame.sample_id,
    )
    return PipelineResult(detections, tracks)


class _Source:
    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        self.closed = False

    def __iter__(self):
        return iter(self.frames)

    def close(self) -> None:
        self.closed = True


class _Pipeline:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def step(self, frame: Frame) -> PipelineResult:
        return _result(frame)


def test_tracking_runner_owns_sequence_resets_timing_and_cleanup() -> None:
    source = _Source([_frame("a0", "a", 0), _frame("a1", "a", 1), _frame("b0", "b", 0)])
    pipeline = _Pipeline()

    runner = TrackingRunner(source, pipeline)  # type: ignore[arg-type]
    output = list(runner.run())

    assert [frame.sample_id for frame, _ in output] == ["a0", "a1", "b0"]
    assert pipeline.resets == 2
    assert source.closed
    assert runner.summary is not None
    assert runner.summary.frames == 3
    assert runner.summary.sequences == 2
    assert runner.summary.unique_track_ids == 2
    assert runner.summary.interrupted is False
    assert [timing.sample_id for timing in runner.summary.timings] == ["a0", "a1", "b0"]


@pytest.mark.parametrize(
    ("interrupt_index", "completed_samples"),
    ((0, []), (1, ["a0"])),
)
def test_tracking_runner_treats_keyboard_interrupt_as_an_orderly_partial_run(
    interrupt_index: int,
    completed_samples: list[str],
) -> None:
    frames = [_frame("a0", "a", 0), _frame("a1", "a", 1)]
    source = _Source(frames)

    class InterruptingPipeline(_Pipeline):
        def step(self, frame: Frame) -> PipelineResult:
            if frame.frame_index == interrupt_index:
                raise KeyboardInterrupt
            return super().step(frame)

    class RecordingSink:
        def __init__(self) -> None:
            self.samples: list[str] = []
            self.closed = False

        def write(self, frame: Frame, result: PipelineResult) -> None:
            del result
            self.samples.append(frame.sample_id)

        def close(self) -> None:
            self.closed = True

    sink = RecordingSink()
    runner = TrackingRunner(source, InterruptingPipeline(), sinks=(sink,))  # type: ignore[arg-type]

    output = list(runner.run())

    assert [frame.sample_id for frame, _ in output] == completed_samples
    assert source.closed
    assert sink.closed
    assert runner.summary is not None
    assert runner.summary.frames == len(completed_samples)
    assert runner.summary.sequences == 1
    assert runner.summary.interrupted is True
    attempted_samples = [*completed_samples, frames[interrupt_index].sample_id]
    assert [timing.sample_id for timing in runner.summary.timings] == attempted_samples
    assert [timing.completed for timing in runner.summary.timings] == [
        *([True] * len(completed_samples)),
        False,
    ]


def test_track_reporter_labels_an_interrupted_summary_as_stopped() -> None:
    reporter = TrackWorkflowReporter(type("Args", (), {})())
    run = type(
        "Run",
        (),
        {
            "summary": RunSummary(frames=3, sequences=1, elapsed_ms=125.0, interrupted=True),
            "video_path": None,
            "mot_path": None,
            "json_path": None,
        },
    )()

    rendered = capture_renderable(reporter.result(run), width=100)

    assert "Tracking stopped by user" in rendered
    assert "3 frames" in rendered


def test_display_sink_q_stops_tracking_and_closes_the_window(monkeypatch) -> None:
    source = _Source([_frame("a0", "a", 0), _frame("a1", "a", 1)])
    pipeline = _Pipeline()
    shown: list[str] = []
    closed: list[str] = []
    monkeypatch.setattr("boxmot.engine.tracking.sinks.cv2.imshow", lambda name, _image: shown.append(name))
    monkeypatch.setattr("boxmot.engine.tracking.sinks.cv2.waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr("boxmot.engine.tracking.sinks.cv2.destroyWindow", lambda name: closed.append(name))

    runner = TrackingRunner(source, pipeline, sinks=(DisplaySink(),))  # type: ignore[arg-type]
    output = list(runner.run())

    assert [frame.sample_id for frame, _result in output] == ["a0"]
    assert shown == ["BoxMOT"]
    assert closed == ["BoxMOT"]
    assert source.closed
    assert runner.summary is not None
    assert runner.summary.frames == 1


def test_default_save_and_show_render_once_and_share_the_same_image(
    monkeypatch,
    tmp_path,
) -> None:
    render_calls: list[str] = []
    shown: list[np.ndarray] = []
    closed_windows: list[str] = []

    def fake_render(frame: Frame, _result: PipelineResult, **_kwargs) -> np.ndarray:
        render_calls.append(frame.sample_id)
        return np.zeros((frame.height, frame.width, 3), dtype=np.uint8)

    class Writer:
        def __init__(self, *_args) -> None:
            self.frames: list[np.ndarray] = []
            self.released = False

        def isOpened(self) -> bool:
            return True

        def write(self, image: np.ndarray) -> None:
            self.frames.append(image)

        def release(self) -> None:
            self.released = True

    writers: list[Writer] = []

    def make_writer(*args) -> Writer:
        writer = Writer(*args)
        writers.append(writer)
        return writer

    monkeypatch.setattr(sink_module, "render_result", fake_render)
    monkeypatch.setattr(sink_module.cv2, "VideoWriter", make_writer)
    monkeypatch.setattr(sink_module.cv2, "VideoWriter_fourcc", lambda *_codec: 0)
    monkeypatch.setattr(sink_module.cv2, "imshow", lambda _name, image: shown.append(image))
    monkeypatch.setattr(sink_module.cv2, "waitKey", lambda _delay: ord("q"))
    monkeypatch.setattr(sink_module.cv2, "destroyWindow", lambda name: closed_windows.append(name))

    configured_sinks, video_path, _mot_path, _json_path = workflow._default_sinks(
        SimpleNamespace(
            project=tmp_path,
            name="shared-render",
            save=True,
            show=True,
            save_txt=False,
            save_json=False,
            fps=30.0,
            line_width=2,
        )
    )
    source = _Source([_frame("a0", "a", 0), _frame("a1", "a", 1)])
    runner = TrackingRunner(source, _Pipeline(), sinks=configured_sinks)  # type: ignore[arg-type]

    output = list(runner.run())

    assert video_path == tmp_path / "shared-render" / "tracks.mp4"
    assert [frame.sample_id for frame, _result in output] == ["a0"]
    assert render_calls == ["a0"]
    assert len(writers) == 1
    assert writers[0].frames[0] is shown[0]
    assert writers[0].released
    assert closed_windows == ["BoxMOT"]
    assert runner.summary is not None
    assert runner.summary.frames == 1
    assert runner.summary.interrupted is True


def test_text_sinks_serialize_only_at_the_engine_boundary() -> None:
    frame = _frame("sample", "sequence", 0)
    result = _result(frame)
    json_output = StringIO()
    mot_output = StringIO()

    JsonLinesSink(json_output).write(frame, result)
    MotSink(mot_output).write(frame, result)

    assert '"sample_id":"sample"' in json_output.getvalue()
    assert '"track_id":3' in json_output.getvalue()
    assert mot_output.getvalue().startswith("1,3,1.0,2.0,4.0,5.0,0.75,2,-1")


def test_rendering_sink_exposes_a_boundary_image_without_mutating_frame() -> None:
    frame = _frame("sample", "sequence", 0)
    original = frame.image.clone()
    rendered = []

    RenderingSink(lambda _frame, _result, image: rendered.append(image)).write(frame, _result(frame))

    assert rendered[0].shape == (8, 10, 3)
    assert torch.equal(frame.image, original)


def _spatial_result(sample_id: str) -> Tracks3D:
    """One camera-space estimate sharing an image track's identity."""
    return Tracks3D(
        geometry=Boxes3D(torch.tensor([[0, 1, 8, 0, 2, 2, 2]], dtype=torch.float32)),
        track_ids=torch.tensor([3], dtype=torch.int64),
        scores=torch.tensor([0.8], dtype=torch.float32),
        class_ids=torch.tensor([2], dtype=torch.int64),
        detection_indices=torch.tensor([-1], dtype=torch.int64),
        sample_id=sample_id,
    )


def test_spatial_overlay_changes_only_rendered_pixels() -> None:
    frame = _frame("sample", "sequence", 0)
    result = _result(frame)
    spatial = _spatial_result(frame.sample_id)
    original = spatial.geometry.values.clone()
    camera = CameraModel(torch.tensor([[8, 0, 5, 0], [0, 8, 4, 0], [0, 0, 1, 0]], dtype=torch.float32), (8, 10))
    plain = sink_module.render_result(frame, result)

    rendered = sink_module.render_result(frame, result, spatial_tracks=spatial, camera=camera)

    assert np.any(rendered != plain)
    assert not bool(frame.image.any())
    torch.testing.assert_close(spatial.geometry.values, original)
    assert result.tracks.track_ids.tolist() == spatial.track_ids.tolist()


@pytest.mark.parametrize("mismatch", ["missing_camera", "missing_tracks", "sample", "dimensions"])
def test_spatial_overlay_rejects_missing_or_misaligned_camera_inputs(mismatch: str) -> None:
    frame = _frame("sample", "sequence", 0)
    spatial = _spatial_result("different" if mismatch == "sample" else frame.sample_id)
    camera = CameraModel(
        torch.tensor([[8, 0, 5, 0], [0, 8, 4, 0], [0, 0, 1, 0]], dtype=torch.float32),
        (10, 10) if mismatch == "dimensions" else (8, 10),
    )
    with pytest.raises(ValueError):
        sink_module.render_result(
            frame,
            _result(frame),
            spatial_tracks=None if mismatch == "missing_tracks" else spatial,
            camera=None if mismatch == "missing_camera" else camera,
        )


def test_live_workflow_composes_injected_components_through_runner() -> None:
    frames = [_frame("a0", "a", 0), _frame("a1", "a", 1)]

    class Detector:
        capabilities = DetectorCapabilities()

        def predict(self, batch):
            return [_result(frame).detections for frame in batch]

    class Tracker:
        name = "fixture"
        supports_obb = False
        requirements = TrackerRequirements()

        def update(self, detections, frame=None):
            del frame
            return Tracks(
                detections.geometry,
                torch.tensor([3], dtype=torch.int64),
                detections.scores,
                detections.class_ids,
                torch.tensor([0], dtype=torch.int64),
                detections.sample_id,
            )

        def reset(self):
            return None

    source = _Source(frames)

    class Workflow:
        def __init__(self) -> None:
            self.updates: list[str] = []
            self.advances: list[str] = []

        def update(self, detail: str) -> None:
            self.updates.append(detail)

        def advance(self, detail: str) -> None:
            self.advances.append(detail)

    workflow = Workflow()
    run = run_track(
        type("Args", (), {"geometry": "aabb"})(),
        detector=Detector(),
        tracker=Tracker(),
        source=source,
        sinks=(NullSink(),),
        ui_pipeline=workflow,  # type: ignore[arg-type]
    )

    assert run.summary.frames == 2
    assert run.summary.sequences == 1
    assert source.closed
    assert workflow.advances == ["Processing frames…"]
    assert workflow.updates == [
        "Processed 1 frame(s) • a0",
        "Processed 2 frame(s) • a1",
    ]


def test_live_component_specs_apply_device_and_reid_precision_controls(monkeypatch) -> None:
    detector_spec = DetectorSpec(
        "fixture",
        device="profile-device",
        precision="fp32",
        geometry_mode="aabb",
    )
    reid_spec = ReIDEncoderSpec(
        "fixture",
        device="profile-device",
        precision="fp16",
    )
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (detector_spec, {}),
    )
    monkeypatch.setattr(
        workflow,
        "resolve_reid_spec",
        lambda _reference: (reid_spec, {}),
    )

    args = type(
        "Args",
        (),
        {
            "detector": "detector",
            "reid": "reid",
            "device": "mps",
            "half": False,
            "conf": None,
            "iou": None,
            "classes": None,
            "imgsz": None,
            "agnostic_nms": False,
        },
    )()

    resolved_detector = workflow._detector_spec(args, "aabb")
    resolved_reid = workflow._reid_spec(args)

    assert resolved_detector.device == "mps"
    assert resolved_detector.precision == "fp32"
    assert resolved_reid.device == "mps"
    assert resolved_reid.precision == "fp32"

    args.half = True
    assert workflow._reid_spec(args).precision == "fp16"
    assert workflow._detector_spec(args, "aabb").precision == "fp32"


def test_live_detector_spec_expands_single_image_size_to_square(monkeypatch) -> None:
    detector_spec = DetectorSpec("fixture", geometry_mode="aabb")
    monkeypatch.setattr(
        workflow,
        "resolve_detector_spec",
        lambda _reference, *, geometry: (detector_spec, {}),
    )
    args = type(
        "Args",
        (),
        {
            "detector": "detector",
            "device": "cpu",
            "conf": None,
            "iou": None,
            "classes": None,
            "imgsz": 640,
            "agnostic_nms": False,
        },
    )()

    resolved = workflow._detector_spec(args, "aabb")

    assert resolved.option_values()["image_size"] == (640, 640)
