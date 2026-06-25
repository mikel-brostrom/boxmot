from __future__ import annotations

from types import SimpleNamespace

import pytest

import boxmot.engine.tracking.workflow as tracker_module
import boxmot.utils.rich.core.ui as ui_module
from boxmot.engine.workflows import support as workflow_support_module


def test_should_consume_result_for_finite_sources_without_output():
    args = SimpleNamespace(
        source="assets/DOTA8-MOT/train/P1142__1024__0___824/img1",
        save=False,
        save_txt=False,
        show=False,
    )

    assert tracker_module._should_consume_result(args) is True


def test_consume_run_iterates_results_and_refreshes():
    events = []

    class _FakeResults:
        def __iter__(self):
            events.append(("iter", None))

            def _gen():
                yield object()

            return _gen()

    class _FakeRun:
        def __init__(self):
            self.results = _FakeResults()

        def show(self):
            events.append(("show", None))

        def refresh(self):
            events.append(("refresh", None))

    tracker_module._consume_run(_FakeRun())

    assert ("iter", None) in events
    assert ("refresh", None) in events
    assert ("show", None) not in events


def test_should_consume_result_keeps_live_and_output_sources_lazy():
    assert (
        tracker_module._should_consume_result(
            SimpleNamespace(source="0", save=False, save_txt=False, show=False)
        )
        is False
    )
    assert (
        tracker_module._should_consume_result(
            SimpleNamespace(source="rtsp://camera/stream", save=False, save_txt=False, show=False)
        )
        is False
    )
    assert (
        tracker_module._should_consume_result(
            SimpleNamespace(source="video.mp4", save=True, save_txt=False, show=False)
        )
        is False
    )
    assert (
        tracker_module._should_consume_result(
            SimpleNamespace(source="video.mp4", save=False, save_txt=True, show=False)
        )
        is False
    )
    assert (
        tracker_module._should_consume_result(
            SimpleNamespace(source="video.mp4", save=False, save_txt=False, show=True)
        )
        is False
    )


def test_workflow_support_routes_cpp_live_tracker_backend(monkeypatch):
    seen = {}

    class _FakeNativeBackend:
        def create_tracker(self, cfg, **kwargs):
            seen["cfg"] = cfg
            seen["kwargs"] = kwargs
            return "native-botsort"

    monkeypatch.setattr(workflow_support_module, "get_native_live_backend", lambda name: _FakeNativeBackend())

    tracker = workflow_support_module.build_tracker_from_spec("botsort", tracker_backend="cpp")

    assert tracker == "native-botsort"
    assert "track_high_thresh" in seen["cfg"]
    assert seen["kwargs"] == {"reid_weights": None, "reid_preprocess": None}


def test_workflow_support_routes_cpp_live_bytetrack_backend(monkeypatch):
    seen = {}

    class _FakeNativeBackend:
        def create_tracker(self, cfg, **kwargs):
            seen["cfg"] = cfg
            seen["kwargs"] = kwargs
            return "native-bytetrack"

    monkeypatch.setattr(workflow_support_module, "get_native_live_backend", lambda name: _FakeNativeBackend())

    tracker = workflow_support_module.build_tracker_from_spec("bytetrack", tracker_backend="cpp")

    assert tracker == "native-bytetrack"
    assert "track_thresh" in seen["cfg"]
    assert seen["kwargs"] == {"reid_weights": None, "reid_preprocess": None}


def test_workflow_support_routes_cpp_live_sfsort_backend(monkeypatch):
    seen = {}

    class _FakeNativeBackend:
        def create_tracker(self, cfg, **kwargs):
            seen["cfg"] = cfg
            seen["kwargs"] = kwargs
            return "native-sfsort"

    monkeypatch.setattr(workflow_support_module, "get_native_live_backend", lambda name: _FakeNativeBackend())

    tracker = workflow_support_module.build_tracker_from_spec("sfsort", tracker_backend="cpp")

    assert tracker == "native-sfsort"
    assert "high_th" in seen["cfg"]
    assert seen["kwargs"] == {"reid_weights": None, "reid_preprocess": None}


def test_workflow_support_routes_cpp_live_ocsort_backend(monkeypatch):
    seen = {}

    class _FakeNativeBackend:
        def create_tracker(self, cfg, **kwargs):
            seen["cfg"] = cfg
            seen["kwargs"] = kwargs
            return "native-ocsort"

    monkeypatch.setattr(workflow_support_module, "get_native_live_backend", lambda name: _FakeNativeBackend())

    tracker = workflow_support_module.build_tracker_from_spec("ocsort", tracker_backend="cpp")

    assert tracker == "native-ocsort"
    assert "det_thresh" in seen["cfg"]
    assert seen["kwargs"] == {"reid_weights": None, "reid_preprocess": None}


def test_workflow_support_rejects_unsupported_cpp_live_tracker_backend():
    with pytest.raises(
        ValueError,
        match=r"Available native live trackers: botsort, bytetrack, occluboost, ocsort, sfsort",
    ):
        workflow_support_module.build_tracker_from_spec("deepocsort", tracker_backend="cpp")


def test_build_tracker_with_reid_spec_skips_python_reid_when_native_tracker_provides_it(monkeypatch):
    calls = []

    def fake_build_reid(*args, **kwargs):
        calls.append((args, kwargs))
        return "python-reid"

    tracker = SimpleNamespace(with_reid=True, provides_reid=True)
    monkeypatch.setattr(workflow_support_module, "build_reid_from_spec", fake_build_reid)

    reid = workflow_support_module.build_tracker_with_reid_spec(
        "botsort",
        tracker,
        "models/lmbn_n_duke.onnx",
    )

    assert reid is None
    assert calls == []


def test_build_tracker_with_reid_spec_skips_reid_for_nonreid_tracker(monkeypatch):
    calls = []

    def fake_build_reid(*args, **kwargs):
        calls.append((args, kwargs))
        return "python-reid"

    tracker = SimpleNamespace()
    monkeypatch.setattr(workflow_support_module, "build_reid_from_spec", fake_build_reid)

    reid = workflow_support_module.build_tracker_with_reid_spec(
        "bytetrack",
        tracker,
        "models/osnet_x0_25_msmt17.onnx",
    )

    assert reid is None
    assert calls == []


def test_run_track_routes_progress_into_workflow(monkeypatch, tmp_path):
    created = {}

    class _FakeResults:
        def __init__(self, *args, verbose=True, drawer=None, progress_callback=None, **kwargs):
            _ = (args, drawer, kwargs)
            created["verbose"] = verbose
            created["progress_callback"] = progress_callback
            self._cache_results = True

        def __iter__(self):
            def _gen():
                created["progress_callback"]("Frame 1 | Det: 1.0ms | Track: 2.0ms | Total: 3.0ms")
                yield object()

            return _gen()

        def summary(self):
            return {
                "source": str(tmp_path / "video.mp4"),
                "frames": 1,
                "detections": 2,
                "tracks": 1,
                "unique_tracks": 1,
                "timings_ms": {
                    "det": 1.0,
                    "reid": 0.0,
                    "track": 2.0,
                    "total": 3.0,
                    "avg_total": 3.0,
                },
            }

        def format_summary(self):
            return "TRACKING SUMMARY"

        def show(self):
            raise AssertionError("show() should not be called for finite non-display runs")

    class _FakeWorkflow:
        def __init__(self):
            self.steps = [
                (tracker_module.TRACK_SETUP_STEP, "active"),
                (tracker_module.TRACK_RUN_STEP, "todo"),
            ]
            self.details = []
            self.renderable_details = []
            self.completed = []
            self.detail_renderable = None
            self.detail_text = None

        def set_detail(self, title, text, *, render=True):
            self.details.append((title, text, render))

        def set_detail_renderable(self, title, renderable, *, render=True):
            self.renderable_details.append((title, ui_module.capture_renderable(renderable, width=120), render))

        def complete(self, label, *, render=True):
            self.completed.append((label, render))

        def transition(self, done, next_step, detail=None):
            self.completed.append((done, False))
            if detail:
                self.details.append((next_step, detail, True))

    from boxmot.utils.rich.workflow.pipeline import PipelineTracker

    monkeypatch.setattr(tracker_module, "Results", _FakeResults)

    workflow = _FakeWorkflow()
    pipeline = PipelineTracker(workflow, wire_status_fns=False)
    result = tracker_module.run_track(
        SimpleNamespace(
            source=str(tmp_path / "video.mp4"),
            verbose=True,
            project=tmp_path / "runs",
            save=False,
            save_txt=False,
            show=False,
        ),
        detector=object(),
        reid=None,
        tracker=object(),
        pipeline=pipeline,
    )

    assert created["verbose"] is False
    assert created["progress_callback"] is not None
    # pipeline.advance() completes SETUP and sets detail on RUN
    assert workflow.completed[0] == (tracker_module.TRACK_SETUP_STEP, False)
    assert (
        tracker_module.TRACK_RUN_STEP,
        "Frame 1 | Det: 1.0ms | Track: 2.0ms | Total: 3.0ms",
        True,
    ) in workflow.details
    assert len(workflow.renderable_details) == 1
    assert workflow.renderable_details[0][0] == "Summary"
    assert "TRACKING SUMMARY" in workflow.renderable_details[0][1]
    assert "Stage" in workflow.renderable_details[0][1]
    assert result.summary["frames"] == 1


def test_main_starts_and_stops_tracking_workflow(monkeypatch, tmp_path):
    workflows = []
    calls = []

    class _FakeWorkflow:
        def __init__(self, title, fields, steps, stderr=False, transient=False):
            self.title = title
            self.fields = list(fields)
            self.steps = list(steps)
            self.stderr = stderr
            self.transient = transient
            self.started = False
            self.stopped = False
            self.prefer_alt_screen = False
            self.prefer_compact_layout = False
            self._live = None

        def start(self):
            self.started = True
            return self

        def stop(self):
            self.stopped = True

        def __enter__(self):
            self.start()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val is not None:
                self.fail(error=exc_val)
            self.stop()

        def fail(self, label=None, error=None, *, render=True):
            return None

        def renderable(self, **kwargs):
            return ""

    def fake_create_workflow_progress(title, fields, *, steps=(), stderr=False, transient=False):
        workflow = _FakeWorkflow(title, fields, steps, stderr=stderr, transient=transient)
        workflows.append(workflow)
        return workflow

    def fake_run_track(args, **kwargs):
        calls.append((args, kwargs))
        return "track-result"

    monkeypatch.setattr(ui_module, "create_workflow_progress", fake_create_workflow_progress)
    monkeypatch.setattr(ui_module, "print_renderable", lambda *a, **kw: None)
    monkeypatch.setattr(tracker_module, "run_track", fake_run_track)

    result = tracker_module.main(
        SimpleNamespace(
            detector=tmp_path / "detector.pt",
            reid=tmp_path / "reid.onnx",
            tracker="botsort",
            tracker_backend="python",
            source="0",
            device="cpu",
            half=False,
            imgsz=None,
            conf=None,
            iou=0.7,
            show=True,
            save=False,
            save_txt=False,
        )
    )

    assert result == "track-result"
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.title == "Tracking"
    assert workflow.started is True
    assert workflow.stopped is True
    # Cards are now subsystem-based (like eval view)
    tracker_card = dict(workflow.fields).get("__panel__:Tracker")
    assert tracker_card is not None
    assert ("Name", "botsort") in tracker_card
    source_card = dict(workflow.fields).get("__panel__:Source")
    assert source_card is not None
    assert ("Input", "0") in source_card
    assert (tracker_module.TRACK_SETUP_STEP, "active") in workflow.steps
    assert (tracker_module.TRACK_RUN_STEP, "todo") in workflow.steps
    assert calls == [
        (
            calls[0][0],
            {
                "detector_spec": tmp_path / "detector.pt",
                "reid_spec": tmp_path / "reid.onnx",
                "tracker_spec": "botsort",
                "classes": None,
                "pipeline": calls[0][1]["pipeline"],
            },
        )
    ]
