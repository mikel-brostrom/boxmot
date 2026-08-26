from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import boxmot.engine.tracking.results as tracking_results_module
import boxmot.engine.tracking.runtime as tracker_runtime_module
import boxmot.engine.tracking.workflow as tracker_module
import boxmot.engine.workflows.results as workflow_results_module
import boxmot.utils.rich.core.ui as ui_module
from boxmot.engine.workflows import support as workflow_support_module
from boxmot.trackers import OccluBoost
from boxmot.trackers.common.geometry.obb import xywha_to_corners


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


def test_run_track_reports_setup_timings_and_progress(monkeypatch, tmp_path):
    detector = object()
    tracker = object()
    details = []

    class _FakePipeline:
        def update(self, detail, step=None):
            details.append(("update", detail, step))

        def advance(self, detail=None):
            details.append(("advance", detail, None))

        @staticmethod
        def callback():
            return lambda _message: None

        def complete_step(self):
            details.append(("complete", None, None))

        def set_detail_renderable(self, title, _renderable, **_kwargs):
            details.append(("renderable", title, None))

    monkeypatch.setattr(tracker_module, "_build_detector", lambda *args, **kwargs: detector)
    monkeypatch.setattr(
        tracker_module,
        "resolve_tracker_class_metadata",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(tracker_module, "_build_tracker", lambda *args, **kwargs: tracker)
    monkeypatch.setattr(tracker_module, "_build_reid", lambda *args, **kwargs: None)
    counter_values = iter([0.0, 0.1, 0.1, 0.3, 0.3, 0.35, 0.35, 0.375])
    monkeypatch.setattr(tracker_module.time, "perf_counter", lambda: next(counter_values))

    result = tracker_module.run_track(
        SimpleNamespace(
            source="0",
            verbose=False,
            project=tmp_path / "runs",
            save=False,
            save_txt=False,
            show=False,
        ),
        detector_spec="detector",
        reid_spec="reid",
        tracker_spec="tracker",
        pipeline=_FakePipeline(),
    )

    expected = {
        "detector_load": 100.0,
        "tracker_reid_load": 200.0,
        "reid_adapter": 50.0,
        "output_prepare": 25.0,
        "source_first_frame": 0.0,
        "total": 375.0,
    }
    assert result.summary["setup_timings_ms"] == pytest.approx(expected)
    assert result.setup_timings == pytest.approx(expected)
    assert all(isinstance(value, float) for value in result.timings.values())
    assert [detail[1] for detail in details[:5]] == [
        "Loading detector...",
        "Loading tracker and ReID model...",
        "Preparing ReID inference stage...",
        "Preparing tracking outputs...",
        "Opening source and waiting for the first frame...",
    ]
    assert ("renderable", "No frames processed.", None) in details
    rendered = ui_module.capture_renderable(result.renderable(), width=120)
    assert "Startup total" in rendered
    assert "First source frame" in rendered
    setup_rendered = ui_module.capture_renderable(
        workflow_results_module._build_tracking_startup_timing_table(result.summary),
        width=120,
    )
    assert "Avg (ms)" not in setup_rendered
    assert "FPS" not in setup_rendered
    text_summary = result.format_summary()
    startup_text = text_summary.split("Startup\n", 1)[1].split("Stage", 1)[0]
    assert "Startup total" in startup_text
    assert "FPS" not in startup_text


def test_results_times_existing_iterator_until_first_source_frame(monkeypatch):
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    messages = []

    class _FakeDetector:
        def __call__(self, _frame):
            return np.empty((0, 6), dtype=np.float32)

    class _FakeTracker:
        @staticmethod
        def reset():
            return None

        @staticmethod
        def update(_dets, _frame):
            return np.empty((0, 8), dtype=np.float32)

    monkeypatch.setattr(
        tracking_results_module,
        "iter_source",
        lambda _source: iter([("camera", frame)]),
    )
    monotonic_values = iter([10.0, 10.125])
    monkeypatch.setattr(
        tracking_results_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    results = tracking_results_module.Results(
        "0",
        detector=_FakeDetector(),
        reid=None,
        tracker=_FakeTracker(),
        verbose=False,
        progress_callback=messages.append,
    )
    list(results)

    setup = results.summary()["setup_timings_ms"]
    assert setup["source_first_frame"] == pytest.approx(125.0)
    assert setup["total"] == pytest.approx(125.0)
    assert messages[0] == "Opening source and waiting for the first frame..."


def test_results_records_source_acquisition_when_source_is_empty(monkeypatch):
    monkeypatch.setattr(
        tracking_results_module,
        "iter_source",
        lambda _source: iter(()),
    )
    monotonic_values = iter([20.0, 20.05])
    monkeypatch.setattr(
        tracking_results_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    results = tracking_results_module.Results(
        "missing",
        detector=object(),
        reid=None,
        tracker=object(),
        verbose=False,
    )

    assert list(results) == []
    summary = results.summary()
    assert summary["frames"] == 0
    assert summary["setup_timings_ms"]["source_first_frame"] == pytest.approx(50.0)
    assert summary["setup_timings_ms"]["total"] == pytest.approx(50.0)


def test_should_consume_result_keeps_live_and_output_sources_lazy():
    assert (
        tracker_module._should_consume_result(SimpleNamespace(source="0", save=False, save_txt=False, show=False))
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


@pytest.mark.parametrize("source", [0, "0", "rtsp://camera/stream"])
def test_resolve_output_fps_does_not_open_live_sources(source):
    class _NoLiveCapture:
        CAP_PROP_FPS = 5

        @staticmethod
        def VideoCapture(_source):
            raise AssertionError("live source must not be opened just to resolve output FPS")

    assert workflow_support_module.resolve_output_fps(source, fallback=24.0, cv2_module=_NoLiveCapture) == 24.0


def test_run_track_reuses_saved_render_for_display_and_stops_on_quit(tmp_path, monkeypatch):
    rendered_frame = np.full((12, 16, 3), 7, dtype=np.uint8)
    calls = {"frames": 0, "render": 0, "show": 0, "write": 0, "release": 0, "destroy": 0}

    class _FakeFrameResult:
        def render(self):
            calls["render"] += 1
            return rendered_frame

        def show(self, rendered=None):
            calls["show"] += 1
            assert rendered is rendered_frame
            return False

    class _FakeResults:
        def __init__(self, source, detector, reid, tracker, **kwargs):
            self.source = source
            self.tracker = tracker
            self.totals = {
                "det": 0.0,
                "reid": 0.0,
                "track": 0.0,
                "total": 0.0,
                "frames": 0,
                "detections": 0,
                "tracks": 0,
            }

        def __iter__(self):
            for _ in range(2):
                calls["frames"] += 1
                yield _FakeFrameResult()

    class _FakeVideoWriter:
        def __init__(self, path, fourcc, fps, size):
            assert Path(path) == tmp_path / "tracks.mp4"
            assert fourcc == 1234
            assert fps == 24.0
            assert size == (16, 12)

        def write(self, frame):
            assert frame is rendered_frame
            calls["write"] += 1

        def release(self):
            calls["release"] += 1

    monkeypatch.setattr(tracker_module, "Results", _FakeResults)
    monkeypatch.setattr(tracker_module, "resolve_tracker_class_metadata", lambda *args: (None, None))
    monkeypatch.setattr(tracker_module, "resolve_track_output_dir", lambda project, source: tmp_path)
    monkeypatch.setattr(
        tracker_module,
        "resolve_output_fps",
        lambda _source: (_ for _ in ()).throw(AssertionError("explicit --fps must bypass source probing")),
    )
    monkeypatch.setattr(tracker_module.cv2, "VideoWriter_fourcc", lambda *args: 1234)
    monkeypatch.setattr(tracker_module.cv2, "VideoWriter", _FakeVideoWriter)
    monkeypatch.setattr(
        tracker_module.cv2,
        "destroyAllWindows",
        lambda: calls.__setitem__("destroy", calls["destroy"] + 1),
    )

    tracker_module.run_track(
        SimpleNamespace(
            source="0",
            verbose=False,
            project=tmp_path,
            save=True,
            save_txt=False,
            show=True,
            fps=24,
        ),
        detector=object(),
        reid=None,
        tracker=object(),
    )

    assert calls == {"frames": 1, "render": 1, "show": 1, "write": 1, "release": 1, "destroy": 1}


def test_run_track_formats_flushed_obb_gta_as_mmot(tmp_path, monkeypatch):
    gta_rows = np.array(
        [[3, 50, 40, 20, 10, 0.3, 7, 0.8, 2, -1]],
        dtype=np.float32,
    )

    class _FakeTracker:
        def flush_gta(self):
            return gta_rows.copy()

    class _FakeFrameResult:
        @staticmethod
        def to_mot():
            return np.empty((0, 13), dtype=np.float32)

    class _FakeResults:
        def __init__(self, source, detector, reid, tracker, **kwargs):
            self.tracker = tracker

        def __iter__(self):
            yield _FakeFrameResult()

    monkeypatch.setattr(tracker_module, "Results", _FakeResults)
    monkeypatch.setattr(tracker_module, "resolve_tracker_class_metadata", lambda *args: (None, None))
    monkeypatch.setattr(tracker_module, "resolve_track_output_dir", lambda project, source: tmp_path)

    tracker_module.run_track(
        SimpleNamespace(
            source="video.mp4",
            verbose=False,
            project=tmp_path,
            save=False,
            save_txt=True,
            show=False,
        ),
        detector=object(),
        reid=object(),
        tracker=_FakeTracker(),
    )

    saved = np.loadtxt(tmp_path / "tracks.txt", delimiter=",", ndmin=2)
    assert saved.shape == (1, 13)
    np.testing.assert_allclose(saved[0, 2:10], xywha_to_corners(gta_rows[0, 1:6]), atol=1e-5)
    assert int(saved[0, 0]) == 3
    assert int(saved[0, 1]) == 7
    assert int(saved[0, -1]) == -1


def test_tracker_runtime_forwards_precomputed_reid(monkeypatch):
    seen = {}

    class _FakeTracker:
        def update(self, dets, img, **kwargs):
            return []

    def fake_create_tracker(**kwargs):
        seen.update(kwargs)
        return _FakeTracker()

    monkeypatch.setattr(tracker_runtime_module, "create_tracker", fake_create_tracker)

    runtime = tracker_runtime_module.TrackerRuntime.create(
        tracker_name="occluboost",
        reid_weights="unused.pt",
        device="cpu",
        half=False,
        per_class=False,
        precomputed_reid=True,
    )

    assert isinstance(runtime, tracker_runtime_module.TrackerRuntime)
    assert seen["precomputed_reid"] is True


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


def test_resolve_tracker_class_metadata_uses_detector_config_classes():
    args = SimpleNamespace(
        dataset_detector_cfg={"classes": {0: "car", 7: "awning-bike"}},
        classes=None,
    )

    class_ids, class_names = workflow_support_module.resolve_tracker_class_metadata(args)

    assert class_ids == (0, 7)
    assert class_names == {0: "car", 7: "awning-bike"}


def test_resolve_tracker_class_metadata_respects_selected_classes():
    args = SimpleNamespace(
        dataset_detector_cfg={"classes": {0: "car", 7: "awning-bike"}},
        classes=[7],
    )

    class_ids, class_names = workflow_support_module.resolve_tracker_class_metadata(args)

    assert class_ids == (7,)
    assert class_names == {0: "car", 7: "awning-bike"}


def test_resolve_tracker_class_metadata_falls_back_to_detector_backend_names():
    detector = SimpleNamespace(backend=SimpleNamespace(names={3: "van"}))
    args = SimpleNamespace(dataset_detector_cfg=None, classes=None)

    class_ids, class_names = workflow_support_module.resolve_tracker_class_metadata(args, detector)

    assert class_ids == (3,)
    assert class_names == {3: "van"}


def test_build_tracker_from_spec_forwards_class_metadata(monkeypatch):
    seen = {}

    def fake_create_tracker(**kwargs):
        seen.update(kwargs)
        return "tracker"

    monkeypatch.setattr(workflow_support_module, "create_tracker", fake_create_tracker)
    monkeypatch.setattr(workflow_support_module, "select_device", lambda device: device)

    tracker = workflow_support_module.build_tracker_from_spec(
        "bytetrack",
        class_ids=(0, 7),
        class_names={0: "car", 7: "awning-bike"},
    )

    assert tracker == "tracker"
    assert seen["class_ids"] == (0, 7)
    assert seen["class_names"] == {0: "car", 7: "awning-bike"}


def test_build_tracker_from_spec_accepts_tracker_class_and_kwargs(monkeypatch):
    seen = {}

    def fake_create_tracker(**kwargs):
        seen.update(kwargs)
        return "tracker"

    monkeypatch.setattr(workflow_support_module, "create_tracker", fake_create_tracker)
    monkeypatch.setattr(workflow_support_module, "select_device", lambda device: device)

    tracker = workflow_support_module.build_tracker_from_spec(
        OccluBoost,
        reid_model="reid-runtime",
        tracker_kwargs={"with_reid": True, "max_age": 30},
    )

    assert tracker == "tracker"
    assert seen["tracker_type"] == "occluboost"
    assert seen["reid_model"] == "reid-runtime"
    assert seen["tracker_kwargs"] == {"with_reid": True, "max_age": 30}


def test_build_tracker_from_spec_rejects_kwargs_for_initialized_tracker():
    with pytest.raises(ValueError, match="tracker_kwargs"):
        workflow_support_module.build_tracker_from_spec(object(), tracker_kwargs={"with_reid": True})


def test_tracker_reid_model_from_spec_prefers_get_features_object():
    class ReIDRuntime:
        def get_features(self, boxes, image):
            return None

    runtime = ReIDRuntime()

    assert workflow_support_module.tracker_reid_model_from_spec(runtime) is runtime


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
