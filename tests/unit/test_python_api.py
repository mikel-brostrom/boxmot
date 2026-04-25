from __future__ import annotations

import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import boxmot
import boxmot.api as api_module
from boxmot.engine import cache as cache_module
from boxmot.engine import evaluator as evaluator_module
from boxmot.engine import export as export_module
from boxmot.engine import research as research_engine_module
from boxmot.engine import tracker as tracker_module
from boxmot.engine import tuner as tuner_module
from boxmot.engine import workflow_reporting as reporting_module
from boxmot.engine import workflow_support as workflow_support_module
import boxmot.engine.results as results_module
from boxmot.configs import BOXMOT_DEFAULTS, DEFAULT_DETECTOR, DEFAULT_REID, get_mode_default
from boxmot.detectors import Detector
from boxmot.detectors.base import Detections
from boxmot.reid import ReID
from boxmot.utils.timing import TimingStats
import boxmot.utils.ui as ui_module


_DUMMY_IMG = np.zeros((32, 32, 3), dtype=np.uint8)


def test_package_root_lazily_reexports_python_api():
    assert "__version__" in boxmot.__all__
    assert "Boxmot" in boxmot.__all__
    assert "GenerateResult" in boxmot.__all__
    assert "ResearchResult" in boxmot.__all__
    assert "track" in boxmot.__all__
    assert boxmot.Boxmot is api_module.Boxmot
    assert boxmot.GenerateResult is api_module.GenerateResult
    assert boxmot.ResearchResult is api_module.ResearchResult
    assert boxmot.track is api_module.track
    assert boxmot.ValidationResult is api_module.ValidationResult


def test_boxmot_defaults_follow_shared_configs():
    model = api_module.Boxmot()

    assert model.detector == DEFAULT_DETECTOR == BOXMOT_DEFAULTS.shared.detector
    assert model.reid == DEFAULT_REID == BOXMOT_DEFAULTS.shared.reid
    assert model._tracker_name() == get_mode_default("track", "tracker") == BOXMOT_DEFAULTS.track.tracker
    assert model.project == Path(get_mode_default("track", "project")) == BOXMOT_DEFAULTS.track.project


def test_boxmot_eval_namespace_uses_shared_reid_default_when_reid_is_none(tmp_path):
    model = api_module.Boxmot(reid=None, project=tmp_path / "runs")

    args = model._base_eval_args("mot17-ablation")

    assert args.reid == [DEFAULT_REID]


def test_boxmot_eval_namespace_treats_inherited_defaults_as_non_explicit():
    model = api_module.Boxmot()

    args = model._base_eval_args("mot17-ablation")

    assert args.detector_explicit is False
    assert args.reid_explicit is False
    assert args.tracker_explicit is False
    assert args.device_explicit is False
    assert args.half_explicit is False
    assert args.tracker_backend == "python"
    assert args.tracking_backend == "thread"


def test_boxmot_eval_namespace_preserves_explicit_constructor_overrides():
    model = api_module.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")

    args = model._base_eval_args("mot17-ablation")

    assert args.detector_explicit is True
    assert args.reid_explicit is True
    assert args.tracker_explicit is True


def test_boxmot_eval_namespace_normalizes_inline_tracker_backend():
    model = api_module.Boxmot(tracker="botsort:cpp")

    args = model._base_eval_args("mot17-ablation")

    assert args.tracker == "botsort"
    assert args.tracker_backend == "cpp"


def test_boxmot_eval_namespace_allows_benchmark_runtime_to_override_inherited_defaults(monkeypatch):
    evaluator_module = importlib.import_module("boxmot.engine.evaluator")
    model = api_module.Boxmot()
    args = model._base_eval_args("mot17-mini")

    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_detector_model",
        lambda _cfg: Path("models/yolox_x_mot17_ablation.pt"),
    )
    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_reid_model",
        lambda _cfg: Path("models/lmbn_n_duke.pt"),
    )

    evaluator_module._configure_benchmark_runtime(args)

    assert args.detector[0].name == "yolox_x_mot17_ablation.pt"
    assert args.reid[0].name == "lmbn_n_duke.pt"
    assert args.reid_half is True


def test_public_reid_supports_boxes_and_crops(monkeypatch):
    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.half = False
            self.input_shape = (8, 4)
            self.mean_array = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            self.std_array = torch.ones((1, 3, 1, 1), dtype=torch.float32)

        def get_features(self, boxes, image):
            return np.full((len(boxes), 2), image.shape[0], dtype=np.float32)

        def inference_preprocess(self, batch):
            return batch

        def forward(self, batch):
            return torch.ones((batch.shape[0], 3), dtype=torch.float32)

        def inference_postprocess(self, features):
            return features.cpu().numpy()

    monkeypatch.setattr(ReID, "get_backend", lambda self: _FakeModel())

    reid = ReID("lmbn_n_duke.pt")

    from_boxes = reid(_DUMMY_IMG, boxes=np.array([[0, 0, 10, 10, 0.9, 0]], dtype=np.float32))
    from_crops = reid([_DUMMY_IMG, _DUMMY_IMG])

    assert from_boxes.shape == (1, 2)
    assert np.allclose(from_boxes, np.full((1, 2), 32.0, dtype=np.float32))
    assert from_crops.shape == (2, 3)
    assert np.all(np.isfinite(from_crops))


def test_public_reid_reuses_preselected_torch_device(monkeypatch):
    reid_module = importlib.import_module("boxmot.reid.core.reid")

    class _FakeBackend:
        pass

    monkeypatch.setattr(ReID, "get_backend", lambda self: _FakeBackend())
    monkeypatch.setattr(
        reid_module,
        "select_device",
        lambda device: (_ for _ in ()).throw(AssertionError("select_device should not run for torch.device inputs")),
    )

    device = torch.device("cpu")
    reid = ReID("lmbn_n_duke.pt", device=device)

    assert reid.device is device


def test_public_detector_and_reid_allow_stage_overrides(monkeypatch):
    class _FakeDetectorBackend:
        def __init__(self, model, device, imgsz):
            self.model = object()

        def __call__(self, images, conf, iou, classes, agnostic_nms):
            return [Detections(dets=np.array([[0, 0, 4, 4, 0.5, 0]], dtype=np.float32), orig_img=images[0])]

    class _PublicDetector(Detector):
        @classmethod
        def _get_backend_class(cls, path):
            return _FakeDetectorBackend

    detector = _PublicDetector("fake.pt")
    detector_calls = []

    def detector_preprocess(frame, **kwargs):
        detector_calls.append("pre")
        return frame

    def detector_process(frame, **kwargs):
        detector_calls.append("proc")
        return Detections(dets=np.array([[1, 2, 6, 8, 0.9, 1]], dtype=np.float32), orig_img=frame)

    def detector_postprocess(result, **kwargs):
        detector_calls.append("post")
        return result.dets + 1

    detector.preprocess = detector_preprocess
    detector.process = detector_process
    detector.postprocess = detector_postprocess

    detector_output = detector(_DUMMY_IMG)

    assert detector_calls == ["pre", "proc", "post"]
    np.testing.assert_array_equal(
        detector_output,
        np.array([[2, 3, 7, 9, 1.9, 2]], dtype=np.float32),
    )

    class _FakeReIDModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.half = False
            self.input_shape = (8, 4)
            self.mean_array = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            self.std_array = torch.ones((1, 3, 1, 1), dtype=torch.float32)

        def get_features(self, boxes, image):
            return np.ones((len(boxes), 2), dtype=np.float32)

        def inference_preprocess(self, batch):
            return batch

        def forward(self, batch):
            return batch

        def inference_postprocess(self, features):
            return features

    monkeypatch.setattr(ReID, "get_backend", lambda self: _FakeReIDModel())

    reid = ReID("fake_reid.pt")
    reid_calls = []

    def reid_preprocess(inputs, boxes=None, **kwargs):
        reid_calls.append("pre")
        return {"features": np.ones((1, 4), dtype=np.float32)}

    def reid_process(payload, **kwargs):
        reid_calls.append("proc")
        return payload["features"]

    def reid_postprocess(features, **kwargs):
        reid_calls.append("post")
        return features + 2

    reid.preprocess = reid_preprocess
    reid.process = reid_process
    reid.postprocess = reid_postprocess

    reid_output = reid(_DUMMY_IMG, boxes=np.array([[0, 0, 4, 4]], dtype=np.float32))

    assert reid_calls == ["pre", "proc", "post"]
    np.testing.assert_array_equal(reid_output, np.full((1, 4), 3.0, dtype=np.float32))


def test_results_save_summary_and_evaluate(tmp_path):
    for index in range(2):
        image_path = tmp_path / f"{index + 1:06d}.jpg"
        cv2.imwrite(str(image_path), _DUMMY_IMG)

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            assert boxes is not None
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    results = api_module.track(tmp_path, _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)
    first = next(iter(results))

    results.drawer = lambda frame, tracks: np.full_like(frame, 127)

    assert first.frame_idx == 1
    assert first.num_tracks == 1
    assert first.render().shape == _DUMMY_IMG.shape
    assert np.all(first.render() == 127)

    output_path = tmp_path / "tracks.txt"
    saved = results.save(output_path)
    summary = results.summary()
    evaluation = api_module.evaluate([results], metrics=True, speed=True)

    assert saved == output_path
    assert output_path.read_text(encoding="utf-8").count("\n") == 2
    assert summary["frames"] == 2
    assert summary["tracks"] == 2
    assert summary["unique_tracks"] == 2
    assert evaluation["metrics"]["frames"] == 2
    assert evaluation["metrics"]["tracks"] == 2


def test_boxmot_track_returns_paths_and_timings(tmp_path, monkeypatch):
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"{index + 1:06d}.jpg"), _DUMMY_IMG)

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    frames_written = []

    class _FakeVideoWriter:
        def __init__(self, path, fourcc, fps, frame_size):
            self.path = path
            self.opened = True

        def write(self, frame):
            frames_written.append(frame.copy())

        def release(self):
            Path(self.path).touch()

    monkeypatch.setattr(workflow_support_module.cv2, "VideoWriter", _FakeVideoWriter)

    model = api_module.Boxmot(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path, save=True, save_txt=True)

    assert run.source == tmp_path
    assert run.video_path is not None and run.video_path.exists()
    assert run.text_path is not None and run.text_path.exists()
    assert run.summary["frames"] == 2
    assert run.summary["unique_tracks"] == 2
    assert run.timings["fps"] >= 0
    assert len(frames_written) == 2


def test_boxmot_track_reuses_tracker_reid_backend_and_suppresses_setup_logs(monkeypatch, tmp_path):
    frames = [("0", _DUMMY_IMG.copy())]
    monkeypatch.setattr(results_module, "iter_source", lambda source: iter(frames))

    suppress_calls = []

    def fake_suppress(enabled, level="WARNING"):
        suppress_calls.append((enabled, level))
        return nullcontext()

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeTrackerBackend:
        def __init__(self):
            self.calls = []

        def get_features(self, boxes, image):
            self.calls.append(np.asarray(boxes, dtype=np.float32).copy())
            return np.full((len(boxes), 2), 7.0, dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.with_reid = True
            self.model = _FakeTrackerBackend()
            self.embeddings = []

        def reset(self):
            return None

        def update(self, dets, frame, embs=None):
            self.embeddings.append(None if embs is None else np.asarray(embs, dtype=np.float32).copy())
            return np.array([[1, 2, 10, 12, 1, 0.9, 0, 0]], dtype=np.float32)

    fake_tracker = _FakeTracker()

    monkeypatch.setattr(tracker_module, "suppress_boxmot_logs", fake_suppress)
    monkeypatch.setattr(tracker_module, "build_detector_from_spec", lambda *args, **kwargs: _FakeDetector())
    monkeypatch.setattr(tracker_module, "build_tracker_from_spec", lambda *args, **kwargs: fake_tracker)

    def fail_build_track_reid(*args, **kwargs):
        raise AssertionError("track() should reuse the tracker ReID backend for built-in ReID trackers")

    monkeypatch.setattr(workflow_support_module, "build_reid_from_spec", fail_build_track_reid)

    model = api_module.Boxmot(
        detector="yolov8n",
        reid="lmbn_n_duke",
        tracker="botsort",
        project=tmp_path / "runs",
    )

    run = model.track(source="0", verbose=False)
    output = list(run)

    assert len(output) == 1
    assert suppress_calls == [(True, "WARNING")]
    assert len(fake_tracker.model.calls) == 1
    np.testing.assert_array_equal(
        fake_tracker.model.calls[0],
        np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32),
    )
    assert len(fake_tracker.embeddings) == 1
    np.testing.assert_array_equal(
        fake_tracker.embeddings[0],
        np.full((1, 2), 7.0, dtype=np.float32),
    )


def test_boxmot_track_keeps_live_sources_lazy(monkeypatch, tmp_path):
    class _FakeResults:
        def __init__(self):
            self.totals = {
                "det": 0.0,
                "reid": 0.0,
                "track": 0.0,
                "total": 0.0,
                "frames": 0,
                "detections": 0,
                "tracks": 0,
            }
            self.materialized = False

        def __iter__(self):
            def _gen():
                self.totals.update({
                    "det": 1.0,
                    "reid": 2.0,
                    "track": 3.0,
                    "total": 6.0,
                    "frames": 1,
                    "detections": 4,
                    "tracks": 5,
                })
                yield SimpleNamespace(frame_idx=1, num_tracks=5, render=lambda: _DUMMY_IMG)

            return _gen()

        def materialize(self):
            self.materialized = True
            raise AssertionError("live sources should not be materialized before iteration")

        def show(self):
            return None

    fake_results = _FakeResults()
    monkeypatch.setattr(tracker_module, "Results", lambda *args, **kwargs: fake_results)

    model = api_module.Boxmot(detector=object(), reid=object(), tracker=object(), project=tmp_path / "runs")
    run = model.track(source="0")

    assert fake_results.materialized is False
    assert run.summary["frames"] == 0
    assert run.summary["unique_tracks"] == 0

    frames = list(run)

    assert len(frames) == 1
    assert run.summary["frames"] == 1
    assert run.summary["tracks"] == 5
    assert run.summary["unique_tracks"] == 0
    assert run.timings["fps"] > 0


def test_results_summary_does_not_resume_live_source_after_partial_iteration(monkeypatch):
    frames = [("0", _DUMMY_IMG.copy()), ("0", _DUMMY_IMG.copy())]
    monkeypatch.setattr(results_module, "iter_source", lambda source: iter(frames))

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    results = api_module.track("0", _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)

    first = next(iter(results))
    summary = results.summary()

    assert first.frame_idx == 1
    assert summary["frames"] == 1
    assert summary["tracks"] == 1
    assert summary["unique_tracks"] == 1


def test_results_live_sources_do_not_cache_frames(monkeypatch):
    frames = [("0", _DUMMY_IMG.copy()), ("0", _DUMMY_IMG.copy())]
    monkeypatch.setattr(results_module, "iter_source", lambda source: iter(frames))

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    results = api_module.track("0", _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)

    first = next(iter(results))

    assert first.frame_idx == 1
    assert results._cache == []

    second = next(results)

    assert second.frame_idx == 2
    assert results._cache == []


def test_boxmot_track_eagerly_consumes_finite_sources_for_uniform_cli_behavior(monkeypatch, tmp_path):
    class _FakeResults:
        def __init__(self):
            self.totals = {
                "det": 0.0,
                "reid": 0.0,
                "track": 0.0,
                "total": 0.0,
                "frames": 0,
                "detections": 0,
                "tracks": 0,
            }
            self.iterated = False

        def __iter__(self):
            def _gen():
                self.iterated = True
                self.totals.update({
                    "det": 1.0,
                    "reid": 2.0,
                    "track": 3.0,
                    "total": 6.0,
                    "frames": 1,
                    "detections": 4,
                    "tracks": 5,
                })
                yield SimpleNamespace(frame_idx=1, num_tracks=5, render=lambda: _DUMMY_IMG)

            return _gen()

        def save(self, output_path):
            raise AssertionError("save should not be called when save_txt is disabled")

        def show(self):
            return None

        def stop(self, reason=None):
            return None

        def format_summary(self):
            return ""

        def print_summary(self):
            return None

    fake_results = _FakeResults()
    monkeypatch.setattr(tracker_module, "Results", lambda *args, **kwargs: fake_results)

    model = api_module.Boxmot(detector=object(), reid=object(), tracker=object(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    assert fake_results.iterated is True
    assert run.summary["frames"] == 1
    assert run.summary["tracks"] == 5


def test_boxmot_track_returns_summary_for_eagerly_consumed_finite_sources(tmp_path):
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"{index + 1:06d}.jpg"), _DUMMY_IMG)

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    model = api_module.Boxmot(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    summary = run.summary

    assert summary["frames"] == 2
    assert summary["tracks"] == 2
    assert summary["unique_tracks"] == 2
    assert run.timings["fps"] >= 0


def test_boxmot_track_show_flag_displays_results(monkeypatch, tmp_path):
    class _FakeResults:
        def __init__(self):
            self.totals = {
                "det": 0.0,
                "reid": 0.0,
                "track": 0.0,
                "total": 0.0,
                "frames": 0,
                "detections": 0,
                "tracks": 0,
            }
            self.shown = False

        def summary(self):
            frames = int(self.totals["frames"])
            avg_total = (self.totals["total"] / frames) if frames else 0.0
            return {
                "source": str(tmp_path),
                "frames": frames,
                "detections": int(self.totals["detections"]),
                "tracks": int(self.totals["tracks"]),
                "unique_tracks": 0,
                "timings_ms": {
                    "det": float(self.totals["det"]),
                    "reid": float(self.totals["reid"]),
                    "track": float(self.totals["track"]),
                    "total": float(self.totals["total"]),
                    "avg_total": float(avg_total),
                },
            }

        def show(self):
            self.shown = True
            self.totals.update({
                "det": 1.0,
                "reid": 2.0,
                "track": 3.0,
                "total": 6.0,
                "frames": 1,
                "detections": 4,
                "tracks": 5,
            })

        def save(self, output_path):
            raise AssertionError("save should not be called when only show=True")

        def stop(self, reason=None):
            return None

        def format_summary(self):
            return ""

        def print_summary(self):
            return None

    fake_results = _FakeResults()
    monkeypatch.setattr(tracker_module, "Results", lambda *args, **kwargs: fake_results)

    model = api_module.Boxmot(detector=object(), reid=object(), tracker=object(), project=tmp_path / "runs")
    run = model.track(source=tmp_path, show=True)

    assert fake_results.shown is True
    assert run.summary["frames"] == 1
    assert run.summary["tracks"] == 5


def test_results_keyboard_interrupt_stops_live_tracking_cleanly(monkeypatch):
    frames = [("0", _DUMMY_IMG.copy()), ("0", _DUMMY_IMG.copy())]
    monkeypatch.setattr(results_module, "iter_source", lambda source: iter(frames))

    class _InterruptingDetector:
        def __init__(self):
            self.calls = 0

        def __call__(self, frame):
            self.calls += 1
            if self.calls > 1:
                raise KeyboardInterrupt()
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    results = api_module.track("0", _InterruptingDetector(), _FakeReID(), _FakeTracker(), verbose=False)

    output = list(results)
    summary = results.summary()

    assert len(output) == 1
    assert summary["frames"] == 1
    assert summary["tracks"] == 1
    assert summary["unique_tracks"] == 1


def test_tracks_show_stops_live_results_on_q(monkeypatch):
    frames = [("0", _DUMMY_IMG.copy()), ("0", _DUMMY_IMG.copy())]
    monkeypatch.setattr(results_module, "iter_source", lambda source: iter(frames))
    monkeypatch.setattr(results_module.cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(results_module.cv2, "waitKey", lambda delay: ord("q"))

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    results = api_module.track("0", _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)

    first = next(iter(results))

    assert first.show() is False
    assert results._interrupted is True
    assert results.summary()["frames"] == 1
    assert results.summary()["unique_tracks"] == 1


def test_track_run_result_formats_summary_block(tmp_path, monkeypatch):
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"{index + 1:06d}.jpg"), _DUMMY_IMG)

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    model = api_module.Boxmot(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    summary_text = run.format_summary()

    assert "TRACKING SUMMARY" in summary_text
    assert "Detection" in summary_text
    assert "Total" in summary_text
    assert "Track rows" in summary_text
    assert "Unique IDs" in summary_text


def test_track_run_result_renderable_uses_rich_summary_layout(tmp_path, monkeypatch):
    for index in range(2):
        cv2.imwrite(str(tmp_path / f"{index + 1:06d}.jpg"), _DUMMY_IMG)

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeReID:
        def __call__(self, frame, boxes=None):
            return np.ones((len(boxes), 4), dtype=np.float32)

    class _FakeTracker:
        def __init__(self):
            self.count = 0

        def reset(self):
            self.count = 0

        def update(self, dets, frame, embs=None):
            self.count += 1
            return np.array([[1, 2, 10, 12, self.count, 0.9, 0, 0]], dtype=np.float32)

    model = api_module.Boxmot(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    rendered = ui_module.capture_renderable(run.renderable(), width=120)

    assert "TRACKING SUMMARY" in rendered
    assert "Track rows" in rendered
    assert "Unique IDs" in rendered
    assert "Component" in rendered
    assert "Detection" in rendered
    assert "ReID" in rendered
    assert "Tracking" in rendered
    assert "Total (ms)" in rendered
    assert "Source" not in rendered


def test_results_summary_splits_tracker_owned_reid_time(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "000001.jpg"), _DUMMY_IMG)

    class _FakeDetector:
        def __call__(self, frame):
            return np.array([[1, 2, 10, 12, 0.9, 0]], dtype=np.float32)

    class _FakeNativeTracker:
        def reset(self):
            self.last_reid_time_ms = 0.0

        def update(self, dets, frame, embs=None):
            self.last_reid_time_ms = 4.0
            return np.array([[1, 2, 10, 12, 1, 0.9, 0, 0]], dtype=np.float32)

        def get_last_reid_time_ms(self):
            return self.last_reid_time_ms

    perf_counter_values = iter([0.0, 0.010, 0.010, 0.020])
    monkeypatch.setattr(results_module.time, "perf_counter", lambda: next(perf_counter_values))

    results = api_module.track(tmp_path, _FakeDetector(), None, _FakeNativeTracker(), verbose=False)
    summary = results.summary()

    assert summary["frames"] == 1
    assert summary["timings_ms"]["det"] == pytest.approx(10.0, abs=1e-6)
    assert summary["timings_ms"]["reid"] == pytest.approx(4.0, abs=1e-6)
    assert summary["timings_ms"]["track"] == pytest.approx(6.0, abs=1e-6)
    assert summary["timings_ms"]["total"] == pytest.approx(20.0, abs=1e-6)


def test_validation_result_formats_sequence_and_combined_report():
    raw = {
        "HOTA": 69.445,
        "MOTA": 78.243,
        "IDF1": 81.937,
        "AssA": 71.0,
        "AssRe": 82.0,
        "IDSW": 12,
        "IDs": 123,
        "per_sequence": {
            "MOT17-02": {
                "HOTA": 70.1,
                "MOTA": 79.2,
                "IDF1": 82.3,
                "AssA": 72.0,
                "AssRe": 83.0,
                "IDSW": 3,
                "IDs": 40,
            },
            "MOT17-04": {
                "HOTA": 68.8,
                "MOTA": 77.9,
                "IDF1": 81.4,
                "AssA": 70.5,
                "AssRe": 81.0,
                "IDSW": 4,
                "IDs": 41,
            },
        },
    }
    result = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw=raw,
        summary_label="single_class",
        summary={"HOTA": 69.445, "MOTA": 78.243, "IDF1": 81.937},
    )

    report = result.format_report()

    assert "VAL RESULTS" in report
    assert "Sequence" in report
    assert "MOT17-02" in report
    assert "MOT17-04" in report
    assert "COMBINED" in report
    assert "69.44" in report or "69.45" in report


def test_validation_result_str_renders_cli_style_report():
    result = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "HOTA": 69.445,
            "MOTA": 78.243,
            "IDF1": 81.937,
            "AssA": 71.0,
            "AssRe": 82.0,
            "IDSW": 12,
            "IDs": 123,
            "per_sequence": {
                "MOT17-02": {
                    "HOTA": 70.1,
                    "MOTA": 79.2,
                    "IDF1": 82.3,
                    "AssA": 72.0,
                    "AssRe": 83.0,
                    "IDSW": 3,
                    "IDs": 40,
                },
                "MOT17-04": {
                    "HOTA": 68.8,
                    "MOTA": 77.9,
                    "IDF1": 81.4,
                    "AssA": 70.5,
                    "AssRe": 81.0,
                    "IDSW": 4,
                    "IDs": 41,
                },
            },
        },
        summary_label="single_class",
        summary={"HOTA": 69.445, "MOTA": 78.243, "IDF1": 81.937},
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )

    rendered = str(result)

    assert "📊 RESULTS SUMMARY" in rendered
    assert "person" in rendered
    assert "COMBINED (person)" in rendered
    assert "Sequence                  HOTA       MOTA       IDF1" in rendered
    assert "ValidationResult(" in repr(result)


def test_validation_result_str_keeps_multiclass_obb_sections():
    result = api_module.ValidationResult(
        benchmark="dota8-mot",
        raw={
            "plane": {
                "HOTA": 59.546,
                "MOTA": 0.0,
                "IDF1": 66.667,
                "AssA": 84.211,
                "AssRe": 84.211,
                "IDSW": 0,
                "IDs": 2,
                "per_sequence": {
                    "P1053__1024__0___90": {
                        "HOTA": 0.0,
                        "MOTA": 0.0,
                        "IDF1": 0.0,
                        "AssA": 0.0,
                        "AssRe": 0.0,
                        "IDSW": 0,
                        "IDs": 0,
                    },
                    "P1142__1024__0___824": {
                        "HOTA": 59.546,
                        "MOTA": 0.0,
                        "IDF1": 66.667,
                        "AssA": 84.211,
                        "AssRe": 84.211,
                        "IDSW": 0,
                        "IDs": 2,
                    },
                },
            },
            "tennis court": {
                "HOTA": 90.805,
                "MOTA": 87.5,
                "IDF1": 94.118,
                "AssA": 96.431,
                "AssRe": 97.295,
                "IDSW": 0,
                "IDs": 9,
                "per_sequence": {},
            },
            "cls_comb_det_av": {
                "HOTA": 83.617,
                "MOTA": 78.571,
                "IDF1": 90.323,
                "AssA": 96.14,
                "AssRe": 97.098,
                "IDSW": 0,
                "IDs": 17,
                "per_sequence": {},
            },
        },
        summary_label="cls_comb_det_av",
        summary={"HOTA": 83.617, "MOTA": 78.571, "IDF1": 90.323},
        args=SimpleNamespace(
            remapped_class_names=None,
            translated_benchmark_class_names=None,
            eval_box_type="obb",
            classes=None,
            benchmark="dota8-mot",
        ),
    )

    rendered = str(result)

    assert "Per-Class Combined Metrics" in rendered
    assert "plane" in rendered
    assert "tennis court" in rendered
    assert "Class Avg (Det)" in rendered
    assert "COMBINED (plane)" in rendered
    assert "COMBINED (results)" not in rendered


def test_tune_result_formats_best_report():
    metrics = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "HOTA": 69.445,
            "MOTA": 78.243,
            "IDF1": 81.937,
            "AssA": 71.0,
            "AssRe": 82.0,
            "IDSW": 12,
            "IDs": 123,
            "per_sequence": {
                "MOT17-02": {
                    "HOTA": 70.1,
                    "MOTA": 79.2,
                    "IDF1": 82.3,
                    "AssA": 72.0,
                    "AssRe": 83.0,
                    "IDSW": 3,
                    "IDs": 40,
                }
            },
        },
        summary_label="single_class",
        summary={"HOTA": 69.445, "MOTA": 78.243, "IDF1": 81.937},
    )
    tune = api_module.TuneResult(
        benchmark="mot17-ablation",
        tracker="botsort",
        trials=[],
        best=api_module.TuneTrialResult(index=1, config={}, metrics=metrics, score=(69.445, 78.243, 81.937)),
        best_config={},
        best_yaml=Path("best.yaml"),
    )

    report = tune.format_best_report()

    assert "TUNE BEST RESULTS" in report
    assert "MOT17-02" in report
    assert "COMBINED" in report


def test_tune_results_expose_validation_like_accessors():
    metrics = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={"all": {"HOTA": 69.445}},
        summary_label="all",
        summary={"HOTA": 69.445, "MOTA": 78.243, "IDF1": 81.937},
        timings={"frames": 10},
        exp_dir=Path("runs/eval"),
        args=SimpleNamespace(device="cpu"),
    )
    trial = api_module.TuneTrialResult(
        index=2,
        config={"track_buffer": 40},
        metrics=metrics,
        score=(69.445, 78.243, 81.937),
    )
    tune = api_module.TuneResult(
        benchmark="mot17-ablation",
        tracker="bytetrack",
        trials=[trial],
        best=trial,
        best_config={"track_buffer": 40},
        best_yaml=Path("best.yaml"),
    )

    assert trial.summary == metrics.summary
    assert trial.raw == metrics.raw
    assert trial.timings == metrics.timings
    assert trial.exp_dir == metrics.exp_dir
    assert "📊 RESULTS SUMMARY" in str(trial)
    assert "TuneTrialResult(index=2" in repr(trial)
    assert trial.to_dict()["metrics"]["summary"] == metrics.summary

    assert tune.summary == metrics.summary
    assert tune.raw == metrics.raw
    assert tune.timings == metrics.timings
    assert tune.exp_dir == metrics.exp_dir
    assert tune.format_report() == tune.format_best_report()
    assert "📊 BEST TRIAL SUMMARY" in str(tune)
    assert "TuneResult(benchmark='mot17-ablation'" in repr(tune)
    assert tune.to_dict()["summary"] == metrics.summary
    assert tune.to_dict(include_trials=True)["trials"][0]["metrics"]["summary"] == metrics.summary


def test_tune_result_str_shows_delta_vs_baseline(monkeypatch):
    class _TTYStdout:
        def isatty(self):
            return True

    monkeypatch.setattr(reporting_module.sys, "stdout", _TTYStdout())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)

    baseline_metrics = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "HOTA": 66.0,
            "MOTA": 77.0,
            "IDF1": 78.0,
            "AssA": 68.0,
            "AssRe": 74.0,
            "IDSW": 200,
            "IDs": 400,
            "per_sequence": {
                "MOT17-02": {
                    "HOTA": 45.0,
                    "MOTA": 54.0,
                    "IDF1": 56.0,
                    "AssA": 44.0,
                    "AssRe": 50.0,
                    "IDSW": 70,
                    "IDs": 80,
                },
                "MOT17-04": {
                    "HOTA": 78.0,
                    "MOTA": 88.0,
                    "IDF1": 90.0,
                    "AssA": 80.0,
                    "AssRe": 84.0,
                    "IDSW": 20,
                    "IDs": 90,
                },
            },
        },
        summary_label="single_class",
        summary={"HOTA": 66.0, "MOTA": 77.0, "IDF1": 78.0},
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )
    best_metrics = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "HOTA": 67.5,
            "MOTA": 78.2,
            "IDF1": 80.0,
            "AssA": 69.4,
            "AssRe": 75.1,
            "IDSW": 185,
            "IDs": 383,
            "per_sequence": {
                "MOT17-02": {
                    "HOTA": 47.0,
                    "MOTA": 55.5,
                    "IDF1": 58.0,
                    "AssA": 46.0,
                    "AssRe": 52.0,
                    "IDSW": 63,
                    "IDs": 64,
                },
                "MOT17-04": {
                    "HOTA": 79.6,
                    "MOTA": 89.4,
                    "IDF1": 91.9,
                    "AssA": 81.3,
                    "AssRe": 85.4,
                    "IDSW": 19,
                    "IDs": 91,
                },
            },
        },
        summary_label="single_class",
        summary={"HOTA": 67.5, "MOTA": 78.2, "IDF1": 80.0},
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )
    baseline_trial = api_module.TuneTrialResult(index=1, config={"track_buffer": 30}, metrics=baseline_metrics, score=(66.0,))
    best_trial = api_module.TuneTrialResult(index=2, config={"track_buffer": 40}, metrics=best_metrics, score=(67.5,))
    tune = api_module.TuneResult(
        benchmark="mot17-ablation",
        tracker="bytetrack",
        trials=[baseline_trial, best_trial],
        best=best_trial,
        best_config={"track_buffer": 40},
        best_yaml=Path("best.yaml"),
    )

    rendered = str(tune)

    assert "📊 BEST TRIAL SUMMARY" in rendered
    assert "Sequence                  HOTA       MOTA       IDF1" in rendered
    assert "COMBINED (person)        67.50      78.20      80.00" in rendered
    assert "\x1b[32m(+1.50)\x1b[0m" in rendered
    assert "\x1b[32m(-15)\x1b[0m" in rendered
    assert "\x1b[32m(+2.00)\x1b[0m" in rendered
    assert "\x1b[32m(-7)\x1b[0m" in rendered
    assert "\x1b[31m(+1)\x1b[0m" in rendered


def test_validation_result_renderable_shows_delta_vs_baseline() -> None:
    baseline_metrics = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "person": {
                "HOTA": 66.0,
                "MOTA": 78.0,
                "IDF1": 79.0,
                "AssA": 68.0,
                "AssRe": 74.0,
                "IDSW": 230,
                "IDs": 435,
                "per_sequence": {
                    "MOT17-02": {
                        "HOTA": 46.0,
                        "MOTA": 55.0,
                        "IDF1": 57.0,
                        "AssA": 45.0,
                        "AssRe": 50.0,
                        "IDSW": 80,
                        "IDs": 90,
                    }
                },
            },
        },
        summary_label="single_class",
        summary={"HOTA": 66.0, "MOTA": 78.0, "IDF1": 79.0},
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )
    best_metrics = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "person": {
                "HOTA": 67.5,
                "MOTA": 78.2,
                "IDF1": 80.0,
                "AssA": 70.0,
                "AssRe": 75.0,
                "IDSW": 215,
                "IDs": 428,
                "per_sequence": {
                    "MOT17-02": {
                        "HOTA": 47.5,
                        "MOTA": 55.5,
                        "IDF1": 57.2,
                        "AssA": 47.0,
                        "AssRe": 51.0,
                        "IDSW": 65,
                        "IDs": 77,
                    }
                },
            },
        },
        summary_label="single_class",
        summary={"HOTA": 67.5, "MOTA": 78.2, "IDF1": 80.0},
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )

    rendered = ui_module.capture_renderable(
        best_metrics.renderable(
            title=reporting_module.CLI_TUNE_BEST_SUMMARY_TITLE,
            compare_raw=baseline_metrics.raw,
            compare_args=baseline_metrics.args,
        ),
        width=140,
    )

    assert "📊 BEST TRIAL SUMMARY" in rendered
    assert "(+1.50)" in rendered
    assert "(-15)" in rendered
    assert "(-7)" in rendered


def test_validation_result_str_colorizes_base_table_when_tty(monkeypatch):
    class _TTYStdout:
        def isatty(self):
            return True

    monkeypatch.setattr(reporting_module.sys, "stdout", _TTYStdout())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)

    result = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "HOTA": 69.445,
            "MOTA": 78.243,
            "IDF1": 81.937,
            "AssA": 72.34,
            "AssRe": 77.58,
            "IDSW": 137,
            "IDs": 367,
            "per_sequence": {
                "MOT17-02": {
                    "HOTA": 49.23,
                    "MOTA": 54.55,
                    "IDF1": 58.94,
                    "AssA": 50.56,
                    "AssRe": 55.79,
                    "IDSW": 63,
                    "IDs": 72,
                },
                "MOT17-04": {
                    "HOTA": 80.37,
                    "MOTA": 89.75,
                    "IDF1": 92.51,
                    "AssA": 82.85,
                    "AssRe": 86.41,
                    "IDSW": 22,
                    "IDs": 82,
                },
            },
        },
        summary_label="single_class",
        summary={"HOTA": 69.445, "MOTA": 78.243, "IDF1": 81.937},
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )

    rendered = str(result)

    assert "\x1b[1;36m" in rendered
    assert "\x1b[1;34mSequence" in rendered
    assert "\x1b[1;33m" in rendered
    assert "COMBINED (person)" in rendered


def test_validation_result_print_report_matches_cli_style(capsys):
    result = api_module.ValidationResult(
        benchmark="mot17-ablation",
        raw={
            "HOTA": 69.445,
            "MOTA": 78.243,
            "IDF1": 81.937,
            "AssA": 72.34,
            "AssRe": 77.58,
            "IDSW": 137,
            "IDs": 367,
            "per_sequence": {
                "MOT17-02": {
                    "HOTA": 49.23,
                    "MOTA": 54.55,
                    "IDF1": 58.94,
                    "AssA": 50.56,
                    "AssRe": 55.79,
                    "IDSW": 63,
                    "IDs": 72,
                },
                "MOT17-04": {
                    "HOTA": 80.37,
                    "MOTA": 89.75,
                    "IDF1": 92.51,
                    "AssA": 82.85,
                    "AssRe": 86.41,
                    "IDSW": 22,
                    "IDs": 82,
                },
            },
        },
        summary_label="single_class",
        summary={"HOTA": 69.445, "MOTA": 78.243, "IDF1": 81.937},
        timings={
            "frames": 2652,
            "totals_ms": {
                "preprocess": 0.0,
                "inference": 0.0,
                "postprocess": 0.0,
                "reid": 0.0,
                "track": 29945.9,
                "plot": 0.0,
                "total": 29945.9,
            },
        },
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )

    result.print_report()

    combined = capsys.readouterr().out
    assert "📊 RESULTS SUMMARY" in combined
    assert "person" in combined
    assert "COMBINED (person)" in combined
    assert "📊 TIMING SUMMARY" not in combined

    result.print_report(include_timings=True)
    combined = capsys.readouterr().out
    assert "📊 TIMING SUMMARY" in combined


def test_track_run_result_str_and_print_summary_use_plain_stdout(monkeypatch, capsys, tmp_path):
    class _FakeResults:
        def __init__(self):
            self.summary_calls = 0

        def summary(self):
            self.summary_calls += 1
            return {
                "source": str(tmp_path),
                "frames": 1,
                "detections": 2,
                "tracks": 1,
                "unique_tracks": 1,
                "timings_ms": {
                    "det": 1.0,
                    "reid": 2.0,
                    "track": 3.0,
                    "total": 6.0,
                    "avg_total": 6.0,
                },
            }

        def format_summary(self):
            return "TRACKING SUMMARY\nTotal"

        def print_summary(self):
            raise AssertionError("TrackRunResult.print_summary should render via plain stdout")

        def show(self):
            return None

        def stop(self, reason=None):
            return None

    run = api_module.TrackRunResult(
        source=tmp_path,
        results=_FakeResults(),
        video_path=None,
        text_path=None,
    )

    assert "TRACKING SUMMARY" in str(run)
    assert "TrackRunResult(" in repr(run)

    run.print_summary()
    out = capsys.readouterr().out
    assert "TRACKING SUMMARY" in out


def test_boxmot_val_tune_and_export_facades(monkeypatch, tmp_path):
    calls = {}

    def fake_run_eval(args, *, evolve_config=None, **kwargs):
        calls["eval"] = (args, evolve_config, kwargs)
        return api_module.ValidationResult(
            benchmark=str(args.benchmark),
            raw={"all": {"HOTA": 50.0, "MOTA": 45.0, "IDF1": 40.0}},
            summary_label="all",
            summary={"HOTA": 50.0, "MOTA": 45.0, "IDF1": 40.0},
            exp_dir=tmp_path / "eval",
            timings={"frames": 2},
            args=args,
        )

    def fake_run_tune(args, *, baseline_config=None):
        calls["tune"] = (args, baseline_config)
        metrics = api_module.ValidationResult(
            benchmark=str(args.benchmark),
            raw={"all": {"HOTA": 53.0, "MOTA": 48.0, "IDF1": 43.0}},
            summary_label="all",
            summary={"HOTA": 53.0, "MOTA": 48.0, "IDF1": 43.0},
            exp_dir=tmp_path / "tune",
            timings={},
            args=args,
        )
        best_trial = api_module.TuneTrialResult(
            index=1,
            config={"track_buffer": 40},
            metrics=metrics,
            score=(53.0,),
        )
        return api_module.TuneResult(
            benchmark=str(args.benchmark),
            tracker=args.tracker,
            trials=[best_trial],
            best=best_trial,
            best_config={"track_buffer": 40},
            best_yaml=tmp_path / "best.yaml",
        )

    def fake_run_export(args):
        calls["export"] = args
        return api_module.ExportResult(weights=Path(args.weights), files={"onnx": tmp_path / "exported.onnx"})

    monkeypatch.setattr(evaluator_module, "run_eval", fake_run_eval)
    monkeypatch.setattr(tuner_module, "run_tune", fake_run_tune)
    monkeypatch.setattr(export_module, "run_export", fake_run_export)

    model = api_module.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack", classes=[0, 1], project=tmp_path / "runs")

    metrics = model.val(benchmark="mot17-mini", device="cpu")

    assert metrics.summary["HOTA"] == 50.0
    eval_args, eval_config, eval_kwargs = calls["eval"]
    assert eval_args.data == "mot17-mini"
    assert eval_args.detector[0].name == "yolov8n.pt"
    assert eval_args.reid[0].name == "lmbn_n_duke.pt"
    assert eval_args.classes == [0, 1]
    assert eval_args.show_progress is True
    assert eval_config is None
    assert "workflow" in eval_kwargs
    assert eval_kwargs["workflow"] is not None

    tune_results = model.tune(benchmark="mot17-mini", n_trials=3, device="cpu")

    assert tune_results.best_config["track_buffer"] == 40
    assert tune_results.best.metrics.summary["HOTA"] == 53.0
    assert tune_results.workflow_rendered is True
    assert str(tune_results) == ""
    tune_args, tune_baseline = calls["tune"]
    assert tune_args.data == "mot17-mini"
    assert tune_args.n_trials == 3
    assert tune_args.seed == 0
    assert tune_args.compare_to_first_trial is True
    assert tune_baseline is None

    export_results = model.export(include=("onnx",), device="cpu")

    export_args = calls["export"]
    assert export_results.weights.name == "lmbn_n_duke.pt"
    assert export_results.files["onnx"] == tmp_path / "exported.onnx"
    assert export_args.include == ("onnx",)
    assert export_args.weights.name == "lmbn_n_duke.pt"


def test_boxmot_val_logs_cli_like_intro_without_printing_report(monkeypatch, tmp_path, capsys):
    workflows = []

    class _FakeWorkflow:
        def __init__(self, title, fields, steps, stderr=False, transient=False):
            self.title = title
            self.fields = list(fields)
            self.steps = list(steps)
            self.stderr = stderr
            self.transient = transient
            self.started = False
            self.stopped = False

        def start(self):
            self.started = True
            return self

        def stop(self):
            self.stopped = True

        def activate(self, label):
            self.steps = [
                (step_label, "active" if step_label == label else ("todo" if step_state == "active" else step_state))
                for step_label, step_state in self.steps
            ]

        def complete(self, label):
            self.steps = [
                (step_label, "done" if step_label == label else step_state)
                for step_label, step_state in self.steps
            ]

    def fake_create_workflow_progress(title, fields, *, steps=(), stderr=False, transient=False):
        workflow = _FakeWorkflow(title, fields, steps, stderr=stderr, transient=transient)
        workflows.append(workflow)
        return workflow

    def fake_run_eval(args, *, evolve_config=None, **kwargs):
        return api_module.ValidationResult(
            benchmark=str(args.benchmark),
            raw={"all": {"HOTA": 50.0, "MOTA": 45.0, "IDF1": 40.0}},
            summary_label="all",
            summary={"HOTA": 50.0, "MOTA": 45.0, "IDF1": 40.0},
            exp_dir=tmp_path / "eval",
            timings={"frames": 2},
            args=args,
        )

    monkeypatch.setattr(evaluator_module.ui, "create_workflow_progress", fake_create_workflow_progress)
    monkeypatch.setattr(evaluator_module, "run_eval", fake_run_eval)

    model = api_module.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="botsort", project=tmp_path / "runs")

    metrics = model.val(benchmark="mot17-mini", device="cpu")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.title == "Evaluation"
    assert workflow.started is True
    assert workflow.stopped is True
    assert ("Tracker", "botsort") in workflow.fields
    assert ("Dataset", "mot17-mini") in workflow.fields
    assert (evaluator_module.EVAL_GENERATE_STEP, "active") in workflow.steps
    assert metrics.summary["HOTA"] == 50.0
    assert metrics.workflow_rendered is True
    assert str(metrics) == ""


def test_boxmot_generate_and_research_facades(monkeypatch, tmp_path):
    calls = {}

    def fake_run_generate(args):
        calls["generate"] = args
        timing_stats = TimingStats()
        timing_stats.frames = 4
        timing_stats.totals["inference"] = 20.0
        timing_stats.totals["reid"] = 12.0
        timing_stats.totals["total"] = 40.0
        args.benchmark = "mot17-mini"
        args.source = tmp_path / "datasets" / "mot17-mini" / "train"
        return timing_stats

    def fake_run_research(args):
        calls["research"] = args
        return research_engine_module.ResearchResult(
            tracker=args.tracker,
            benchmark=str(args.data),
            proposal_model=args.proposal_model,
            run_dir=tmp_path / "runs" / "research" / "bytetrack_mot17_mini",
            best_candidate_dir=tmp_path / "runs" / "research" / "best",
            editable_files=("boxmot/trackers/bytetrack/bytetrack.py",),
            train_sequences=("MOT17-02",),
            val_sequences=("MOT17-04",),
            baseline_summary={"HOTA": 60.0, "IDF1": 70.0, "MOTA": 80.0},
            best_summary={"HOTA": 61.5, "IDF1": 71.0, "MOTA": 80.2},
            delta_summary={"HOTA": 1.5, "IDF1": 1.0, "MOTA": 0.2},
        )

    monkeypatch.setattr(cache_module, "run_generate", fake_run_generate)
    monkeypatch.setattr(research_engine_module, "run_research", fake_run_research)

    model = api_module.Boxmot(tracker="bytetrack", project=tmp_path / "runs")

    generated = model.generate(benchmark="mot17-mini", device="cpu", batch_size=8, resume=False)

    generate_args = calls["generate"]
    assert generated.benchmark == "mot17-mini"
    assert generated.source == tmp_path / "datasets" / "mot17-mini" / "train"
    assert generated.cache_dir == tmp_path / "runs" / "dets_n_embs" / "mot17-mini"
    assert generated.timings["frames"] == 4
    assert generated.detectors[0].name == "yolov8n.pt"
    assert generated.reid_models[0].name == "osnet_x0_25_msmt17.pt"
    assert generate_args.data == "mot17-mini"
    assert generate_args.benchmark == "mot17-mini"
    assert generate_args.batch_size == 8
    assert generate_args.resume is False
    assert "TIMING SUMMARY" in str(generated)

    researched = model.research(
        benchmark="mot17-mini",
        proposal_model="openai/gpt-5.4",
        max_metric_calls=6,
        keep_workspace=True,
        idf1_penalty=2.0,
    )

    research_args = calls["research"]
    assert researched.delta_summary["HOTA"] == 1.5
    assert research_args.data == "mot17-mini"
    assert research_args.proposal_model == "openai/gpt-5.4"
    assert research_args.max_metric_calls == 6
    assert research_args.keep_workspace is True
    assert research_args.idf1_penalty == 2.0
    assert "RESEARCH SUMMARY" in str(researched)


def test_boxmot_generate_requires_exactly_one_input(tmp_path):
    model = api_module.Boxmot(project=tmp_path / "runs")

    with pytest.raises(ValueError, match="exactly one of benchmark=... or source=..."):
        model.generate()

    with pytest.raises(ValueError, match="exactly one of benchmark=... or source=..."):
        model.generate(benchmark="mot17-mini", source=tmp_path / "dataset")


def test_boxmot_tune_forwards_optimization_targets_and_seed(monkeypatch, tmp_path):
    captured = {}

    def fake_run_tune(args, *, baseline_config=None):
        captured["args"] = args
        captured["baseline_config"] = baseline_config
        metrics = api_module.ValidationResult(
            benchmark=str(args.benchmark),
            raw={},
            summary_label="all",
            summary={"HOTA": 51.0, "MOTA": 46.0, "IDF1": 41.0},
            exp_dir=None,
            timings={},
            args=args,
        )
        trial = api_module.TuneTrialResult(index=1, config={}, metrics=metrics, score=(51.0, -0.2))
        return api_module.TuneResult(
            benchmark=str(args.benchmark),
            tracker=args.tracker,
            trials=[trial],
            best=trial,
            best_config={},
            best_yaml=tmp_path / "best.yaml",
        )

    monkeypatch.setattr(tuner_module, "run_tune", fake_run_tune)
    model = api_module.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack", project=tmp_path / "runs")

    tuned = model.tune(
        benchmark="mot17-mini",
        n_trials=2,
        device="cpu",
        maximize=("HOTA", "IDF1"),
        minimize=("IDSW_rate",),
        seed=7,
    )

    assert tuned.best.index == 1
    assert captured["args"].n_trials == 2
    assert captured["args"].maximize == ("HOTA", "IDF1")
    assert captured["args"].minimize == ("IDSW_rate",)
    assert captured["args"].seed == 7
    assert captured["baseline_config"] is None


def test_extract_summary_handles_single_class_results_with_per_sequence_first():
    raw = {
        "per_sequence": {"MOT17-02": {"HOTA": 11.0}},
        "HOTA": 62.5,
        "MOTA": 70.0,
        "IDF1": 65.0,
        "AssA": 61.0,
    }

    label, summary = reporting_module.extract_summary(raw)

    assert label == "single_class"
    assert summary["HOTA"] == 62.5
    assert summary["MOTA"] == 70.0
    assert summary["IDF1"] == 65.0
    assert summary["AssA"] == 61.0
