from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import boxmot
import boxmot.api as api_module
import boxmot.engine.results as results_module
from boxmot.configs import BOXMOT_DEFAULTS, DEFAULT_DETECTOR, DEFAULT_REID, get_mode_default
from boxmot.detectors import Detector
from boxmot.detectors.base import Detections
from boxmot.reid import ReID


_DUMMY_IMG = np.zeros((32, 32, 3), dtype=np.uint8)


def test_package_root_only_exports_metadata():
    assert boxmot.__all__ == ("__version__",)
    assert not hasattr(boxmot, "Boxmot")


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
    assert args.tracking_backend == "thread"


def test_boxmot_eval_namespace_preserves_explicit_constructor_overrides():
    model = api_module.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")

    args = model._base_eval_args("mot17-ablation")

    assert args.detector_explicit is True
    assert args.reid_explicit is True
    assert args.tracker_explicit is True


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

    monkeypatch.setattr(api_module.cv2, "VideoWriter", _FakeVideoWriter)

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

    monkeypatch.setattr(api_module, "_suppress_boxmot_logs", fake_suppress)
    monkeypatch.setattr(api_module.Boxmot, "_build_detector", lambda self, **kwargs: _FakeDetector())
    monkeypatch.setattr(api_module.Boxmot, "_build_tracker", lambda self, **kwargs: fake_tracker)

    def fail_build_reid(self, **kwargs):
        raise AssertionError("track() should reuse the tracker ReID backend for built-in ReID trackers")

    monkeypatch.setattr(api_module.Boxmot, "_build_reid", fail_build_reid)

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
    monkeypatch.setattr(api_module, "track", lambda *args, **kwargs: fake_results)

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


def test_boxmot_track_keeps_finite_sources_lazy_without_save(monkeypatch, tmp_path):
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

        def materialize(self):
            self.materialized = True
            raise AssertionError("finite sources should stay lazy until save/show/summary needs them")

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
    monkeypatch.setattr(api_module, "track", lambda *args, **kwargs: fake_results)

    model = api_module.Boxmot(detector=object(), reid=object(), tracker=object(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    assert fake_results.materialized is False
    assert run.summary["frames"] == 0


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
    assert "TuneTrialResult(index=2" in str(trial)
    assert trial.to_dict()["metrics"]["summary"] == metrics.summary

    assert tune.summary == metrics.summary
    assert tune.raw == metrics.raw
    assert tune.timings == metrics.timings
    assert tune.exp_dir == metrics.exp_dir
    assert tune.format_report() == tune.format_best_report()
    assert "TuneResult(benchmark='mot17-ablation'" in str(tune)
    assert tune.to_dict()["summary"] == metrics.summary
    assert tune.to_dict(include_trials=True)["trials"][0]["metrics"]["summary"] == metrics.summary


def test_validation_result_print_report_matches_cli_style(monkeypatch):
    logs = []
    complete_calls = []

    class _FakeLogger:
        def opt(self, **kwargs):
            return self

        def info(self, message=""):
            logs.append(message)

        def complete(self):
            complete_calls.append(True)

    fake_logger = _FakeLogger()
    results_utils_module = importlib.import_module("boxmot.utils.evaluation.results")
    timing_module = importlib.import_module("boxmot.utils.timing")

    monkeypatch.setattr(api_module, "LOGGER", fake_logger)
    monkeypatch.setattr(results_utils_module, "LOGGER", fake_logger)
    monkeypatch.setattr(timing_module, "LOGGER", fake_logger)

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

    combined = "\n".join(str(entry) for entry in logs)
    assert "📊 RESULTS SUMMARY" in combined
    assert "person" in combined
    assert "COMBINED (person)" in combined
    assert "📊 TIMING SUMMARY" in combined
    assert complete_calls == [True]


def test_boxmot_val_tune_and_export_facades(monkeypatch, tmp_path):
    state = {"last_config": None}
    evaluator_calls = []
    replay_calls = []

    def fake_eval_setup(args):
        args.source = tmp_path / "benchmark" / "train"
        args.project = tmp_path / "runs"
        args.project.mkdir(parents=True, exist_ok=True)

    def fake_run_generate_dets_embs(args, timing_stats=None):
        evaluator_calls.append(("generate", args.data, args.detector[0].name, args.reid[0].name, args.classes))
        if timing_stats is not None:
            timing_stats.frames = 2
            timing_stats.totals["inference"] = 12.0
            timing_stats.totals["reid"] = 4.0
            timing_stats.totals["total"] = 20.0

    def fake_run_generate_mot_results(args, evolve_config=None, timing_stats=None, quiet=False):
        state["last_config"] = evolve_config or {}
        replay_calls.append(("track", quiet, state["last_config"], getattr(args, "show_progress", None)))
        args.exp_dir = tmp_path / f"exp_{len([call for call in evaluator_calls if call[0] == 'track'])}"
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        if timing_stats is not None:
            timing_stats.totals["track"] = 6.0 + float(state["last_config"].get("trial_score", 0.0))

    def fake_run_trackeval(args, verbose=False):
        hota = 50.0 + float(state["last_config"].get("trial_score", 0.0))
        return {"all": {"HOTA": hota, "MOTA": hota - 5.0, "IDF1": hota - 10.0}}

    fake_evaluator = SimpleNamespace(
        eval_setup=fake_eval_setup,
        run_generate_dets_embs=fake_run_generate_dets_embs,
        run_trackeval=fake_run_trackeval,
    )
    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", fake_evaluator)
    monkeypatch.setitem(sys.modules, "boxmot.engine.replay", SimpleNamespace(run_generate_mot_results=fake_run_generate_mot_results))

    def fake_setup_model(args):
        args.weights = Path(args.weights)
        return object(), object()

    def fake_create_export_tasks(args, model, dummy_input):
        return {"onnx": (True, object, ())}

    def fake_perform_exports(export_tasks):
        return {"onnx": tmp_path / "exported.onnx"}

    fake_export = SimpleNamespace(
        setup_model=fake_setup_model,
        create_export_tasks=fake_create_export_tasks,
        perform_exports=fake_perform_exports,
    )
    monkeypatch.setitem(sys.modules, "boxmot.engine.export", fake_export)

    model = api_module.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack", classes=[0, 1], project=tmp_path / "runs")

    metrics = model.val(benchmark="mot17-mini", device="cpu")

    assert metrics.summary["HOTA"] == 50.0
    assert evaluator_calls[0] == ("generate", "mot17-mini", "yolov8n.pt", "lmbn_n_duke.pt", [0, 1])
    assert replay_calls[0] == ("track", False, {}, True)

    monkeypatch.setattr(
        api_module.Boxmot,
        "_iter_tune_configs",
        lambda self, n_trials, rng: iter([
            {"trial_score": 1.0},
            {"trial_score": 3.0},
            {"trial_score": 2.0},
        ]),
    )

    tune_results = model.tune(benchmark="mot17-mini", n_trials=3, device="cpu")

    assert tune_results.best_config["trial_score"] == 3.0
    assert tune_results.best.metrics.summary["HOTA"] == 53.0
    assert tune_results.best_yaml.exists()
    assert replay_calls[1:] == [
        ("track", True, {"trial_score": 1.0}, False),
        ("track", True, {"trial_score": 3.0}, False),
        ("track", True, {"trial_score": 2.0}, False),
    ]

    export_results = model.export(include=("onnx",), device="cpu")

    assert export_results.weights.name == "lmbn_n_duke.pt"
    assert export_results.files["onnx"] == tmp_path / "exported.onnx"


def test_boxmot_tune_logs_trial_progress(monkeypatch, tmp_path):
    writes = []
    suppress_calls = []

    def fake_suppress(enabled, level="WARNING"):
        suppress_calls.append((enabled, level))
        return nullcontext()

    monkeypatch.setattr(api_module, "_suppress_boxmot_logs", fake_suppress)
    monkeypatch.setattr(
        api_module,
        "_write_progress_line",
        lambda message, previous_width, stream=None, final=False: writes.append((message, final)) or max(previous_width, len(message)),
    )
    perf_counter_values = iter([100.0, 101.2, 101.2, 103.6])
    monkeypatch.setattr(api_module.time, "perf_counter", lambda: next(perf_counter_values))

    model = api_module.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack", project=tmp_path / "runs")

    monkeypatch.setattr(
        api_module.Boxmot,
        "_iter_tune_configs",
        lambda self, n_trials, rng: iter([
            {"trial_score": 1.0},
            {"trial_score": 3.0},
        ]),
    )

    def fake_run_validation_pipeline(self, **kwargs):
        score = float(kwargs["evolve_config"]["trial_score"])
        assert kwargs["verbose"] is False
        assert kwargs["show_progress"] is False
        return api_module.ValidationResult(
            benchmark=str(kwargs["benchmark"]),
            raw={},
            summary_label="all",
            summary={"HOTA": 50.0 + score, "MOTA": 45.0 + score, "IDF1": 40.0 + score},
            exp_dir=None,
            timings={},
            args=None,
        )

    monkeypatch.setattr(api_module.Boxmot, "_run_validation_pipeline", fake_run_validation_pipeline)

    tuned = model.tune(benchmark="mot17-mini", n_trials=2, device="cpu")

    assert tuned.best.index == 2
    assert len(writes) == 4
    assert suppress_calls == [(True, "WARNING"), (True, "WARNING")]
    assert writes[0][0].startswith("  Tune")
    assert "0%  (0/2)" in writes[0][0]
    assert "running trial 1/2" in writes[0][0]
    assert writes[0][0].endswith("remaining --:--")
    assert "50%  (1/2)" in writes[1][0]
    assert "HOTA=51.000" in writes[1][0]
    assert "best" in writes[1][0]
    assert writes[1][0].endswith("remaining 00:02")
    assert writes[2][0].startswith("  Tune")
    assert "50%  (1/2)" in writes[2][0]
    assert "running trial 2/2" in writes[2][0]
    assert "last HOTA=51.000" in writes[2][0]
    assert "best" in writes[2][0]
    assert writes[2][0].endswith("remaining 00:02")
    assert "100%  (2/2)" in writes[3][0]
    assert "HOTA=53.000" in writes[3][0]
    assert writes[3][0].endswith("remaining 00:00")
    assert writes[0][1] is False
    assert writes[1][1] is False
    assert writes[2][1] is False
    assert writes[3][1] is True


def test_extract_summary_handles_single_class_results_with_per_sequence_first():
    raw = {
        "per_sequence": {"MOT17-02": {"HOTA": 11.0}},
        "HOTA": 62.5,
        "MOTA": 70.0,
        "IDF1": 65.0,
        "AssA": 61.0,
    }

    label, summary = api_module._extract_summary(raw)

    assert label == "single_class"
    assert summary["HOTA"] == 62.5
    assert summary["MOTA"] == 70.0
    assert summary["IDF1"] == 65.0
    assert summary["AssA"] == 61.0
