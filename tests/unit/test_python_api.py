import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot import BoxMOT, boxmot, track as stream_track
from boxmot.engine.results import Results
from boxmot.model import ExportResults, TrackEvalMetrics, TrackResults, TuneResults
from boxmot.utils import WEIGHTS


def test_boxmot_val_requires_benchmark():
    model = BoxMOT()

    with pytest.raises(ValueError, match="requires a benchmark config"):
        model.val()


def test_boxmot_factory_returns_boxmot_instance():
    tracker = boxmot(tracker="boosttrack", reid="osnet_x0_25_msmt17")

    assert isinstance(tracker, BoxMOT)
    assert tracker.overrides["tracker"] == "boosttrack"
    assert tracker.overrides["reid"] == WEIGHTS / "osnet_x0_25_msmt17.pt"


def test_boxmot_val_remembers_benchmark_and_runtime_overrides(monkeypatch):
    captured = []

    def fake_main(args):
        captured.append(args)
        return {
            "HOTA": 67.1,
            "MOTA": 74.2,
            "IDF1": 80.3,
            "per_sequence": {"MOT17-04-FRCNN": {"HOTA": 67.1}},
        }

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    model = BoxMOT(tracker="boosttrack")

    first = model.val(benchmark="mot17-mini", device="cpu")
    second = model.val()

    assert isinstance(first, TrackEvalMetrics)
    assert first.HOTA == pytest.approx(67.1)
    assert first.hota == pytest.approx(67.1)
    assert first.per_sequence["MOT17-04-FRCNN"]["HOTA"] == pytest.approx(67.1)
    assert second.MOTA == pytest.approx(74.2)

    assert captured[0].data == "mot17-mini"
    assert captured[0].tracking_method == "boosttrack"
    assert captured[0].tracking_backend == "thread"
    assert captured[0].device == "cpu"
    assert captured[0].device_explicit is True
    assert captured[1].data == "mot17-mini"
    assert captured[1].device == "cpu"
    assert captured[1].tracking_backend == "thread"


def test_boxmot_val_accepts_click_style_kwargs(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args
        return {"HOTA": 62.5, "MOTA": 71.0, "IDF1": 79.5}

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    tracker = boxmot(tracker="boosttrack", reid="something")
    metrics = tracker.val(
        data="mot17-ablation",
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.7,
        device="0",
    )

    assert metrics.HOTA == pytest.approx(62.5)
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].imgsz == 640
    assert captured["args"].batch_size == 16
    assert captured["args"].conf == pytest.approx(0.25)
    assert captured["args"].iou == pytest.approx(0.7)
    assert captured["args"].device == "0"
    assert captured["args"].reid_model[0] == WEIGHTS / "something.pt"


def test_boxmot_val_respects_explicit_tracking_backend(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args
        return {"HOTA": 62.5, "MOTA": 71.0, "IDF1": 79.5}

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    model = BoxMOT(tracker="boosttrack")
    model.val(data="mot17-mini", tracking_backend="process")

    assert captured["args"].tracking_backend == "process"


def test_boxmot_val_normalizes_bare_detector_and_reid_names(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args
        return {"HOTA": 61.0, "MOTA": 72.0, "IDF1": 83.0}

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    model = BoxMOT(
        detector="yolo11s_obb",
        reid="lmbn_n_duke",
        tracker="botsort",
        benchmark="mot17-mini",
    )
    model.val()

    assert captured["args"].yolo_model[0] == WEIGHTS / "yolo11s_obb.pt"
    assert captured["args"].reid_model[0] == WEIGHTS / "lmbn_n_duke.pt"
    assert captured["args"].yolo_model_explicit is True
    assert captured["args"].reid_model_explicit is True


def test_boxmot_val_accepts_multi_model_inputs(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args
        return {"HOTA": 61.0, "MOTA": 72.0, "IDF1": 83.0}

    monkeypatch.setitem(sys.modules, "boxmot.engine.evaluator", SimpleNamespace(main=fake_main))

    tracker = boxmot(
        detector=["yolo11s_obb", "yolo11m"],
        reid=["lmbn_n_duke", "osnet_x0_25_msmt17"],
        tracker="botsort",
    )
    tracker.val(data="mot17-mini")

    assert captured["args"].yolo_model == [WEIGHTS / "yolo11s_obb.pt", WEIGHTS / "yolo11m.pt"]
    assert captured["args"].reid_model == [WEIGHTS / "lmbn_n_duke.pt", WEIGHTS / "osnet_x0_25_msmt17.pt"]


def test_boxmot_tune_requires_benchmark():
    model = BoxMOT()

    with pytest.raises(ValueError, match="BoxMOT.tune\\(\\) requires a benchmark config"):
        model.tune()


def test_boxmot_tune_remembers_benchmark_and_tune_overrides(monkeypatch):
    captured = []

    def fake_main(args):
        captured.append(args)
        return {
            "tracking_method": args.tracking_method,
            "benchmark": args.data,
            "objectives": list(args.objectives),
            "maximize": list(args.maximize) or [args.objectives[0]],
            "minimize": list(args.minimize),
            "best_trial_id": "trial_001",
            "best_config": {"max_age": 42},
            "best_metrics": {"HOTA": 68.4, "MOTA": 75.1, "IDF1": 80.6, "IDSW_rate": 0.03},
            "best_trial": {
                "trial_id": "trial_001",
                "trial_dir": "/tmp/trial_001",
                "config": {"max_age": 42},
                "metrics": {"HOTA": 68.4, "MOTA": 75.1, "IDF1": 80.6, "IDSW_rate": 0.03},
            },
            "trials": [
                {
                    "trial_id": "trial_001",
                    "trial_dir": "/tmp/trial_001",
                    "config": {"max_age": 42},
                    "metrics": {"HOTA": 68.4, "MOTA": 75.1, "IDF1": 80.6, "IDSW_rate": 0.03},
                }
            ],
        }

    monkeypatch.setitem(sys.modules, "boxmot.engine.tuner", SimpleNamespace(main=fake_main))

    model = BoxMOT(tracker="boosttrack")

    first = model.tune(benchmark="mot17-mini", device="cpu", n_trials=7, objectives="HOTA IDF1")
    second = model.tune()

    assert isinstance(first, TuneResults)
    assert first.HOTA == pytest.approx(68.4)
    assert first.best.metrics.idsw_rate == pytest.approx(0.03)
    assert first.best_config["max_age"] == 42
    assert first.trials[0].trial_id == "trial_001"
    assert second.MOTA == pytest.approx(75.1)

    assert captured[0].data == "mot17-mini"
    assert captured[0].tracking_method == "boosttrack"
    assert captured[0].tracking_backend == "thread"
    assert captured[0].device == "cpu"
    assert captured[0].n_trials == 7
    assert captured[0].objectives == ["HOTA", "IDF1"]
    assert captured[1].data == "mot17-mini"
    assert captured[1].n_trials == 7
    assert captured[1].objectives == ["HOTA", "IDF1"]
    assert captured[1].tracking_backend == "thread"


def test_boxmot_tune_accepts_data_alias(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args
        return {
            "tracking_method": args.tracking_method,
            "benchmark": args.data,
            "objectives": list(args.objectives),
            "maximize": list(args.maximize) or [args.objectives[0]],
            "minimize": list(args.minimize),
            "best_trial_id": "trial_002",
            "best_config": {"max_age": 20},
            "best_metrics": {"HOTA": 65.0, "MOTA": 73.0, "IDF1": 79.0, "IDSW_rate": 0.04},
            "best_trial": {
                "trial_id": "trial_002",
                "trial_dir": "/tmp/trial_002",
                "config": {"max_age": 20},
                "metrics": {"HOTA": 65.0, "MOTA": 73.0, "IDF1": 79.0, "IDSW_rate": 0.04},
            },
            "trials": [],
        }

    monkeypatch.setitem(sys.modules, "boxmot.engine.tuner", SimpleNamespace(main=fake_main))

    tracker = boxmot(tracker="boosttrack", reid="something")
    results = tracker.tune(data="mot17-ablation", n_trials=8)

    assert results.benchmark == "mot17-ablation"
    assert results.best_trial_id == "trial_002"
    assert captured["args"].data == "mot17-ablation"
    assert captured["args"].tracking_method == "boosttrack"
    assert captured["args"].reid_model[0] == WEIGHTS / "something.pt"
    assert captured["args"].n_trials == 8


def test_boxmot_track_requires_source():
    tracker = BoxMOT()

    with pytest.raises(ValueError, match="BoxMOT.track\\(\\) requires a tracking source"):
        tracker.track()


def test_boxmot_track_remembers_source_and_runtime_overrides(monkeypatch):
    captured = []

    def fake_main(args):
        captured.append(args)
        return {
            "source": args.source,
            "tracking_method": args.tracking_method,
            "detector": args.yolo_model,
            "reid": args.reid_model,
            "frames": 12,
            "video_path": "/tmp/video_tracked.mp4",
            "text_path": "/tmp/video.txt",
            "timings": {"frames": 12, "track": 42.0, "total": 84.0},
        }

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracker", SimpleNamespace(main=fake_main))

    tracker = boxmot(tracker="boosttrack", reid="something")

    first = tracker.track(source="video.mp4", device="cpu", save=True)
    second = tracker.track()

    assert isinstance(first, TrackResults)
    assert first.frames == 12
    assert first.video_path == Path("/tmp/video_tracked.mp4")
    assert first.text_path == Path("/tmp/video.txt")
    assert second.source == "video.mp4"

    assert captured[0].source == "video.mp4"
    assert captured[0].tracking_method == "boosttrack"
    assert captured[0].reid_model == WEIGHTS / "something.pt"
    assert captured[0].device == "cpu"
    assert captured[0].save is True
    assert captured[1].source == "video.mp4"
    assert captured[1].device == "cpu"


def test_boxmot_track_accepts_click_style_kwargs(monkeypatch):
    captured = {}

    def fake_main(args):
        captured["args"] = args
        return {
            "source": args.source,
            "tracking_method": args.tracking_method,
            "detector": args.yolo_model,
            "reid": args.reid_model,
            "save_dir": "/tmp/runs/track",
            "video_path": "/tmp/runs/track/video_tracked.mp4",
            "text_path": "/tmp/runs/track/video.txt",
            "frames": 5,
            "user_quit": False,
            "timings": {"frames": 5, "track": 10.0, "total": 20.0},
        }

    monkeypatch.setitem(sys.modules, "boxmot.engine.tracker", SimpleNamespace(main=fake_main))

    tracker = boxmot(tracker="boosttrack", detector="yolo11s", reid="something")
    results = tracker.track(
        source="video.mp4",
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device="0",
        save=True,
        save_txt=True,
    )

    assert results.source == "video.mp4"
    assert results.tracking_method == "boosttrack"
    assert results.save_dir == Path("/tmp/runs/track")
    assert results.video_path == Path("/tmp/runs/track/video_tracked.mp4")
    assert results.text_path == Path("/tmp/runs/track/video.txt")
    assert captured["args"].imgsz == 640
    assert captured["args"].conf == pytest.approx(0.25)
    assert captured["args"].iou == pytest.approx(0.7)
    assert captured["args"].device == "0"
    assert captured["args"].save is True
    assert captured["args"].save_txt is True
    assert captured["args"].yolo_model == WEIGHTS / "yolo11s.pt"
    assert captured["args"].reid_model == WEIGHTS / "something.pt"


def test_boxmot_export_requires_weights_or_reid():
    tracker = BoxMOT()

    with pytest.raises(ValueError, match="BoxMOT.export\\(\\) requires model weights"):
        tracker.export()


def test_boxmot_export_uses_reid_weights_and_returns_artifacts(monkeypatch):
    captured = []

    def fake_main(args):
        captured.append(args)
        return {
            "weights": args.weights,
            "include": list(args.include),
            "output_dir": "/tmp/exports",
            "elapsed_time": 1.2,
            "input_shape": (1, 3, 256, 128),
            "output_shape": (1, 512),
            "files": {
                "onnx": "/tmp/exports/something.onnx",
                "engine": "/tmp/exports/something.engine",
            },
        }

    monkeypatch.setitem(sys.modules, "boxmot.engine.export", SimpleNamespace(main=fake_main))

    tracker = boxmot(reid="something")

    first = tracker.export(include=("onnx", "engine"), device="0", half=True)
    second = tracker.export()

    assert isinstance(first, ExportResults)
    assert first.weights == WEIGHTS / "something.pt"
    assert first.include == ["onnx", "engine"]
    assert first.output_dir == Path("/tmp/exports")
    assert first.onnx == Path("/tmp/exports/something.onnx")
    assert first["engine"] == Path("/tmp/exports/something.engine")
    assert second.weights == WEIGHTS / "something.pt"

    assert captured[0].weights == WEIGHTS / "something.pt"
    assert captured[0].include == ("onnx", "engine")
    assert captured[0].device == "0"
    assert captured[0].half is True
    assert captured[1].weights == WEIGHTS / "something.pt"
    assert captured[1].include == ("onnx", "engine")


def test_trackeval_metrics_exposes_summary_and_class_metrics():
    metrics = TrackEvalMetrics(
        {
            "plane": {"HOTA": 11.0, "MOTA": 12.0, "IDF1": 13.0},
            "all": {"HOTA": 21.0, "MOTA": 22.0, "IDF1": 23.0},
        }
    )

    assert metrics.summary_name == "all"
    assert metrics.HOTA == pytest.approx(21.0)
    assert metrics["MOTA"] == pytest.approx(22.0)
    assert metrics.classes["plane"].IDF1 == pytest.approx(13.0)
    assert metrics["plane"].HOTA == pytest.approx(11.0)


def test_tune_results_exposes_best_trial_metrics():
    results = TuneResults(
        {
            "tracking_method": "boosttrack",
            "benchmark": "mot17-mini",
            "best_trial_id": "trial_007",
            "best_config": {"max_age": 30},
            "best_metrics": {"HOTA": 70.1, "MOTA": 76.2, "IDF1": 81.3, "IDSW_rate": 0.02},
            "best_trial": {
                "trial_id": "trial_007",
                "trial_dir": "/tmp/trial_007",
                "config": {"max_age": 30},
                "metrics": {"HOTA": 70.1, "MOTA": 76.2, "IDF1": 81.3, "IDSW_rate": 0.02},
            },
            "trials": [
                {
                    "trial_id": "trial_007",
                    "trial_dir": "/tmp/trial_007",
                    "config": {"max_age": 30},
                    "metrics": {"HOTA": 70.1, "MOTA": 76.2, "IDF1": 81.3, "IDSW_rate": 0.02},
                }
            ],
        }
    )

    assert results.tracking_method == "boosttrack"
    assert results.benchmark == "mot17-mini"
    assert results.HOTA == pytest.approx(70.1)
    assert results.best.metrics["MOTA"] == pytest.approx(76.2)
    assert results.best.metrics.idsw_rate == pytest.approx(0.02)
    assert results.trials[0].config["max_age"] == 30


def test_stream_track_yields_structured_aabb_frame_results(monkeypatch):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

    def fake_get_frames(self):
        yield from frames

    monkeypatch.setattr(Results, "_get_frames", fake_get_frames)

    class DummyTracker:
        def update(self, dets, frame, features=None):
            return np.array([[1, 2, 5, 6, 7, 0.9, 0, 0]], dtype=np.float32)

    def detector(frame):
        return np.array([[1, 2, 5, 6, 0.9, 0]], dtype=np.float32)

    results = stream_track("video.mp4", detector, None, DummyTracker(), verbose=False)
    chunk = next(iter(results))

    assert chunk.frame_id == 1
    assert chunk.is_obb is False
    assert chunk.tracks.shape == (1, 8)
    np.testing.assert_allclose(chunk.xyxy, np.array([[1, 2, 5, 6]], dtype=np.float32))
    assert chunk.xywha is None
    np.testing.assert_array_equal(chunk.id, np.array([7], dtype=np.int32))
    np.testing.assert_allclose(chunk.conf, np.array([0.9], dtype=np.float32))
    np.testing.assert_array_equal(chunk.cls, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(chunk.det_ind, np.array([0], dtype=np.int32))


def test_stream_track_yields_structured_obb_frame_results(monkeypatch):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

    def fake_get_frames(self):
        yield from frames

    monkeypatch.setattr(Results, "_get_frames", fake_get_frames)

    class DummyTracker:
        def update(self, dets, frame, features=None):
            return np.array([[10, 20, 4, 8, 0.0, 3, 0.8, 1, 5]], dtype=np.float32)

    def detector(frame):
        return np.array([[10, 20, 4, 8, 0.0, 0.8, 1]], dtype=np.float32)

    results = stream_track("video.mp4", detector, None, DummyTracker(), verbose=False)
    chunk = next(iter(results))

    assert chunk.frame_id == 1
    assert chunk.is_obb is True
    np.testing.assert_allclose(
        chunk.xywha,
        np.array([[10, 20, 4, 8, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        chunk.xyxy,
        np.array([[8, 16, 12, 24]], dtype=np.float32),
    )
    np.testing.assert_array_equal(chunk.id, np.array([3], dtype=np.int32))
    np.testing.assert_allclose(chunk.conf, np.array([0.8], dtype=np.float32))
    np.testing.assert_array_equal(chunk.cls, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(chunk.det_ind, np.array([5], dtype=np.int32))


def test_stream_track_save_writes_cached_and_remaining_chunks(monkeypatch, tmp_path):
    frames = [np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.uint8)]

    def fake_get_frames(self):
        yield from frames

    monkeypatch.setattr(Results, "_get_frames", fake_get_frames)

    class DummyTracker:
        def __init__(self):
            self.calls = 0

        def update(self, dets, frame, features=None):
            self.calls += 1
            if self.calls == 1:
                return np.array([[1, 2, 5, 6, 7, 0.9, 0, 0]], dtype=np.float32)
            return np.empty((0, 8), dtype=np.float32)

    def detector(frame):
        return np.array([[1, 2, 5, 6, 0.9, 0]], dtype=np.float32)

    results = stream_track("video.mp4", detector, None, DummyTracker(), verbose=False)
    first = next(iter(results))
    np.testing.assert_array_equal(first.id, np.array([7], dtype=np.int32))

    output_path = results.save(tmp_path / "tracks.txt")

    assert output_path == tmp_path / "tracks.txt"
    assert output_path.read_text() == "1,7,1,2,4,4,0.900000,1,0\n"
