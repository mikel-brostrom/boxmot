from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "python_api_all.py"


def _load_example_module():
    spec = importlib.util.spec_from_file_location("python_api_all_example", EXAMPLE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_args(**overrides):
    defaults = {
        "modes": ["val", "tune", "track", "export", "stream"],
        "benchmark": "mot17-mini",
        "source": "video.mp4",
        "detector": "yolov8n.pt",
        "reid": "osnet_x0_25_msmt17.pt",
        "tracker": "strongsort",
        "device": "cpu",
        "half": False,
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.7,
        "classes": [0],
        "n_trials": 3,
        "include": ["onnx"],
        "save": False,
        "save_txt": False,
        "stream_output": Path("runs/python_api_stream.txt"),
        "stream_limit": 2,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_python_api_example_runs_high_level_workflows(monkeypatch, capsys):
    module = _load_example_module()
    args = _make_args(modes=["val", "tune", "track", "export"])
    calls = {}

    class DummyModel:
        def val(self, **kwargs):
            calls["val"] = kwargs
            return SimpleNamespace(summary={"HOTA": 70.0})

        def tune(self, **kwargs):
            calls["tune"] = kwargs
            return SimpleNamespace(
                best_config={"max_age": 30},
                best_yaml=Path("/tmp/best.yaml"),
                best=SimpleNamespace(metrics=SimpleNamespace(summary={"HOTA": 71.0})),
            )

        def track(self, **kwargs):
            calls["track"] = kwargs
            return SimpleNamespace(
                source=kwargs["source"],
                video_path=Path("/tmp/tracked.mp4"),
                text_path=Path("/tmp/tracked.txt"),
                timings={"track": 10.0},
            )

        def export(self, **kwargs):
            calls["export"] = kwargs
            return SimpleNamespace(
                weights=Path("/tmp/osnet_x0_25_msmt17.pt"),
                files={"onnx": Path("/tmp/osnet_x0_25_msmt17.onnx")},
            )

    def fake_boxmot(**kwargs):
        calls["boxmot"] = kwargs
        return DummyModel()

    monkeypatch.setattr(module, "boxmot", fake_boxmot)

    module.run_high_level_api(args)
    stdout = capsys.readouterr().out

    assert calls["boxmot"] == {
        "detector": "yolov8n.pt",
        "reid": "osnet_x0_25_msmt17.pt",
        "tracker": "strongsort",
        "classes": [0],
    }
    assert calls["val"]["benchmark"] == "mot17-mini"
    assert calls["val"]["imgsz"] == 640
    assert calls["tune"]["n_trials"] == 3
    assert calls["track"]["source"] == "video.mp4"
    assert calls["export"]["include"] == ("onnx",)
    assert "[val]" in stdout
    assert "[tune]" in stdout
    assert "[track]" in stdout
    assert "[export]" in stdout


def test_python_api_example_runs_streaming_workflow(monkeypatch, capsys, tmp_path):
    module = _load_example_module()
    args = _make_args(
        modes=["stream"],
        stream_output=tmp_path / "stream.txt",
        stream_limit=2,
    )
    calls = {}

    detector_callable = object()
    reid_callable = object()
    tracker_obj = object()

    monkeypatch.setattr(module, "make_detector", lambda weights: detector_callable)

    def fake_reid(weights, device="cpu", half=False):
        calls["reid"] = {"weights": weights, "device": device, "half": half}
        return reid_callable

    monkeypatch.setattr(module, "ReID", fake_reid)
    monkeypatch.setattr(module, "get_tracker_config", lambda tracker: Path(f"/tmp/{tracker}.yaml"))

    def fake_create_tracker(**kwargs):
        calls["create_tracker"] = kwargs
        return tracker_obj

    monkeypatch.setattr(module, "create_tracker", fake_create_tracker)

    class DummyFrameTracks:
        def __init__(self, frame_id: int, track_id: int):
            self.frame_id = frame_id
            self.xyxy = np.array([[1, 2, 5, 6]], dtype=np.float32)
            self.xywha = None
            self.conf = np.array([0.9], dtype=np.float32)
            self.cls = np.array([0], dtype=np.int32)
            self.id = np.array([track_id], dtype=np.int32)
            self.det_ind = np.array([0], dtype=np.int32)

        def __len__(self):
            return len(self.id)

    class DummyStream:
        def __init__(self):
            self.saved_to = None

        def __iter__(self):
            yield DummyFrameTracks(1, 10)
            yield DummyFrameTracks(2, 11)
            yield DummyFrameTracks(3, 12)

        def save(self, output_path):
            self.saved_to = Path(output_path)
            return self.saved_to

    stream = DummyStream()

    def fake_stream_track(source, detector, reid, tracker, verbose=False):
        calls["stream_track"] = {
            "source": source,
            "detector": detector,
            "reid": reid,
            "tracker": tracker,
            "verbose": verbose,
        }
        return stream

    monkeypatch.setattr(module, "stream_track", fake_stream_track)

    module.run_streaming_api(args)
    stdout = capsys.readouterr().out

    assert calls["reid"] == {
        "weights": "osnet_x0_25_msmt17.pt",
        "device": "cpu",
        "half": False,
    }
    assert calls["create_tracker"] == {
        "tracker_type": "strongsort",
        "tracker_config": Path("/tmp/strongsort.yaml"),
        "reid_weights": Path("osnet_x0_25_msmt17.pt"),
        "device": "cpu",
        "half": False,
        "per_class": False,
    }
    assert calls["stream_track"]["source"] == "video.mp4"
    assert calls["stream_track"]["detector"] is detector_callable
    assert calls["stream_track"]["reid"] is reid_callable
    assert calls["stream_track"]["tracker"] is tracker_obj
    assert calls["stream_track"]["verbose"] is False
    assert stream.saved_to == tmp_path / "stream.txt"
    assert "frame=1 tracks=1 ids=[10]" in stdout
    assert "frame=2 tracks=1 ids=[11]" in stdout
    assert "saved stream output to:" in stdout
