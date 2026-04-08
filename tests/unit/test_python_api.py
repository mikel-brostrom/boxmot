from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

import boxmot
import boxmot.api as api_module
from boxmot.configs import BOXMOT_DEFAULTS, DEFAULT_DETECTOR, DEFAULT_REID, get_mode_default
from boxmot.detectors import Detector
from boxmot.detectors.base import Detections
from boxmot.reid import ReID


_DUMMY_IMG = np.zeros((32, 32, 3), dtype=np.uint8)


def test_boxmot_defaults_follow_shared_configs():
    model = boxmot.Boxmot()

    assert model.detector == DEFAULT_DETECTOR == BOXMOT_DEFAULTS.shared.detector
    assert model.reid == DEFAULT_REID == BOXMOT_DEFAULTS.shared.reid
    assert model._tracker_name() == get_mode_default("track", "tracker") == BOXMOT_DEFAULTS.track.tracker
    assert model.project == Path(get_mode_default("track", "project")) == BOXMOT_DEFAULTS.track.project


def test_boxmot_eval_namespace_uses_shared_reid_default_when_reid_is_none(tmp_path):
    model = boxmot.Boxmot(reid=None, project=tmp_path / "runs")

    args = model._base_eval_args("mot17-ablation")

    assert args.reid_model == [DEFAULT_REID]


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

    results = boxmot.track(tmp_path, _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)
    first = next(iter(results))

    results.drawer = lambda frame, tracks: np.full_like(frame, 127)

    assert first.frame_idx == 1
    assert first.num_tracks == 1
    assert first.render().shape == _DUMMY_IMG.shape
    assert np.all(first.render() == 127)

    output_path = tmp_path / "tracks.txt"
    saved = results.save(output_path)
    summary = results.summary()
    evaluation = boxmot.evaluate([results], metrics=True, speed=True)

    assert saved == output_path
    assert output_path.read_text(encoding="utf-8").count("\n") == 2
    assert summary["frames"] == 2
    assert summary["tracks"] == 2
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

    model = boxmot.Boxmot(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path, save=True, save_txt=True)

    assert run.source == tmp_path
    assert run.video_path is not None and run.video_path.exists()
    assert run.text_path is not None and run.text_path.exists()
    assert run.summary["frames"] == 2
    assert run.timings["fps"] >= 0
    assert len(frames_written) == 2


def test_boxmot_val_tune_and_export_facades(monkeypatch, tmp_path):
    state = {"last_config": None}
    evaluator_calls = []
    replay_calls = []

    def fake_eval_setup(args):
        args.source = tmp_path / "benchmark" / "train"
        args.project = tmp_path / "runs"
        args.project.mkdir(parents=True, exist_ok=True)

    def fake_run_generate_dets_embs(args, timing_stats=None):
        evaluator_calls.append(("generate", args.data, args.yolo_model[0].name, args.reid_model[0].name, args.classes))
        if timing_stats is not None:
            timing_stats.frames = 2
            timing_stats.totals["inference"] = 12.0
            timing_stats.totals["reid"] = 4.0
            timing_stats.totals["total"] = 20.0

    def fake_run_generate_mot_results(args, evolve_config=None, timing_stats=None, quiet=False):
        state["last_config"] = evolve_config or {}
        replay_calls.append(("track", quiet, state["last_config"]))
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

    model = boxmot.Boxmot(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack", classes=[0, 1], project=tmp_path / "runs")

    metrics = model.val(benchmark="mot17-mini", device="cpu")

    assert metrics.summary["HOTA"] == 50.0
    assert evaluator_calls[0] == ("generate", "mot17-mini", "yolov8n.pt", "lmbn_n_duke.pt", [0, 1])
    assert replay_calls[0] == ("track", True, {})

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

    export_results = model.export(include=("onnx",), device="cpu")

    assert export_results.weights.name == "lmbn_n_duke.pt"
    assert export_results.files["onnx"] == tmp_path / "exported.onnx"