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
import boxmot.api.functional as api_functional_module
import boxmot.api.results as api_results_module
from boxmot.api import _args as api_args_module
import boxmot.engine.tracking.results as results_module
import boxmot.utils.rich.core.ui as ui_module
from boxmot.configs import BOXMOT_DEFAULTS, DEFAULT_DETECTOR, DEFAULT_REID, get_mode_default
from boxmot.detectors import Detector
from boxmot.detectors.base import Detections
from boxmot.engine import research as research_engine_module
from boxmot.engine.eval import cache as cache_module
from boxmot.engine.eval import evaluator as evaluator_module
from boxmot.engine.reid import evaluator as reid_evaluator_module
from boxmot.engine.reid import export as export_module
from boxmot.engine.reid import trainer as reid_trainer_module
from boxmot.engine.tracking import workflow as tracker_module
from boxmot.engine.tuning import tuner as tuner_module
from boxmot.engine.workflows import reporting as reporting_module
from boxmot.engine.workflows import support as workflow_support_module
from boxmot.reid import ReID
from boxmot.trackers import OccluBoost
from boxmot.utils.timing import TimingStats

_DUMMY_IMG = np.zeros((32, 32, 3), dtype=np.uint8)


def test_package_root_lazily_reexports_python_api():
    assert isinstance(boxmot.__version__, str)
    assert set(boxmot.__all__) == {"BoxMOT", "Detector", "ReIDModel"}
    assert "BoxMOT" in boxmot.__all__
    assert "Detector" in boxmot.__all__
    assert "ReIDModel" in boxmot.__all__
    assert not hasattr(boxmot, "Boxmot")
    assert not hasattr(boxmot, "Detections")
    assert not hasattr(boxmot, "track")
    assert boxmot.BoxMOT is api_module.BoxMOT
    assert boxmot.Detector is api_module.Detector
    assert boxmot.ReIDModel is api_module.ReIDModel
    assert importlib.import_module("boxmot.models.detector").Detector is api_module.Detector
    assert importlib.import_module("boxmot.models.reid").ReIDModel is api_module.ReIDModel
    assert importlib.import_module("boxmot.pipeline").BoxMOT is api_module.BoxMOT


def test_api_module_exports_only_canonical_classes():
    assert set(api_module.__all__) == {"BoxMOT", "Detector", "ReIDModel"}
    assert not hasattr(api_module, "Boxmot")
    assert not hasattr(api_module, "Detections")
    assert not hasattr(api_module, "track")
    assert not hasattr(api_module, "GenerateResult")


def test_trackers_package_exports_occluboost_only():
    trackers_module = importlib.import_module("boxmot.trackers")

    assert trackers_module.__all__ == ("OccluBoost",)
    assert trackers_module.OccluBoost is importlib.import_module("boxmot.trackers.occluboost").OccluBoost


def test_boxmot_defaults_follow_shared_configs():
    model = api_module.BoxMOT()

    assert model.detector == DEFAULT_DETECTOR == BOXMOT_DEFAULTS.shared.detector
    assert model.reid == DEFAULT_REID == BOXMOT_DEFAULTS.shared.reid
    assert model._tracker_name() == get_mode_default("track", "tracker") == BOXMOT_DEFAULTS.track.tracker
    assert model.project == Path(get_mode_default("track", "project")) == BOXMOT_DEFAULTS.track.project


def test_boxmot_accepts_grouped_component_kwargs_and_tracker_class(monkeypatch, tmp_path):
    captured = {}

    def fake_run_track(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return "run"

    monkeypatch.setattr(tracker_module, "run_track", fake_run_track)

    model = api_module.BoxMOT(
        detector="yolox_x_MOT17_ablation",
        reid="models/lmbn_n_duke.onnx",
        tracker=OccluBoost,
        detector_kwargs={"confidence": 0.25, "image_size": 640, "half": True},
        reid_kwargs={"half": True},
        tracker_kwargs={"with_reid": True},
        project=tmp_path / "runs",
    )

    run = model.track(source="0")

    assert run == "run"
    assert model._tracker_name() == "occluboost"
    assert captured["detector_spec"] == "yolox_x_MOT17_ablation"
    assert captured["reid_spec"] == "models/lmbn_n_duke.onnx"
    assert captured["tracker_spec"] is OccluBoost
    assert captured["detector_kwargs"] == {"confidence": 0.25, "image_size": 640, "half": True}
    assert captured["reid_kwargs"] == {"half": True}
    assert captured["tracker_kwargs"] == {"with_reid": True}


def test_boxmot_rejects_kwargs_with_initialized_components():
    with pytest.raises(ValueError, match="tracker_kwargs"):
        api_module.BoxMOT(tracker=object(), tracker_kwargs={"with_reid": True})

    with pytest.raises(ValueError, match="detector_kwargs"):
        api_module.BoxMOT(detector=object(), detector_kwargs={"confidence": 0.25})

    with pytest.raises(ValueError, match="reid_kwargs"):
        api_module.BoxMOT(reid=object(), reid_kwargs={"half": True})


def test_component_kwargs_reject_internal_aliases():
    with pytest.raises(ValueError, match="image_size"):
        workflow_support_module.build_detector_from_spec("fake.pt", detector_kwargs={"imgsz": 640})

    with pytest.raises(ValueError, match="confidence"):
        workflow_support_module.build_detector_from_spec("fake.pt", detector_kwargs={"conf": 0.25})

    with pytest.raises(ValueError, match="preprocess"):
        workflow_support_module.build_reid_from_spec("fake.pt", reid_kwargs={"preprocess_name": "resize"})


def test_boxmot_tracker_kwargs_override_eval_tracker_config():
    model = api_module.BoxMOT(
        tracker="occluboost",
        tracker_kwargs={"with_reid": True, "max_age": 42},
    )

    tracker_config = model._tracker_config_from_spec()

    assert tracker_config["with_reid"] is True
    assert tracker_config["max_age"] == 42
    assert "iou_threshold" in tracker_config


def test_boxmot_eval_namespace_uses_shared_reid_default_when_reid_is_none(tmp_path):
    model = api_module.BoxMOT(reid=None, project=tmp_path / "runs")

    args = model._base_eval_args("mot17-mini")

    assert args.reid == [DEFAULT_REID]


def test_boxmot_eval_namespace_treats_inherited_defaults_as_non_explicit():
    model = api_module.BoxMOT()

    args = model._base_eval_args("mot17-mini")

    assert args.detector_explicit is False
    assert args.reid_explicit is False
    assert args.tracker_explicit is False
    assert args.device_explicit is False
    assert args.half_explicit is False
    assert args.tracker_backend == "python"
    assert args.tracking_backend == "thread"


def test_boxmot_eval_namespace_preserves_explicit_constructor_overrides():
    model = api_module.BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack")

    args = model._base_eval_args("mot17-mini")

    assert args.detector_explicit is True
    assert args.reid_explicit is True
    assert args.tracker_explicit is True


def test_boxmot_val_namespace_accepts_explicit_split():
    model = api_module.BoxMOT(
        detector="yolox_n",
        reid="models/lmbn_n_duke.pt",
        tracker="occluboost",
    )

    args = model._base_eval_args(
        "mot17",
        split="ablation",
        conf=0.25,
        imgsz=640,
        half=True,
    )

    assert args.benchmark == "mot17"
    assert args.split == "ablation"
    assert args.split_explicit is True
    assert args.conf == 0.25
    assert args.imgsz == 640
    assert args.half is True
    assert args.detector[0].name == "yolox_n.pt"
    assert args.reid[0] == Path("models/lmbn_n_duke.pt")
    assert args.tracker == "occluboost"


def test_boxmot_tune_namespace_accepts_explicit_split():
    model = api_module.BoxMOT(
        detector="yolox_x_MOT17_ablation",
        reid="models/lmbn_n_duke.pt",
        tracker="occluboost",
    )

    args = api_args_module.build_tune_args(
        model,
        "mot17",
        split="ablation",
        n_trials=1,
    )

    assert args.benchmark == "mot17"
    assert args.split == "ablation"
    assert args.split_explicit is True
    assert args.n_trials == 1
    assert args.show_progress is False
    assert args.objectives == tuple(BOXMOT_DEFAULTS.tune.objectives)
    assert args.detector[0].name == "yolox_x_MOT17_ablation.pt"
    assert args.reid[0] == Path("models/lmbn_n_duke.pt")
    assert args.tracker == "occluboost"


def test_boxmot_constructor_keeps_detector_positional_for_detector_names():
    model = api_module.BoxMOT("yolov8n")

    assert model.detector == "yolov8n"
    assert model.train_model is None
    assert model.train_recipe is None
    assert model._detector_explicit is True


def test_boxmot_constructor_accepts_training_model_positional():
    model = api_module.BoxMOT("mobilenetv4")

    assert model.detector == DEFAULT_DETECTOR
    assert model.train_model is None
    assert model.train_recipe == "mobilenetv4"
    assert model._detector_explicit is False
    assert model._train_recipe_explicit is True


def test_boxmot_constructor_accepts_reid_weight_and_training_profile():
    model = api_module.BoxMOT(reid="mobilenetv4.pt")

    assert model.detector == DEFAULT_DETECTOR
    assert model.reid == "mobilenetv4.pt"
    assert model.train_model is None
    assert model.train_recipe == "mobilenetv4"
    assert model._detector_explicit is False
    assert model._reid_explicit is True
    assert model._train_recipe_explicit is True


def test_boxmot_eval_namespace_uses_explicit_tracker_backend():
    model = api_module.BoxMOT(tracker="botsort")

    args = model._base_eval_args("mot17-mini", tracker_backend="cpp")

    assert args.tracker == "botsort"
    assert args.tracker_backend == "cpp"


def test_boxmot_eval_namespace_allows_benchmark_runtime_to_override_inherited_defaults(monkeypatch):
    evaluator_module = importlib.import_module("boxmot.engine.eval.evaluator")
    model = api_module.BoxMOT()
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


def test_public_reid_empty_boxes_skip_backend_feature_call(monkeypatch):
    class _FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.half = False
            self.input_shape = (8, 4)
            self.mean_array = torch.zeros((1, 3, 1, 1), dtype=torch.float32)
            self.std_array = torch.ones((1, 3, 1, 1), dtype=torch.float32)

        def get_features(self, boxes, image):
            raise AssertionError("empty boxes should not call backend feature extraction")

        def inference_preprocess(self, batch):
            return batch

        def forward(self, batch):
            raise AssertionError("empty boxes should not run model forward")

        def inference_postprocess(self, features):
            return features.cpu().numpy()

    monkeypatch.setattr(ReID, "get_backend", lambda self: _FakeModel())

    reid = ReID("lmbn_n_duke.pt")
    embeddings = reid(_DUMMY_IMG, boxes=np.empty((0, 4), dtype=np.float32))

    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32


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


def test_public_detector_predicts_with_public_constructor_names(monkeypatch):
    class _FakeDetectorBackend:
        def __init__(self, model, device, imgsz):
            self.model = model
            self.device = device
            self.imgsz = imgsz

        def __call__(self, images, conf, iou, classes, agnostic_nms):
            return [Detections(dets=np.array([[1, 2, 3, 4, conf, 0]], dtype=np.float32), orig_img=images[0])]

    monkeypatch.setattr(api_module.Detector, "_get_backend_class", classmethod(lambda cls, path: _FakeDetectorBackend))

    detector = api_module.Detector("fake.pt", confidence=0.25, image_size=640, half=True)
    predictions = detector.predict(_DUMMY_IMG)

    assert detector.confidence == 0.25
    assert detector.image_size == 640
    assert detector.half is True
    np.testing.assert_array_equal(
        predictions.dets,
        np.array([[1, 2, 3, 4, 0.25, 0]], dtype=np.float32),
    )


def test_public_root_detector_returns_detections(monkeypatch):
    class _FakeDetectorBackend:
        def __init__(self, model, device, imgsz):
            self.model = model
            self.device = device
            self.imgsz = imgsz

        def __call__(self, images, conf, iou, classes, agnostic_nms):
            return [Detections(dets=np.array([[1, 2, 3, 4, conf, 0]], dtype=np.float32), orig_img=images[0])]

    monkeypatch.setattr(api_module.Detector, "_get_backend_class", classmethod(lambda cls, path: _FakeDetectorBackend))

    detector = api_module.Detector("fake", confidence=0.25, image_size=640, half=True)
    detections = detector.predict(_DUMMY_IMG)

    assert isinstance(detections, Detections)
    assert detector.confidence == 0.25
    assert detector.image_size == 640
    assert detector.half is True
    assert detections.shape == (1, 6)
    np.testing.assert_array_equal(detections.xyxy, np.array([[1, 2, 3, 4]], dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(detections), detections.dets)


def test_public_reid_model_embed_export_and_tracker_contract(monkeypatch, tmp_path):
    calls = {}

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

    def fake_boxmot_export(self, **kwargs):
        calls["boxmot_reid"] = self.reid
        calls["export_kwargs"] = kwargs
        exported_path = tmp_path / "fake_reid.onnx"
        exported_path.touch()
        return SimpleNamespace(embedding_weights=exported_path, half=bool(kwargs["half"]))

    monkeypatch.setattr(ReID, "get_backend", lambda self: _FakeModel())
    monkeypatch.setattr(api_module.BoxMOT, "export", fake_boxmot_export)

    reid = api_module.ReIDModel("fake_reid.pt")
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)

    embeddings = reid.embed(_DUMMY_IMG, boxes=boxes)
    tracker_features = reid.get_features(boxes, _DUMMY_IMG)
    exported = reid.export(format="onnx", half=True)

    assert embeddings.shape == (1, 2)
    np.testing.assert_array_equal(embeddings, tracker_features)
    assert isinstance(exported, api_module.ReIDModel)
    assert calls["boxmot_reid"] == reid.path
    assert calls["export_kwargs"]["dynamic"] is True
    assert exported.path.name == "fake_reid.onnx"
    assert exported.half is True


def test_tracker_update_accepts_image_and_embeddings_aliases():
    from boxmot.trackers.base import BaseTracker

    class _AliasTracker(BaseTracker):
        def __init__(self):
            super().__init__(det_thresh=0.1)
            self.captured = None

        def _update_impl(self, dets, img, embs=None, masks=None):
            self.captured = (dets, img, embs, masks)
            return np.empty((0, 8), dtype=np.float32)

    detections = Detections(
        dets=np.array([[1, 2, 3, 4, 0.9, 0]], dtype=np.float32),
        orig_img=_DUMMY_IMG,
    )
    embeddings = np.ones((1, 2), dtype=np.float32)

    tracker = _AliasTracker()
    tracks = tracker.update(detections, image=_DUMMY_IMG, embeddings=embeddings)

    assert tracks.shape == (0, 8)
    captured_dets, captured_img, captured_embs, captured_masks = tracker.captured
    np.testing.assert_array_equal(captured_dets, detections.dets)
    assert captured_img is _DUMMY_IMG
    assert captured_embs is embeddings
    assert captured_masks is None


def test_public_detector_predict_normalizes_none_to_empty_detections(monkeypatch):
    class _NoneDetectorBackend:
        def __init__(self, model, device, imgsz):
            self.model = model
            self.device = device
            self.imgsz = imgsz

        def __call__(self, images, conf, iou, classes, agnostic_nms):
            return None

    monkeypatch.setattr(Detector, "_get_backend_class", classmethod(lambda cls, path: _NoneDetectorBackend))

    detector = api_module.Detector("fake.pt")
    predictions = detector.predict(_DUMMY_IMG)

    assert predictions.shape == (0, 6)
    assert predictions.dets.dtype == np.float32


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

    # Test iteration and FrameResult properties
    results = api_functional_module.track(tmp_path, _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)
    first = next(iter(results))

    results.drawer = lambda frame, tracks: np.full_like(frame, 127)

    assert first.frame_idx == 1
    assert first.num_tracks == 1
    assert first.render().shape == _DUMMY_IMG.shape
    assert np.all(first.render() == 127)

    # save() streams remaining frames (1 already consumed above)
    output_path = tmp_path / "tracks.txt"
    saved = results.save(output_path)
    summary = results.summary()

    assert saved == output_path
    assert output_path.read_text(encoding="utf-8").count("\n") == 1
    assert summary["frames"] == 2
    assert summary["tracks"] == 2
    assert summary["unique_tracks"] == 2

    # Fresh Results for full save + evaluate
    results2 = api_functional_module.track(tmp_path, _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)
    output_path2 = tmp_path / "tracks2.txt"
    results2.save(output_path2)
    assert output_path2.read_text(encoding="utf-8").count("\n") == 2

    evaluation = api_functional_module.evaluate([results2], metrics=True, speed=True)
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

    model = api_module.BoxMOT(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
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

    model = api_module.BoxMOT(
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

    model = api_module.BoxMOT(detector=object(), reid=object(), tracker=object(), project=tmp_path / "runs")
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

    results = api_functional_module.track("0", _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)

    first = next(iter(results))
    summary = results.summary()

    assert first.frame_idx == 1
    assert summary["frames"] == 1
    assert summary["tracks"] == 1
    assert summary["unique_tracks"] == 1


def test_results_streaming_does_not_cache_frames(monkeypatch):
    """Results is purely streaming - no frames stored in RAM."""
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

    results = api_functional_module.track("0", _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)

    first = next(iter(results))
    assert first.frame_idx == 1
    assert not hasattr(results, "_cache")

    second = next(results)
    assert second.frame_idx == 2


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

    model = api_module.BoxMOT(detector=object(), reid=object(), tracker=object(), project=tmp_path / "runs")
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

    model = api_module.BoxMOT(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    summary = run.summary

    assert summary["frames"] == 2
    assert summary["tracks"] == 2
    assert summary["unique_tracks"] == 2
    assert run.timings["fps"] >= 0


def test_boxmot_track_show_flag_displays_results(monkeypatch, tmp_path):
    shown_frames = []

    class _FakeFrameResult:
        def to_mot(self):
            import numpy as np
            return np.empty((0, 0), dtype=np.float32)

        def render(self):
            return _DUMMY_IMG

        def show(self):
            shown_frames.append(1)
            return True  # continue

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
            self._exhausted = False
            self._interrupted = False

        def __iter__(self):
            self.totals.update({
                "det": 1.0,
                "reid": 2.0,
                "track": 3.0,
                "total": 6.0,
                "frames": 1,
                "detections": 4,
                "tracks": 5,
            })
            self._exhausted = True
            yield _FakeFrameResult()

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

    model = api_module.BoxMOT(detector=object(), reid=object(), tracker=object(), project=tmp_path / "runs")
    run = model.track(source=tmp_path, show=True)

    assert len(shown_frames) == 1
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

    results = api_functional_module.track("0", _InterruptingDetector(), _FakeReID(), _FakeTracker(), verbose=False)

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

    results = api_functional_module.track("0", _FakeDetector(), _FakeReID(), _FakeTracker(), verbose=False)

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

    model = api_module.BoxMOT(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    summary_text = run.format_summary()

    assert "TRACKING SUMMARY" in summary_text
    assert "Detector" in summary_text
    assert "Tracker" in summary_text
    assert "association/update" in summary_text
    assert "Overall total" in summary_text
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

    model = api_module.BoxMOT(detector=_FakeDetector(), reid=_FakeReID(), tracker=_FakeTracker(), project=tmp_path / "runs")
    run = model.track(source=tmp_path)

    rendered = ui_module.capture_renderable(run.renderable(), width=120)

    assert "TRACKING SUMMARY" in rendered
    assert "Track rows" in rendered
    assert "Unique IDs" in rendered
    assert "Stage" in rendered
    assert "Detector" in rendered
    assert "Tracker" in rendered
    assert "association/update" in rendered
    assert "Overall total" in rendered
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

    results = api_functional_module.track(tmp_path, _FakeDetector(), None, _FakeNativeTracker(), verbose=False)
    list(results)  # consume the stream
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
    result = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    result = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    result = api_results_module.ValidationResult(
        benchmark="mmot-mini",
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
            benchmark="mmot-mini",
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
    metrics = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    tune = api_results_module.TuneResult(
        benchmark="mot17-mini",
        tracker="botsort",
        trials=[],
        best=api_results_module.TuneTrialResult(index=1, config={}, metrics=metrics, score=(69.445, 78.243, 81.937)),
        best_config={},
        best_yaml=Path("best.yaml"),
    )

    report = tune.format_best_report()

    assert "TUNE BEST RESULTS" in report
    assert "MOT17-02" in report
    assert "COMBINED" in report


def test_tune_results_expose_validation_like_accessors():
    metrics = api_results_module.ValidationResult(
        benchmark="mot17-mini",
        raw={"all": {"HOTA": 69.445}},
        summary_label="all",
        summary={"HOTA": 69.445, "MOTA": 78.243, "IDF1": 81.937},
        timings={"frames": 10},
        exp_dir=Path("runs/eval"),
        args=SimpleNamespace(device="cpu"),
    )
    trial = api_results_module.TuneTrialResult(
        index=2,
        config={"track_buffer": 40},
        metrics=metrics,
        score=(69.445, 78.243, 81.937),
    )
    tune = api_results_module.TuneResult(
        benchmark="mot17-mini",
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
    assert "TuneResult(benchmark='mot17-mini'" in repr(tune)
    assert tune.to_dict()["summary"] == metrics.summary
    assert tune.to_dict(include_trials=True)["trials"][0]["metrics"]["summary"] == metrics.summary


def test_tune_result_str_shows_delta_vs_baseline(monkeypatch):
    class _TTYStdout:
        def isatty(self):
            return True

    monkeypatch.setattr(reporting_module.sys, "stdout", _TTYStdout())
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)

    baseline_metrics = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    best_metrics = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    baseline_trial = api_results_module.TuneTrialResult(index=1, config={"track_buffer": 30}, metrics=baseline_metrics, score=(66.0,))
    best_trial = api_results_module.TuneTrialResult(index=2, config={"track_buffer": 40}, metrics=best_metrics, score=(67.5,))
    tune = api_results_module.TuneResult(
        benchmark="mot17-mini",
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
    baseline_metrics = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    best_metrics = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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

    result = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    result = api_results_module.ValidationResult(
        benchmark="mot17-mini",
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
    assert "association/update" in combined
    assert "Tracker total" in combined


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

    run = api_results_module.TrackRunResult(
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
        return api_results_module.ValidationResult(
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
        metrics = api_results_module.ValidationResult(
            benchmark=str(args.benchmark),
            raw={"all": {"HOTA": 53.0, "MOTA": 48.0, "IDF1": 43.0}},
            summary_label="all",
            summary={"HOTA": 53.0, "MOTA": 48.0, "IDF1": 43.0},
            exp_dir=tmp_path / "tune",
            timings={},
            args=args,
        )
        best_trial = api_results_module.TuneTrialResult(
            index=1,
            config={"track_buffer": 40},
            metrics=metrics,
            score=(53.0,),
        )
        return api_results_module.TuneResult(
            benchmark=str(args.benchmark),
            tracker=args.tracker,
            trials=[best_trial],
            best=best_trial,
            best_config={"track_buffer": 40},
            best_yaml=tmp_path / "best.yaml",
        )

    def fake_run_export(args):
        calls["export"] = args
        return api_results_module.ExportResult(weights=Path(args.weights), files={"onnx": tmp_path / "exported.onnx"})

    monkeypatch.setattr(evaluator_module, "run_eval", fake_run_eval)
    monkeypatch.setattr(tuner_module, "run_tune", fake_run_tune)
    monkeypatch.setattr(export_module, "run_export", fake_run_export)

    model = api_module.BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack", classes=[0, 1], project=tmp_path / "runs")

    metrics = model.val(benchmark="mot17-mini", device="cpu")

    assert metrics.summary["HOTA"] == 50.0
    eval_args, eval_config, eval_kwargs = calls["eval"]
    assert eval_args.data == "mot17-mini"
    assert eval_args.detector[0].name == "yolov8n.pt"
    assert eval_args.reid[0].name == "lmbn_n_duke.pt"
    assert eval_args.classes == [0, 1]
    assert eval_args.show_progress is True
    assert eval_config is None
    assert "pipeline" in eval_kwargs
    assert eval_kwargs["pipeline"] is not None

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

    export_results = model.export(
        include=("onnx",),
        device="cpu",
        tflite_quantize="static",
        tflite_calibration_data=tmp_path / "calibration",
        tflite_calibration_samples=32,
        tflite_calibration_seed=7,
        tflite_calibration_update="moving_average",
        tflite_static_activation_bits=8,
    )

    export_args = calls["export"]
    assert export_results.weights.name == "lmbn_n_duke.pt"
    assert export_results.files["onnx"] == tmp_path / "exported.onnx"
    assert export_args.include == ("onnx",)
    assert export_args.weights.name == "lmbn_n_duke.pt"
    assert export_args.tflite_quantize == "static"
    assert export_args.tflite_calibration_data == tmp_path / "calibration"
    assert export_args.tflite_calibration_samples == 32
    assert export_args.tflite_calibration_seed == 7
    assert export_args.tflite_calibration_update == "moving_average"
    assert export_args.tflite_static_activation_bits == 8


def test_boxmot_export_accepts_format_alias_and_half_flag(monkeypatch, tmp_path):
    import boxmot.reid as reid_module

    calls = {}
    expected = np.ones((1, 4), dtype=np.float32)

    def fake_run_export(args):
        calls["export"] = args
        return api_results_module.ExportResult(
            weights=Path(args.weights),
            files={"onnx": tmp_path / "mobilenetv4.onnx"},
            half=args.half,
        )

    class FakeReID:
        def __init__(self, path, *, device="cpu", half=False, preprocess_name=None):
            calls["reid"] = {
                "path": path,
                "device": device,
                "half": half,
                "preprocess_name": preprocess_name,
            }

        def __call__(self, source, boxes=None):
            calls["embed"] = {"source": source, "boxes": boxes}
            return expected

    monkeypatch.setattr(export_module, "run_export", fake_run_export)
    monkeypatch.setattr(reid_module, "ReID", FakeReID)

    model = api_module.BoxMOT(reid="mobilenetv4.pt", project=tmp_path / "runs")
    model = model.export(format="onnx", half=True)
    embeddings = model.embed(source=tmp_path / "image.jpg")

    export_args = calls["export"]
    assert model.files["onnx"] == tmp_path / "mobilenetv4.onnx"
    assert embeddings is expected
    assert export_args.weights == Path.cwd() / "models" / "mobilenetv4.pt"
    assert export_args.include == ("onnx",)
    assert export_args.half is True
    assert export_args.dynamic is True
    assert calls["reid"] == {
        "path": tmp_path / "mobilenetv4.onnx",
        "device": "cpu",
        "half": True,
        "preprocess_name": None,
    }
    assert calls["embed"] == {"source": tmp_path / "image.jpg", "boxes": None}


def test_export_result_embed_prefers_exported_onnx(monkeypatch, tmp_path):
    import boxmot.reid as reid_module

    calls = {}
    expected = np.ones((1, 4), dtype=np.float32)

    class FakeReID:
        def __init__(self, path, *, device="cpu", half=False, preprocess_name=None):
            calls["init"] = {
                "path": path,
                "device": device,
                "half": half,
                "preprocess_name": preprocess_name,
            }

        def __call__(self, source, boxes=None):
            calls["call"] = {"source": source, "boxes": boxes}
            return expected

    monkeypatch.setattr(reid_module, "ReID", FakeReID)

    result = api_results_module.ExportResult(
        weights=tmp_path / "lmbn_n_duke.pt",
        files={"onnx": tmp_path / "lmbn_n_duke.onnx"},
        half=True,
    )
    embeddings = result.embed(source=tmp_path / "image.jpg", preprocess="resize")

    assert embeddings is expected
    assert result.embedding_weights == tmp_path / "lmbn_n_duke.onnx"
    assert calls["init"] == {
        "path": tmp_path / "lmbn_n_duke.onnx",
        "device": "cpu",
        "half": True,
        "preprocess_name": "resize",
    }
    assert calls["call"] == {"source": tmp_path / "image.jpg", "boxes": None}


def test_export_result_embed_falls_back_to_source_weights(monkeypatch, tmp_path):
    import boxmot.reid as reid_module

    calls = {}
    expected = np.ones((1, 4), dtype=np.float32)

    class FakeReID:
        def __init__(self, path, *, device="cpu", half=False, preprocess_name=None):
            calls["init"] = {"path": path, "device": device, "half": half, "preprocess_name": preprocess_name}

        def __call__(self, source, boxes=None):
            return expected

    monkeypatch.setattr(reid_module, "ReID", FakeReID)

    result = api_results_module.ExportResult(weights=tmp_path / "lmbn_n_duke.pt", files={})
    embeddings = result.embed(source=tmp_path / "image.jpg", half=False)

    assert embeddings is expected
    assert result.embedding_weights == tmp_path / "lmbn_n_duke.pt"
    assert calls["init"] == {
        "path": tmp_path / "lmbn_n_duke.pt",
        "device": "cpu",
        "half": False,
        "preprocess_name": None,
    }


def test_boxmot_embed_uses_reid_factory_weight(monkeypatch, tmp_path):
    import boxmot.reid as reid_module

    calls = {}
    expected = np.ones((1, 4), dtype=np.float32)

    class FakeReID:
        def __init__(self, path, *, device="cpu", half=False, preprocess_name=None):
            calls["init"] = {
                "path": path,
                "device": device,
                "half": half,
                "preprocess_name": preprocess_name,
            }

        def __call__(self, source, boxes=None):
            calls["call"] = {"source": source, "boxes": boxes}
            return expected

    monkeypatch.setattr(reid_module, "ReID", FakeReID)

    image = tmp_path / "image.jpg"
    model = api_module.BoxMOT(reid="mobilenetv4.pt", project=tmp_path / "runs")
    embeddings = model.embed(
        source=image,
        boxes=np.array([[0, 0, 16, 16]], dtype=np.float32),
        device="cpu",
        half=True,
        preprocess="resize",
    )

    assert embeddings is expected
    assert calls["init"] == {
        "path": Path.cwd() / "models" / "mobilenetv4.pt",
        "device": "cpu",
        "half": True,
        "preprocess_name": "resize",
    }
    assert calls["call"]["source"] == image
    np.testing.assert_array_equal(calls["call"]["boxes"], np.array([[0, 0, 16, 16]], dtype=np.float32))


def test_boxmot_train_and_eval_reid_facades(monkeypatch, tmp_path):
    calls = {}

    expected_train = api_results_module.TrainResult(
        best_epoch=1,
        best_mAP=0.75,
        best_rank1=0.50,
        weights_path=tmp_path / "runs" / "reid_train" / "exp" / "best.pt",
    )

    def fake_train_main(args):
        calls["train"] = args
        return expected_train

    def fake_eval_reid_main(args):
        calls["eval_reid"] = args
        return {
            "model": "mobilenetv2_x1_0",
            "dataset": "market1501",
            "mAP": 0.75,
            "rank1": 0.5,
            "rank5": 0.0,
            "rank10": 0.0,
        }

    monkeypatch.setattr(reid_trainer_module, "main", fake_train_main)
    monkeypatch.setattr(reid_evaluator_module, "main", fake_eval_reid_main)

    model = api_module.BoxMOT(project=tmp_path / "runs")

    trained = model.train(
        model="mobilenetv2_x1_0",
        dataset="market1501",
        data_dir=tmp_path / "assets" / "reid-mini",
        device="cpu",
        epochs=1,
        warmup_epochs=0,
        eval_interval=1,
        batch_size=4,
        p_ids=2,
        k_instances=2,
        num_workers=0,
        project=tmp_path / "runs" / "reid_train",
        name="exp",
        pretrained=False,
        seed=123,
        deterministic=False,
    )

    assert trained is expected_train
    train_args = calls["train"]
    assert train_args.model == "mobilenetv2_x1_0"
    assert train_args.dataset == "market1501"
    assert train_args.data_dir == str(tmp_path / "assets" / "reid-mini")
    assert train_args.epochs == 1
    assert train_args.warmup_epochs == 0
    assert train_args.eval_interval == 1
    assert train_args.batch_size == 4
    assert train_args.p_ids == 2
    assert train_args.k_instances == 2
    assert train_args.pretrained is False
    assert train_args.seed == 123
    assert train_args.deterministic is False
    assert train_args.project == tmp_path / "runs" / "reid_train"
    assert train_args.name == "exp"

    evaluated = model.eval_reid(
        weights=tmp_path / "runs" / "reid_train" / "exp" / "best.pt",
        model="mobilenetv2_x1_0",
        dataset="market1501",
        data_dir=tmp_path / "assets" / "reid-mini",
        preprocess="resize",
        imgsz=(384, 128),
        inference_feature="raw_mean",
        flip_tta=True,
        device="cpu",
        batch_size=2,
        num_workers=0,
        output=tmp_path / "runs" / "reid_eval",
    )

    assert evaluated["mAP"] == 0.75
    eval_args = calls["eval_reid"]
    assert eval_args.weights == str(tmp_path / "runs" / "reid_train" / "exp" / "best.pt")
    assert eval_args.model == "mobilenetv2_x1_0"
    assert eval_args.dataset == "market1501"
    assert eval_args.data_dir == str(tmp_path / "assets" / "reid-mini")
    assert eval_args.preprocess == "resize"
    assert eval_args.imgsz == (384, 128)
    assert eval_args.inference_feature == "raw_mean"
    assert eval_args.flip_tta is True
    assert eval_args.batch_size == 2
    assert eval_args.num_workers == 0
    assert eval_args.output == str(tmp_path / "runs" / "reid_eval")


def test_boxmot_train_accepts_training_cfg(monkeypatch, tmp_path):
    calls = {}
    expected_train = api_results_module.TrainResult(
        best_epoch=1,
        best_mAP=0.75,
        best_rank1=0.50,
        weights_path=tmp_path / "runs" / "reid_train" / "exp" / "best.pt",
    )

    def fake_train_main(args):
        calls["train"] = args
        return expected_train

    monkeypatch.setattr(reid_trainer_module, "main", fake_train_main)

    train_cfg = tmp_path / "custom_config.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "run:",
                "  model_name: csl_tinyvit_7m",
                "data:",
                "  dataset: duke",
                f"  data_dir: {tmp_path}",
                "  img_size: [384, 128]",
                "model:",
                "  head:",
                "    parts: [1, 2, 4]",
                "optimization:",
                "  epochs: 5",
                "  lr: 0.001",
                "system:",
                "  device: cpu",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model = api_module.BoxMOT(project=tmp_path / "runs")
    trained = model.train(cfg=train_cfg, epochs=2)

    assert trained is expected_train
    train_args = calls["train"]
    assert train_args.model == "csl_tinyvit_7m"
    assert train_args.dataset == "duke"
    assert train_args.data_dir == str(tmp_path)
    assert train_args.imgsz == (384, 128)
    assert train_args.head_parts == (1, 2, 4)
    assert train_args.epochs == 2
    assert train_args.lr == 0.001
    assert train_args.device == "cpu"
    assert {"cfg", "epochs"} <= set(train_args.train_explicit_keys)


def test_boxmot_train_applies_constructor_training_profile_to_cfg(monkeypatch, tmp_path):
    calls = {}
    expected_train = api_results_module.TrainResult(
        best_epoch=1,
        best_mAP=0.75,
        best_rank1=0.50,
        weights_path=tmp_path / "runs" / "reid_train" / "exp" / "best.pt",
    )

    def fake_train_main(args):
        calls["train"] = args
        return expected_train

    monkeypatch.setattr(reid_trainer_module, "main", fake_train_main)

    train_cfg = tmp_path / "mobilenetv4_custom.yaml"
    train_cfg.write_text(
        "\n".join(
            [
                "data:",
                "  dataset: duke",
                f"  data_dir: {tmp_path}",
                "optimization:",
                "  epochs: 9",
                "  lr: 0.0008",
                "system:",
                "  device: cpu",
                "",
            ]
        ),
        encoding="utf-8",
    )

    model = api_module.BoxMOT("mobilenetv4", project=tmp_path / "runs")
    trained = model.train(cfg=train_cfg, epochs=2)

    assert trained is expected_train
    train_args = calls["train"]
    assert train_args.model == "mobilenetv4_conv_small"
    assert train_args.dataset == "duke"
    assert train_args.data_dir == str(tmp_path)
    assert train_args.epochs == 2
    assert train_args.lr == 0.0008
    assert train_args.device == "cpu"
    assert train_args.feature_fusion == "final"
    assert {"cfg", "recipe", "epochs"} <= set(train_args.train_explicit_keys)


def test_boxmot_train_accepts_reid_data_yaml_list(monkeypatch, tmp_path):
    calls = {}
    expected_train = api_results_module.TrainResult(
        best_epoch=1,
        best_mAP=0.75,
        best_rank1=0.50,
        weights_path=tmp_path / "runs" / "reid_train" / "exp" / "best.pt",
    )

    def fake_train_main(args):
        calls["train"] = args
        return expected_train

    monkeypatch.setattr(reid_trainer_module, "main", fake_train_main)

    data_root = tmp_path / "datasets"
    market_root = data_root / "Market-1501-v15.09.15"
    duke_root = data_root / "DukeMTMC-reID"
    market_root.mkdir(parents=True)
    duke_root.mkdir(parents=True)
    market_yaml = tmp_path / "market1501.yaml"
    duke_yaml = tmp_path / "duke.yaml"
    market_yaml.write_text(f"dataset: market1501\npath: {market_root}\n", encoding="utf-8")
    duke_yaml.write_text(f"dataset: duke\npath: {duke_root}\n", encoding="utf-8")

    model = api_module.BoxMOT(project=tmp_path / "runs")
    trained = model.train(
        model="mobilenetv2_x1_0",
        data=[market_yaml, duke_yaml],
        device="cpu",
        epochs=1,
        project=tmp_path / "runs" / "reid_train",
        name="exp",
    )

    assert trained is expected_train
    train_args = calls["train"]
    assert train_args.dataset == "market1501,duke"
    assert train_args.data_dir == str(data_root)
    assert train_args.data_specs[0]["root"] == str(market_root)
    assert train_args.data_specs[1]["root"] == str(duke_root)


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
            self._live = None
            self.detail_renderable = None
            self.detail_text = None
            self.detail_title = None

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

        def renderable(self, *, compact=False, include_setup=True):
            return ""

    def fake_create_workflow_progress(title, fields, *, steps=(), stderr=False, transient=False):
        workflow = _FakeWorkflow(title, fields, steps, stderr=stderr, transient=transient)
        workflows.append(workflow)
        return workflow

    def fake_run_eval(args, *, evolve_config=None, **kwargs):
        return api_results_module.ValidationResult(
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

    model = api_module.BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="botsort", project=tmp_path / "runs")

    metrics = model.val(benchmark="mot17-mini", device="cpu")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.title == "Evaluation"
    assert workflow.started is True
    assert workflow.stopped is True
    assert ("__panel__:Tracker", [("Name", "botsort"), ("Backend", "python")]) in workflow.fields
    assert ("__panel__:Dataset", [("Benchmark", "mot17-mini")]) in workflow.fields
    assert (evaluator_module.EVAL_SETUP_STEP, "active") in workflow.steps
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
        args.split = "train"
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

    model = api_module.BoxMOT(tracker="bytetrack", project=tmp_path / "runs")

    generated = model.generate(benchmark="mot17-mini", device="cpu", batch_size=8, resume=False)

    generate_args = calls["generate"]
    assert generated.benchmark == "mot17-mini"
    assert generated.source == tmp_path / "datasets" / "mot17-mini" / "train"
    assert generated.cache_dir == tmp_path / "runs" / "dets_n_embs" / "mot17-mini" / "train"
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
    model = api_module.BoxMOT(project=tmp_path / "runs")

    with pytest.raises(ValueError, match="exactly one of benchmark=... or source=..."):
        model.generate()

    with pytest.raises(ValueError, match="exactly one of benchmark=... or source=..."):
        model.generate(benchmark="mot17-mini", source=tmp_path / "dataset")


def test_boxmot_tune_forwards_optimization_targets_and_seed(monkeypatch, tmp_path):
    captured = {}

    def fake_run_tune(args, *, baseline_config=None):
        captured["args"] = args
        captured["baseline_config"] = baseline_config
        metrics = api_results_module.ValidationResult(
            benchmark=str(args.benchmark),
            raw={},
            summary_label="all",
            summary={"HOTA": 51.0, "MOTA": 46.0, "IDF1": 41.0},
            exp_dir=None,
            timings={},
            args=args,
        )
        trial = api_results_module.TuneTrialResult(index=1, config={}, metrics=metrics, score=(51.0, -0.2))
        return api_results_module.TuneResult(
            benchmark=str(args.benchmark),
            tracker=args.tracker,
            trials=[trial],
            best=trial,
            best_config={},
            best_yaml=tmp_path / "best.yaml",
        )

    monkeypatch.setattr(tuner_module, "run_tune", fake_run_tune)
    model = api_module.BoxMOT(detector="yolov8n", reid="lmbn_n_duke", tracker="boosttrack", project=tmp_path / "runs")

    tuned = model.tune(
        benchmark="mot17-mini",
        n_trials=2,
        device="cpu",
        objectives=("hota", "id_switches"),
        maximize=("HOTA", "IDF1"),
        minimize=("IDSW_rate",),
        seed=7,
    )

    assert tuned.best.index == 1
    assert captured["args"].n_trials == 2
    assert captured["args"].objectives == ("HOTA", "IDSW")
    assert captured["args"].maximize == ("HOTA", "IDF1")
    assert captured["args"].minimize == ("IDSW_rate",)
    assert captured["args"].seed == 7
    assert captured["baseline_config"] is None

    tuned = model.tune(
        benchmark="mot17-mini",
        split="ablation",
        n_trials=1,
        objectives=("hota", "id_switches"),
    )

    assert tuned.best.index == 1
    assert captured["args"].split == "ablation"
    assert captured["args"].objectives == ("HOTA", "IDSW")
    assert captured["args"].maximize == ("HOTA",)
    assert captured["args"].minimize == ("IDSW",)


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
