import importlib
import queue
import re
from contextlib import nullcontext
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from rich.console import Console

import boxmot.data.loaders as loaders_module
import boxmot.detectors.detector as detector_module
import boxmot.detectors.ultralytics as ultralytics_detector_module
import boxmot.engine.eval.evaluator as evaluator_module
import boxmot.engine.eval.replay as cached_tracking_module
import boxmot.engine.tracking.inference as pipeline_module
import boxmot.engine.tracking.runtime as tracker_runtime_module
import boxmot.engine.workflows.reporting as workflow_reporting_module
import boxmot.reid.core as reid_core_module
import boxmot.utils.rich.core.ui as ui_module
from boxmot.configs import ensure_model_extension
from boxmot.data import iter_source
from boxmot.data.cache import (
    AppendableNpyWriter,
    _existing_cache_path,
    _max_frame_id,
    _saved_detection_column_count,
)
from boxmot.detectors import default_conf, default_imgsz, get_detector_url, get_runtime_detector_cfg, load_detector_cfg
from boxmot.detectors.base import Detections
from boxmot.detectors.ultralytics import UltralyticsDetector
from boxmot.engine.tracking.inference import prepare_detections
from boxmot.engine.tracking.mot import convert_to_mot_format, write_mot_results
from boxmot.engine.workflows import support as workflow_support_module
from boxmot.trackers.base import BaseTracker
from boxmot.trackers.common.association.iou import iou_obb_pair
from boxmot.trackers.common.detections.layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.trackers.common.motion import (
    xysr_state_to_xywha,
    xywha_to_xysr_measurement,
)
from boxmot.utils import WEIGHTS
from boxmot.utils.timing import TimingStats

_DUMMY_IMG = np.zeros((64, 64, 3), dtype=np.uint8)


class _DummyTracker(BaseTracker):
    def _update_impl(
        self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None, masks: np.ndarray = None
    ) -> np.ndarray:
        self.check_inputs(dets, img, embs)
        return np.empty((0, 9 if self.is_obb else 8), dtype=np.float32)

    def _display_groups(self):
        return []


class _DummyOBBTracker(_DummyTracker):
    supports_obb = True


def test_prepare_detections_reads_aabb_result():
    dets = np.array([[10, 20, 30, 40, 0.9, 0]], dtype=np.float32)
    result = Detections(dets=dets, orig_img=_DUMMY_IMG)

    out = prepare_detections(result)

    assert out.shape == (1, 6)
    np.testing.assert_array_equal(out[0], np.array([10, 20, 30, 40, 0.9, 0], dtype=np.float32))


def test_prepare_detections_reads_obb_result():
    dets = np.array([[10, 20, 30, 40, 0.5, 0.8, 1]], dtype=np.float32)
    result = Detections(dets=dets, orig_img=_DUMMY_IMG)

    out = prepare_detections(result)

    assert out.shape == (1, 7)
    np.testing.assert_array_equal(out[0], dets[0])


def test_prepare_detections_preserves_empty_obb_width():
    dets = np.empty((0, 7), dtype=np.float32)
    result = Detections(dets=dets, orig_img=_DUMMY_IMG)

    out = prepare_detections(result)

    assert out.shape == (0, 7)


def test_prepare_detections_filters_invalid_obb_boxes():
    dets = np.array(
        [
            [100, 100, 20, 10, 0.2, 0.9, 0],  # valid: area = 200
            [100, 100, 0, 10, 0.2, 0.9, 0],  # invalid: w = 0
        ],
        dtype=np.float32,
    )
    result = Detections(dets=dets, orig_img=_DUMMY_IMG)

    out = prepare_detections(result)

    assert out.shape == (1, 7)
    np.testing.assert_array_equal(out[0], dets[0])


def test_ultralytics_detector_preserves_obb_results(monkeypatch):
    class _FakeOBB:
        def __init__(self, values):
            tensor = torch.tensor(values, dtype=torch.float32)
            self.xywhr = tensor[:, :5]
            self.conf = tensor[:, 5]
            self.cls = tensor[:, 6]

        def __len__(self):
            return len(self.conf)

    class _FakeResult:
        def __init__(self, values):
            self.obb = _FakeOBB(values)
            self.boxes = None
            self.orig_img = _DUMMY_IMG
            self.path = "frame.jpg"
            self.names = {0: "plane"}

    class _FakePredictor:
        def __init__(self):
            self.batch = None
            self.args = SimpleNamespace(conf=0.25, iou=0.7, classes=None, agnostic_nms=False)

        def preprocess(self, ims):
            return torch.zeros((len(ims), 3, 64, 64), dtype=torch.float32)

        def inference(self, preprocessed):
            return torch.zeros((preprocessed.shape[0], 1, 7), dtype=torch.float32)

        def postprocess(self, raw_preds, preprocessed, orig_imgs):
            return [_FakeResult([[32.0, 24.0, 20.0, 10.0, 0.25, 0.9, 0.0]])]

    class _FakeYOLO:
        def __init__(self, model):
            self.model = model
            self.names = {0: "plane"}
            self.predictor = None

        def predict(self, **_kwargs):
            self.predictor = _FakePredictor()
            return [_FakeResult([[32.0, 24.0, 20.0, 10.0, 0.25, 0.9, 0.0]])]

    monkeypatch.setattr(ultralytics_detector_module, "YOLO", _FakeYOLO)

    detector = UltralyticsDetector(model="fake-obb.pt", device="cpu", imgsz=[64, 64])
    results = detector(
        [_DUMMY_IMG],
        conf=0.25,
        iou=0.7,
        classes=None,
        agnostic_nms=False,
    )

    assert len(results) == 1
    assert results[0].is_obb is True
    np.testing.assert_array_equal(
        results[0].dets,
        np.array([[32.0, 24.0, 20.0, 10.0, 0.25, 0.9, 0.0]], dtype=np.float32),
    )


def test_ultralytics_detector_scales_segmentation_masks_to_original_shape():
    class _FakeBoxes:
        def __init__(self):
            self.xyxy = torch.tensor([[10.0, 10.0, 20.0, 20.0]], dtype=torch.float32)
            self.conf = torch.tensor([0.9], dtype=torch.float32)
            self.cls = torch.tensor([0.0], dtype=torch.float32)

        def __len__(self):
            return len(self.conf)

    class _FakeMasks:
        def __init__(self):
            self.data = torch.zeros((1, 80, 80), dtype=torch.uint8)
            # Original image is 40x80 letterboxed into 80x80, so rows 0:20
            # and 60:80 are padding. This object should land at y=10:20
            # after de-letterboxing, not y=15:20 from naive resize.
            self.data[0, 30:40, 10:20] = 1
            self.orig_shape = (40, 80)

        def __len__(self):
            return int(self.data.shape[0])

    result = SimpleNamespace(
        obb=None,
        boxes=_FakeBoxes(),
        masks=_FakeMasks(),
        orig_img=np.zeros((40, 80, 3), dtype=np.uint8),
    )
    detector = UltralyticsDetector.__new__(UltralyticsDetector)

    dets, masks = detector._extract_dets(result)

    np.testing.assert_array_equal(
        dets,
        np.array([[10.0, 10.0, 20.0, 20.0, 0.9, 0.0]], dtype=np.float32),
    )
    assert masks.shape == (1, 40, 80)
    assert masks[0, 10:20, 10:20].all()
    assert masks[0, :10, 10:20].sum() == 0
    assert masks[0, 20:, 10:20].sum() == 0


def test_default_detector_fallbacks_preserve_legacy_runtime_behavior():
    assert default_imgsz("yolox_s.pt") == [1080, 1920]
    assert default_imgsz("yolov8n.pt") == [640, 640]
    assert default_conf("yolox_s.pt") == 0.01


def test_model_config_detector_defaults_override_runtime_defaults_by_model_name():
    detector_cfg = load_detector_cfg("yolo11l_3ch.pt")

    assert detector_cfg["classes"][0] == "car"
    assert default_imgsz("yolo11l_3ch.pt") == detector_cfg["imgsz"]
    assert default_conf("yolo11l_3ch.pt") == detector_cfg["conf"]


def test_model_config_detector_defaults_match_separator_variants():
    detector_cfg = load_detector_cfg("yolo11l_3ch.pt")

    assert detector_cfg["id"] == "yolo11l_3ch"
    assert default_imgsz("yolo11l_3ch.pt") == detector_cfg["imgsz"]
    assert get_detector_url("yolo11l_3ch.pt") == detector_cfg["url"]


def test_runtime_detector_cfg_uses_model_config_to_override_dataset_values():
    detector_cfg = load_detector_cfg("yolo11l_3ch.pt")
    benchmark_cfg = {
        "default_model": "models/yolo11l_3ch.pt",
        "imgsz": [1024, 1024],
        "conf": 0.2,
        "classes": {0: "person"},
    }

    resolved = get_runtime_detector_cfg("yolo11l_3ch.pt", benchmark_cfg)

    assert resolved["default_model"] == "models/yolo11l_3ch.pt"
    assert resolved["imgsz"] == detector_cfg["imgsz"]
    assert resolved["conf"] == detector_cfg["conf"]
    assert resolved["classes"][0] == detector_cfg["classes"][0]


def test_ultralytics_detector_downloads_missing_configured_weights(monkeypatch, tmp_path):
    calls = {}

    class _FakeYOLO:
        def __init__(self, model):
            calls["model"] = model
            self.names = {0: "car"}

    monkeypatch.setattr(ultralytics_detector_module, "YOLO", _FakeYOLO)
    monkeypatch.setattr(
        ultralytics_detector_module,
        "download_file",
        lambda url, dest, overwrite=False, **_kwargs: calls.update({"url": url, "dest": dest, "overwrite": overwrite})
        or dest,
        raising=False,
    )

    missing_model = tmp_path / "yolo11l-3ch.pt"
    detector = UltralyticsDetector(model=missing_model, device="cpu", imgsz=[64, 64])

    assert detector.names == {0: "car"}
    assert calls == {
        "url": "https://drive.google.com/uc?id=15gmA4-Yclvh5EZvTJYhcyV1CVdNRGIkR",
        "dest": missing_model,
        "overwrite": False,
        "model": str(missing_model),
    }


def test_ultralytics_detector_downloads_missing_official_weights_into_models_dir(monkeypatch):
    calls = {}

    class _FakeYOLO:
        def __init__(self, model):
            calls["model"] = model
            self.names = {0: "car"}

    monkeypatch.setattr(ultralytics_detector_module, "YOLO", _FakeYOLO)
    monkeypatch.setattr(
        ultralytics_detector_module,
        "attempt_download_asset",
        lambda file, release="latest", **_kwargs: calls.update({"file": Path(file), "release": release}) or str(file),
        raising=False,
    )

    detector = UltralyticsDetector(model="yolo11_codex_missing.pt", device="cpu", imgsz=[64, 64])

    assert detector.names == {0: "car"}
    assert calls["file"] == WEIGHTS / "yolo11_codex_missing.pt"
    assert calls["release"] == "latest"
    assert calls["model"] == str(WEIGHTS / "yolo11_codex_missing.pt")


def test_ultralytics_detector_redownloads_corrupt_official_weights(monkeypatch, tmp_path):
    calls = {"models": []}

    class _FakeYOLO:
        def __init__(self, model):
            calls["models"].append(model)
            if len(calls["models"]) == 1:
                raise RuntimeError("PytorchStreamReader failed reading zip archive: failed finding central directory")
            self.names = {0: "car"}

    monkeypatch.setattr(ultralytics_detector_module, "YOLO", _FakeYOLO)
    monkeypatch.setattr(
        ultralytics_detector_module,
        "attempt_download_asset",
        lambda file, release="latest", **_kwargs: calls.update({"file": Path(file), "release": release}) or str(file),
        raising=False,
    )

    corrupt_model = tmp_path / "yolo11_corrupt.pt"
    corrupt_model.write_bytes(b"broken")

    detector = UltralyticsDetector(model=corrupt_model, device="cpu", imgsz=[64, 64])

    assert detector.names == {0: "car"}
    assert calls["models"] == [str(corrupt_model), str(corrupt_model)]
    assert calls["file"] == corrupt_model
    assert calls["release"] == "latest"


def test_configure_benchmark_runtime_lets_model_config_override_dataset_detector(monkeypatch):
    detector_cfg = load_detector_cfg("yolo11l_3ch.pt")
    benchmark_bundle = {
        "benchmark": {"box_type": "obb"},
        "detector": {
            "default_model": "models/yolo11l_3ch.pt",
            "imgsz": [1024, 1024],
            "conf": 0.2,
            "classes": {0: "person"},
        },
        "reid": {
            "default_model": "models/lmbn_n_duke.pt",
            "device": "cpu",
            "half": True,
        },
    }
    args = SimpleNamespace(
        detector=[Path("models/yolov8n.pt")],
        reid=[Path("models/osnet_x0_25_msmt17.pt")],
        detector_explicit=False,
        reid_explicit=False,
        device="cuda:0",
        half=False,
        imgsz=None,
        conf=None,
        eval_box_type=None,
        dataset_detector_cfg=None,
    )

    monkeypatch.setattr(evaluator_module, "_load_benchmark_cfg", lambda _args: benchmark_bundle)
    monkeypatch.setattr(evaluator_module, "should_use_benchmark_detector", lambda _args, _cfg: True)
    monkeypatch.setattr(evaluator_module, "should_use_benchmark_reid", lambda _args, _cfg: True)
    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_detector_model",
        lambda _cfg: Path("models/yolo11l_3ch.pt"),
    )
    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_reid_model",
        lambda _cfg: Path("models/lmbn_n_duke.pt"),
    )

    _, _, runtime_cfg = evaluator_module._configure_benchmark_runtime(args)

    assert args.detector == [Path("models/yolo11l_3ch.pt")]
    assert args.reid == [Path("models/lmbn_n_duke.pt")]
    assert args.reid_device == "cpu"
    assert args.reid_half is True
    assert args.imgsz == detector_cfg["imgsz"]
    assert args.conf == detector_cfg["conf"]
    assert args.eval_box_type == "obb"
    assert runtime_cfg["classes"][0] == detector_cfg["classes"][0]
    assert args.dataset_detector_cfg["classes"][0] == detector_cfg["classes"][0]


def test_configure_benchmark_runtime_reuses_existing_benchmark_model_paths(monkeypatch, tmp_path):
    detector_cfg = load_detector_cfg("yolox_x_mot17_ablation")
    detector_path = tmp_path / "yolox_x_MOT17_ablation.pt"
    reid_path = tmp_path / "lmbn_n_duke.pt"
    detector_path.write_bytes(b"detector")
    reid_path.write_bytes(b"reid")

    benchmark_bundle = {
        "benchmark": {"box_type": "aabb"},
        "detector": detector_cfg,
        "reid": {
            "default_model": "models/lmbn_n_duke.pt",
            "device": "",
            "half": True,
        },
    }
    args = SimpleNamespace(
        detector=[detector_path],
        reid=[reid_path],
        detector_explicit=True,
        reid_explicit=True,
        device="cpu",
        half=False,
        imgsz=None,
        conf=None,
        eval_box_type=None,
        dataset_detector_cfg=None,
    )

    monkeypatch.setattr(evaluator_module, "_load_benchmark_cfg", lambda _args: benchmark_bundle)
    monkeypatch.setattr(evaluator_module, "should_use_benchmark_detector", lambda _args, _cfg: True)
    monkeypatch.setattr(evaluator_module, "should_use_benchmark_reid", lambda _args, _cfg: True)
    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_detector_model",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("detector download should not be triggered")),
    )
    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_reid_model",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("reid download should not be triggered")),
    )

    _, _, runtime_cfg = evaluator_module._configure_benchmark_runtime(args)

    assert args.detector == [detector_path]
    assert args.reid == [reid_path]
    assert args.imgsz == detector_cfg["imgsz"]
    assert args.conf == detector_cfg["conf"]
    assert runtime_cfg["id"] == detector_cfg["id"]


def test_configure_benchmark_runtime_uses_explicit_component_configs_without_benchmark(monkeypatch):
    detector_cfg = load_detector_cfg("yolo11l_3ch.pt")
    args = SimpleNamespace(
        detector=[Path("models/yolo11l_3ch.pt")],
        reid=[Path("models/lmbn_n_duke.pt")],
        detector_explicit=True,
        reid_explicit=True,
        device="cuda:0",
        half=False,
        device_explicit=False,
        half_explicit=False,
        imgsz=None,
        conf=None,
        eval_box_type=None,
        dataset_detector_cfg=None,
    )

    monkeypatch.setattr(evaluator_module, "_load_benchmark_cfg", lambda _args: {})

    _, benchmark_cfg, runtime_cfg = evaluator_module._configure_benchmark_runtime(args)

    assert benchmark_cfg == {}
    assert args.reid_device == "cuda:0"
    assert args.reid_half is True
    assert args.imgsz == detector_cfg["imgsz"]
    assert args.conf == detector_cfg["conf"]
    assert args.eval_box_type == "obb"
    assert runtime_cfg["box_type"] == "obb"


def test_apply_class_remap_skips_obb_without_loading_aabb_mapping(monkeypatch):
    def fail_remap(*_args, **_kwargs):
        raise AssertionError("OBB eval must not run AABB class remapping")

    monkeypatch.setattr(evaluator_module, "build_gt_class_remap", fail_remap)
    args = SimpleNamespace(
        eval_box_type="obb",
        benchmark_id=None,
        dataset_id=None,
        benchmark="local-mmot",
        data="/tmp/local-mmot.yaml",
    )

    evaluator_module.apply_class_remap(args, {"id": "yolo11l_3ch"})

    assert not hasattr(args, "gt_class_remap")


def test_iter_source_expands_globs(tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "000001.jpg"
    import cv2

    cv2.imwrite(str(img_path), img)

    frames = list(iter_source(str(tmp_path / "*.jpg")))

    assert len(frames) == 1
    assert frames[0][0] == str(img_path)
    assert frames[0][1].shape == img.shape


def test_detection_cache_helpers_support_npy(tmp_path):
    det_npy = tmp_path / "SEQ.npy"
    dets = np.array(
        [
            [1, 10, 20, 30, 40, 0.9, 0],
            [3, 11, 21, 31, 41, 0.8, 0],
        ],
        dtype=np.float32,
    )
    np.save(det_npy, dets)

    assert _existing_cache_path(det_npy) == det_npy
    assert _saved_detection_column_count(det_npy) == 7
    assert _max_frame_id(det_npy) == 3


def test_appendable_npy_writer_appends_rows_without_buffering_full_array(tmp_path):
    path = tmp_path / "stream.npy"
    writer = AppendableNpyWriter(
        path,
        dtype=np.float32,
        trailing_shape=(3,),
        empty_trailing_shape=(3,),
    )

    writer.append(np.array([[1, 2, 3]], dtype=np.float32))
    writer.append(np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float32))
    writer.close()

    np.testing.assert_allclose(
        np.load(path),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
    )


def test_appendable_npy_writer_can_resume_existing_file(tmp_path):
    path = tmp_path / "resume.npy"
    writer = AppendableNpyWriter(
        path,
        dtype=np.float32,
        trailing_shape=(2,),
        empty_trailing_shape=(2,),
    )
    writer.append(np.array([[1, 2]], dtype=np.float32))
    writer.close()

    resumed = AppendableNpyWriter(
        path,
        dtype=np.float32,
        trailing_shape=(2,),
        empty_trailing_shape=(2,),
    )
    resumed.append(np.array([[3, 4], [5, 6]], dtype=np.float32))
    resumed.close()

    np.testing.assert_allclose(
        np.load(path),
        np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32),
    )


def test_appendable_npy_writer_creates_empty_file_for_lazy_writer(tmp_path):
    path = tmp_path / "empty.npy"
    writer = AppendableNpyWriter(
        path,
        dtype=np.float32,
        trailing_shape=None,
        empty_trailing_shape=(0,),
    )
    writer.close()

    arr = np.load(path)
    assert arr.shape == (0, 0)
    assert arr.dtype == np.float32


def test_iter_source_preserves_stream_urls(monkeypatch):
    seen = {}

    class _FakeCapture:
        def __init__(self, value):
            seen["source"] = value

        def isOpened(self):
            return False

        def release(self):
            seen["released"] = True

    monkeypatch.setattr(loaders_module.cv2, "VideoCapture", _FakeCapture)

    frames = list(iter_source("rtsp://camera/stream"))

    assert frames == []
    assert seen["source"] == "rtsp://camera/stream"
    assert seen["released"] is True


def test_iter_source_reads_line_based_source_lists(tmp_path):
    import cv2

    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "000001.jpg"
    cv2.imwrite(str(img_path), img)

    manifest = tmp_path / "list.txt"
    manifest.write_text("000001.jpg\n", encoding="utf-8")

    frames = list(iter_source(manifest))

    assert len(frames) == 1
    assert frames[0][0] == str(img_path.resolve())
    assert frames[0][1].shape == img.shape


def test_detector_wrapper_process_uses_detector_backend(monkeypatch):
    captured = {}

    class _FakeDetector:
        def __init__(self, model, device, imgsz):
            captured["init"] = {"model": model, "device": str(device), "imgsz": imgsz}

        def __call__(self, images, conf, iou, classes, agnostic_nms):
            captured["call"] = {
                "n_images": len(images),
                "conf": conf,
                "iou": iou,
                "classes": classes,
                "agnostic_nms": agnostic_nms,
            }
            return "ok"

    monkeypatch.setattr(detector_module, "get_detector_class", lambda _path: _FakeDetector)

    detector = detector_module.Detector("det.pt", device=torch.device("cpu"), imgsz=[64, 64])
    results = detector.process([_DUMMY_IMG], conf=0.25, iou=0.7, agnostic_nms=False, classes=[0])

    assert results == "ok"
    assert Path(captured["init"]["model"]) == Path("det.pt")
    assert captured["call"] == {
        "n_images": 1,
        "conf": 0.25,
        "iou": 0.7,
        "classes": [0],
        "agnostic_nms": False,
    }


def test_reid_runtime_returns_named_feature_sets(monkeypatch):
    class _FakeReIDModel:
        def get_features(self, xyxys, img):
            return np.full((len(xyxys), 4), float(img.shape[0]), dtype=np.float32)

    class _FakeBackend:
        def __init__(self, weights, device, half, **kwargs):
            _ = (weights, device, half)
            self.model = _FakeReIDModel()

    class _FakeDetector:
        def __init__(self, model, device, imgsz):
            _ = (device, imgsz)
            self.model = model

        def __call__(self, images, conf, iou, classes, agnostic_nms):
            _ = (conf, iou, classes, agnostic_nms)
            return [Detections(dets=np.empty((0, 6), dtype=np.float32), orig_img=images[0], path="")]

    monkeypatch.setattr(pipeline_module, "get_detector_class", lambda _path: _FakeDetector)
    monkeypatch.setattr(reid_core_module, "ReID", _FakeBackend)

    pipeline = pipeline_module.DetectorReIDPipeline(
        "det.pt",
        reid_paths=[Path("alpha.pt"), Path("beta.pt")],
        device="cpu",
        imgsz=[64, 64],
    )
    features = pipeline.get_all_reid_features(np.array([[0, 0, 1, 1]], dtype=np.float32), _DUMMY_IMG)

    assert set(features) == {"alpha.pt", "beta.pt"}
    np.testing.assert_array_equal(features["alpha.pt"], np.full((1, 4), 64.0, dtype=np.float32))


def test_pipeline_delegates_to_detector_backend_and_reid_models(monkeypatch):
    detector_calls = []
    reid_calls = []

    class _FakeDetector:
        def __init__(self, model, device, imgsz):
            detector_calls.append(("init", Path(model), str(device), imgsz))

        def __call__(self, images, conf, iou, classes, agnostic_nms):
            detector_calls.append((len(images), conf, iou, agnostic_nms, classes))
            return [Detections(dets=np.empty((0, 6), dtype=np.float32), orig_img=image, path="") for image in images]

    class _FakeReIDModel:
        def get_features(self, xyxys, img):
            reid_calls.append((len(xyxys), img.shape[0]))
            return np.ones((len(xyxys), 2), dtype=np.float32)

    class _FakeBackend:
        def __init__(self, weights, device, half, **kwargs):
            _ = (weights, device, half)
            self.model = _FakeReIDModel()

    monkeypatch.setattr(pipeline_module, "get_detector_class", lambda _path: _FakeDetector)
    monkeypatch.setattr(pipeline_module, "select_device", lambda device: device)
    monkeypatch.setattr(pipeline_module, "iter_source", lambda source, vid_stride=1: iter([("frame.jpg", _DUMMY_IMG)]))
    monkeypatch.setattr(reid_core_module, "ReID", _FakeBackend)

    pipeline = pipeline_module.DetectorReIDPipeline("det.pt", reid_paths=["alpha.pt"], device="cuda:0", imgsz=[64, 64])
    batch_results = pipeline.predict_batch([_DUMMY_IMG], conf=0.25, iou=0.7, agnostic_nms=False, classes=None)
    assert len(batch_results) == 1
    pipeline.warmup()
    assert pipeline.autotune_batch_size(8) == 8
    assert list(pipeline.predict("video.mp4", conf=0.25, iou=0.7, agnostic_nms=False, classes=None))
    np.testing.assert_array_equal(
        pipeline.get_reid_features(np.array([[0, 0, 1, 1]], dtype=np.float32), _DUMMY_IMG),
        np.ones((1, 2), dtype=np.float32),
    )
    assert set(pipeline.get_all_reid_features(np.array([[0, 0, 1, 1]], dtype=np.float32), _DUMMY_IMG)) == {"alpha.pt"}
    assert detector_calls[0] == ("init", Path("det.pt"), "cuda:0", [64, 64])
    assert detector_calls[1:5] == [
        (1, 0.25, 0.7, False, None),
        (1, 0.25, 0.7, False, None),
        (8, 0.25, 0.7, False, None),
        (1, 0.25, 0.7, False, None),
    ]
    assert reid_calls == [(1, 64), (1, 64)]


def test_pipeline_records_detector_and_reid_phase_timings(monkeypatch):
    class _FakeDetector:
        def __init__(self, model, device, imgsz):
            _ = (model, device, imgsz)

        def preprocess(self, images):
            return [image + 1 for image in images]

        def process(self, preprocessed, conf, iou, classes, agnostic_nms):
            _ = (conf, iou, classes, agnostic_nms)
            return preprocessed

        def postprocess(self, detections):
            return [
                Detections(dets=np.empty((0, 6), dtype=np.float32), orig_img=image, path="") for image in detections
            ]

    class _FakeReIDModel:
        def get_crops(self, xyxys, img):
            _ = (xyxys, img)
            return [np.ones((4, 4, 3), dtype=np.uint8)]

        def inference_preprocess(self, crops):
            return np.stack(crops).astype(np.float32)

        def forward(self, crops):
            _ = crops
            return np.array([[3.0, 4.0]], dtype=np.float32)

        def inference_postprocess(self, features):
            return features

    class _FakeBackend:
        def __init__(self, weights, device, half, **kwargs):
            _ = (weights, device, half, kwargs)
            self.model = _FakeReIDModel()

    monkeypatch.setattr(pipeline_module, "get_detector_class", lambda _path: _FakeDetector)
    monkeypatch.setattr(pipeline_module, "select_device", lambda device: device)
    monkeypatch.setattr(reid_core_module, "ReID", _FakeBackend)

    pipeline = pipeline_module.DetectorReIDPipeline("det.pt", reid_paths=["alpha.pt"], device="cpu", imgsz=[64, 64])

    pipeline.predict_batch([_DUMMY_IMG], conf=0.25, iou=0.7, agnostic_nms=False, classes=None)
    features = pipeline.get_reid_features(np.array([[0, 0, 1, 1]], dtype=np.float32), _DUMMY_IMG)

    assert features.shape == (1, 2)
    assert pipeline.timing_stats.totals["detector_preprocess"] > 0.0
    assert pipeline.timing_stats.totals["detector_process"] > 0.0
    assert pipeline.timing_stats.totals["detector_postprocess"] > 0.0
    assert pipeline.timing_stats.totals["reid_preprocess"] > 0.0
    assert pipeline.timing_stats.totals["reid_process"] > 0.0
    assert pipeline.timing_stats.totals["reid_postprocess"] > 0.0
    assert pipeline.timing_stats.totals["preprocess"] == pytest.approx(
        pipeline.timing_stats.totals["detector_preprocess"],
        abs=1e-6,
    )
    assert pipeline.timing_stats.totals["inference"] == pytest.approx(
        pipeline.timing_stats.totals["detector_process"],
        abs=1e-6,
    )
    assert pipeline.timing_stats.totals["postprocess"] == pytest.approx(
        pipeline.timing_stats.totals["detector_postprocess"],
        abs=1e-6,
    )
    assert pipeline.timing_stats.totals["reid"] == pytest.approx(
        pipeline.timing_stats.totals["reid_preprocess"]
        + pipeline.timing_stats.totals["reid_process"]
        + pipeline.timing_stats.totals["reid_postprocess"],
        rel=1e-6,
    )


def test_aabb_text_output_uses_conf_class_det_ind_columns(tmp_path):
    tracks = np.array([[10, 20, 30, 45, 7, 0.85, 3, 11]], dtype=np.float32)

    mot_rows = convert_to_mot_format(tracks, frame_idx=5)
    write_mot_results(tmp_path / "tracks.txt", mot_rows)

    np.testing.assert_array_equal(
        mot_rows[0, :6],
        np.array([5, 7, 10, 20, 20, 25], dtype=np.float32),
    )
    assert np.isclose(mot_rows[0, 6], 0.85)
    assert mot_rows[0, 7] == 4
    assert mot_rows[0, 8] == 11
    assert (tmp_path / "tracks.txt").read_text().strip() == "5,7,10,20,20,25,0.850000,4,11"


def test_tracker_infers_obb_mode_on_empty_followup_frame():
    tracker = _DummyOBBTracker(asso_func="iou")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    obb_dets = np.array([[32, 32, 20, 10, 0.1, 0.9, 0]], dtype=np.float32)

    tracker.update(obb_dets, image)
    tracker.update(np.empty((0, 7), dtype=np.float32), image)

    assert tracker.is_obb is True
    assert tracker.detection_layout is OBB_DETECTIONS
    assert tracker.asso_func_name == "iou_obb"


def test_tracker_layout_helpers_switch_with_detection_mode():
    tracker = _DummyOBBTracker(asso_func="iou")

    tracker._set_detection_mode(False)
    assert tracker.detection_layout is AABB_DETECTIONS
    assert tracker.empty_detections().shape == (0, 6)
    assert tracker.empty_output().shape == (0, 8)

    tracker._set_detection_mode(True)
    assert tracker.detection_layout is OBB_DETECTIONS
    assert tracker.empty_detections().shape == (0, 7)
    assert tracker.empty_output().shape == (0, 9)


def test_tracker_rejects_obb_when_not_supported():
    tracker = _DummyTracker(asso_func="iou")
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    obb_dets = np.array([[32, 32, 20, 10, 0.1, 0.9, 0]], dtype=np.float32)

    try:
        tracker.update(obb_dets, image)
    except AssertionError as exc:
        assert str(exc) == (
            "_DummyTracker does not support OBB detections. "
            "Use an OBB-capable tracker such as ByteTrack, BotSort, OCSort, or SFSORT."
        )
    else:
        raise AssertionError("Expected unsupported OBB trackers to fail fast")


def test_ocsort_obb_state_roundtrip_handles_column_vectors():
    obb = np.array([32, 24, 20, 10, 0.25], dtype=np.float32)

    state = xywha_to_xysr_measurement(obb).reshape((5, 1))
    decoded = xysr_state_to_xywha(state)

    assert decoded.shape == (1, 5)
    np.testing.assert_allclose(decoded[0], obb, rtol=1e-6, atol=1e-6)


def test_iou_obb_pair_accepts_column_like_inputs_and_radians():
    dets = np.array([[32, 24, 20, 10, 0.25, 0.9]], dtype=object)
    trks = np.array([[32, 24, 20, 10, 0.25, 0.8]], dtype=object)

    iou = iou_obb_pair(0, 0, dets, trks)

    assert iou > 0.99


def test_ensure_model_extension_preserves_explicit_export_paths():
    model_path = "models/osnet_x0_25_msmt17_saved_model/osnet_x0_25_msmt17_float32.tflite"

    resolved = ensure_model_extension(model_path)

    assert resolved == Path(model_path)


def test_ensure_model_extension_keeps_bare_reid_names_in_weights_dir():
    resolved = ensure_model_extension("osnet_x0_25_msmt17")

    assert resolved == WEIGHTS / "osnet_x0_25_msmt17.pt"


def test_parse_mot_results_preserves_multiword_class_names():
    # Legacy fixed-width format: name column %-35s, each value %-10s
    results = (
        "\nHOTA: tracker-storage tank HOTA      DetA      AssA      DetRe     DetPr     "
        "AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)\n"
        "COMBINED                           51.0      52.0      61.0      53.0      "
        "54.0      71.0      72.0      73.0      74.0      75.0      76.0      77.0      \n"
        "CLEAR: tracker-storage tank MOTA      MOTP      MODA      CLR_Re    CLR_Pr    "
        "MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      "
        "MT        PT        ML        Frag      \n"
        "COMBINED                           41.0      42.0      43.0      44.0      "
        "45.0      46.0      47.0      48.0      52.0      49        50        51        "
        "3         4         5         6         7         \n"
        "Identity: tracker-storage tank IDF1      IDR       IDP       IDTP      IDFN      IDFP      \n"
        "COMBINED                           31.0      32.0      33.0      34        35        36        \n"
        "Count: tracker-storage tank Dets      GT_Dets   IDs       GT_IDs    \n"
        "COMBINED                           80        81        7         82        \n"
    )

    parsed = evaluator_module.parse_mot_results(results, known_classes=["storage tank"])

    assert list(parsed) == ["storage tank"]
    assert parsed["storage tank"]["HOTA"] == 51.0
    assert parsed["storage tank"]["DetA"] == 52.0
    assert parsed["storage tank"]["AssA"] == 61.0
    assert parsed["storage tank"]["AssRe"] == 71.0
    assert parsed["storage tank"]["LocA"] == 73.0
    assert parsed["storage tank"]["HOTA(0)"] == 75.0
    assert parsed["storage tank"]["MOTA"] == 41.0
    assert parsed["storage tank"]["MOTP"] == 42.0
    assert parsed["storage tank"]["sMOTA"] == 52.0
    assert parsed["storage tank"]["CLR_TP"] == 49
    assert parsed["storage tank"]["Frag"] == 7
    assert parsed["storage tank"]["IDSW"] == 3
    assert parsed["storage tank"]["IDF1"] == 31.0
    assert parsed["storage tank"]["IDR"] == 32.0
    assert parsed["storage tank"]["IDTP"] == 34
    assert parsed["storage tank"]["Dets"] == 80
    assert parsed["storage tank"]["GT_Dets"] == 81
    assert parsed["storage tank"]["IDs"] == 7
    assert parsed["storage tank"]["GT_IDs"] == 82


def test_build_mot_feedback_keeps_summary_and_per_sequence_metrics():
    results_module = importlib.import_module("boxmot.engine.eval.results")
    raw = {
        "all": {
            "HOTA": 62.5,
            "MOTA": 70.0,
            "IDF1": 65.0,
            "DetA": 60.0,
            "CLR_TP": 123,
            "per_sequence": {
                "MOT17-02": {"HOTA": 61.0, "MOTA": 69.0, "IDSW": 4, "CLR_TP": 50},
            },
        },
        "person": {
            "HOTA": 58.0,
            "MOTA": 66.0,
            "IDF1": 60.0,
            "CLR_TP": 100,
            "per_sequence": {
                "MOT17-02": {"HOTA": 57.0, "MOTA": 65.0, "IDSW": 5},
            },
        },
    }

    feedback = results_module.build_mot_feedback(raw)

    assert feedback["summary_label"] == "all"
    assert feedback["summary"]["HOTA"] == 62.5
    assert feedback["summary"]["CLR_TP"] == 123
    assert feedback["per_sequence_metrics"]["MOT17-02"]["IDSW"] == 4
    assert feedback["per_class_metrics"]["all"]["MOTA"] == 70.0
    assert feedback["per_class_metrics"]["person"]["CLR_TP"] == 100


def test_eval_runtime_metrics_are_generated_by_motmetrics_module():
    forbidden_module = "track" + "eval"

    assert evaluator_module._LAZY_EXPORTS["motmetrics_runner"] == (
        "boxmot.engine.eval.motmetrics",
        "run_motmetrics",
    )
    assert all(forbidden_module not in module.lower() for module, _ in evaluator_module._LAZY_EXPORTS.values())
    assert not (Path(evaluator_module.__file__).parent / forbidden_module).exists()


def test_ordered_benchmark_eval_class_names_preserve_multiword_names():
    bench_cfg = {"eval_classes": {"1": "storage tank", "2": "ground track field"}}

    class_names = evaluator_module._ordered_benchmark_eval_class_names(bench_cfg)

    assert class_names == ["storage tank", "ground track field"]


def test_select_plot_metrics_data_prefers_explicit_aggregate_rows():
    results = {
        "plane": {"HOTA": 11.0},
        "all": {"HOTA": 22.0},
    }

    plot_class, metrics = evaluator_module._select_plot_metrics_data(results)

    assert plot_class == "all"
    assert metrics == {"HOTA": 22.0}


def test_select_plot_metrics_data_skips_ambiguous_multiclass_rows():
    results = {
        "plane": {"HOTA": 11.0},
        "ship": {"HOTA": 22.0},
    }

    plot_class, metrics = evaluator_module._select_plot_metrics_data(results)

    assert plot_class == ""
    assert metrics == {}


def test_mmot_mini_obb_gt_uses_zero_based_eval_class_ids():
    expected = {0, 1, 2, 3, 6, 7}
    found = set()

    for path in sorted(Path("assets/mmot-mini/train/mot").glob("*.txt")):
        matrix = evaluator_module._load_obb_gt_matrix(path)
        found.update(matrix[:, 11].astype(int).tolist())

    assert found == expected


def test_load_obb_gt_matrix_rejects_legacy_xywha_format(tmp_path):
    gt_path = tmp_path / "gt_obb.txt"
    gt_path.write_text("1,2,10,20,30,40,0.5,1,3,0\n")

    try:
        evaluator_module._load_obb_gt_matrix(gt_path)
    except ValueError as exc:
        assert "expected 13 columns in corner format" in str(exc)
    else:
        raise AssertionError("Expected legacy xywha OBB GT to be rejected")


def test_run_generate_mot_results_quiet_mode_skips_manager_queue(tmp_path, monkeypatch):
    source = tmp_path / "train"
    for seq_name in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        img_dir = source / seq_name / "img1"
        img_dir.mkdir(parents=True)
        (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        tracker="boosttrack",
        fps=None,
        device="cpu",
        n_threads=2,
        tracking_backend="thread",
        postprocessing="none",
        conf=0.25,
    )
    queue_types = []
    executor_kwargs = {}
    manager_calls = []

    class FakeManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def Queue(self):
            progress_queue = queue.Queue()
            queue_types.append(type(progress_queue).__name__)
            return progress_queue

    class FakeSpawnContext:
        def Manager(self):
            manager_calls.append(True)
            return FakeManager()

    spawn_context = FakeSpawnContext()

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeThreadPoolExecutor:
        def __init__(self, *_args, **kwargs):
            executor_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, _func, *task_arg):
            seq_name = task_arg[0]
            progress_queue = task_arg[-2]
            assert progress_queue is None
            return FakeFuture((seq_name, [1], {"track_time_ms": 5.0, "num_frames": 1}))

    monkeypatch.setattr(cached_tracking_module.mp, "get_context", lambda method: spawn_context)
    monkeypatch.setattr(cached_tracking_module.concurrent.futures, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(
        cached_tracking_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (set(pending), set()),
    )

    cached_tracking_module.run_generate_mot_results(args, quiet=True)

    assert args.seq_frame_nums == {
        "MOT17-02-FRCNN": [1],
        "MOT17-04-FRCNN": [1],
    }
    assert manager_calls == []
    assert queue_types == []
    assert executor_kwargs["max_workers"] == 2


def test_evaluator_dependency_check_is_lazy(monkeypatch):
    calls = []

    from boxmot.utils import checks as checks_module

    monkeypatch.setattr(
        checks_module.RequirementsChecker,
        "check_packages",
        lambda self, packages: calls.append(tuple(packages)),
    )

    importlib.reload(evaluator_module)

    assert calls == []

    evaluator_module._ensure_eval_dependencies()

    assert calls == [("ultralytics",)]


@pytest.mark.parametrize(
    ("single_class_mode", "expected_include_sequences"),
    [(False, False), (True, True)],
)
def test_run_motmetrics_only_expands_sequences_for_single_class(
    monkeypatch, tmp_path, single_class_mode, expected_include_sequences
):
    source = tmp_path / "test"
    source.mkdir()
    parsed = {
        "car": {"HOTA": 70.0, "per_sequence": {"seq": {"HOTA": 69.0}}},
        "bike": {"HOTA": 50.0, "per_sequence": {"seq": {"HOTA": 49.0}}},
    }
    render_kwargs = {}

    def render_report(*args, **kwargs):
        render_kwargs.update(kwargs)
        return "report"

    dependencies = {
        "_collect_seq_info": lambda _: ([source / "seq"], {"seq": 1}),
        "filter_obb_mot_results": lambda results, args, cfg: (results, single_class_mode),
        "log_mot_report": lambda report: None,
        "render_mot_report": render_report,
        "motmetrics_runner": lambda *args, **kwargs: parsed,
    }
    monkeypatch.setattr(evaluator_module, "_get_lazy_export", dependencies.__getitem__)
    monkeypatch.setattr(evaluator_module, "_load_benchmark_cfg", lambda args: {"benchmark": {}})
    monkeypatch.setattr(evaluator_module, "_resolve_eval_box_type", lambda args, cfg=None: "obb")

    args = SimpleNamespace(
        source=source,
        project=tmp_path,
        benchmark="mmot-mini",
        name="run",
        tracker="botsort",
        ci=False,
    )
    evaluator_module.run_motmetrics(args, verbose=True)

    assert render_kwargs["include_sequences"] is expected_include_sequences


def test_evaluator_main_prints_validation_report_without_verbose(monkeypatch, tmp_path, capsys):
    result = evaluator_module.ValidationResult(
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
        exp_dir=tmp_path,
        timings={},
        args=SimpleNamespace(remapped_class_names=["person"], eval_box_type=None, classes=None),
    )
    calls = []

    def fake_run_eval(args, **kwargs):
        calls.append(kwargs)
        return result

    class _FakePlotter:
        def __init__(self, exp_dir):
            self.exp_dir = exp_dir

        def plot_radar_chart(self, *args, **kwargs):
            return None

    workflows = []

    class _FakeWorkflow:
        def __init__(self, title, fields, steps, stderr=False, transient=False):
            self.title = title
            self.fields = list(fields)
            self.steps = list(steps)
            self.details = []
            self.stderr = stderr
            self.transient = transient
            self.started = False
            self.stopped = False
            self.prefer_alt_screen = False
            self.prefer_compact_layout = False
            self._live = None
            self.detail_renderable = None
            self.detail_text = None
            self.detail_title = None

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

        def activate(self, label):
            self.steps = [
                (step_label, "active" if step_label == label else ("todo" if step_state == "active" else step_state))
                for step_label, step_state in self.steps
            ]

        def complete(self, label):
            self.steps = [
                (step_label, "done" if step_label == label else step_state) for step_label, step_state in self.steps
            ]

        def fail(self, label=None, error=None, *, render=True):
            return None

        def set_detail(self, title, text, *, render=True):
            self.details.append((title, text, render))

        def renderable(self, *, compact=False, include_setup=True):
            return ""

    def fake_create_workflow_progress(title, fields, *, steps=(), stderr=False, transient=False):
        workflow = _FakeWorkflow(title, fields, steps, stderr=stderr, transient=transient)
        workflows.append(workflow)
        return workflow

    monkeypatch.setattr(evaluator_module, "run_eval", fake_run_eval)
    monkeypatch.setattr(evaluator_module, "MetricsPlotter", _FakePlotter)
    monkeypatch.setattr(evaluator_module.ui, "create_workflow_progress", fake_create_workflow_progress)

    args = SimpleNamespace(
        detector=[tmp_path / "detector.pt"],
        reid=[tmp_path / "reid.pt"],
        tracker="bytetrack",
        data="mot17-mini",
        benchmark="mot17-mini",
        source=None,
        imgsz=None,
        show_timing=False,
        verbose=False,
    )

    evaluator_module.main(args)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow.title == "Evaluation"
    assert workflow.started is True
    assert workflow.stopped is True
    assert ("__panel__:Dataset", [("Benchmark", "mot17-mini")]) in workflow.fields
    assert (evaluator_module.EVAL_SETUP_STEP, "active") in workflow.steps
    assert workflow.details == []
    assert "pipeline" in calls[0]
    assert calls[0]["verbose"] is False


def test_run_eval_marks_workflow_steps_done(monkeypatch, tmp_path):
    actions = []

    class _FakeWorkflow:
        steps = [
            (evaluator_module.EVAL_SETUP_STEP, "active"),
            (evaluator_module.EVAL_GENERATE_STEP, "todo"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
            (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
        ]
        detail_renderable = None
        detail_text = None
        detail_title = None

        def activate(self, label, *, render=True):
            actions.append(("active", label))

        def complete(self, label, *, render=True):
            actions.append(("done", label))

        def set_detail(self, title, text, *, render=True):
            actions.append(("detail", title, text))

        def set_detail_renderable(self, title, renderable, *, render=True):
            actions.append(("detail", title, str(renderable)))

        def transition(self, done, next_step, detail=None):
            actions.append(("done", done))
            actions.append(("active", next_step))
            if detail:
                actions.append(("detail", next_step, detail))

    from boxmot.utils.rich.workflow.pipeline import PipelineTracker

    monkeypatch.setattr(evaluator_module, "_ensure_eval_dependencies", lambda: None)
    monkeypatch.setattr(evaluator_module, "_normalize_eval_models", lambda args: None)
    monkeypatch.setattr(
        evaluator_module,
        "eval_setup",
        lambda args, pipeline=None: None,
    )
    monkeypatch.setattr(
        evaluator_module,
        "run_generate_dets_embs",
        lambda args, timing_stats=None, progress_callback=None: progress_callback(
            "Generating detections and embeddings: 2/2 frames\n  MOT17-02-FRCNN ████████████████████  100%  (done)"
        )
        if progress_callback
        else None,
    )
    monkeypatch.setattr(
        evaluator_module,
        "run_generate_mot_results",
        lambda *args, progress_callback=None, **kwargs: progress_callback("Tracking: 1/1 sequences done")
        if progress_callback
        else None,
    )
    motmetrics_calls = []

    def fake_run_motmetrics(args, verbose=True):
        motmetrics_calls.append(verbose)
        return {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0}

    monkeypatch.setattr(evaluator_module, "run_motmetrics", fake_run_motmetrics)
    monkeypatch.setattr(
        evaluator_module, "extract_summary", lambda raw: ("all", {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0})
    )

    args = SimpleNamespace(
        detector=[tmp_path / "detector.pt"],
        reid=[tmp_path / "reid.pt"],
        benchmark="mot17-mini",
        data="mot17-mini",
        show_progress=True,
        verbose=False,
    )

    pipeline = PipelineTracker(_FakeWorkflow(), wire_status_fns=False)
    result = evaluator_module.run_eval(args, verbose=False, pipeline=pipeline)

    assert result.summary == {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0}
    assert motmetrics_calls == [False]
    # Pipeline.advance() calls workflow.transition(), which records (done, active, detail)
    assert actions[:11] == [
        ("done", evaluator_module.EVAL_SETUP_STEP),
        ("active", evaluator_module.EVAL_GENERATE_STEP),
        ("detail", evaluator_module.EVAL_GENERATE_STEP, "Generating detections & embeddings..."),
        (
            "detail",
            evaluator_module.EVAL_GENERATE_STEP,
            "Generating detections and embeddings: 2/2 frames\n  MOT17-02-FRCNN ████████████████████  100%  (done)",
        ),
        ("done", evaluator_module.EVAL_GENERATE_STEP),
        ("active", evaluator_module.EVAL_TRACK_STEP),
        ("detail", evaluator_module.EVAL_TRACK_STEP, "Starting tracker..."),
        ("detail", evaluator_module.EVAL_TRACK_STEP, "Tracking: 1/1 sequences done"),
        ("done", evaluator_module.EVAL_TRACK_STEP),
        ("active", evaluator_module.EVAL_EVALUATE_STEP),
        ("detail", evaluator_module.EVAL_EVALUATE_STEP, "Computing metrics..."),
    ]
    assert actions[11] == ("done", evaluator_module.EVAL_EVALUATE_STEP)


def test_run_eval_advances_from_postprocess_to_evaluate(monkeypatch, tmp_path):
    actions = []
    postprocess_step = "Postprocess tracks"

    class _FakeWorkflow:
        steps = [
            (evaluator_module.EVAL_SETUP_STEP, "active"),
            (evaluator_module.EVAL_GENERATE_STEP, "todo"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
            (postprocess_step, "todo"),
            (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
        ]
        detail_renderable = None
        detail_text = None
        detail_title = None

        def activate(self, label, *, render=True):
            actions.append(("active", label))

        def complete(self, label, *, render=True):
            actions.append(("done", label))

        def set_detail(self, title, text, *, render=True):
            actions.append(("detail", title, text))

        def set_detail_renderable(self, title, renderable, *, render=True):
            actions.append(("renderable", title))

        def transition(self, done, next_step, detail=None):
            actions.append(("done", done))
            actions.append(("active", next_step))
            if detail:
                actions.append(("detail", next_step, detail))

    from boxmot.utils.rich.workflow.pipeline import PipelineTracker

    monkeypatch.setattr(evaluator_module, "_ensure_eval_dependencies", lambda: None)
    monkeypatch.setattr(evaluator_module, "_normalize_eval_models", lambda args: None)
    monkeypatch.setattr(evaluator_module, "eval_setup", lambda args, pipeline=None: None)
    monkeypatch.setattr(evaluator_module, "run_generate_dets_embs", lambda *args, **kwargs: None)

    def fake_run_generate_mot_results(*args, progress_callback=None, postprocess_callback=None, **kwargs):
        if progress_callback:
            progress_callback("Tracking: 1/1 sequences done")
        if postprocess_callback:
            postprocess_callback("GBRC: 1/1 sequences done")

    monkeypatch.setattr(evaluator_module, "run_generate_mot_results", fake_run_generate_mot_results)
    monkeypatch.setattr(
        evaluator_module, "run_motmetrics", lambda args, verbose=True: {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0}
    )
    monkeypatch.setattr(
        evaluator_module, "extract_summary", lambda raw: ("all", {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0})
    )

    args = SimpleNamespace(
        detector=[tmp_path / "detector.pt"],
        reid=[tmp_path / "reid.pt"],
        benchmark="mot17-mini",
        data="mot17-mini",
        postprocessing="gbrc",
        show_progress=True,
        verbose=False,
    )

    pipeline = PipelineTracker(_FakeWorkflow(), wire_status_fns=False)
    evaluator_module.run_eval(args, verbose=False, pipeline=pipeline)

    assert ("done", evaluator_module.EVAL_TRACK_STEP) in actions
    assert ("active", postprocess_step) in actions
    assert ("detail", postprocess_step, "GBRC: 1/1 sequences done") in actions
    assert ("done", postprocess_step) in actions
    assert ("active", evaluator_module.EVAL_EVALUATE_STEP) in actions
    assert ("detail", evaluator_module.EVAL_EVALUATE_STEP, "Computing metrics...") in actions
    assert actions[-2:] == [
        ("done", evaluator_module.EVAL_EVALUATE_STEP),
        ("renderable", evaluator_module.EVAL_EVALUATE_STEP),
    ]


def test_run_eval_advances_through_tune_kf_step(monkeypatch, tmp_path):
    actions = []

    from boxmot.motion.kalman_filters import calibration as calibration_module
    from boxmot.utils.rich.workflow import steps as step_labels
    from boxmot.utils.rich.workflow.pipeline import PipelineTracker

    class _FakeWorkflow:
        steps = [
            (evaluator_module.EVAL_SETUP_STEP, "active"),
            (evaluator_module.EVAL_GENERATE_STEP, "todo"),
            (step_labels.TUNE_KF, "todo"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
            (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
        ]
        detail_renderable = None
        detail_text = None
        detail_title = None

        def activate(self, label, *, render=True):
            actions.append(("active", label))

        def complete(self, label, *, render=True):
            actions.append(("done", label))

        def set_detail(self, title, text, *, render=True):
            actions.append(("detail", title, text))

        def set_detail_renderable(self, title, renderable, *, render=True):
            actions.append(("renderable", title))

        def transition(self, done, next_step, detail=None):
            actions.append(("done", done))
            actions.append(("active", next_step))
            if detail:
                actions.append(("detail", next_step, detail))

    monkeypatch.setattr(evaluator_module, "_ensure_eval_dependencies", lambda: None)
    monkeypatch.setattr(evaluator_module, "_normalize_eval_models", lambda args: None)
    monkeypatch.setattr(evaluator_module, "eval_setup", lambda args, pipeline=None: None)
    monkeypatch.setattr(evaluator_module, "run_generate_dets_embs", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluator_module, "run_generate_mot_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(calibration_module, "tracker_kf_type", lambda tracker: "xywh")
    monkeypatch.setattr(
        calibration_module,
        "run_kf_tuning",
        lambda *args, **kwargs: ({"std_weight_position": 0.1, "std_weight_velocity": 0.01}, None),
    )
    monkeypatch.setattr(
        evaluator_module, "run_motmetrics", lambda args, verbose=True: {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0}
    )
    monkeypatch.setattr(
        evaluator_module, "extract_summary", lambda raw: ("all", {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0})
    )

    args = SimpleNamespace(
        detector=[tmp_path / "detector.pt"],
        reid=[tmp_path / "reid.pt"],
        benchmark="mot17-mini",
        data="mot17-mini",
        tracker="botsort",
        tune_kf=True,
        show_progress=True,
        verbose=False,
    )

    pipeline = PipelineTracker(_FakeWorkflow(), wire_status_fns=False)
    evaluator_module.run_eval(args, verbose=False, pipeline=pipeline)

    assert [label for action, label, *detail in actions if action == "active"] == [
        evaluator_module.EVAL_GENERATE_STEP,
        step_labels.TUNE_KF,
        evaluator_module.EVAL_TRACK_STEP,
        evaluator_module.EVAL_EVALUATE_STEP,
    ]
    assert ("detail", step_labels.TUNE_KF, "Calibrating Kalman filter...") in actions
    assert ("detail", evaluator_module.EVAL_TRACK_STEP, "Starting tracker...") in actions
    assert actions[-2:] == [
        ("done", evaluator_module.EVAL_EVALUATE_STEP),
        ("renderable", evaluator_module.EVAL_EVALUATE_STEP),
    ]


def test_run_eval_suppresses_inner_logs_when_workflow_is_active(monkeypatch, tmp_path):
    suppress_calls = []

    def fake_suppress(enabled, level="WARNING"):
        suppress_calls.append((enabled, level))
        return nullcontext()

    class _FakeWorkflow:
        steps = [
            (evaluator_module.EVAL_SETUP_STEP, "active"),
            (evaluator_module.EVAL_GENERATE_STEP, "todo"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
            (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
        ]
        detail_renderable = None
        detail_text = None
        detail_title = None

        def activate(self, label, *, render=True):
            return None

        def complete(self, label, *, render=True):
            return None

        def set_detail(self, title, text, *, render=True):
            return None

        def set_detail_renderable(self, title, renderable, *, render=True):
            return None

        def transition(self, done, next_step, detail=None):
            return None

    from boxmot.utils.rich.workflow.pipeline import PipelineTracker

    monkeypatch.setattr(evaluator_module, "suppress_boxmot_logs", fake_suppress)
    monkeypatch.setattr(evaluator_module, "_ensure_eval_dependencies", lambda: None)
    monkeypatch.setattr(evaluator_module, "_normalize_eval_models", lambda args: None)
    monkeypatch.setattr(evaluator_module, "eval_setup", lambda args, pipeline=None: None)
    monkeypatch.setattr(
        evaluator_module,
        "run_generate_dets_embs",
        lambda args, timing_stats=None, progress_callback=None: None,
    )
    monkeypatch.setattr(evaluator_module, "run_generate_mot_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluator_module, "run_motmetrics", lambda args, verbose=True: {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0}
    )
    monkeypatch.setattr(
        evaluator_module, "extract_summary", lambda raw: ("all", {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0})
    )

    args = SimpleNamespace(
        detector=[tmp_path / "detector.pt"],
        reid=[tmp_path / "reid.pt"],
        benchmark="mot17-mini",
        data="mot17-mini",
        show_progress=True,
        verbose=False,
    )

    pipeline = PipelineTracker(_FakeWorkflow(), wire_status_fns=False)
    evaluator_module.run_eval(args, verbose=False, pipeline=pipeline)

    assert suppress_calls == [(True, "WARNING"), (True, "WARNING")]


def test_build_eval_workflow_fields_reports_effective_cpp_backend(tmp_path):
    args = SimpleNamespace(
        detector=[Path("models/yolox_x_MOT17_ablation.pt")],
        reid=[Path("models/lmbn_n_duke.pt")],
        tracker="botsort",
        tracker_backend=None,
        tracking_backend="cpp",
        data="mot17-mini",
        benchmark="",
        dataset_id="",
        benchmark_id="",
        source=None,
        imgsz=[800, 1440],
        device="cpu",
        half=False,
        conf=0.25,
        n_threads=2,
        postprocessing="none",
        track_high_thresh=0.6,
        track_low_thresh=0.1,
        new_track_thresh=0.7,
        track_buffer=30,
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        cmc_method="ecc",
        det_thresh=None,
        with_reid=True,
        split="ablation",
    )

    fields = dict(evaluator_module._build_eval_workflow_fields(args))

    # Tracker subsystem card
    tracker_items = fields["__panel__:Tracker"]
    assert ("Name", "botsort") in tracker_items
    assert ("Backend", "cpp") in tracker_items
    assert ("CMC", "ecc") in tracker_items
    assert ("ReID", "✓") in tracker_items
    assert ("New track", "0.70") in tracker_items

    # Detector subsystem card
    detector_items = fields["__panel__:Detector"]
    assert ("Model", "yolox_x_MOT17_ablation") in detector_items
    assert ("Size", "800×1440") in detector_items
    assert ("Conf", "≥ 0.25") in detector_items

    # ReID subsystem card
    assert ("Model", "lmbn_n_duke") in fields["__panel__:ReID"]

    # Dataset card
    assert ("Benchmark", "mot17-mini") in fields["__panel__:Dataset"]
    assert ("Split", "ablation") in fields["__panel__:Dataset"]

    # Runtime card — no replay since tracking_backend is cpp (shown in Tracker)
    runtime_items = fields["__panel__:Runtime"]
    assert ("Device", "cpu") in runtime_items
    assert ("Precision", "fp32") in runtime_items
    assert ("Threads", 2) in runtime_items
    assert all(label != "Replay" for label, _ in runtime_items)


def test_run_eval_refreshes_workflow_fields_after_setup(monkeypatch, tmp_path):
    refreshed_fields = []

    class _FakeWorkflow:
        steps = [
            (evaluator_module.EVAL_SETUP_STEP, "active"),
            (evaluator_module.EVAL_GENERATE_STEP, "todo"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
            (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
        ]
        detail_renderable = None
        detail_text = None
        detail_title = None

        def set_fields(self, fields, *, render=True):
            refreshed_fields.append(list(fields))

        def activate(self, label, *, render=True):
            return None

        def complete(self, label, *, render=True):
            return None

        def set_detail(self, title, text, *, render=True):
            return None

        def set_detail_renderable(self, title, renderable, *, render=True):
            return None

        def transition(self, done, next_step, detail=None):
            return None

    from boxmot.utils.rich.workflow.pipeline import PipelineTracker

    monkeypatch.setattr(evaluator_module, "_ensure_eval_dependencies", lambda: None)
    monkeypatch.setattr(evaluator_module, "_normalize_eval_models", lambda args: None)

    def fake_eval_setup(args, pipeline=None):
        args.detector = [tmp_path / "yolox_x.pt"]
        args.reid = [tmp_path / "lmbn_n_duke.pt"]
        args.benchmark = "mot17-mini"
        args.tracker_backend = "cpp"

    monkeypatch.setattr(evaluator_module, "eval_setup", fake_eval_setup)
    monkeypatch.setattr(
        evaluator_module, "run_generate_dets_embs", lambda args, timing_stats=None, progress_callback=None: None
    )
    monkeypatch.setattr(evaluator_module, "run_generate_mot_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluator_module, "run_motmetrics", lambda args, verbose=True: {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0}
    )
    monkeypatch.setattr(
        evaluator_module, "extract_summary", lambda raw: ("all", {"HOTA": 1.0, "MOTA": 2.0, "IDF1": 3.0})
    )

    args = SimpleNamespace(
        detector=[tmp_path / "detector.pt"],
        reid=[tmp_path / "reid.pt"],
        tracker="botsort",
        tracker_backend=None,
        tracking_backend="process",
        benchmark="",
        data="mot17-mini",
        source=None,
        imgsz=None,
        show_progress=False,
        verbose=False,
    )

    pipeline = PipelineTracker(_FakeWorkflow(), wire_status_fns=False)
    evaluator_module.run_eval(args, verbose=False, pipeline=pipeline)

    assert refreshed_fields
    refreshed = dict(refreshed_fields[-1])
    # New card-based structure: check subsystem panels
    tracker_items = dict(refreshed["__panel__:Tracker"])
    assert tracker_items["Name"] == "botsort"
    assert tracker_items["Backend"] == "cpp"
    detector_items = dict(refreshed["__panel__:Detector"])
    assert detector_items["Model"] == "yolox_x"
    reid_items = dict(refreshed["__panel__:ReID"])
    assert reid_items["Model"] == "lmbn_n_duke"
    dataset_items = dict(refreshed["__panel__:Dataset"])
    assert dataset_items["Benchmark"] == "mot17-mini"


def test_workflow_progress_renders_single_stateful_block(monkeypatch):
    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        width=120,
        stderr=True,
        highlight=False,
        soft_wrap=False,
        theme=ui_module.BOXMOT_THEME,
        no_color=False,
    )

    monkeypatch.setattr(ui_module, "get_console", lambda *, stderr=False: console)

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "bytetrack")],
        steps=(
            (evaluator_module.EVAL_GENERATE_STEP, "active"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
            (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
        ),
        stderr=True,
    )

    workflow.start()
    workflow.set_detail(evaluator_module.EVAL_GENERATE_STEP, "Setting up evaluation data...")
    workflow.complete(evaluator_module.EVAL_GENERATE_STEP, render=False)
    workflow.activate(evaluator_module.EVAL_TRACK_STEP, render=False)
    workflow.set_detail(evaluator_module.EVAL_TRACK_STEP, "Tracking: 1/1 sequences done")
    workflow.complete(evaluator_module.EVAL_TRACK_STEP, render=False)
    workflow.activate(evaluator_module.EVAL_EVALUATE_STEP, render=False)
    workflow.set_detail(
        evaluator_module.EVAL_EVALUATE_STEP,
        "📊 RESULTS SUMMARY\nCOMBINED (person) 50.00 45.00 40.00",
        render=False,
    )
    workflow.complete(evaluator_module.EVAL_EVALUATE_STEP, render=False)
    workflow.stop()

    rendered = buffer.getvalue()

    assert "[✓] Generate / Track / Evaluate" in rendered
    assert "[x] Evaluate results" not in rendered
    assert "[>] Generate detections and embeddings" not in rendered
    assert "Setting up evaluation data..." not in rendered
    assert "Tracking: 1/1 sequences done" not in rendered
    assert "📊 RESULTS SUMMARY" in rendered
    assert "COMBINED (person) 50.00 45.00 40.00" in rendered


def test_workflow_progress_preserves_rich_styles_on_terminal(monkeypatch):
    buffer = StringIO()
    base_console = Console(
        file=buffer,
        force_terminal=True,
        color_system="standard",
        width=100,
        stderr=True,
        highlight=False,
        soft_wrap=False,
        theme=ui_module.BOXMOT_THEME,
        no_color=False,
    )

    monkeypatch.setattr(
        ui_module,
        "get_console",
        lambda *, stderr=False: base_console,
    )

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "bytetrack")],
        steps=((evaluator_module.EVAL_EVALUATE_STEP, "active"),),
        stderr=True,
    )

    workflow.start()
    workflow.set_detail(
        evaluator_module.EVAL_EVALUATE_STEP,
        "\x1b[1;33mCOMBINED (person)\x1b[0m 50.00 45.00 40.00",
    )
    workflow.stop()

    rendered = buffer.getvalue()

    assert "\x1b[36m" in rendered or "\x1b[1;36m" in rendered
    assert "\x1b[1;33mCOMBINED (person)\x1b[0m" in rendered
    assert "\x1b[?25l" in rendered
    assert "\x1b[?25h" in rendered
    assert re.search(r"\x1b\[\d+F\x1b\[J", rendered) is None


def test_workflow_progress_uses_full_live_overflow(monkeypatch):
    captured = {}

    class _FakeLive:
        def __init__(self, renderable, **kwargs):
            captured["renderable"] = renderable
            captured["vertical_overflow"] = kwargs.get("vertical_overflow")

        def start(self, *, refresh=True):
            captured["start_refresh"] = refresh

        def update(self, renderable, *, refresh=True):
            captured["updated"] = (renderable, refresh)

        def stop(self):
            captured["stopped"] = True

    monkeypatch.setattr(ui_module, "Live", _FakeLive)

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "bytetrack")],
        steps=((evaluator_module.EVAL_EVALUATE_STEP, "active"),),
        stderr=True,
    )

    workflow.start()
    workflow.stop()

    assert captured["vertical_overflow"] == "visible"
    assert captured["start_refresh"] is True
    assert captured["stopped"] is True


def test_workflow_progress_uses_alt_screen_when_oversized(monkeypatch):
    init_calls: list[dict] = []
    update_calls: list[tuple[object, bool]] = []
    stop_calls: list[bool] = []

    class _FakeLive:
        def __init__(self, renderable, **kwargs):
            init_calls.append(kwargs)
            self.renderable = renderable

        def start(self, *, refresh=True):
            return None

        def update(self, renderable, *, refresh=True):
            update_calls.append((renderable, refresh))

        def stop(self):
            stop_calls.append(True)

    monkeypatch.setattr(ui_module, "Live", _FakeLive)

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "hybridsort")],
        steps=(
            (evaluator_module.EVAL_GENERATE_STEP, "active"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
        ),
        stderr=True,
    )

    monkeypatch.setattr(
        ui_module.WorkflowProgress,
        "_renderable_exceeds_console",
        lambda self, renderable: True,
    )

    workflow.start()
    workflow.set_detail(evaluator_module.EVAL_GENERATE_STEP, "step 1")
    workflow.set_detail(evaluator_module.EVAL_GENERATE_STEP, "step 2")
    workflow.stop()

    # Live must be created on the alternate screen so in-place refreshes
    # don't pollute scrollback. Mid-flight updates ARE allowed there because
    # the alt buffer doesn't scroll the regular history.
    assert init_calls[0].get("screen") is True
    assert init_calls[0].get("vertical_overflow") == "visible"
    assert len(update_calls) >= 2
    assert stop_calls == [True]


def test_workflow_progress_prefers_alt_screen_from_first_render(monkeypatch):
    init_calls: list[dict] = []

    class _FakeLive:
        def __init__(self, renderable, **kwargs):
            init_calls.append(kwargs)

        def start(self, *, refresh=True):
            return None

        def update(self, renderable, *, refresh=True):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(ui_module, "Live", _FakeLive)

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "deepocsort")],
        steps=((evaluator_module.EVAL_TRACK_STEP, "active"),),
        stderr=True,
    )
    workflow.prefer_alt_screen = True

    monkeypatch.setattr(
        ui_module.WorkflowProgress,
        "_renderable_exceeds_console",
        lambda self, renderable: False,
    )

    workflow.start()

    assert init_calls[0].get("screen") is True
    assert init_calls[0].get("vertical_overflow") == "visible"


def test_workflow_progress_stays_on_normal_screen_when_compact_live_fits(monkeypatch):
    init_calls: list[dict] = []

    class _FakeLive:
        def __init__(self, renderable, **kwargs):
            init_calls.append(kwargs)

        def start(self, *, refresh=True):
            return None

        def update(self, renderable, *, refresh=True):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(ui_module, "Live", _FakeLive)

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "hybridsort")],
        steps=((evaluator_module.EVAL_TRACK_STEP, "active"),),
        stderr=True,
    )

    full_renderable = object()
    compact_renderable = object()

    workflow.renderable = lambda *, compact=False: compact_renderable if compact else full_renderable
    workflow._live_renderable = lambda: compact_renderable

    monkeypatch.setattr(
        ui_module.WorkflowProgress,
        "_renderable_exceeds_console",
        lambda self, renderable: renderable is full_renderable,
    )

    workflow.start()

    assert init_calls[0].get("screen") is False
    assert init_calls[0].get("vertical_overflow") == "visible"


def test_workflow_progress_keeps_compact_layout_for_final_render(monkeypatch):
    update_calls: list[tuple[object, bool]] = []

    class _FakeLive:
        def __init__(self, renderable, **kwargs):
            self.renderable = renderable

        def start(self, *, refresh=True):
            return None

        def update(self, renderable, *, refresh=True):
            update_calls.append((renderable, refresh))

        def stop(self):
            return None

    monkeypatch.setattr(ui_module, "Live", _FakeLive)

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "hybridsort")],
        steps=((evaluator_module.EVAL_TRACK_STEP, "active"),),
        stderr=True,
    )

    full_renderable = object()
    compact_renderable = object()
    workflow.renderable = (
        lambda *, compact=False, include_setup=True: compact_renderable if compact else full_renderable
    )

    monkeypatch.setattr(
        ui_module.WorkflowProgress,
        "_renderable_exceeds_console",
        lambda self, renderable: renderable is full_renderable,
    )

    workflow.start()
    workflow.set_detail(evaluator_module.EVAL_TRACK_STEP, "step 1")
    workflow.set_detail(evaluator_module.EVAL_EVALUATE_STEP, "done", render=False)
    workflow.stop()

    assert workflow._compact_layout is True
    # In compact mode, _renderable_with_limit builds a Panel via
    # build_workflow_intro instead of calling self.renderable(compact=True).
    from rich.panel import Panel

    assert isinstance(update_calls[-1][0], Panel)


def test_workflow_progress_prefers_compact_layout_from_first_render(monkeypatch):
    init_renderables: list[object] = []

    class _FakeLive:
        def __init__(self, renderable, **kwargs):
            init_renderables.append(renderable)

        def start(self, *, refresh=True):
            return None

        def update(self, renderable, *, refresh=True):
            return None

        def stop(self):
            return None

    monkeypatch.setattr(ui_module, "Live", _FakeLive)

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "deepocsort")],
        steps=((evaluator_module.EVAL_TRACK_STEP, "active"),),
        stderr=True,
    )
    workflow.prefer_compact_layout = True

    full_renderable = object()
    compact_renderable = object()
    workflow.renderable = (
        lambda *, compact=False, include_setup=True: compact_renderable if compact else full_renderable
    )

    workflow.start()

    assert workflow._compact_layout is True
    # In compact mode, _renderable_with_limit builds a Panel via
    # build_workflow_intro instead of calling self.renderable(compact=True).
    from rich.panel import Panel

    assert isinstance(init_renderables[0], Panel)


def test_build_checklist_uses_semantic_state_colors():
    rendered = ui_module._capture_renderable(
        ui_module.build_checklist(
            (
                ("Done step", "done"),
                ("Active step", "active"),
                ("Queued step", "todo"),
            )
        ),
        width=80,
        force_terminal=True,
        color_system="truecolor",
    )

    assert "\x1b[1;38;2;63;185;80m[✓] " in rendered
    assert "\x1b[1;38;2;227;179;65m[>] " in rendered or "\x1b[1;93m[>] " in rendered
    assert "\x1b[1;38;2;139;148;158m[ ] " in rendered


def test_hybridsort_eval_intro_fits_terminal_height():
    args = SimpleNamespace(
        detector=[Path("models/yolox_x_MOT17_ablation.pt")],
        reid=[Path("models/lmbn_n_duke.pt")],
        tracker="hybridsort",
        tracker_backend=None,
        tracking_backend="process",
        data="mot17-mini",
        benchmark="mot17-mini",
        dataset_id="",
        benchmark_id="",
        source=None,
        imgsz=[800, 1440],
        device="cpu",
        half=False,
        conf=0.01,
        n_threads=8,
        postprocessing="none",
    )

    renderable = ui_module.build_workflow_intro(
        "Evaluation",
        evaluator_module._build_eval_workflow_fields(args),
        steps=(
            (evaluator_module.EVAL_GENERATE_STEP, "active"),
            (evaluator_module.EVAL_TRACK_STEP, "todo"),
            (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
        ),
    )
    rendered = ui_module.capture_renderable(renderable, stderr=True, width=80)

    assert rendered.count("\n") + 1 <= 30


def test_build_workflow_intro_uses_compact_setup_panel_and_completed_pipeline_summary():
    rendered = ui_module.capture_renderable(
        ui_module.build_workflow_intro(
            "Evaluation",
            [
                ("Detector", Path("models/yolox_x_MOT17_ablation.pt")),
                ("ReID", Path("models/lmbn_n_duke.pt")),
                ("Tracker", "bytetrack"),
                ("Dataset", "mot17-mini"),
                ("__panel__:Tracker Parameters", [("Min Conf", 0.1), ("Track Thresh", 0.6)]),
                ("__panel__:Pipeline Parameters", [("Tracker backend", "python"), ("Replay backend", "thread")]),
            ],
            steps=(
                (evaluator_module.EVAL_GENERATE_STEP, "done"),
                (evaluator_module.EVAL_TRACK_STEP, "done"),
                (evaluator_module.EVAL_EVALUATE_STEP, "done"),
            ),
        ),
        width=120,
    )

    assert "Setup" in rendered
    assert "Configuration" not in rendered
    assert "Tracker Parameters" not in rendered
    assert "Pipeline Parameters" not in rendered
    assert "yolox_x_MOT17_ablation.pt" in rendered
    assert "lmbn_n_duke.pt" in rendered
    assert "models/yolox_x_MOT17_ablation.pt" not in rendered
    assert "models/lmbn_n_duke.pt" not in rendered
    assert "Generate / Track / Evaluate" in rendered
    assert "[x] Evaluate results" not in rendered


def test_build_workflow_intro_compact_live_layout_shows_all_progress_rows():
    fields = [
        ("Detector", Path("models/yolox_x_MOT17_ablation.pt")),
        ("ReID", Path("models/lmbn_n_duke.pt")),
        ("Tracker", "hybridsort"),
        ("Dataset", "mot17-mini"),
        (
            "__panel__:Tracker Parameters",
            [
                ("Low Thresh", 0.1),
                ("Delta T", 3),
                ("Inertia", 0.05),
                ("Use Byte", True),
                ("Use Custom KF", True),
                ("Longterm Bank Length", 30),
                ("Alpha", 0.9),
                ("Adapfs", False),
                ("Track Thresh", 0.5),
                ("EG Weight High Score", 4.6),
                ("EG Weight Low Score", 1.3),
                ("TCM First Step", True),
                ("TCM Byte Step", True),
                ("TCM Byte Step Weight", 1.0),
                ("High Score Matching Thresh", 0.7),
                ("With Longterm ReID", True),
                ("Longterm ReID Weight", 0.0),
                ("With Longterm ReID Correction", True),
                ("Longterm ReID Correction Thresh", 0.4),
                ("Longterm ReID Correction Thresh Low", 0.4),
            ],
        ),
        (
            "__panel__:Pipeline Parameters",
            [
                ("Tracker backend", "python"),
                ("Replay backend", "process"),
                ("Device", "cpu"),
                ("Precision", "fp32"),
                ("Image size", "[800, 1440]"),
                ("Confidence", 0.01),
                ("Threads", 8),
                ("Postprocessing", "none"),
            ],
        ),
    ]
    progress_detail = "\n".join(
        [
            "Tracking: 2/7 sequences done",
            "  MOT17-02 ██████████████░░░░░░   73%  (217/299)",
            "  MOT17-04 ██████░░░░░░░░░░░░░░   33%  (172/524)",
            "  MOT17-05 ████████████████████  100%  (done)",
            "  MOT17-09 ████████████████████  100%  (done)",
            "  MOT17-10 ██████████████░░░░░░   75%  (243/326)",
            "  MOT17-11 ████████████░░░░░░░░   62%  (277/449)",
            "  MOT17-13 ███████████░░░░░░░░░   56%  (210/374)",
        ]
    )

    rendered = ui_module.capture_renderable(
        ui_module.build_workflow_intro(
            "Evaluation",
            fields,
            steps=(
                (evaluator_module.EVAL_GENERATE_STEP, "done"),
                (evaluator_module.EVAL_TRACK_STEP, "active"),
                (evaluator_module.EVAL_EVALUATE_STEP, "todo"),
            ),
            detail_title=evaluator_module.EVAL_TRACK_STEP,
            detail_text=progress_detail,
            compact=True,
        ),
        width=80,
    )

    assert rendered.count("MOT17-") == 7
    assert rendered.count("\n") + 1 <= 30
    assert "[>] Track" in rendered


def test_build_timing_renderable_shows_detector_reid_tracker_breakdown():
    timings = {
        "frames": 100,
        "fps": 111.1,
        "totals_ms": {
            "preprocess": 100.0,
            "inference": 120.0,
            "postprocess": 30.0,
            "detector_preprocess": 100.0,
            "detector_process": 120.0,
            "detector_postprocess": 30.0,
            "reid": 200.0,
            "reid_preprocess": 40.0,
            "reid_process": 150.0,
            "reid_postprocess": 10.0,
            "track": 600.0,
            "plot": 0.0,
            "total": 1050.0,
        },
        "avg_ms": {
            "preprocess": 1.0,
            "inference": 1.2,
            "postprocess": 0.3,
            "reid": 2.0,
            "track": 6.0,
            "plot": 0.0,
            "total": 10.5,
        },
    }

    rendered = ui_module.capture_renderable(
        workflow_reporting_module._build_timing_renderable(timings),
        width=120,
    )

    assert "Frames" in rendered
    assert "Stage" in rendered
    assert "Detector" in rendered
    assert "Tracker" in rendered
    assert "  preprocess" in rendered
    assert "  process" in rendered
    assert "  postprocess" in rendered
    assert "ReID preprocess" in rendered
    assert "ReID process" in rendered
    assert "ReID postprocess" in rendered
    assert "association/update" in rendered
    assert "Overall total" in rendered
    assert "120.0" in rendered
    assert "250.0" in rendered
    assert "800.0" in rendered
    assert "1050.0" in rendered
    assert "111.1" in rendered


def test_build_timing_renderable_marks_cached_detector_and_embeddings():
    timings = {
        "frames": 12,
        "fps": 59405.9,
        "metadata": {
            "detector_from_cache": True,
            "reid_from_cache": True,
        },
        "totals_ms": {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "detector_preprocess": 0.0,
            "detector_process": 0.0,
            "detector_postprocess": 0.0,
            "reid": 0.0,
            "reid_preprocess": 0.0,
            "reid_process": 0.0,
            "reid_postprocess": 0.0,
            "track": 0.2,
            "plot": 0.0,
            "total": 0.2,
        },
        "avg_ms": {
            "preprocess": 0.0,
            "inference": 0.0,
            "postprocess": 0.0,
            "reid": 0.0,
            "track": 0.02,
            "plot": 0.0,
            "total": 0.02,
        },
    }

    rendered = ui_module.capture_renderable(
        workflow_reporting_module._build_timing_renderable(timings),
        width=120,
    )

    assert "detections loaded from cache" in rendered
    assert "embeddings loaded from cache" in rendered
    assert "association/update" in rendered
    assert "Detector total" in rendered
    assert "Tracker total" in rendered
    assert "Overall total" in rendered
    assert "  preprocess" not in rendered
    assert "ReID preprocess" not in rendered


def test_workflow_progress_supports_renderable_detail(monkeypatch):
    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=False,
        width=120,
        stderr=True,
        highlight=False,
        soft_wrap=False,
        theme=ui_module.BOXMOT_THEME,
    )

    monkeypatch.setattr(ui_module, "get_console", lambda *, stderr=False: console)

    detail = ui_module.Table.grid(expand=True)
    detail.add_column()
    detail.add_column(justify="right")
    detail.add_row("HOTA", "50.00")

    workflow = ui_module.create_workflow_progress(
        "Evaluation",
        [("Tracker", "bytetrack")],
        steps=((evaluator_module.EVAL_EVALUATE_STEP, "active"),),
        stderr=True,
    )

    workflow.start()
    workflow.set_detail_renderable("Results", detail)
    workflow.stop()

    rendered = buffer.getvalue()

    assert "Results" in rendered
    assert "HOTA" in rendered
    assert "50.00" in rendered


def test_build_validation_cli_renderable_contains_sequence_table_without_metric_cards():
    raw = {
        "person": {
            "HOTA": 69.43,
            "MOTA": 78.26,
            "IDF1": 82.00,
            "AssA": 72.29,
            "AssRe": 77.48,
            "IDSW": 136,
            "IDs": 367,
            "per_sequence": {
                "MOT17-02": {
                    "HOTA": 49.84,
                    "MOTA": 54.60,
                    "IDF1": 60.20,
                    "AssA": 51.80,
                    "AssRe": 56.49,
                    "IDSW": 61,
                    "IDs": 72,
                }
            },
        }
    }

    renderable = workflow_reporting_module.build_validation_cli_renderable(raw, title=None)
    rendered = ui_module.capture_renderable(renderable, width=120)

    assert "person" in rendered
    assert "HOTA" in rendered
    assert "MOTA" in rendered
    assert "IDF1" in rendered
    assert "AssA" in rendered
    assert "AssRe" in rendered
    assert "IDSW" in rendered
    assert "IDs" in rendered
    assert "MOT17-02" in rendered
    assert "COMBINED (person)" in rendered
    assert "•" in rendered
    assert "╭───────╮" not in rendered

    sequence_header = next(line for line in rendered.splitlines() if line.startswith("Sequence"))
    first_sequence = next(line for line in rendered.splitlines() if line.startswith("MOT17-02"))
    combined_index = next(i for i, line in enumerate(rendered.splitlines()) if line.startswith("COMBINED (person)"))
    assert len(sequence_header) >= 100
    assert len(first_sequence) >= 100
    assert set(rendered.splitlines()[combined_index - 1].strip()) == {"─"}


def test_build_validation_cli_renderable_keeps_multiclass_obb_sections():
    raw = {
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
    }

    renderable = workflow_reporting_module.build_validation_cli_renderable(
        raw,
        title=None,
        args=SimpleNamespace(
            remapped_class_names=None,
            translated_benchmark_class_names=None,
            eval_box_type="obb",
            classes=None,
            benchmark="mmot-mini",
        ),
    )
    rendered = ui_module.capture_renderable(renderable, width=140)

    assert "Per-Class Combined Metrics" in rendered
    assert "plane" in rendered
    assert "tennis court" in rendered
    assert "Class Avg (Det)" in rendered
    assert "COMBINED (plane)" in rendered


def test_build_validation_cli_renderable_compacts_multiclass_without_sequences():
    raw = {
        "car": {
            "HOTA": 70.0,
            "MOTA": 75.0,
            "IDF1": 80.0,
            "per_sequence": {"seq-1": {"HOTA": 69.0, "MOTA": 74.0, "IDF1": 79.0}},
        },
        "bike": {
            "HOTA": 50.0,
            "MOTA": 55.0,
            "IDF1": 60.0,
            "per_sequence": {"seq-1": {"HOTA": 49.0, "MOTA": 54.0, "IDF1": 59.0}},
        },
        "cls_comb_det_av": {"HOTA": 60.0, "MOTA": 65.0, "IDF1": 70.0},
    }

    renderable = workflow_reporting_module.build_validation_cli_renderable(
        raw,
        title=None,
        include_sequences=False,
        args=SimpleNamespace(
            remapped_class_names=None,
            translated_benchmark_class_names=None,
            eval_box_type="obb",
            classes=None,
            benchmark="mmot-mini",
        ),
    )
    rendered = ui_module.capture_renderable(renderable, width=140)

    assert "Per-Class Combined Metrics" in rendered
    assert "Aggregate Groups" in rendered
    assert "seq-1" not in rendered
    assert "COMBINED (car)" not in rendered


def test_run_generate_mot_results_nonquiet_mode_uses_manager_queue(tmp_path, monkeypatch):
    source = tmp_path / "train"
    for seq_name in ("MOT17-02-FRCNN", "MOT17-04-FRCNN"):
        img_dir = source / seq_name / "img1"
        img_dir.mkdir(parents=True)
        (img_dir / "000001.jpg").write_bytes(b"")

    args = SimpleNamespace(
        project=tmp_path,
        benchmark="mot17-mini",
        source=source,
        detector=[Path("det.pt")],
        reid=[Path("reid.pt")],
        tracker="boosttrack",
        fps=None,
        device="cpu",
        n_threads=2,
        tracking_backend="process",
        postprocessing="none",
        conf=0.25,
    )

    queue_types = []
    executor_kwargs = {}
    manager_calls = []

    class FakeManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def Queue(self):
            progress_queue = queue.Queue()
            queue_types.append(type(progress_queue).__name__)
            return progress_queue

    class FakeSpawnContext:
        def Manager(self):
            manager_calls.append(True)
            return FakeManager()

    spawn_context = FakeSpawnContext()

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeProcessPoolExecutor:
        def __init__(self, *_args, **kwargs):
            executor_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, _func, *task_arg):
            seq_name = task_arg[0]
            progress_queue = task_arg[-2]
            if progress_queue is not None:
                progress_queue.put_nowait((seq_name, 1, 1))
            return FakeFuture((seq_name, [1], {"track_time_ms": 5.0, "num_frames": 1}))

    monkeypatch.setattr(cached_tracking_module.mp, "get_context", lambda method: spawn_context)
    monkeypatch.setattr(cached_tracking_module.concurrent.futures, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(
        cached_tracking_module.concurrent.futures,
        "wait",
        lambda pending, timeout, return_when: (set(pending), set()),
    )

    cached_tracking_module.run_generate_mot_results(args, quiet=False)

    assert args.seq_frame_nums == {
        "MOT17-02-FRCNN": [1],
        "MOT17-04-FRCNN": [1],
    }
    assert manager_calls == [True]
    assert queue_types == ["Queue"]
    assert executor_kwargs["mp_context"] is spawn_context
    assert executor_kwargs["max_workers"] == 2


def test_resolve_output_stem_handles_stream_and_camera_sources():
    assert workflow_support_module.resolve_output_stem("rtsp://camera/stream") == "rtsp_camera_stream"
    assert workflow_support_module.resolve_output_stem("0") == "camera_0"


def test_resolve_output_fps_uses_video_fallback_when_source_cannot_open():
    assert workflow_support_module.resolve_output_fps("video.mp4") == 30


def test_tracker_runtime_update_measures_elapsed_time_and_passes_embeddings():
    calls = []

    class _FakeTracker:
        def update(self, dets, img, embs=None):
            calls.append((dets.shape, img.shape, None if embs is None else embs.shape))
            return np.array([[1, 2, 5, 6, 7, 0.9, 0, 0]], dtype=np.float32)

        def plot_results(self, img, show_trajectories, *, thickness, show_kf_preds):
            _ = (show_trajectories, thickness, show_kf_preds)
            return img

    timing_stats = TimingStats()
    runtime = tracker_runtime_module.TrackerRuntime(_FakeTracker(), timing_stats=timing_stats)

    tracks, elapsed_ms = runtime.update(
        np.array([[1, 2, 5, 6, 0.9, 0]], dtype=np.float32),
        _DUMMY_IMG,
        np.ones((1, 4), dtype=np.float32),
    )

    assert tracks.shape == (1, 8)
    assert elapsed_ms >= 0
    assert timing_stats.get_last_track_time() == elapsed_ms
    assert calls == [((1, 6), _DUMMY_IMG.shape, (1, 4))]


def test_initialize_trackers_rejects_unknown_tracker():
    with pytest.raises(ValueError, match="registered tracker name"):
        workflow_support_module.tracker_name_from_spec("unknown")
