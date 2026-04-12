import importlib
import queue
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.configs import ensure_model_extension
from boxmot.data import iter_source
from boxmot.detectors import default_conf, default_imgsz, get_detector_url, get_runtime_detector_cfg, load_detector_cfg
from boxmot.detectors.base import Detections
import boxmot.data.loaders as loaders_module
import boxmot.detectors.detector as detector_module
import boxmot.detectors.ultralytics as ultralytics_detector_module
import boxmot.engine.evaluator as evaluator_module
import boxmot.engine.inference as pipeline_module
import boxmot.engine.replay as cached_tracking_module
import boxmot.engine.tracker as tracker_module
import boxmot.engine.tracker as tracker_runtime_module
import boxmot.reid.core as reid_core_module
from boxmot.detectors.ultralytics import UltralyticsDetector
from boxmot.engine.inference import prepare_detections
from boxmot.trackers.ocsort.ocsort import convert_obb_to_z, convert_x_to_obb
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.detection_layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.data.cache import (
    AppendableNpyWriter,
    _existing_cache_path,
    _existing_embedding_cache_path,
    _load_embedding_cache_array,
    _load_numeric_cache_array,
    _max_frame_id,
    _migrate_legacy_embedding_cache,
    _saved_detection_column_count,
)
from boxmot.utils.iou import iou_obb_pair
from boxmot.utils.mot_utils import convert_to_mot_format, write_mot_results
from boxmot.utils import WEIGHTS

_DUMMY_IMG = np.zeros((64, 64, 3), dtype=np.uint8)


class _DummyTracker(BaseTracker):
    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
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
            [100, 100, 0, 10, 0.2, 0.9, 0],   # invalid: w = 0
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

    class _FakeYOLO:
        def __init__(self, model):
            self.model = model
            self.names = {0: "plane"}

        def predict(self, **_kwargs):
            return [
                _FakeResult(
                    [[32.0, 24.0, 20.0, 10.0, 0.25, 0.9, 0.0]]
                )
            ]

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


def test_default_detector_fallbacks_preserve_legacy_runtime_behavior():
    assert default_imgsz("yolox_s.pt") == [1080, 1920]
    assert default_imgsz("yolov8n.pt") == [640, 640]
    assert default_conf("yolox_s.pt") == 0.01


def test_model_config_detector_defaults_override_runtime_defaults_by_model_name():
    detector_cfg = load_detector_cfg("yolo11s-obb.pt")

    assert detector_cfg["classes"][0] == "plane"
    assert default_imgsz("yolo11s-obb.pt") == detector_cfg["imgsz"]
    assert default_conf("yolo11s-obb.pt") == detector_cfg["conf"]


def test_model_config_detector_defaults_match_separator_variants():
    detector_cfg = load_detector_cfg("yolo11l_3ch.pt")

    assert detector_cfg["id"] == "yolo11l_3ch"
    assert default_imgsz("yolo11l_3ch.pt") == detector_cfg["imgsz"]
    assert get_detector_url("yolo11l_3ch.pt") == detector_cfg["url"]


def test_runtime_detector_cfg_uses_model_config_to_override_dataset_values():
    detector_cfg = load_detector_cfg("yolo11s-obb.pt")
    benchmark_cfg = {
        "default_model": "models/yolo11s-obb.pt",
        "imgsz": [1024, 1024],
        "conf": 0.2,
        "classes": {0: "person"},
    }

    resolved = get_runtime_detector_cfg("yolo11s-obb.pt", benchmark_cfg)

    assert resolved["default_model"] == "models/yolo11s-obb.pt"
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
        lambda url, dest, overwrite=False, **_kwargs: calls.update(
            {"url": url, "dest": dest, "overwrite": overwrite}
        ) or dest,
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
        lambda file, release="latest", **_kwargs: calls.update(
            {"file": Path(file), "release": release}
        ) or str(file),
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
        lambda file, release="latest", **_kwargs: calls.update(
            {"file": Path(file), "release": release}
        ) or str(file),
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
    detector_cfg = load_detector_cfg("yolo11s-obb.pt")
    benchmark_bundle = {
        "benchmark": {"box_type": "obb"},
        "detector": {
            "default_model": "models/yolo11s-obb.pt",
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
        lambda _cfg: Path("models/yolo11s-obb.pt"),
    )
    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_reid_model",
        lambda _cfg: Path("models/lmbn_n_duke.pt"),
    )

    _, _, runtime_cfg = evaluator_module._configure_benchmark_runtime(args)

    assert args.detector == [Path("models/yolo11s-obb.pt")]
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
    detector_cfg = load_detector_cfg("yolo11s-obb.pt")
    args = SimpleNamespace(
        detector=[Path("models/yolo11s-obb.pt")],
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


def test_iter_source_expands_globs(tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "000001.jpg"
    import cv2
    cv2.imwrite(str(img_path), img)

    frames = list(iter_source(str(tmp_path / "*.jpg")))

    assert len(frames) == 1
    assert frames[0][0] == str(img_path)
    assert frames[0][1].shape == img.shape


def test_existing_embedding_cache_path_falls_back_to_legacy_txt(tmp_path):
    emb_txt = tmp_path / "SEQ.txt"
    np.savetxt(emb_txt, np.arange(4, dtype=np.float32)[None, :], fmt="%f")

    resolved = _existing_embedding_cache_path(tmp_path / "SEQ.npy")

    assert resolved == emb_txt


def test_load_embedding_cache_array_normalizes_single_row_txt_to_2d(tmp_path):
    emb_txt = tmp_path / "SEQ.txt"
    expected = np.arange(8, dtype=np.float32)[None, :]
    np.savetxt(emb_txt, expected, fmt="%f")

    loaded = _load_embedding_cache_array(emb_txt)

    assert loaded.shape == (1, 8)
    np.testing.assert_allclose(loaded, expected)


def test_migrate_legacy_embedding_cache_writes_npy(tmp_path):
    emb_txt = tmp_path / "SEQ.txt"
    target_npy = tmp_path / "SEQ.npy"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savetxt(emb_txt, expected, fmt="%f")

    migrated = _migrate_legacy_embedding_cache(emb_txt, target_npy)

    assert migrated is True
    np.testing.assert_allclose(np.load(target_npy), expected)


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
    np.testing.assert_allclose(
        _load_numeric_cache_array(det_npy),
        dets,
    )


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
        def __init__(self, weights, device, half):
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

    assert set(features) == {"alpha", "beta"}
    np.testing.assert_array_equal(features["alpha"], np.full((1, 4), 64.0, dtype=np.float32))


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
        def __init__(self, weights, device, half):
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
    assert set(pipeline.get_all_reid_features(np.array([[0, 0, 1, 1]], dtype=np.float32), _DUMMY_IMG)) == {"alpha"}
    assert detector_calls[0] == ("init", Path("det.pt"), "cuda:0", [64, 64])
    assert detector_calls[1:5] == [
        (1, 0.25, 0.7, False, None),
        (1, 0.25, 0.7, False, None),
        (8, 0.25, 0.7, False, None),
        (1, 0.25, 0.7, False, None),
    ]
    assert reid_calls == [(1, 64), (1, 64)]


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

    state = convert_obb_to_z(obb)
    decoded = convert_x_to_obb(state)

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
    # TrackEval fixed-width format: name column %-35s, each value %-10s
    results = (
        "\nHOTA: tracker-storage tank HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)\n"
        "COMBINED                           51.0      52.0      61.0      53.0      54.0      71.0      72.0      73.0      74.0      75.0      76.0      77.0      \n"
        "CLEAR: tracker-storage tank MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag      \n"
        "COMBINED                           41.0      42.0      43.0      44.0      45.0      46.0      47.0      48.0      52.0      49        50        51        3         4         5         6         7         \n"
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


def test_build_trackeval_feedback_keeps_summary_and_per_sequence_metrics():
    results_module = importlib.import_module("boxmot.utils.evaluation.results")
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

    feedback = results_module.build_trackeval_feedback(raw)

    assert feedback["summary_label"] == "all"
    assert feedback["summary"]["HOTA"] == 62.5
    assert feedback["summary"]["CLR_TP"] == 123
    assert feedback["per_sequence_metrics"]["MOT17-02"]["IDSW"] == 4
    assert feedback["per_class_metrics"]["all"]["MOTA"] == 70.0
    assert feedback["per_class_metrics"]["person"]["CLR_TP"] == 100


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


def test_dota8_obb_gt_uses_zero_based_eval_class_ids():
    expected = {0, 4, 10, 14}
    found = set()

    for path in sorted(Path("assets/DOTA8-MOT/train").glob("*/gt/gt_obb.txt")):
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

    class FakeProcessPoolExecutor:
        def __init__(self, *_args, **kwargs):
            executor_kwargs.update(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, _func, *task_arg):
            seq_name = task_arg[0]
            progress_queue = task_arg[-1]
            assert progress_queue is None
            return FakeFuture((seq_name, [1], {"track_time_ms": 5.0, "num_frames": 1}))

    monkeypatch.setattr(cached_tracking_module.mp, "get_context", lambda method: spawn_context)
    monkeypatch.setattr(cached_tracking_module.concurrent.futures, "ProcessPoolExecutor", FakeProcessPoolExecutor)
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
    assert executor_kwargs["mp_context"] is spawn_context
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
            progress_queue = task_arg[-1]
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
    monkeypatch.setattr(cached_tracking_module.LOGGER, "opt", lambda **kwargs: SimpleNamespace(info=lambda message: None))

    cached_tracking_module.run_generate_mot_results(args, quiet=False)

    assert args.seq_frame_nums == {
        "MOT17-02-FRCNN": [1],
        "MOT17-04-FRCNN": [1],
    }
    assert manager_calls == [True]
    assert queue_types == ["Queue"]
    assert executor_kwargs["mp_context"] is spawn_context
    assert executor_kwargs["max_workers"] == 2


def test_tracking_session_output_stem_handles_stream_and_camera_sources():
    session = tracker_module.TrackingSession(SimpleNamespace(source="rtsp://camera/stream", fps=None))

    assert session._resolve_output_stem() == "rtsp_camera_stream"

    session.args.source = "0"
    assert session._resolve_output_stem() == "camera_0"


def test_tracking_session_resolves_output_fps_from_args():
    session = tracker_module.TrackingSession(SimpleNamespace(source="video.mp4", fps=12))

    assert session._resolve_output_fps() == 12

    session.args.fps = None
    assert session._resolve_output_fps() == 30


def test_tracker_runtime_update_measures_elapsed_time_and_passes_embeddings():
    calls = []

    class _FakeTracker:
        def update(self, dets, img, embs=None):
            calls.append((dets.shape, img.shape, None if embs is None else embs.shape))
            return np.array([[1, 2, 5, 6, 7, 0.9, 0, 0]], dtype=np.float32)

        def plot_results(self, img, show_trajectories, *, thickness, show_kf_preds):
            _ = (show_trajectories, thickness, show_kf_preds)
            return img

    timing_stats = tracker_runtime_module.TimingStats()
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
    predictor = SimpleNamespace(dataset=SimpleNamespace(bs=1), device="cpu")
    args = SimpleNamespace(tracker="unknown", reid=Path("reid.pt"), half=False, per_class=False, target_id=None)

    with pytest.raises(ValueError, match="not supported"):
        tracker_module.TrackingSession.initialize_trackers(predictor, args)
