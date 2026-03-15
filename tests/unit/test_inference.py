from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from boxmot.detectors import default_conf, default_imgsz, get_runtime_detector_cfg, load_detector_cfg
from boxmot.detectors.detector import Detections
import boxmot.detectors.ultralytics as ultralytics_detector_module
from boxmot.detectors.ultralytics import UltralyticsDetector
from boxmot.engine.cli import ensure_model_extension
import boxmot.engine.evaluator as evaluator_module
from boxmot.engine.inference import _iter_source, prepare_detections
from boxmot.trackers.ocsort.ocsort import convert_obb_to_z, convert_x_to_obb
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.detection_layout import AABB_DETECTIONS, OBB_DETECTIONS
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

    out = prepare_detections(result, _DUMMY_IMG)

    assert out.shape == (1, 6)
    np.testing.assert_array_equal(out[0], np.array([10, 20, 30, 40, 0.9, 0], dtype=np.float32))


def test_prepare_detections_reads_obb_result():
    dets = np.array([[10, 20, 30, 40, 0.5, 0.8, 1]], dtype=np.float32)
    result = Detections(dets=dets, orig_img=_DUMMY_IMG)

    out = prepare_detections(result, _DUMMY_IMG)

    assert out.shape == (1, 7)
    np.testing.assert_array_equal(out[0], dets[0])


def test_prepare_detections_preserves_empty_obb_width():
    dets = np.empty((0, 7), dtype=np.float32)
    result = Detections(dets=dets, orig_img=_DUMMY_IMG)

    out = prepare_detections(result, _DUMMY_IMG)

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

    out = prepare_detections(result, _DUMMY_IMG)

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

        def predict(self, **kwargs):
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


def test_detector_yaml_overrides_runtime_defaults_by_model_name():
    detector_cfg = load_detector_cfg("yolo11s-obb.pt")

    assert detector_cfg["classes"][0] == "plane"
    assert default_imgsz("yolo11s-obb.pt") == detector_cfg["imgsz"]
    assert default_conf("yolo11s-obb.pt") == detector_cfg["conf"]


def test_runtime_detector_cfg_uses_model_yaml_to_override_benchmark_values():
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


def test_configure_benchmark_runtime_lets_model_yaml_override_benchmark_detector(monkeypatch):
    detector_cfg = load_detector_cfg("yolo11s-obb.pt")
    benchmark_bundle = {
        "benchmark": {"box_type": "obb"},
        "detector": {
            "default_model": "models/yolo11s-obb.pt",
            "imgsz": [1024, 1024],
            "conf": 0.2,
            "classes": {0: "person"},
        },
    }
    args = SimpleNamespace(
        yolo_model=[Path("models/yolov8n.pt")],
        imgsz=None,
        conf=None,
        eval_box_type=None,
        dataset_detector_cfg=None,
    )

    monkeypatch.setattr(evaluator_module, "_load_benchmark_cfg", lambda _args: benchmark_bundle)
    monkeypatch.setattr(evaluator_module, "should_use_benchmark_detector", lambda _args, _cfg: True)
    monkeypatch.setattr(
        evaluator_module,
        "ensure_benchmark_detector_model",
        lambda _cfg: Path("models/yolo11s-obb.pt"),
    )

    _, _, runtime_cfg = evaluator_module._configure_benchmark_runtime(args)

    assert args.yolo_model == [Path("models/yolo11s-obb.pt")]
    assert args.imgsz == detector_cfg["imgsz"]
    assert args.conf == detector_cfg["conf"]
    assert args.eval_box_type == "obb"
    assert runtime_cfg["classes"][0] == detector_cfg["classes"][0]
    assert args.dataset_detector_cfg["classes"][0] == detector_cfg["classes"][0]


def test_iter_source_expands_globs(tmp_path):
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img_path = tmp_path / "000001.jpg"
    import cv2
    cv2.imwrite(str(img_path), img)

    frames = list(_iter_source(str(tmp_path / "*.jpg")))

    assert len(frames) == 1
    assert frames[0][0] == str(img_path)
    assert frames[0][1].shape == img.shape


def test_iter_source_preserves_stream_urls(monkeypatch):
    seen = {}

    class _FakeCapture:
        def __init__(self, value):
            seen["source"] = value

        def isOpened(self):
            return False

        def release(self):
            seen["released"] = True

    import boxmot.engine.inference as inference_module

    monkeypatch.setattr(inference_module.cv2, "VideoCapture", _FakeCapture)

    frames = list(_iter_source("rtsp://camera/stream"))

    assert frames == []
    assert seen["source"] == "rtsp://camera/stream"
    assert seen["released"] is True


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
    results = """
HOTA: tracker-storage tank HOTA DetA AssA DetRe DetPr AssRe AssPr LocA OWTA HOTA(0) LocA(0) HOTALocA(0)
COMBINED 51.0 0 61.0 0 0 71.0 0 0 0 0 0
CLEAR: tracker-storage tank MOTA MOTP MODA CLR_Re CLR_Pr MTR PTR MLR CLR_TP CLR_FN CLR_FP IDSW MT PT ML Frag sMOTA
COMBINED 41.0 0 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 0
Identity: tracker-storage tank IDF1 IDR IDP IDTP IDFN IDFP
COMBINED 31.0 0 0 0 0 0
Count: tracker-storage tank Dets GT_Dets IDs GT_IDs
COMBINED 0 0 7 0
"""

    parsed = evaluator_module.parse_mot_results(results, known_classes=["storage tank"])

    assert list(parsed) == ["storage tank"]
    assert parsed["storage tank"]["HOTA"] == 51.0
    assert parsed["storage tank"]["AssA"] == 61.0
    assert parsed["storage tank"]["AssRe"] == 71.0
    assert parsed["storage tank"]["MOTA"] == 41.0
    assert parsed["storage tank"]["IDSW"] == 3
    assert parsed["storage tank"]["IDF1"] == 31.0
    assert parsed["storage tank"]["IDs"] == 7


def test_ordered_benchmark_eval_class_names_preserve_multiword_legacy_classes():
    bench_cfg = {"classes": ["storage tank", "ground track field"]}

    class_names = evaluator_module._ordered_benchmark_eval_class_names(bench_cfg)

    assert class_names == ["storage tank", "ground track field"]


def test_dota8_obb_gt_uses_zero_based_eval_class_ids():
    expected = {0, 4, 10, 14}
    found = set()

    for path in sorted(Path("assets/DOTA8-MOT/train").glob("*/gt/gt_obb.txt")):
        matrix = evaluator_module._load_obb_gt_matrix(path)
        found.update(matrix[:, 11].astype(int).tolist())

    assert found == expected
