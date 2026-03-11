import numpy as np

from boxmot.engine.inference import extract_detections, filter_detections, resolve_yolo_model_path
from boxmot.trackers.ocsort.ocsort import convert_obb_to_z, convert_x_to_obb
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.detection_layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.utils.iou import iou_obb_pair
from boxmot.utils import WEIGHTS


class _TensorWrapper:
    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    @property
    def shape(self):
        return self._array.shape


class _PredictionWrapper:
    def __init__(self, array: np.ndarray):
        self.data = _TensorWrapper(array)

    def __len__(self):
        return len(self.data.numpy())


class _Result:
    def __init__(self, boxes=None, obb=None):
        self.boxes = boxes
        self.obb = obb


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


def test_extract_detections_reads_aabb_results():
    boxes = _PredictionWrapper([[1, 2, 3, 4, 0.9, 0]])
    result = _Result(boxes=boxes)

    dets = extract_detections(result)

    assert dets.shape == (1, 6)
    np.testing.assert_array_equal(dets[0], np.array([1, 2, 3, 4, 0.9, 0], dtype=np.float32))


def test_extract_detections_reads_obb_results():
    obb = _PredictionWrapper([[10, 20, 30, 40, 0.5, 0.8, 1]])
    result = _Result(obb=obb)

    dets = extract_detections(result)

    assert dets.shape == (1, 7)
    np.testing.assert_array_equal(
        dets[0],
        np.array([10, 20, 30, 40, 0.5, 0.8, 1], dtype=np.float32),
    )


def test_extract_detections_preserves_empty_obb_width():
    result = _Result(obb=_PredictionWrapper(np.empty((0, 7), dtype=np.float32)))

    dets = extract_detections(result)

    assert dets.shape == (0, 7)


def test_filter_detections_keeps_valid_obb_boxes():
    dets = np.array(
        [
            [100, 100, 20, 10, 0.2, 0.9, 0],
            [100, 100, 0, 10, 0.2, 0.9, 0],
        ],
        dtype=np.float32,
    )

    filtered = filter_detections(dets, min_area=50.0)

    assert filtered.shape == (1, 7)
    np.testing.assert_array_equal(filtered[0], dets[0])


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


def test_resolve_yolo_model_path_routes_non_rtdetr_to_weights_dir():
    resolved = resolve_yolo_model_path("yolov8n.pt")

    assert resolved == WEIGHTS / "yolov8n.pt"


def test_resolve_yolo_model_path_keeps_rtdetr_name_without_path_prefix():
    resolved = resolve_yolo_model_path("/tmp/models/rtdetr_v2_r18vd.pt")

    assert resolved.name == "rtdetr_v2_r18vd.pt"
    assert str(resolved.parent) == "."
