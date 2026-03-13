from pathlib import Path

import numpy as np

from boxmot.detectors.detector import Detections
from boxmot.engine.cli import ensure_model_extension
from boxmot.engine.inference import prepare_detections
from boxmot.trackers.ocsort.ocsort import convert_obb_to_z, convert_x_to_obb
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.detection_layout import AABB_DETECTIONS, OBB_DETECTIONS
from boxmot.utils.iou import iou_obb_pair
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
