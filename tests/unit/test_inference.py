from pathlib import Path

import numpy as np

from boxmot.engine.cli import ensure_model_extension
from boxmot.engine.inference import prepare_detections
from boxmot.detectors.detector import Detections
from boxmot.utils.iou import iou_obb_pair



_DUMMY_IMG = np.zeros((64, 64, 3), dtype=np.uint8)


def _make_result(array):
    return Detections(dets=np.asarray(array, dtype=np.float32), orig_img=_DUMMY_IMG)


def test_extract_detections_reads_aabb_results():
    result = _make_result([[10, 20, 30, 40, 0.9, 0]])

    dets = prepare_detections(result, _DUMMY_IMG)

    assert dets.shape == (1, 6)
    np.testing.assert_array_equal(dets[0], np.array([10, 20, 30, 40, 0.9, 0], dtype=np.float32))


def test_extract_detections_reads_obb_results():
    result = _make_result([[10, 20, 30, 40, 0.5, 0.8, 1]])

    dets = prepare_detections(result, _DUMMY_IMG)

    assert dets.shape == (1, 7)
    np.testing.assert_array_equal(
        dets[0],
        np.array([10, 20, 30, 40, 0.5, 0.8, 1], dtype=np.float32),
    )


def test_extract_detections_preserves_empty_obb_width():
    result = _make_result(np.empty((0, 7), dtype=np.float32))

    dets = prepare_detections(result, _DUMMY_IMG)

    assert dets.shape == (0, 7)


def test_filter_detections_keeps_valid_obb_boxes():
    result = _make_result([
        [100, 100, 20, 10, 0.2, 0.9, 0],
        [100, 100,  0, 10, 0.2, 0.9, 0],
    ])

    filtered = prepare_detections(result, _DUMMY_IMG)

    assert filtered.shape == (1, 7)
    np.testing.assert_array_equal(filtered[0], np.array([100, 100, 20, 10, 0.2, 0.9, 0], dtype=np.float32))


def test_iou_obb_pair_accepts_column_like_inputs_and_radians():
    dets = np.array([[32, 24, 20, 10, 0.25, 0.9]], dtype=object)
    trks = np.array([[32, 24, 20, 10, 0.25, 0.8]], dtype=object)

    iou = iou_obb_pair(0, 0, dets, trks)

    assert iou > 0.99


def test_ensure_model_extension_preserves_explicit_export_paths():
    model_path = "models/osnet_x0_25_msmt17_saved_model/osnet_x0_25_msmt17_float32.tflite"

    resolved = ensure_model_extension(model_path)

    assert resolved == Path(model_path)
