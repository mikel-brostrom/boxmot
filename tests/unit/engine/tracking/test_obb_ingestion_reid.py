from __future__ import annotations

import cv2
import numpy as np
import pytest

from boxmot.data.cache import REID_CROP_SCHEMA_VERSION, reid_cache_key, reid_preprocess_cache_key
from boxmot.detectors.base import Detections
from boxmot.engine.tracking.results import FrameResult, Results
from boxmot.trackers.common.geometry.obb import xywha_to_corners
from boxmot.trackers.results import TrackResults


def test_obb_detection_and_track_accessors_return_native_and_enclosing_geometry():
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    row = np.array([[64, 64, 40, 20, 0.5, 0.9, 2]], dtype=np.float32)
    detections = Detections(row, image)
    tracks = TrackResults(np.array([[64, 64, 40, 20, 0.5, 7, 0.9, 2, 0]], dtype=np.float32))
    expected_xyxy = cv2.boxPoints(((64.0, 64.0), (40.0, 20.0), np.degrees(0.5)))
    expected_xyxy = np.array(
        [
            expected_xyxy[:, 0].min(),
            expected_xyxy[:, 1].min(),
            expected_xyxy[:, 0].max(),
            expected_xyxy[:, 1].max(),
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(detections.boxes, row[:, :5])
    np.testing.assert_allclose(detections.xyxy[0], expected_xyxy, atol=1e-5)
    np.testing.assert_array_equal(tracks.boxes, row[:, :5])
    np.testing.assert_array_equal(tracks.xywh, row[:, :4])
    np.testing.assert_allclose(tracks.xyxy[0], expected_xyxy, atol=1e-5)


def test_track_results_slicing_keeps_masks_row_aligned():
    rows = np.array(
        [
            [20, 20, 12, 8, 0.1, 1, 0.9, 0, 0],
            [40, 20, 12, 8, 0.2, 2, 0.8, 1, 1],
            [60, 20, 12, 8, 0.3, 3, 0.7, 0, 2],
        ],
        dtype=np.float32,
    )
    masks = np.stack([np.full((4, 5), value, dtype=np.uint8) for value in (1, 2, 3)])
    tracks = TrackResults(rows, masks=masks)

    np.testing.assert_array_equal(tracks[:1].masks, masks[:1])
    np.testing.assert_array_equal(tracks[[2, 0]].masks, masks[[2, 0]])
    np.testing.assert_array_equal(tracks[np.array([False, True, True])].masks, masks[1:])
    np.testing.assert_array_equal(tracks[1].masks, masks[1:2])


def test_frame_result_prefers_tracker_refined_masks_over_detector_masks():
    rows = np.array([[32, 32, 20, 10, 0.2, 1, 0.9, 0, 0]], dtype=np.float32)
    tracker_masks = np.ones((1, 8, 8), dtype=np.uint8)
    detector_masks = np.zeros((1, 8, 8), dtype=np.uint8)
    tracks = TrackResults(rows, masks=tracker_masks)

    result = FrameResult(
        frame_idx=1,
        frame=np.zeros((32, 32, 3), dtype=np.uint8),
        tracks=tracks,
        detections=np.array([[32, 32, 20, 10, 0.2, 0.9, 0]], dtype=np.float32),
        source_path="",
        get_drawer=lambda: None,
        masks=detector_masks,
    )

    np.testing.assert_array_equal(result.masks, tracker_masks)


def test_frame_result_rejects_misaligned_tracker_masks():
    rows = np.array([[32, 32, 20, 10, 0.2, 1, 0.9, 0, 0]], dtype=np.float32)
    tracks = TrackResults(rows, masks=np.ones((2, 8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="mask count must match"):
        FrameResult(
            frame_idx=1,
            frame=np.zeros((32, 32, 3), dtype=np.uint8),
            tracks=tracks,
            detections=None,
            source_path="",
            get_drawer=lambda: None,
        )


def test_frame_result_bounds_checks_detection_indices():
    tracks = np.array(
        [
            [1, 2, 3, 4, 10, 0.9, 0, 5],
            [5, 6, 7, 8, 11, 0.8, 0, -1],
        ],
        dtype=np.float32,
    )
    detections = np.array([[1, 2, 3, 4, 0.9, 0]], dtype=np.float32)
    embeddings = np.ones((1, 3), dtype=np.float32)
    masks = np.ones((1, 4, 4), dtype=np.uint8)

    result = FrameResult(
        frame_idx=1,
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        tracks=tracks,
        detections=detections,
        source_path="",
        get_drawer=lambda: None,
        embeddings=embeddings,
        masks=masks,
    )

    np.testing.assert_array_equal(result.detections, np.zeros((2, 6), dtype=np.float32))
    np.testing.assert_array_equal(result.embeddings, np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(result.masks, np.zeros((2, 4, 4), dtype=np.uint8))


def test_frame_result_save_txt_uses_canonical_mmot_obb_schema(tmp_path):
    tracks = np.array([[50, 40, 20, 10, 0.3, 7, 0.8, 2, 4]], dtype=np.float32)
    result = FrameResult(
        frame_idx=12,
        frame=np.zeros((80, 100, 3), dtype=np.uint8),
        tracks=tracks,
        detections=None,
        source_path="",
        get_drawer=lambda: None,
    )
    output_path = tmp_path / "tracks.txt"

    result.save_txt(output_path)

    saved = np.loadtxt(output_path, delimiter=",", ndmin=2)
    expected = result.to_mot()
    assert saved.shape == (1, 13)
    np.testing.assert_allclose(saved, expected, atol=1e-5)
    np.testing.assert_allclose(
        np.loadtxt([str(result)], delimiter=",", ndmin=2),
        expected,
        atol=1e-5,
    )
    assert int(saved[0, 1]) == 7
    assert int(saved[0, -1]) == 4

    direct_path = tmp_path / "direct-tracks.txt"
    result.tracks.save_mot(direct_path, frame_id=result.frame_idx)
    direct = np.loadtxt(direct_path, delimiter=",", ndmin=2)
    assert direct.shape == (1, 13)
    np.testing.assert_allclose(direct, expected, atol=1e-5)


def test_results_save_flushes_gta_through_canonical_mmot_schema(tmp_path):
    gta_rows = np.array([[5, 50, 40, 20, 10, 0.3, 7, 0.8, 2, -1]], dtype=np.float32)

    class _FakeTracker:
        @staticmethod
        def flush_gta():
            return gta_rows.copy()

    results = Results([], detector=object(), reid=None, tracker=_FakeTracker(), verbose=False)
    results._generator = iter(())
    output_path = results.save(tmp_path / "tracks.txt")

    saved = np.loadtxt(output_path, delimiter=",", ndmin=2)
    assert saved.shape == (1, 13)
    np.testing.assert_allclose(saved[0, 2:10], xywha_to_corners(gta_rows[0, 1:6]), atol=1e-5)
    assert int(saved[0, 0]) == 5
    assert int(saved[0, 1]) == 7
    assert int(saved[0, -1]) == -1


def test_live_results_sanitize_obb_before_reid_and_keep_masks_aligned():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    raw_dets = np.array(
        [
            [32, 32, 20, 10, 0.2, 0.9, 0],
            [32, 32, np.inf, 10, 0.2, 0.8, 0],
            [32, 32, 500, 10, 0.2, 0.7, 0],
            [32, 32, 0, 10, 0.2, 0.6, 0],
        ],
        dtype=np.float32,
    )
    raw_masks = np.stack([np.full((8, 8), value, dtype=np.uint8) for value in range(1, 5)])

    class FakeDetector:
        def __call__(self, _frame):
            return Detections(raw_dets, frame, masks=raw_masks)

    class FakeReID:
        def __init__(self):
            self.boxes = None

        def __call__(self, _frame, boxes=None):
            self.boxes = np.asarray(boxes).copy()
            return np.ones((len(boxes), 4), dtype=np.float32)

    class FakeTracker:
        def __init__(self):
            self.inputs = None

        def reset(self):
            return None

        def update(self, dets, _frame, embs=None, masks=None):
            self.inputs = (dets.copy(), embs.copy(), masks.copy())
            return np.array([[*dets[0, :5], 1, dets[0, 5], dets[0, 6], 0]], dtype=np.float32)

    reid = FakeReID()
    tracker = FakeTracker()
    results = Results([], detector=FakeDetector(), reid=reid, tracker=tracker, verbose=False)
    results._iter_frames = lambda: iter([("frame.jpg", frame)])

    result = next(iter(results))

    np.testing.assert_array_equal(reid.boxes, raw_dets[:1])
    np.testing.assert_array_equal(tracker.inputs[0], raw_dets[:1])
    np.testing.assert_array_equal(tracker.inputs[1], np.ones((1, 4), dtype=np.float32))
    np.testing.assert_array_equal(tracker.inputs[2], raw_masks[:1])
    np.testing.assert_array_equal(result.detections, raw_dets[:1])
    np.testing.assert_array_equal(result.masks, raw_masks[:1])


def test_reid_cache_key_versions_crop_semantics():
    key = reid_cache_key("lmbn_n_duke.onnx", tracker_backend="cpp")

    assert key.startswith("cpp/lmbn_n_duke-onnx-")
    assert reid_preprocess_cache_key("resize").endswith(f"-cropv{REID_CROP_SCHEMA_VERSION}")


def test_reid_cache_key_fingerprints_weight_contents(tmp_path):
    first = tmp_path / "first" / "model.onnx"
    second = tmp_path / "second" / "model.onnx"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first model")
    second.write_bytes(b"second model")

    first_key = reid_cache_key(first)
    second_key = reid_cache_key(second)

    assert first_key != second_key
    assert "-w" in first_key
    assert first_key.startswith("python/model-onnx-")
