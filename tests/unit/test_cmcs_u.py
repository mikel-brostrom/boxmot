import cv2
import numpy as np
import pytest

from boxmot.motion.cmc.ecc import ECC
from boxmot.motion.cmc.orb import ORB
from boxmot.motion.cmc.sift import SIFT
from boxmot.motion.cmc.sof import SOF


# Fixture for creating CMC objects
@pytest.fixture
def cmc_object(request):
    cmc_class = request.param
    return cmc_class()


# Define the test function
@pytest.mark.parametrize("cmc_object", [ECC, ORB, SIFT, SOF], indirect=True)
def test_cmc_apply(cmc_object):
    # Create dummy images and detections
    prev_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = np.array([[0, 0, 10, 10]])
    # Apply the CMC algorithm
    result = cmc_object.apply(prev_img, dets)
    # Assert the type of result
    assert isinstance(result, np.ndarray)


# Test preprocessing function
@pytest.mark.parametrize("cmc_object", [ECC, ORB, SIFT, SOF], indirect=True)
def test_cmc_preprocess(cmc_object):
    # Create a dummy image
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    processed_img = cmc_object.preprocess(img)
    # Assert the shape of the processed image, scale is 0.1 by default
    assert processed_img.shape == (30, 30)


# Test apply function with empty detections
@pytest.mark.parametrize("cmc_object", [ECC, ORB, SIFT, SOF], indirect=True)
def test_cmc_apply_empty_detections(cmc_object):
    # Create dummy images and empty detections
    prev_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = np.array([])
    # Apply the CMC algorithm
    result = cmc_object.apply(prev_img, dets)
    # Assert that result is an identity matrix
    assert np.array_equal(result, np.eye(2, 3, dtype=np.float32))


def test_sof_uses_detection_mask_for_keypoints(monkeypatch):
    captured_masks = []

    def fake_good_features(img, mask=None, **kwargs):
        captured_masks.append(None if mask is None else mask.copy())
        return np.array(
            [[[8.0, 8.0]], [[10.0, 8.0]], [[8.0, 10.0]], [[10.0, 10.0]]],
            dtype=np.float32,
        )

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", fake_good_features)
    monkeypatch.setattr(cv2, "cornerSubPix", lambda *args, **kwargs: None)

    sof = SOF(scale=0.15)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = np.array([[0, 0, 50, 50]], dtype=np.float32)

    sof.apply(img, dets)

    assert captured_masks
    mask = captured_masks[0]
    assert mask is not None
    assert mask[3, 3] == 0
    assert mask[10, 10] == 255


def test_sof_rejects_weak_ransac_estimate(monkeypatch):
    sof = SOF(scale=1.0, min_inliers=3, min_inlier_ratio=0.75)
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    prev_keypoints = np.array(
        [[[1.0, 1.0]], [[1.0, 5.0]], [[5.0, 1.0]], [[5.0, 5.0]]],
        dtype=np.float32,
    )
    next_keypoints = prev_keypoints + np.array([[[10.0, 0.0]]], dtype=np.float32)

    sof.prev_frame = sof.preprocess(img)
    sof.prev_keypoints = prev_keypoints.copy()
    sof.initialized = True

    monkeypatch.setattr(
        cv2,
        "calcOpticalFlowPyrLK",
        lambda *args, **kwargs: (next_keypoints, np.ones((4, 1), dtype=np.uint8), None),
    )
    monkeypatch.setattr(
        cv2,
        "estimateAffinePartial2D",
        lambda *args, **kwargs: (
            np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.array([[1], [0], [0], [0]], dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        cv2,
        "goodFeaturesToTrack",
        lambda *args, **kwargs: prev_keypoints.copy(),
    )

    result = sof.apply(img, np.empty((0, 4), dtype=np.float32))

    assert np.array_equal(result, np.eye(2, 3, dtype=np.float32))
