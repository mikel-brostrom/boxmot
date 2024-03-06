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
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    processed_img = cmc_object.preprocess(img)
    # Assert the shape of the processed image, scale is 0.1 by default
    assert processed_img.shape == (10, 10)


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
