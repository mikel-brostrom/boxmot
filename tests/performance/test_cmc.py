import cv2
import time
import numpy as np
import pytest
from boxmot.motion.cmc.ecc import ECC
from boxmot.motion.cmc.orb import ORB
from boxmot.motion.cmc.sift import SIFT
from boxmot.motion.cmc.sof import SOF
from boxmot.utils import ROOT



# Fixture for creating CMC objects
@pytest.fixture
def cmc_object(request):
    cmc_class = request.param
    return cmc_class()


# Define the test function
@pytest.mark.parametrize("cmc_object", [ECC, ORB, SIFT, SOF], indirect=True)
def test_cmc_apply(cmc_object):

    # Create dummy images and detections
    curr_img = cv2.imread(str(ROOT / 'assets/MOT17-mini/train/MOT17-04-FRCNN/img1/000005.jpg'))
    prev_img = cv2.imread(str(ROOT / 'assets/MOT17-mini/train/MOT17-04-FRCNN/img1/000001.jpg'))
    
    print(curr_img.shape)
    print(prev_img.shape)
    
    dets = np.array([[0, 0, 10, 10]])

    n_runs = 100
    start = time.process_time()
    for i in range(0, n_runs):
        warp_matrix = cmc_object.apply(prev_img, dets)
        warp_matrix = cmc_object.apply(curr_img, dets)
    end = time.process_time()
    elapsed_time_per_interation = (end - start) / n_runs

    # Define a threshold for the maximum allowed time
    max_allowed_time = 0.1

    # Assert that the elapsed time is within the allowed limit
    assert elapsed_time_per_interation < max_allowed_time, "CMC algorithm processing time exceeds the allowed limit"