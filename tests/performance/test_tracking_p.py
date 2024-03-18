import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import WEIGHTS
import time


from numpy.testing import assert_allclose
from boxmot import (
    StrongSORT, BoTSORT, DeepOCSORT, OCSORT, BYTETracker, get_tracker_config, create_tracker,
)

MOTION_N_APPEARANCE_TRACKING_METHODS=['botsort', 'deepocsort', 'strongsort']
MOTION_ONLY_TRACKING_METHODS=['ocsort', 'bytetrack']

@pytest.mark.parametrize("tracker_type", MOTION_ONLY_TRACKING_METHODS)
def test_motion_tracker_update_time(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    
    n_runs = 100
    start = time.process_time()
    for i in range(0, n_runs):
        output = tracker.update(det, rgb)
    end = time.process_time()
    elapsed_time_per_iteration = (end - start) / n_runs
    
    max_allowed_time = 0.005
    
    assert elapsed_time_per_iteration < max_allowed_time, f"Tracking algorithms processing time exceeds the allowed limit:  {elapsed_time_per_iteration} > {max_allowed_time}"


@pytest.mark.parametrize("tracker_type", MOTION_N_APPEARANCE_TRACKING_METHODS)
def test_motion_n_appearance_tracker_update_time(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=False
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    
    n_runs = 100
    start = time.process_time()
    for i in range(0, n_runs):
        output = tracker.update(det, rgb)
    end = time.process_time()
    elapsed_time_per_iteration = (end - start) / n_runs
    
    max_allowed_time = 6
    
    assert elapsed_time_per_iteration < max_allowed_time, f"Tracking algorithms processing time exceeds the allowed limit:  {elapsed_time_per_iteration} > {max_allowed_time}"
