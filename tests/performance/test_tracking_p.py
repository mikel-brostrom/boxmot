import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import WEIGHTS


from numpy.testing import assert_allclose
from boxmot import (
    StrongSORT, BoTSORT, DeepOCSORT, OCSORT, BYTETracker, get_tracker_config, create_tracker,
)

ALL_TRACKERS=['botsort', 'deepocsort', 'ocsort', 'bytetrack', 'strongsort']


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_tracker_update_time(tracker_type):
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
    elapsed_time_per_interation = (end - start) / n_runs
    
    max_allowed_time = 0.1
    assert elapsed_time_per_interation < max_allowed_time, "Tracking algorithms processing time exceeds the allowed limit"
