import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import WEIGHTS

from boxmot import get_tracker_config, create_tracker
from tests.test_config import ALL_TRACKERS

@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_tracker_output_size(tracker_type):
    cfg = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=cfg,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu", half=False, per_class=False,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    dets = np.array([[144, 212, 400, 480, 0.82, 0],
                     [425, 281, 576, 472, 0.72, 65]])

    out = tracker.update(dets, rgb)
    assert out.shape == (2, 8)
