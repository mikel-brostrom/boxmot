import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import WEIGHTS
import time
import subprocess

from numpy.testing import assert_allclose
from boxmot import (
    StrongSort, BotSort, DeepOcSort, OcSort, ByteTrack, ImprAssocTrack, get_tracker_config, create_tracker,
)
from tests.test_config import MOTION_ONLY_TRACKING_NAMES, MOTION_N_APPEARANCE_TRACKING_NAMES


@pytest.mark.parametrize("tracker_type", MOTION_ONLY_TRACKING_NAMES)
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

    rgb = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    
    n_runs = 100

    # Warm-up iteration to ensure initialization overhead is not measured
    tracker.update(det, rgb)
    
    start = time.perf_counter()
    for _ in range(n_runs):
        tracker.update(det, rgb)
    end = time.perf_counter()
    
    elapsed_time_per_iteration = (end - start) / n_runs
    fps = 1.0 / elapsed_time_per_iteration
    
    # Print FPS for each tracker type
    print(f"Tracker type: {tracker_type} - FPS: {fps:.2f}")
    result = subprocess.run(
        "cat /proc/cpuinfo | grep 'model name' | head -1",
        shell=True,
        capture_output=True,
        text=True
    )
    print(result.stdout.strip())
    max_allowed_time = 0.005  # maximum allowed time per iteration in seconds
    
    assert elapsed_time_per_iteration < max_allowed_time, (
        f"Tracking algorithm's processing time per iteration ({elapsed_time_per_iteration:.6f}s) "
        f"exceeds the allowed limit of {max_allowed_time}s."
    )


@pytest.mark.parametrize("tracker_type", MOTION_N_APPEARANCE_TRACKING_NAMES)
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

    rgb = np.random.randint(0, 255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])
    
    n_runs = 100

    # Warm-up iteration to avoid initialization overhead in timing
    tracker.update(det, rgb)
    
    start = time.perf_counter()
    for _ in range(n_runs):
        tracker.update(det, rgb)
    end = time.perf_counter()
    
    elapsed_time_per_iteration = (end - start) / n_runs
    fps = 1.0 / elapsed_time_per_iteration
    
    # Print FPS for each tracker type
    print(f"Tracker type: {tracker_type} - FPS: {fps:.2f}")
    max_allowed_time = 6  # maximum allowed time per iteration in seconds
    
    assert elapsed_time_per_iteration < max_allowed_time, (
        f"Tracking algorithm's processing time per iteration ({elapsed_time_per_iteration:.4f}s) "
        f"exceeds the allowed limit of {max_allowed_time}s."
    )