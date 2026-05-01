"""FPS measurement with increasing number of detections per frame.

Sweeps over a range of detection counts and reports tracker FPS for each
(tracker_type, n_detections) combination.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config
from boxmot.utils import WEIGHTS
from tests.test_config import (
    MOTION_N_APPEARANCE_TRACKING_NAMES,
    MOTION_ONLY_TRACKING_NAMES,
)

DETECTION_COUNTS = [1, 5, 10, 20, 50, 100]
N_WARMUP = 5
N_RUNS = 50
IMG_H, IMG_W = 640, 640


def _make_detections(n: int) -> np.ndarray:
    """Generate *n* random non-overlapping-ish AABB detections in a 640×640 frame."""
    rng = np.random.default_rng(seed=42)
    x1 = rng.uniform(0, IMG_W - 50, size=n)
    y1 = rng.uniform(0, IMG_H - 50, size=n)
    x2 = x1 + rng.uniform(20, 50, size=n)
    y2 = y1 + rng.uniform(20, 50, size=n)
    conf = rng.uniform(0.5, 1.0, size=n)
    cls = rng.integers(0, 80, size=n).astype(float)
    return np.stack([x1, y1, x2, y2, conf, cls], axis=1).astype(np.float32)


def _measure_fps(tracker, dets: np.ndarray, img: np.ndarray) -> float:
    """Return FPS after *N_WARMUP* warm-up frames and *N_RUNS* timed frames."""
    for _ in range(N_WARMUP):
        tracker.update(dets, img)

    start = time.perf_counter()
    for _ in range(N_RUNS):
        tracker.update(dets, img)
    elapsed = time.perf_counter() - start

    return N_RUNS / elapsed


@pytest.mark.parametrize("n_dets", DETECTION_COUNTS)
@pytest.mark.parametrize("tracker_type", MOTION_ONLY_TRACKING_NAMES)
def test_motion_only_fps_vs_detections(tracker_type: str, n_dets: int):
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=get_tracker_config(tracker_type),
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    img = np.random.randint(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    dets = _make_detections(n_dets)

    fps = _measure_fps(tracker, dets, img)
    print(f"\n[motion-only] tracker={tracker_type:12s}  n_dets={n_dets:4d}  fps={fps:.1f}")


@pytest.mark.parametrize("n_dets", DETECTION_COUNTS)
@pytest.mark.parametrize("tracker_type", MOTION_N_APPEARANCE_TRACKING_NAMES)
def test_motion_n_appearance_fps_vs_detections(tracker_type: str, n_dets: int):
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=get_tracker_config(tracker_type),
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    img = np.random.randint(0, 255, size=(IMG_H, IMG_W, 3), dtype=np.uint8)
    dets = _make_detections(n_dets)

    fps = _measure_fps(tracker, dets, img)
    print(f"\n[motion+app]  tracker={tracker_type:12s}  n_dets={n_dets:4d}  fps={fps:.1f}")
