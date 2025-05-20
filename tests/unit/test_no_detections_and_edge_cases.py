import pytest
import numpy as np
from pathlib import Path

from boxmot import (
    get_tracker_config, create_tracker,
    OcSort, DeepOcSort, ByteTrack, ImprAssocTrack
)
from boxmot.utils import WEIGHTS
from tests.test_config import ALL_TRACKERS, PER_CLASS_TRACKERS, MOTION_ONLY_TRACKING_METHODS

# No detections
@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
@pytest.mark.parametrize("dets", [None, np.array([])])
def test_tracker_with_no_detections(tracker_type, dets):
    cfg = get_tracker_config(tracker_type)
    tr = create_tracker(
        tracker_type, cfg,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu", half=False, per_class=False,
    )
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    embs = np.random.random((2, 512))
    out = tr.update(dets, rgb, embs)
    assert out.size == 0

# ID consistency
@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_id_consistency_over_two_frames(tracker_type):
    cfg = get_tracker_config(tracker_type)
    tr = create_tracker(
        tracker_type, cfg,
        WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu", half=False, per_class=False,
    )
    det1 = np.array([[50, 50, 100, 100, 0.9, 0]])
    out1 = tr.update(det1, np.zeros((640, 640, 3), np.uint8))
    id1 = out1[0, 1]

    det2 = np.array([[52, 52, 102, 102, 0.88, 0]])
    out2 = tr.update(det2, np.zeros((640, 640, 3), np.uint8))
    assert out2[0, 1] == id1

# Track deletion
@pytest.mark.parametrize("Tracker", [OcSort, DeepOcSort, ByteTrack, ImprAssocTrack])
def test_track_deletion_after_max_age(Tracker):
    max_age = 3
    kwargs = {}
    if Tracker is DeepOcSort:
        kwargs = {
            "reid_weights": WEIGHTS / "osnet_x0_25_msmt17.pt",
            "device": "cpu", "half": True
        }
    if Tracker is ByteTrack:
        kwargs = {"track_thresh": 0.5, "match_thresh": 0.1}

    tr = Tracker(max_age=max_age, **kwargs)
    det = np.array([[10, 10, 50, 50, 0.95, 0]])
    out = tr.update(det, np.zeros((640, 640, 3), np.uint8))
    assert out.shape[0] == 1

    # simulate misses
    for _ in range(max_age):
        out = tr.update(None, np.zeros((640, 640, 3), np.uint8))
        assert out.shape[0] == 1
    out = tr.update(None, np.zeros((640, 640, 3), np.uint8))
    assert out.shape[0] == 0

# ByteTrack threshold logic
def test_bytetrack_low_high_threshold_logic():
    from boxmot import ByteTrack
    ht, lt = 0.6, 0.2
    tr = ByteTrack(track_thresh=ht, match_thresh=lt)
    det1 = np.array([[20, 20, 60, 60, 0.8, 0]])
    out1 = tr.update(det1, np.zeros((640, 640, 3), np.uint8))
    assert out1.shape[0] == 1

    det2 = np.array([[22, 22, 62, 62, 0.4, 0]])
    out2 = tr.update(det2, np.zeros((640, 640, 3), np.uint8))
    assert getattr(tr, "_unconfirmed_tracks", None)
    assert out2.shape[0] == 0

# Motion-only image format error
@pytest.mark.parametrize("Tracker", MOTION_ONLY_TRACKING_METHODS)
def test_motion_only_tracker_image_format(Tracker):
    tr = Tracker()
    gray = np.zeros((640, 640), np.uint8)
    with pytest.raises((ValueError, AssertionError)):
        tr.update(np.array([[0,0,10,10,0.5,0]]), gray)

# Prediction drift
def test_prediction_drift_with_kalman():
    tr = OcSort(max_age=2)
    det = np.array([[30,30,80,80,0.9,0]])
    o1 = tr.update(det, np.zeros((640,640,3), np.uint8))
    last = o1[0, 2:6].copy()
    o2 = tr.update(None, np.zeros((640,640,3), np.uint8))
    assert not np.allclose(o2[0,2:6], last)

# Invalid det shape
@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_invalid_det_array_shape(tracker_type):
    tr = create_tracker(
        tracker_type, get_tracker_config(tracker_type),
        WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu", half=False, per_class=False,
    )
    bad = np.random.rand(2, 5)
    with pytest.raises(ValueError):
        tr.update(bad, np.zeros((640, 640, 3), np.uint8))
