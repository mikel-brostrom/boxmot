import pytest
import numpy as np
from pathlib import Path
from boxmot import get_tracker_config, create_tracker
from boxmot.utils import WEIGHTS
from tests.test_config import PER_CLASS_TRACKERS

@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_tracker_output_size(tracker_type):
    cfg = get_tracker_config(tracker_type)
    tr = create_tracker(
        tracker_type, cfg,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu", half=False, per_class=True,
    )
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                     [425, 281, 576, 472, 0.72, 65]])
    embs = np.random.random((2, 512))

    _ = tr.update(dets, rgb, embs)  # first frame
    out = tr.update(dets, rgb, embs)
    assert out.shape == (2, 8)

@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_tracker_active_tracks(tracker_type):
    cfg = get_tracker_config(tracker_type)
    tr = create_tracker(
        tracker_type, cfg,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu", half=False, per_class=True,
    )
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                     [425, 281, 576, 472, 0.72, 65]])
    embs = np.random.random((2, 512))

    tr.update(dets, rgb, embs)
    assert tr.per_class_active_tracks[0]
    assert tr.per_class_active_tracks[65]

@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_requires_embeddings(tracker_type):
    cfg = get_tracker_config(tracker_type)
    tr = create_tracker(
        tracker_type, cfg,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu", half=False, per_class=True,
    )
    det = np.array([[10, 10, 20, 20, 0.7, 0]])
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)

    with pytest.raises(TypeError):
        tr.update(det, rgb)
    with pytest.raises(ValueError):
        tr.update(det, rgb, np.random.rand(3, 512))
