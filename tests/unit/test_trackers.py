import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import WEIGHTS


from numpy.testing import assert_allclose
from boxmot import (
    StrongSORT, BoTSORT, DeepOCSORT, OCSORT, BYTETracker, get_tracker_config, create_tracker,
)


MOTION_ONLY_TRACKING_METHODS=[OCSORT, BYTETracker]
MOTION_N_APPEARANCE_TRACKING_METHODS=[StrongSORT, BoTSORT, DeepOCSORT]
ALL_TRACKERS=['botsort', 'deepocsort', 'ocsort', 'bytetrack', 'strongsort']
PER_CLASS_TRACKERS=['botsort', 'deepocsort', 'ocsort', 'bytetrack']


@pytest.mark.parametrize("Tracker", MOTION_N_APPEARANCE_TRACKING_METHODS)
def test_motion_n_appearance_trackers_instantiation(Tracker):
    Tracker(
        model_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        fp16=True,
    )


@pytest.mark.parametrize("Tracker", MOTION_ONLY_TRACKING_METHODS)
def test_motion_only_trackers_instantiation(Tracker):
    Tracker()


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_tracker_output_size(tracker_type):
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

    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    
    
def test_dynamic_max_obs_based_on_max_age():
    max_age = 400
    ocsort = OCSORT(
        max_age=max_age
    )

    assert ocsort.max_obs == (max_age + 5)


@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_tracker_output_size(tracker_type):

    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=True
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    output = tracker.update(det, rgb)
    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs


@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_tracker_active_tracks(tracker_type):

    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        device='cpu',
        half=False,
        per_class=True
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    tracker.update(det, rgb)

    # check that tracks are created under the class tracks
    assert tracker.per_class_active_tracks[0] and tracker.per_class_active_tracks[65]

