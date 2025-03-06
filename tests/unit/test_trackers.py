import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import WEIGHTS


from numpy.testing import assert_allclose
from boxmot import (
    StrongSort, BotSort, DeepOcSort, OcSort, ByteTrack, ImprAssocTrack, get_tracker_config, create_tracker,
)

from boxmot.trackers.ocsort.ocsort import KalmanBoxTracker as OCSortKalmanBoxTracker
from boxmot.trackers.deepocsort.deepocsort import KalmanBoxTracker as DeepOCSortKalmanBoxTracker
from tests.test_config import MOTION_ONLY_TRACKING_METHODS, MOTION_N_APPEARANCE_TRACKING_METHODS, ALL_TRACKERS, PER_CLASS_TRACKERS


@pytest.mark.parametrize("Tracker", MOTION_N_APPEARANCE_TRACKING_METHODS)
def test_motion_n_appearance_trackers_instantiation(Tracker):
    Tracker(
        reid_weights=Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        device='cpu',
        half=True,
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
    det = np.array([[144, 212, 400, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.72, 65]])

    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)  # two inputs should give two outputs
    
    
def test_dynamic_max_obs_based_on_max_age():
    max_age = 400
    ocsort = OcSort(
        max_age=max_age
    )

    assert ocsort.max_obs == (max_age + 5)


def create_kalman_box_tracker_ocsort(bbox, cls, det_ind, tracker):
    return OCSortKalmanBoxTracker(
        bbox,
        cls,
        det_ind,
        Q_xy_scaling=tracker.Q_xy_scaling,
        Q_s_scaling=tracker.Q_s_scaling
    )


def create_kalman_box_tracker_deepocsort(bbox, cls, det_ind, tracker):
    # DeepOCSort KalmanBoxTracker expects input in different format than OCSort
    det = np.concatenate([bbox, [cls, det_ind]]) 
    return DeepOCSortKalmanBoxTracker(
        det,
        Q_xy_scaling=tracker.Q_xy_scaling,
        Q_s_scaling=tracker.Q_s_scaling
    )


TRACKER_CREATORS = {
    OcSort: create_kalman_box_tracker_ocsort,
    DeepOcSort: create_kalman_box_tracker_deepocsort,
}


@pytest.mark.parametrize("Tracker, init_args", [
    (OcSort, {}),
    (DeepOcSort, {
        'reid_weights': Path(WEIGHTS / 'osnet_x0_25_msmt17.pt'),
        'device': 'cpu',
        'half': True
    }),
])
def test_Q_matrix_scaling(Tracker, init_args):
    bbox = np.array([0, 0, 100, 100, 0.9])
    cls = 1
    det_ind = 0
    Q_xy_scaling = 0.05
    Q_s_scaling = 0.0005

    tracker = Tracker(
        Q_xy_scaling=Q_xy_scaling, 
        Q_s_scaling=Q_s_scaling,
        **init_args
    )

    create_kalman_box_tracker = TRACKER_CREATORS[Tracker]
    kalman_box_tracker = create_kalman_box_tracker(bbox, cls, det_ind, tracker)

    assert kalman_box_tracker.kf.Q[4, 4] == Q_xy_scaling, "Q_xy scaling incorrect for x' velocity"
    assert kalman_box_tracker.kf.Q[5, 5] == Q_xy_scaling, "Q_xy scaling incorrect for y' velocity"
    assert kalman_box_tracker.kf.Q[6, 6] == Q_s_scaling, "Q_s scaling incorrect for s' (scale) velocity"


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
                    [425, 281, 576, 472, 0.72, 65]])
    embs = np.random.random(size=(2, 512))

    output = tracker.update(det, rgb, embs)
    output = tracker.update(det, rgb, embs)
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
                    [425, 281, 576, 472, 0.72, 65]])
    embs = np.random.random(size=(2, 512))

    tracker.update(det, rgb, embs)

    # Check that tracks are created under the class tracks
    assert tracker.per_class_active_tracks[0], "No active tracks for class 0"
    assert tracker.per_class_active_tracks[65], "No active tracks for class 65"


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
@pytest.mark.parametrize("dets", [None, np.array([])])
def test_tracker_with_no_detections(tracker_type, dets):
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
    embs = np.random.random(size=(2, 512))
    
    output = tracker.update(dets, rgb, embs)
    assert output.size == 0, "Output should be empty when no detections are provided"