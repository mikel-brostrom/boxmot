from pathlib import Path

import numpy as np
import pytest

from boxmot import (
    DeepOcSort,
    OcSort,
    create_tracker,
    get_tracker_config,
)
from boxmot.trackers.deepocsort.deepocsort import (
    KalmanBoxTracker as DeepOCSortKalmanBoxTracker,
)
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.botsort_track import STrack as BotSortTrack
from boxmot.trackers.bytetrack.bytetrack import ByteTrack, STrack as ByteTrackTrack
from boxmot.trackers.ocsort.ocsort import KalmanBoxTracker as OCSortKalmanBoxTracker
from boxmot.trackers.sfsort.sfsort import SFSORT
from boxmot.utils.matching import iou_distance
from boxmot.utils import WEIGHTS
from tests.test_config import (
    ALL_TRACKERS,
    MOTION_N_APPEARANCE_TRACKING_METHODS,
    MOTION_N_APPEARANCE_TRACKING_NAMES,
    MOTION_ONLY_TRACKING_METHODS,
    PER_CLASS_TRACKERS,
)

# --- existing tests ---


@pytest.mark.parametrize("Tracker", MOTION_N_APPEARANCE_TRACKING_METHODS)
def test_motion_n_appearance_trackers_instantiation(Tracker):
    Tracker(
        reid_weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
        device="cpu",
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
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 400, 480, 0.82, 0], [425, 281, 576, 472, 0.72, 65]])

    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)


def test_dynamic_max_obs_based_on_max_age():
    max_age = 400
    ocsort = OcSort(max_age=max_age)
    assert ocsort.max_obs == (max_age + 5)


def create_kalman_box_tracker_ocsort(bbox, cls, det_ind, tracker):
    return OCSortKalmanBoxTracker(
        bbox,
        cls,
        det_ind,
        Q_xy_scaling=tracker.Q_xy_scaling,
        Q_s_scaling=tracker.Q_s_scaling,
    )


def create_kalman_box_tracker_deepocsort(bbox, cls, det_ind, tracker):
    det = np.concatenate([bbox, [cls, det_ind]])
    return DeepOCSortKalmanBoxTracker(
        det, Q_xy_scaling=tracker.Q_xy_scaling, Q_s_scaling=tracker.Q_s_scaling
    )


TRACKER_CREATORS = {
    OcSort: create_kalman_box_tracker_ocsort,
    DeepOcSort: create_kalman_box_tracker_deepocsort,
}


@pytest.mark.parametrize(
    "Tracker, init_args",
    [
        (OcSort, {}),
        (
            DeepOcSort,
            {
                "reid_weights": Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
                "device": "cpu",
                "half": True,
            },
        ),
    ],
)
def test_Q_matrix_scaling(Tracker, init_args):
    bbox = np.array([0, 0, 100, 100, 0.9])
    cls = 1
    det_ind = 0
    Q_xy_scaling = 0.05
    Q_s_scaling = 0.0005

    tracker = Tracker(Q_xy_scaling=Q_xy_scaling, Q_s_scaling=Q_s_scaling, **init_args)

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
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=True,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([
        [100, 100, 300, 250, 0.95,   0],  # class 0
        [400, 300, 550, 450, 0.90,  65],  # class 65
    ])
    embs = np.random.random(size=(2, 512))

    _ = tracker.update(det, rgb, embs)
    output = tracker.update(det, rgb, embs)
    assert output.shape == (2, 8)


@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_tracker_active_tracks(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=True,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([
        [100, 100, 300, 250, 0.95,   0],  # class 0
        [400, 300, 550, 450, 0.90,  65],  # class 65
    ])
    embs = np.random.random(size=(2, 512))

    tracker.update(det, rgb, embs)
    assert tracker.per_class_active_tracks[0], "No active tracks for class 0"
    assert tracker.per_class_active_tracks[65], "No active tracks for class 65"


def test_botsort_supports_obb_without_reid():
    tracker = BotSort(
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        with_reid=False,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[320, 240, 80, 40, 0.15, 0.95, 0]], dtype=np.float32)

    out1 = tracker.update(det, rgb)
    out2 = tracker.update(det, rgb)

    assert out1.shape[1] == 9
    assert out2.shape[1] == 9
    np.testing.assert_allclose(out2[0, :5], det[0, :5], atol=1e-2)


def test_botsort_obb_matching_uses_oriented_geometry():
    det = np.array([320, 240, 80, 40, 0.15, 0.95, 0, 0], dtype=np.float32)
    track_a = BotSortTrack(det, max_obs=10, is_obb=True)
    track_b = BotSortTrack(det, max_obs=10, is_obb=True)

    cost = iou_distance([track_a], [track_b], is_obb=True)

    assert cost.shape == (1, 1)
    assert cost[0, 0] < 1e-3


def test_botsort_obb_state_history_follows_rotation_without_flips():
    tracker = BotSort(
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        with_reid=False,
    )
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    angles = np.linspace(0.0, 6.1, 20, dtype=np.float32)

    for angle in angles:
        det = np.array([[320, 240, 90, 40, angle, 0.95, 0]], dtype=np.float32)
        tracker.update(det, rgb)

    assert tracker.active_tracks
    history = np.asarray(tracker.active_tracks[0].history_observations, dtype=np.float32)
    assert history.shape[1] == 8
    centers = history.reshape(-1, 4, 2).mean(axis=1)
    assert np.max(np.abs(centers - centers[0])) < 1e-2
    assert np.max(np.abs(history[-1] - history[0])) > 1.0


def test_bytetrack_supports_obb_outputs():
    tracker = ByteTrack()
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[320, 240, 80, 40, 0.15, 0.95, 0]], dtype=np.float32)

    out1 = tracker.update(det, rgb)
    out2 = tracker.update(det, rgb)

    assert out1.shape == (1, 9)
    assert out2.shape == (1, 9)
    np.testing.assert_allclose(out2[0, :5], det[0, :5], atol=1e-2)


def test_bytetrack_obb_matching_uses_oriented_geometry():
    det = np.array([320, 240, 80, 40, 0.15, 0.95, 0, 0], dtype=np.float32)
    track_a = ByteTrackTrack(det, max_obs=10, is_obb=True)
    track_b = ByteTrackTrack(det, max_obs=10, is_obb=True)

    cost = iou_distance([track_a], [track_b], is_obb=True)

    assert cost.shape == (1, 1)
    assert cost[0, 0] < 1e-3


def test_bytetrack_obb_state_history_follows_rotation_without_flips():
    tracker = ByteTrack(track_thresh=0.1, min_conf=0.01, match_thresh=0.99)
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    angles = np.linspace(0.0, 6.1, 20, dtype=np.float32)

    for angle in angles:
        det = np.array([[320, 240, 90, 40, angle, 0.95, 0]], dtype=np.float32)
        tracker.update(det, rgb)

    assert tracker.active_tracks
    history = np.asarray(tracker.active_tracks[0].history_observations, dtype=np.float32)
    assert history.shape[1] == 8
    centers = history.reshape(-1, 4, 2).mean(axis=1)
    assert np.max(np.abs(centers - centers[0])) < 1e-2
    assert np.max(np.abs(history[-1] - history[0])) > 1.0


def test_ocsort_obb_state_history_uses_state_corners():
    tracker = OcSort(det_thresh=0.1)
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    angles = np.linspace(0.0, 6.1, 20, dtype=np.float32)

    for angle in angles:
        det = np.array([[320, 240, 90, 40, angle, 0.95, 0]], dtype=np.float32)
        tracker.update(det, rgb)

    assert tracker.active_tracks
    history = np.asarray(tracker.active_tracks[0].history_observations, dtype=np.float32)
    assert history.shape[1] == 8
    assert np.max(np.abs(history[-1] - history[0])) > 1.0


def test_ocsort_obb_state_history_uses_post_update_state_center():
    tracker = OcSort(det_thresh=0.1, min_hits=1)
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)

    det1 = np.array([[100, 100, 90, 40, 0.0, 0.95, 0]], dtype=np.float32)
    det2 = np.array([[102, 102, 90, 40, 1.0, 0.95, 0]], dtype=np.float32)

    tracker.update(det1, rgb)
    tracker.update(det2, rgb)

    assert tracker.active_tracks
    track = tracker.active_tracks[0]
    assert len(track.history_observations) >= 1

    history_center = (
        np.asarray(track.history_observations[-1], dtype=np.float32).reshape(4, 2).mean(axis=0)
    )
    state_center = np.asarray(track.get_state()[0][:2], dtype=np.float32)
    np.testing.assert_allclose(history_center, state_center, atol=0.75)


def test_sfsort_obb_state_history_uses_state_corners():
    tracker = SFSORT()
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    angles = np.linspace(0.0, 6.1, 20, dtype=np.float32)

    for angle in angles:
        det = np.array([[320, 240, 90, 40, angle, 0.95, 0]], dtype=np.float32)
        tracker.update(det, rgb)

    assert tracker.active_tracks
    history = np.asarray(tracker.active_tracks[0].history_observations, dtype=np.float32)
    assert history.shape[1] == 8
    assert np.max(np.abs(history[-1] - history[0])) > 1.0


def test_sfsort_supports_obb_outputs():
    tracker = SFSORT()
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[320, 240, 80, 40, 0.15, 0.95, 0]], dtype=np.float32)

    out1 = tracker.update(det, rgb)
    out2 = tracker.update(det, rgb)

    assert out1.shape == (1, 9)
    assert out2.shape == (1, 9)
    np.testing.assert_allclose(out2[0, :5], det[0, :5], atol=1e-2)


def test_sfsort_obb_plotting_draws_tracks():
    tracker = SFSORT()
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    det = np.array([[128, 128, 60, 30, 0.3, 0.95, 0]], dtype=np.float32)

    tracker.update(det, img)
    rendered = tracker.plot_results(img.copy(), show_trajectories=True)

    assert np.any(rendered != 0)


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
@pytest.mark.parametrize("dets", [None, np.array([])])
def test_tracker_with_no_detections(tracker_type, dets):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    embs = np.random.random(size=(0, 512))

    output = tracker.update(dets, rgb, embs)
    assert output.size == 0, "Output should be empty when no detections are provided"


@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_isolation(tracker_type):
    tracker = create_tracker(
        tracker_type,
        get_tracker_config(tracker_type),
        WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=True,
    )
    det = np.array(
        [
            [100, 100, 150, 150, 0.9, 1],
            [102, 102, 152, 152, 0.9, 2],
        ]
    )
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    embs = np.random.rand(2, 512)
    out = tracker.update(det, rgb, embs)
    ids = set(out[:, 1].tolist())
    assert len(ids) == 2, "Each class should get a separate track even if overlapping"


@pytest.mark.parametrize("tracker_type", MOTION_N_APPEARANCE_TRACKING_NAMES)
def test_emb_trackers_requires_embeddings(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )
    det = np.array([[10, 10, 20, 20, 0.7, 0]])
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        tracker.update(det, rgb, np.random.rand(2, 512))


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_invalid_det_array_shape(tracker_type):
    tracker = create_tracker(
        tracker_type,
        get_tracker_config(tracker_type),
        WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    embs = np.random.rand(2, 512)
    bad_det = np.random.rand(2, 5)
    with pytest.raises(AssertionError):
        tracker.update(bad_det, img, embs)


# def test_get_tracker_config_invalid_name():
#     """Requesting config for an unknown tracker should raise a KeyError."""
#     with pytest.raises(KeyError):
#         get_tracker_config("not_a_tracker")


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_track_id_stable_over_frames(tracker_type):
    """
    If the same detection appears in successive frames,
    the tracker should assign the same track ID.
    """
    cfg = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=cfg,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    det = np.array([[50, 50, 100, 100, 0.95, 3]])
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)

    # choose embedding only if needed
    if tracker_type in MOTION_N_APPEARANCE_TRACKING_NAMES:
        embs = np.random.rand(1, 512)
        out1 = tracker.update(det, rgb, embs)
        out2 = tracker.update(det, rgb, embs)
    else:
        out1 = tracker.update(det, rgb)
        out2 = tracker.update(det, rgb)

    assert out1.shape == out2.shape == (1, 8), "Unexpected output shape"
    # track ID is at column 1
    assert out1[0, 4] == out2[0, 4], "Track ID should remain the same across frames"


def test_create_tracker_invalid_tracker_name():
    """Creating a tracker with an unknown name should raise a ValueError."""
    with pytest.raises(ValueError, match="Unknown tracker type: 'nonexistent_tracker'"):
        create_tracker(
            tracker_type="nonexistent_tracker",
            tracker_config=get_tracker_config("botsort"),
            reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
            device="cpu",
            half=False,
            per_class=False,
        )
