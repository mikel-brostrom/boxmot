from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxmot import (
    DeepOcSort,
    OcSort,
    StrongSort,
    create_tracker,
    get_tracker_config,
)
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.trackers.deepocsort.deepocsort import (
    KalmanBoxTracker as DeepOCSortKalmanBoxTracker,
)
from boxmot.trackers.botsort.botsort import BotSort
from boxmot.trackers.botsort.botsort_track import STrack as BotSortTrack
from boxmot.trackers.bytetrack.bytetrack import ByteTrack, STrack as ByteTrackTrack
from boxmot.trackers.ocsort.ocsort import KalmanBoxTracker as OCSortKalmanBoxTracker
from boxmot.trackers.sfsort.sfsort import SFSORT
from boxmot.trackers.strongsort.sort import iou_matching as strongsort_iou_matching
from boxmot.trackers.strongsort.sort.detection import Detection as StrongSortDetection
from boxmot.trackers.strongsort.sort.track import (
    Track as StrongSortTrack,
    TrackState as StrongSortTrackState,
)
from boxmot.utils import WEIGHTS
from boxmot.utils.iou import AssociationFunction
from boxmot.utils.matching import iou_distance
from tests.test_config import (
    ALL_TRACKERS,
    MOTION_N_APPEARANCE_TRACKING_METHODS,
    MOTION_N_APPEARANCE_TRACKING_NAMES,
    MOTION_ONLY_TRACKING_METHODS,
    PER_CLASS_TRACKERS,
)

# --- existing tests ---


class DummyCMC:
    def __init__(self, warp: np.ndarray | None = None):
        self.warp = (
            np.eye(2, 3, dtype=np.float32)
            if warp is None
            else np.asarray(warp, dtype=np.float32)
        )
        self.calls: list[np.ndarray | None] = []

    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        call = None if dets is None else np.asarray(dets, dtype=np.float32).copy()
        self.calls.append(call)
        return self.warp.copy()


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

    output = np.empty((0,))
    for _ in range(10):
        output = tracker.update(det, rgb)
        if output.shape == (2, 8):
            break

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

    output = np.empty((0,))
    for _ in range(10):
        output = tracker.update(det, rgb, embs)
        if output.shape == (2, 8):
            break
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


def test_strongsort_per_class_outputs_unique_ids():
    tracker = StrongSort(
        reid_weights=Path(WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt"),
        device="cpu",
        half=False,
        per_class=True,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([
        [100, 100, 300, 250, 0.95, 0],
        [400, 300, 550, 450, 0.90, 65],
    ])
    embs = np.random.random(size=(2, 512)).astype(np.float32)

    output = np.empty((0,))
    for _ in range(10):
        output = tracker.update(det, rgb, embs)
        if output.shape == (2, 8):
            break

    assert output.shape == (2, 8)
    assert len(set(output[:, 4].astype(int))) == 2


def test_strongsort_track_init_is_not_special_cased_in_ci(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_JOB", "unit-tests")

    detection = StrongSortDetection(
        tlwh=np.array([10, 10, 20, 20], dtype=np.float32),
        conf=0.95,
        cls=0,
        det_ind=0,
        feat=np.ones(4, dtype=np.float32),
    )
    track = StrongSortTrack(
        detection=detection,
        id=1,
        n_init=3,
        max_age=30,
        ema_alpha=0.9,
        max_obs=50,
    )

    assert track.state == StrongSortTrackState.Tentative


def test_strongsort_iou_cost_uses_shared_iou_batch(monkeypatch):
    calls = {"count": 0}
    original_iou_batch = AssociationFunction.iou_batch

    def wrapped_iou_batch(bboxes1, bboxes2):
        calls["count"] += 1
        return original_iou_batch(bboxes1, bboxes2)

    monkeypatch.setattr(
        AssociationFunction,
        "iou_batch",
        staticmethod(wrapped_iou_batch),
    )

    detection = StrongSortDetection(
        tlwh=np.array([10, 10, 20, 20], dtype=np.float32),
        conf=0.95,
        cls=0,
        det_ind=0,
        feat=np.ones(4, dtype=np.float32),
    )
    track = StrongSortTrack(
        detection=detection,
        id=1,
        n_init=3,
        max_age=30,
        ema_alpha=0.9,
        max_obs=50,
    )

    cost = strongsort_iou_matching.iou_cost([track], [detection])

    expected_iou = original_iou_batch(
        np.asarray([track.xyxy], dtype=np.float32),
        np.asarray([[10, 10, 30, 30]], dtype=np.float32),
    )
    assert calls["count"] == 1
    np.testing.assert_allclose(cost, 1.0 - expected_iou)


def test_strongsort_reset_clears_per_class_state():
    tracker = StrongSort(
        reid_weights=Path(WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt"),
        device="cpu",
        half=False,
        per_class=True,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([
        [100, 100, 300, 250, 0.95, 0],
        [400, 300, 550, 450, 0.90, 65],
    ])
    embs = np.random.random(size=(2, 512)).astype(np.float32)

    for _ in range(3):
        tracker.update(det, rgb, embs)

    assert tracker.per_class_active_tracks[0]
    assert tracker.per_class_active_tracks[65]

    tracker.reset()

    assert tracker.frame_count == 0
    assert tracker.active_tracks == []
    assert tracker.tracker.tracks == []
    assert tracker._next_track_id_value == 1
    assert all(not tracks for tracks in tracker.per_class_active_tracks.values())


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


def test_botsort_obb_cmc_uses_enclosing_aabb_boxes():
    tracker = BotSort(
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        with_reid=False,
    )
    tracker.cmc = DummyCMC()

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[320, 240, 80, 40, 0.35, 0.95, 0]], dtype=np.float32)

    tracker.update(det, rgb)

    assert len(tracker.cmc.calls) == 1
    assert tracker.cmc.calls[0] is not None
    np.testing.assert_allclose(
        tracker.cmc.calls[0][0],
        BotSortTrack.obb_to_xyxy(det[0, :5]),
        atol=1e-4,
    )


def test_botsort_obb_cmc_warps_track_state():
    det = np.array([320, 240, 80, 40, 0.15, 0.95, 0, 0], dtype=np.float32)
    track = BotSortTrack(det, max_obs=10, is_obb=True)
    track.activate(KalmanFilterXYWH(ndim=5), frame_id=1)

    BotSortTrack.multi_gmc_obb(
        [track],
        np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -6.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(
        track.xywha[:2], np.array([332.0, 234.0], dtype=np.float32), atol=1e-4
    )
    np.testing.assert_allclose(track.xywha[2:], det[2:5], atol=1e-4)


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


def test_sfsort_obb_angle_update_uses_damping():
    tracker = SFSORT(obb_theta_damping=0.8)
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)

    det1 = np.array([[320, 240, 80, 40, 0.00, 0.95, 0]], dtype=np.float32)
    det2 = np.array([[320, 240, 80, 40, 0.40, 0.95, 0]], dtype=np.float32)

    out1 = tracker.update(det1, rgb)
    out2 = tracker.update(det2, rgb)

    assert out1.shape == (1, 9)
    assert out2.shape == (1, 9)
    assert int(out2[0, 5]) == int(out1[0, 5])

    measured_delta = abs(float(det2[0, 4] - det1[0, 4]))
    tracked_delta = abs(float(out2[0, 4] - out1[0, 4]))
    assert 0.0 < tracked_delta < measured_delta


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
    out = np.empty((0,))
    for _ in range(10):
        out = tracker.update(det, rgb, embs)
        if out.shape == (2, 8):
            break
    assert out.shape == (2, 8)
    assert set(out[:, 6].astype(int).tolist()) == {1, 2}


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

    def update(tracker):
        if tracker_type in MOTION_N_APPEARANCE_TRACKING_NAMES:
            return tracker.update(det, rgb, np.random.rand(1, 512))
        return tracker.update(det, rgb)

    # Warm up until the track is confirmed (handles trackers with n_init > 1)
    out = np.empty((0,))
    for _ in range(10):
        out = update(tracker)
        if out.shape == (1, 8):
            break

    assert out.shape == (1, 8), "Track was not confirmed after warm-up"
    track_id = out[0, 4]

    out2 = update(tracker)
    assert out2.shape == (1, 8), "Unexpected output shape on second frame"
    assert out2[0, 4] == track_id, "Track ID should remain the same across frames"


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
