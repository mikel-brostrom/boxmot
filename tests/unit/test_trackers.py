import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

import boxmot.trackers.registry as tracker_registry
from boxmot.engine.tuning.search_space import flatten_yaml_config
from boxmot.motion.kalman_filters.xywh import KalmanFilterXYWH
from boxmot.reid.core import ReID
from boxmot.trackers.base import BaseTracker
from boxmot.trackers.bbox.botsort import BotSort
from boxmot.trackers.bbox.bytetrack import ByteTrack
from boxmot.trackers.bbox.deepocsort import DeepOcSort
from boxmot.trackers.bbox.hybridsort import HybridSort
from boxmot.trackers.bbox.ocsort import OcSort
from boxmot.trackers.bbox.sfsort import SFSORT
from boxmot.trackers.bbox.strongsort import StrongSort
from boxmot.trackers.common.association.matching import iou_distance
from boxmot.trackers.common.track_models.botsort import STrack as BotSortTrack
from boxmot.trackers.common.track_models.bytetrack import STrack as ByteTrackTrack
from boxmot.trackers.common.track_models.deepocsort import KalmanBoxTracker as DeepOCSortKalmanBoxTracker
from boxmot.trackers.common.track_models.ocsort import KalmanBoxTracker as OCSortKalmanBoxTracker
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.registry import create_tracker, get_tracker_config
from boxmot.utils import WEIGHTS
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
        self.warp = np.eye(2, 3, dtype=np.float32) if warp is None else np.asarray(warp, dtype=np.float32)
        self.calls: list[np.ndarray | None] = []

    def apply(self, img: np.ndarray, dets: np.ndarray | None = None) -> np.ndarray:
        call = None if dets is None else np.asarray(dets, dtype=np.float32).copy()
        self.calls.append(call)
        return self.warp.copy()


@pytest.mark.parametrize("Tracker", MOTION_N_APPEARANCE_TRACKING_METHODS)
def test_motion_n_appearance_trackers_instantiation(Tracker):
    reid_model = ReID(
        weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
        device="cpu",
        half=True,
    ).model
    Tracker(reid_model=reid_model)


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
    det = np.array([[144, 212, 400, 480, 0.92, 0], [425, 281, 576, 472, 0.91, 65]])

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


def test_hybridsort_config_covers_constructor_params_and_conditionals():
    config_path = Path("boxmot/configs/trackers/hybridsort.yaml")
    yaml_config = yaml.safe_load(config_path.read_text())
    flat_config = flatten_yaml_config(yaml_config)

    constructor_params = set(inspect.signature(HybridSort.__init__).parameters)
    expected_params = constructor_params - {"self", "reid_model", "kwargs"}
    expected_params.update(
        {
            "det_thresh",
            "max_age",
            "max_obs",
            "min_hits",
            "iou_threshold",
            "asso_func",
        }
    )

    assert expected_params <= set(flat_config)
    assert set(yaml_config["use_byte"]["activates"]) == {
        "low_thresh",
        "TCM_byte_step",
    }
    assert set(yaml_config["use_byte"]["activates"]["TCM_byte_step"]["activates"]) == {"TCM_byte_step_weight"}
    assert set(yaml_config["TCM_first_step"]["activates"]) == {"inertia"}
    with_reid_children = yaml_config["with_reid"]["activates"]
    assert set(with_reid_children) == {
        "longterm_bank_length",
        "alpha",
        "adapfs",
        "EG_weight_high_score",
        "EG_weight_low_score",
        "high_score_matching_thresh",
        "with_longterm_reid",
        "with_longterm_reid_correction",
    }
    assert set(with_reid_children["with_longterm_reid"]["activates"]) == {"longterm_reid_weight"}
    assert set(with_reid_children["with_longterm_reid_correction"]["activates"]) == {
        "longterm_reid_correction_thresh",
        "longterm_reid_correction_thresh_low",
    }


def test_hybridsort_track_histories_are_bounded_and_resettable():
    tracker = HybridSort(
        with_reid=False,
        min_hits=1,
        max_age=2,
        max_obs=3,
        iou_threshold=0.1,
    )
    tracker.cmc = DummyCMC()

    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    embs = np.ones((1, 4), dtype=np.float32)

    for frame_idx in range(10):
        det = np.array(
            [[10 + frame_idx, 10, 30 + frame_idx, 30, 0.9, 0]],
            dtype=np.float32,
        )
        tracker.update(det, rgb, embs)

    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert len(track.history_observations) == tracker.max_obs == 3
    assert len(track.observations) == tracker.max_obs
    assert len(track.kf.history_obs) <= tracker.max_obs
    assert sorted(track.observations) == [7, 8, 9]

    tracker.reset()

    assert tracker.active_tracks == []
    assert tracker.frame_count == 0
    assert tracker._first_frame_processed is False
    assert tracker._first_dets_processed is False


def create_kalman_box_tracker_ocsort(bbox, cls, det_ind, tracker):
    return OCSortKalmanBoxTracker(
        bbox,
        cls,
        det_ind,
        Q_xy_scaling=tracker.Q_xy_scaling,
        Q_s_scaling=tracker.Q_s_scaling,
        id_allocator=TrackIdAllocator(),
    )


def create_kalman_box_tracker_deepocsort(bbox, cls, det_ind, tracker):
    det = np.concatenate([bbox, [cls, det_ind]])
    return DeepOCSortKalmanBoxTracker(
        det,
        Q_xy_scaling=tracker.Q_xy_scaling,
        Q_s_scaling=tracker.Q_s_scaling,
        id_allocator=TrackIdAllocator(),
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
                "reid_model": ReID(
                    weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
                    device="cpu",
                    half=True,
                ).model,
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
    det = np.array(
        [
            [100, 100, 300, 250, 0.95, 0],  # class 0
            [400, 300, 550, 450, 0.90, 65],  # class 65
        ]
    )
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
    det = np.array(
        [
            [100, 100, 300, 250, 0.95, 0],  # class 0
            [400, 300, 550, 450, 0.90, 65],  # class 65
        ]
    )
    embs = np.random.random(size=(2, 512))

    tracker.update(det, rgb, embs)
    assert tracker.get_class_tracks(0, "active"), "No active tracks for class 0"
    assert tracker.get_class_tracks(65, "active"), "No active tracks for class 65"


def test_per_class_tracking_accepts_arbitrary_detector_class_ids():
    tracker = ByteTrack(per_class=True, track_thresh=0.5, min_conf=0.1)
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    det = np.array([[100, 100, 200, 200, 0.9, 123]], dtype=np.float32)

    tracker.update(det, rgb)

    assert tracker.get_class_tracks(123, "active")


def test_configured_class_catalog_rejects_unknown_detector_class():
    tracker = ByteTrack(class_ids=[0, 7], track_thresh=0.5, min_conf=0.1)
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    det = np.array([[100, 100, 200, 200, 0.9, 8]], dtype=np.float32)

    with pytest.raises(ValueError, match="not present in the tracker class catalog"):
        tracker.update(det, rgb)


def test_class_names_define_tracker_class_catalog():
    tracker = ByteTrack(
        per_class=True,
        class_names={7: "awning-bike"},
        track_thresh=0.5,
        min_conf=0.1,
    )
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    det = np.array([[100, 100, 200, 200, 0.9, 7]], dtype=np.float32)

    tracker.update(det, rgb)

    assert tracker.class_names == {7: "awning-bike"}
    assert tracker.get_class_tracks(7, "active")


def test_strongsort_rejects_obb_with_shared_error_message():
    tracker = StrongSort.__new__(StrongSort)
    BaseTracker.__init__(tracker, asso_func="iou")

    rgb = np.random.randint(255, size=(64, 64, 3), dtype=np.uint8)
    det = np.array([[32, 32, 20, 10, 0.15, 0.95, 0]], dtype=np.float32)

    with pytest.raises(AssertionError, match="StrongSort does not support OBB detections"):
        tracker.update(det, rgb)


def test_botsort_supports_obb_without_reid():
    tracker = BotSort(
        reid_model=None,
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
    track_a = BotSortTrack(
        det,
        max_obs=10,
        is_obb=True,
        id_allocator=TrackIdAllocator(),
    )
    track_b = BotSortTrack(
        det,
        max_obs=10,
        is_obb=True,
        id_allocator=TrackIdAllocator(),
    )

    cost = iou_distance([track_a], [track_b], is_obb=True)

    assert cost.shape == (1, 1)
    assert cost[0, 0] < 1e-3


def test_botsort_obb_cmc_uses_enclosing_aabb_boxes():
    tracker = BotSort(
        reid_model=None,
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
    track = BotSortTrack(
        det,
        max_obs=10,
        is_obb=True,
        id_allocator=TrackIdAllocator(),
    )
    track.activate(KalmanFilterXYWH(ndim=5), frame_id=1)

    BotSortTrack.multi_gmc_obb(
        [track],
        np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -6.0]], dtype=np.float32),
    )

    np.testing.assert_allclose(track.xywha[:2], np.array([332.0, 234.0], dtype=np.float32), atol=1e-4)
    np.testing.assert_allclose(track.xywha[2:], det[2:5], atol=1e-4)


def test_botsort_obb_state_history_follows_rotation_without_flips():
    tracker = BotSort(
        reid_model=None,
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
    track_a = ByteTrackTrack(
        det,
        id_allocator=TrackIdAllocator(),
        max_obs=10,
        is_obb=True,
    )
    track_b = ByteTrackTrack(
        det,
        id_allocator=TrackIdAllocator(),
        max_obs=10,
        is_obb=True,
    )

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

    history_center = np.asarray(track.history_observations[-1], dtype=np.float32).reshape(4, 2).mean(axis=0)
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


def test_create_tracker_uses_precomputed_reid_without_loading_model(monkeypatch):
    def fail_build_reid_model(**kwargs):
        raise AssertionError("precomputed ReID replay must not load a live ReID model")

    monkeypatch.setattr(tracker_registry, "_build_reid_model", fail_build_reid_model)
    tracker = create_tracker(
        tracker_type="occluboost",
        tracker_config=get_tracker_config("occluboost"),
        reid_weights=Path("unused.pt"),
        device="cpu",
        half=False,
        per_class=False,
        tracker_kwargs={
            "use_cmc": False,
            "with_reid": True,
            "min_hits": 1,
            "confirm_hits": 1,
            "instant_confirm_thresh": 0.0,
            "new_track_thresh": 0.1,
            "gta_enabled": False,
        },
        precomputed_reid=True,
    )

    assert tracker.with_reid is True
    assert tracker.reid_model is None

    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    det = np.array([[10, 10, 30, 40, 0.9, 0]], dtype=np.float32)
    embs = np.ones((1, 4), dtype=np.float32)

    output = tracker.update(det, rgb, embs)

    assert output.shape == (1, 8)


def test_create_tracker_disables_optional_reid_without_model_or_precomputed_embeddings():
    tracker = create_tracker(
        tracker_type="occluboost",
        tracker_config=get_tracker_config("occluboost"),
        reid_weights=None,
        device="cpu",
        half=False,
        per_class=False,
        tracker_kwargs={
            "use_cmc": False,
            "with_reid": True,
        },
    )

    assert tracker.with_reid is False
    assert tracker.reid_model is None


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


def test_per_class_state_keeps_lost_tracks_class_local():
    tracker = ByteTrack(per_class=True, track_thresh=0.5, min_conf=0.1, track_buffer=30)
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    empty = np.empty((0, 6), dtype=np.float32)
    class_1_det = np.array([[100, 100, 200, 200, 0.9, 1]], dtype=np.float32)
    class_0_det = np.array([[100, 100, 200, 200, 0.9, 0]], dtype=np.float32)

    tracker.update(class_1_det, rgb)
    class_1_id = tracker.get_class_tracks(1, "active")[0].id

    tracker.update(empty, rgb)
    class_1_lost_ids = {track.id for track in tracker.get_class_tracks(1, "lost")}
    assert class_1_id in class_1_lost_ids

    tracker.update(class_0_det, rgb)
    class_0_ids = {track.id for track in tracker.get_class_tracks(0, "active")}
    class_1_lost_ids = {track.id for track in tracker.get_class_tracks(1, "lost")}

    assert class_1_id not in class_0_ids
    assert class_1_id in class_1_lost_ids

    tracker.reset()
    assert not tracker.get_class_tracks(0, "active")
    assert not tracker.get_class_tracks(1, "lost")


def test_per_class_cmc_is_estimated_once_per_frame():
    tracker = BotSort(
        reid_model=None,
        with_reid=False,
        use_cmc=False,
        per_class=True,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.5,
    )
    tracker.cmc = DummyCMC()
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    det = np.array(
        [
            [100, 100, 200, 200, 0.9, 0],
            [300, 300, 400, 400, 0.9, 1],
        ],
        dtype=np.float32,
    )

    tracker.update(det, rgb)

    assert len(tracker.cmc.calls) == 1
    assert tracker.cmc.calls[0].shape == (2, 4)


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


# ---------------- OccluBoost OBB tests ----------------

from boxmot.trackers.bbox.occluboost import OccluBoost  # noqa: E402
from boxmot.trackers.common.geometry.obb import xywha_to_xyxy  # noqa: E402


def test_occluboost_supports_obb_without_reid():
    tracker = OccluBoost(reid_model=None, with_reid=False, use_cmc=False, min_hits=1)

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[320, 240, 80, 40, 0.15, 0.95, 0]], dtype=np.float32)

    out1 = tracker.update(det, rgb)
    out2 = tracker.update(det, rgb)

    assert tracker.is_obb is True
    assert tracker.supports_obb is True
    assert out1.shape == (1, 9)
    assert out2.shape == (1, 9)
    # cx, cy, w, h, angle should converge close to the (steady) measurement
    np.testing.assert_allclose(out2[0, :5], det[0, :5], atol=5e-2)
    # Same id across both frames
    assert out1[0, 5] == out2[0, 5]


def test_occluboost_obb_emits_nine_column_outputs_for_two_objects():
    tracker = OccluBoost(reid_model=None, with_reid=False, use_cmc=False, min_hits=1)
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    dets = np.array(
        [
            [100, 100, 60, 30, 0.3, 0.9, 0],
            [400, 300, 80, 40, -0.4, 0.85, 0],
        ],
        dtype=np.float32,
    )
    out = tracker.update(dets, rgb)
    out2 = tracker.update(dets + np.array([[2, 2, 0, 0, 0, 0, 0]] * 2, dtype=np.float32), rgb)

    assert out.shape == (2, 9)
    assert out2.shape == (2, 9)
    # IDs should be preserved across frames in the same order
    assert set(out[:, 5]) == set(out2[:, 5])


def test_occluboost_obb_aabb_path_unchanged():
    """The AABB path must remain 8-column and produce stable IDs."""
    tracker = OccluBoost(reid_model=None, with_reid=False, use_cmc=False, min_hits=1)
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    dets = np.array([[80, 80, 130, 130, 0.9, 0]], dtype=np.float32)
    out1 = tracker.update(dets, rgb)
    out2 = tracker.update(dets, rgb)

    assert tracker.is_obb is False
    assert out1.shape == (1, 8)
    assert out2.shape == (1, 8)
    assert out1[0, 4] == out2[0, 4]


def test_xywha_to_xyxy_enclosing_axis_aligned():
    # Zero angle: the enclosing AABB should equal the box itself.
    boxes = np.array([[100, 100, 60, 40, 0.0]], dtype=np.float32)
    xyxy = xywha_to_xyxy(boxes)
    np.testing.assert_allclose(xyxy[0], np.array([70, 80, 130, 120], dtype=np.float32), atol=1e-4)


def test_xywha_to_xyxy_enclosing_45deg_grows_bounds():
    # 45-degree rotation: enclosing AABB should expand symmetrically.
    boxes = np.array([[100, 100, 60, 40, np.pi / 4]], dtype=np.float32)
    xyxy = xywha_to_xyxy(boxes)
    half = 0.5 * (60 + 40) * np.cos(np.pi / 4)  # = 50/sqrt(2) added per axis
    np.testing.assert_allclose(
        xyxy[0],
        np.array([100 - half, 100 - half, 100 + half, 100 + half], dtype=np.float32),
        atol=1e-4,
    )


def test_occluboost_obb_history_follows_smoothly_under_rotation():
    tracker = OccluBoost(reid_model=None, with_reid=False, use_cmc=False, min_hits=1)
    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)

    angles = np.linspace(0.0, 1.5, 12, dtype=np.float32)
    track_id = None
    for angle in angles:
        det = np.array([[320, 240, 90, 40, angle, 0.95, 0]], dtype=np.float32)
        out = tracker.update(det, rgb)
        assert out.shape == (1, 9)
        if track_id is None:
            track_id = out[0, 5]
        else:
            # Single object → ID must persist
            assert out[0, 5] == track_id
