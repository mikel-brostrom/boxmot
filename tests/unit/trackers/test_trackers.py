"""Algorithm-level tracker tests using canonical v24 structures."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from boxmot.engine.tuning.search_space import flatten_yaml_config, load_yaml_config
from boxmot.structures import Boxes, Detections, Frame, MaskBatch, OrientedBoxes, Tracks
from boxmot.trackers import Tracker, TrackerSpec, create_tracker
from boxmot.trackers.box.deepocsort.track import KalmanBoxTracker as DeepOCSortKalmanBoxTracker
from boxmot.trackers.box.deepocsort.tracker import DeepOcSort
from boxmot.trackers.box.hybridsort.tracker import HybridSort
from boxmot.trackers.box.ocsort.track import KalmanBoxTracker as OCSortKalmanBoxTracker
from boxmot.trackers.box.ocsort.tracker import OcSort
from boxmot.trackers.box.sfsort.tracker import SFSORT
from boxmot.trackers.common.geometry.obb import normalize_angle
from boxmot.trackers.common.tracking.track import TrackIdAllocator
from boxmot.trackers.config import load_tracker_defaults
from boxmot.trackers.registry import TRACKER_DEFINITIONS

TRACKER_NAMES = tuple(TRACKER_DEFINITIONS)
EMBEDDING_TRACKER_NAMES = tuple(
    name for name, definition in TRACKER_DEFINITIONS.items() if definition.capabilities.accepts_embeddings
)


def _frame(sample_id: str, frame_index: int, *, height: int = 96, width: int = 128) -> Frame:
    return Frame(
        image=torch.zeros((3, height, width), dtype=torch.uint8),
        sample_id=sample_id,
        sequence_id="sequence",
        frame_index=frame_index,
    )


def _masks_for_rows(rows: np.ndarray, *, height: int = 96, width: int = 128) -> MaskBatch:
    values = torch.zeros((len(rows), height, width), dtype=torch.bool)
    for index, row in enumerate(rows):
        if rows.shape[1] == 7:
            cx, cy, box_width, box_height = row[:4]
            x1, x2 = cx - box_width / 2, cx + box_width / 2
            y1, y2 = cy - box_height / 2, cy + box_height / 2
        else:
            x1, y1, x2, y2 = row[:4]
        left = max(0, min(width - 1, int(np.floor(x1))))
        right = max(left + 1, min(width, int(np.ceil(x2))))
        top = max(0, min(height - 1, int(np.floor(y1))))
        bottom = max(top + 1, min(height, int(np.ceil(y2))))
        values[index, top:bottom, left:right] = True
    return MaskBatch(values)


def _detections(
    rows: np.ndarray,
    *,
    sample_id: str,
    embeddings: bool = False,
    masks: bool = False,
) -> Detections:
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    geometry_width = 5 if rows.shape[1] == 7 else 4
    geometry_values = torch.from_numpy(rows[:, :geometry_width].copy())
    geometry = OrientedBoxes(geometry_values) if geometry_width == 5 else Boxes(geometry_values)
    return Detections(
        geometry=geometry,
        scores=torch.from_numpy(rows[:, geometry_width].copy()),
        class_ids=torch.from_numpy(rows[:, geometry_width + 1].astype(np.int64, copy=True)),
        sample_id=sample_id,
        masks=_masks_for_rows(rows) if masks else None,
        embeddings=(
            torch.nn.functional.normalize(
                torch.arange(len(rows) * 4, dtype=torch.float32).reshape(len(rows), 4) + 1,
                dim=1,
            )
            if embeddings and len(rows)
            else torch.empty((0, 4), dtype=torch.float32)
            if embeddings
            else None
        ),
    )


def _update(tracker: Tracker, rows: np.ndarray, *, frame_index: int) -> Tracks:
    sample_id = f"sequence/{frame_index:06d}"
    detections = _detections(
        rows,
        sample_id=sample_id,
        embeddings=tracker.requirements.embeddings,
        masks=tracker.requirements.masks,
    )
    frame = _frame(sample_id, frame_index) if tracker.requirements.frame else None
    return tracker.update(detections, frame)


def _aabb_rows() -> np.ndarray:
    return np.array(
        [[14, 21, 40, 48, 0.92, 0], [62, 28, 88, 67, 0.91, 65]],
        dtype=np.float32,
    )


def _obb_rows() -> np.ndarray:
    return np.array(
        [[27, 35, 26, 27, 0.2, 0.92, 0], [75, 47, 26, 39, -0.3, 0.91, 65]],
        dtype=np.float32,
    )


def _empty_rows(*, is_obb: bool) -> np.ndarray:
    return np.empty((0, 7 if is_obb else 6), dtype=np.float32)


def _output_after_hits(tracker: Tracker, rows: np.ndarray, *, attempts: int = 6) -> Tracks:
    result = _update(tracker, rows, frame_index=0)
    for frame_index in range(1, attempts):
        if len(result) == len(rows):
            break
        result = _update(tracker, rows, frame_index=frame_index)
    return result


@pytest.mark.parametrize("tracker_name", TRACKER_NAMES)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_trackers_emit_structured_rows_for_both_geometry_modes(tracker_name: str, geometry: str) -> None:
    tracker = create_tracker(TrackerSpec(tracker_name, geometry=geometry, options=(("min_hits", 1),)))
    rows = _obb_rows() if geometry == "obb" else _aabb_rows()

    output = _output_after_hits(tracker, rows)

    assert isinstance(output, Tracks)
    assert output.geometry.values.shape == (len(output), 5 if geometry == "obb" else 4)
    assert (output.to_obb_rows() if geometry == "obb" else output.to_aabb_rows()).shape[1] == (
        9 if geometry == "obb" else 8
    )
    if tracker.requirements.masks:
        assert output.masks is not None
        assert output.masks.values.shape == (len(output), 96, 128)


@pytest.mark.parametrize("tracker_name", TRACKER_NAMES)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_trackers_accept_completed_empty_structured_batches(tracker_name: str, geometry: str) -> None:
    tracker = create_tracker(TrackerSpec(tracker_name, geometry=geometry))

    output = _update(tracker, _empty_rows(is_obb=geometry == "obb"), frame_index=0)

    assert len(output) == 0
    assert output.is_obb is (geometry == "obb")
    if tracker.requirements.masks:
        assert output.masks is not None
        assert output.masks.values.shape == (0, 96, 128)


@pytest.mark.parametrize("tracker_name", TRACKER_NAMES)
def test_track_ids_remain_stable_across_matching_frames(tracker_name: str) -> None:
    tracker = create_tracker(TrackerSpec(tracker_name, options=(("min_hits", 1),)))
    rows = _aabb_rows()[:1]

    first = _output_after_hits(tracker, rows)
    second = _update(tracker, rows, frame_index=7)

    assert len(first) == len(second) == 1
    assert first.track_ids.item() == second.track_ids.item()


def test_dynamic_max_obs_is_large_enough_for_track_lifetime() -> None:
    tracker = OcSort(max_age=400)

    assert tracker.max_obs == 405


def test_hybridsort_config_covers_constructor_and_conditionals() -> None:
    runtime_config = load_tracker_defaults("hybridsort")
    tuning_config = load_yaml_config("hybridsort")
    flat_tuning_config = flatten_yaml_config(tuning_config)
    constructor_params = set(inspect.signature(HybridSort.__init__).parameters)
    expected = constructor_params - {"self", "kwargs"}
    expected.update({"det_thresh", "max_age", "max_obs", "min_hits", "iou_threshold", "asso_func"})

    assert expected <= set(runtime_config)
    assert set(flat_tuning_config) <= set(runtime_config)
    assert set(tuning_config["use_byte"]["activates"]) == {"low_thresh", "TCM_byte_step"}
    assert "longterm_bank_length" in tuning_config["use_embeddings"]["activates"]


@pytest.mark.parametrize("track_type", (OcSort, DeepOcSort))
def test_q_matrix_scaling_is_applied_to_ocsort_family(track_type) -> None:
    q_xy = 0.05
    q_scale = 0.0005
    tracker = (
        track_type(Q_xy_scaling=q_xy, Q_s_scaling=q_scale, use_embeddings=False, cmc_off=True)
        if (track_type is DeepOcSort)
        else track_type(Q_xy_scaling=q_xy, Q_s_scaling=q_scale)
    )
    bbox = np.array([0, 0, 100, 100, 0.9], dtype=np.float32)
    if track_type is DeepOcSort:
        track = DeepOCSortKalmanBoxTracker(
            np.concatenate((bbox, [1, 0])),
            Q_xy_scaling=tracker.Q_xy_scaling,
            Q_s_scaling=tracker.Q_s_scaling,
            id_allocator=TrackIdAllocator(),
        )
    else:
        track = OCSortKalmanBoxTracker(
            bbox,
            cls=1,
            det_ind=0,
            Q_xy_scaling=tracker.Q_xy_scaling,
            Q_s_scaling=tracker.Q_s_scaling,
            id_allocator=TrackIdAllocator(),
        )

    assert track.kf.Q[4, 4] == q_xy
    assert track.kf.Q[5, 5] == q_xy
    assert track.kf.Q[6, 6] == q_scale


def test_per_class_tracking_accepts_sparse_detector_class_ids() -> None:
    tracker = create_tracker(
        TrackerSpec(
            "bytetrack",
            per_class=True,
            class_ids=(0, 65),
            class_names=((0, "person"), (65, "remote")),
            options=(("min_hits", 1),),
        )
    )

    output = _output_after_hits(tracker, _aabb_rows())

    assert sorted(output.class_ids.tolist()) == [0, 65]
    assert sorted(output.detection_indices.tolist()) == [0, 1]


def test_configured_class_catalog_rejects_unknown_detector_class() -> None:
    tracker = create_tracker(TrackerSpec("bytetrack", class_ids=(0,)))
    sample_id = "sequence/000000"
    detections = _detections(_aabb_rows(), sample_id=sample_id)

    with pytest.raises(ValueError, match="not present in the tracker class catalog"):
        tracker.update(detections)


@pytest.mark.parametrize("tracker_name", EMBEDDING_TRACKER_NAMES)
def test_embedding_trackers_never_fall_back_to_model_inference(tracker_name: str) -> None:
    tracker = create_tracker(TrackerSpec(tracker_name))
    sample_id = "sequence/000000"
    detections = _detections(_aabb_rows()[:1], sample_id=sample_id)
    frame = _frame(sample_id, 0) if tracker.requirements.frame else None

    with pytest.raises(ValueError, match="requires detection embeddings"):
        tracker.update(detections, frame)
    assert not hasattr(tracker, "model")
    assert not hasattr(tracker, "reid_model")


def test_sam2mot_preserves_obb_masks_and_detection_alignment() -> None:
    tracker = create_tracker(
        TrackerSpec(
            "sam2mot",
            geometry="obb",
            options=(("det_thresh", 0.1), ("min_hits", 1), ("new_track_thresh", 0.1)),
        )
    )

    output = _update(tracker, _obb_rows(), frame_index=0)

    assert output.to_obb_rows().shape == (2, 9)
    assert output.masks is not None
    assert output.masks.values.dtype is torch.bool
    assert sorted(output.detection_indices.tolist()) == [0, 1]
    assert output.masks.values.flatten(1).any(dim=1).all()


def test_sam2mot_rejects_detection_masks_without_foreground() -> None:
    tracker = create_tracker(TrackerSpec("sam2mot"))
    sample_id = "sequence/000000"
    rows = _aabb_rows()[:1]
    detections = Detections(
        geometry=Boxes(torch.from_numpy(rows[:, :4].copy())),
        scores=torch.from_numpy(rows[:, 4].copy()),
        class_ids=torch.from_numpy(rows[:, 5].astype(np.int64, copy=True)),
        sample_id=sample_id,
        masks=MaskBatch(torch.zeros((1, 96, 128), dtype=torch.bool)),
    )

    with pytest.raises(ValueError, match="foreground in every non-empty detection mask"):
        tracker.update(detections, _frame(sample_id, 0))


def test_sam2mot_aligns_equivalent_obb_forms_without_an_angle_jump() -> None:
    tracker = create_tracker(
        TrackerSpec(
            "sam2mot",
            geometry="obb",
            options=(("det_thresh", 0.1), ("min_hits", 1), ("new_track_thresh", 0.1)),
        )
    )
    first_row = np.array([[32, 32, 20, 10, 0.15, 0.95, 0]], dtype=np.float32)
    equivalent = np.array([[32, 32, 10, 20, 0.15 + np.pi / 2, 0.95, 0]], dtype=np.float32)

    first = _update(tracker, first_row, frame_index=0)
    second = _update(tracker, equivalent, frame_index=1)

    assert first.track_ids.item() == second.track_ids.item()
    torch.testing.assert_close(second.geometry.values[:, 2:4], first.geometry.values[:, 2:4])
    assert abs(float(second.geometry.values[0, 4] - first.geometry.values[0, 4])) < 1e-5


def test_sam2mot_obb_angle_update_is_damped() -> None:
    tracker = create_tracker(
        TrackerSpec(
            "sam2mot",
            geometry="obb",
            options=(
                ("det_thresh", 0.1),
                ("min_hits", 1),
                ("new_track_thresh", 0.1),
                ("obb_theta_damping", 0.75),
            ),
        )
    )
    first_row = np.array([[48, 48, 30, 18, 0.0, 0.95, 0]], dtype=np.float32)
    rotated = np.array([[48, 48, 30, 18, 0.4, 0.95, 0]], dtype=np.float32)

    first = _update(tracker, first_row, frame_index=0)
    second = _update(tracker, rotated, frame_index=1)

    tracked_delta = float(normalize_angle(second.geometry.values[0, 4] - first.geometry.values[0, 4]))
    measured_delta = float(normalize_angle(rotated[0, 4] - first_row[0, 4]))
    assert tracked_delta == pytest.approx(0.25 * measured_delta, abs=1e-5)


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_sfsort_low_score_second_pass_keeps_identity(geometry: str) -> None:
    tracker = SFSORT(
        high_th=0.6,
        low_th=0.1,
        new_track_th=0.5,
        match_th_second=0.3,
        dynamic_tuning=False,
        is_obb=geometry == "obb",
    )
    rows = (_obb_rows() if geometry == "obb" else _aabb_rows())[:1]

    first = _update(tracker, rows, frame_index=0)
    low = rows.copy()
    low[:, -2] = 0.3
    second = _update(tracker, low, frame_index=1)

    assert first.track_ids.item() == second.track_ids.item()
    assert len(tracker.active_tracks) == 1
    assert not tracker.lost_tracks


def test_sfsort_obb_angle_motion_is_damped() -> None:
    tracker = SFSORT(obb_theta_damping=0.8, is_obb=True)
    first_row = np.array([[64, 48, 40, 20, 0.0, 0.95, 0]], dtype=np.float32)
    rotated = first_row.copy()
    rotated[:, 4] = 0.4

    first = _update(tracker, first_row, frame_index=0)
    second = _update(tracker, rotated, frame_index=1)

    assert first.track_ids.item() == second.track_ids.item()
    tracked_delta = abs(float(second.geometry.values[0, 4] - first.geometry.values[0, 4]))
    assert 0.0 < tracked_delta < 0.4
