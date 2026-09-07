"""HybridSORT OBB invariants through canonical structures."""

from __future__ import annotations

import numpy as np
import torch

from boxmot.engine.tuning.search_space import load_yaml_config
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes, Tracks
from boxmot.trackers.box.hybridsort.tracker import HybridSort
from boxmot.trackers.box.ocsort.track import KalmanBoxTracker as OBBKalmanBoxTracker
from boxmot.trackers.box.sfsort.tracker import SFSORT
from boxmot.trackers.common.tracking.track import TrackIdAllocator


def _frame(sample_id: str, frame_index: int) -> Frame:
    return Frame(
        image=torch.zeros((3, 128, 128), dtype=torch.uint8),
        sample_id=sample_id,
        sequence_id="sequence",
        frame_index=frame_index,
    )


def _update(
    tracker: HybridSort,
    rows: np.ndarray,
    *,
    frame_index: int,
    embeddings: np.ndarray | None = None,
) -> Tracks:
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    is_obb = rows.shape[1] == 7
    geometry_width = 5 if is_obb else 4
    geometry_values = torch.from_numpy(rows[:, :geometry_width].copy())
    geometry = OrientedBoxes(geometry_values) if is_obb else Boxes(geometry_values)
    sample_id = f"sequence/{frame_index:06d}"
    detections = Detections(
        geometry=geometry,
        scores=torch.from_numpy(rows[:, geometry_width].copy()),
        class_ids=torch.from_numpy(rows[:, geometry_width + 1].astype(np.int64, copy=True)),
        sample_id=sample_id,
        embeddings=(
            None if embeddings is None else torch.from_numpy(np.ascontiguousarray(embeddings, dtype=np.float32))
        ),
    )
    frame = _frame(sample_id, frame_index) if tracker.requirements.frame else None
    return tracker.update(detections, frame)


def _ids_by_detection_index(output: Tracks) -> dict[int, int]:
    return dict(zip(output.detection_indices.tolist(), output.track_ids.tolist()))


def _tracker(*, use_embeddings: bool = False, **kwargs) -> HybridSort:
    options = {
        "cmc_method": None,
        "use_embeddings": use_embeddings,
        "is_obb": True,
        "min_hits": 1,
        "det_thresh": 0.5,
        "iou_threshold": 0.2,
        "asso_func": "iou",
    }
    options.update(kwargs)
    return HybridSort(**options)


def test_hybridsort_obb_crossed_orientations_keep_geometry_ids() -> None:
    tracker = _tracker()
    first_rows = np.array(
        [
            [64, 64, 52, 8, np.pi / 4, 0.95, 0],
            [64, 64, 52, 8, -np.pi / 4, 0.95, 0],
        ],
        dtype=np.float32,
    )

    first = _update(tracker, first_rows, frame_index=0)
    second = _update(tracker, first_rows[::-1].copy(), frame_index=1)

    first_ids = _ids_by_detection_index(first)
    second_ids = _ids_by_detection_index(second)
    assert second_ids[0] == first_ids[1]
    assert second_ids[1] == first_ids[0]


def test_hybridsort_obb_equivalent_rectangle_forms_keep_ids() -> None:
    tracker = _tracker(iou_threshold=0.8)
    first_row = np.array([[50, 55, 44, 12, 0.37, 0.95, 0]], dtype=np.float32)
    equivalent = first_row.copy()
    equivalent[:, 2:4] = equivalent[:, [3, 2]]
    equivalent[:, 4] += np.pi / 2

    first = _update(tracker, first_row, frame_index=0)
    second = _update(tracker, equivalent, frame_index=1)

    assert len(first) == len(second) == 1
    assert second.track_ids.item() == first.track_ids.item()


def test_hybridsort_obb_consumes_precomputed_embeddings_for_ambiguous_geometry() -> None:
    tracker = _tracker(
        use_embeddings=True,
        EG_weight_high_score=4.0,
        with_longterm_reid_correction=False,
    )
    rows = np.array(
        [[64, 64, 40, 12, 0.2, 0.95, 0], [64, 64, 40, 12, 0.2, 0.94, 0]],
        dtype=np.float32,
    )
    features = np.eye(2, dtype=np.float32)

    first = _update(tracker, rows, frame_index=0, embeddings=features)
    second = _update(tracker, rows[::-1].copy(), frame_index=1, embeddings=features[::-1].copy())

    first_ids = _ids_by_detection_index(first)
    second_ids = _ids_by_detection_index(second)
    assert second_ids[0] == first_ids[1]
    assert second_ids[1] == first_ids[0]


def test_hybridsort_obb_disabled_byte_pass_does_not_update_low_score_track() -> None:
    tracker = _tracker(use_byte=False, det_thresh=0.6, low_thresh=0.1)
    high = np.array([[64, 48, 42, 16, -0.35, 0.95, 0]], dtype=np.float32)
    low = high.copy()
    low[:, 5] = 0.4

    first = _update(tracker, high, frame_index=0)
    second = _update(tracker, low, frame_index=1)

    assert len(first) == 1
    assert len(second) == 0
    assert tracker.active_tracks[0].time_since_update == 1


def test_hybridsort_obb_outputs_latest_matched_observation() -> None:
    tracker = _tracker(det_thresh=0.1, iou_threshold=0.1)
    first_row = np.array([[64, 48, 42, 16, -0.35, 0.95, 0]], dtype=np.float32)
    second_row = np.array([[68, 50, 46, 14, -0.2, 0.95, 0]], dtype=np.float32)

    _update(tracker, first_row, frame_index=0)
    output = _update(tracker, second_row, frame_index=1)

    torch.testing.assert_close(output.geometry.values, torch.from_numpy(second_row[:, :5]))


def test_hybridsort_obb_discards_tracks_with_invalid_predictions() -> None:
    tracker = _tracker()
    row = np.array([[64, 64, 40, 12, 0.2, 0.95, 0]], dtype=np.float32)
    first = _update(tracker, row, frame_index=0)
    first_id = first.track_ids.item()
    tracker.active_tracks[0].predict = lambda: np.full((1, 6), np.nan, dtype=float)

    second = _update(tracker, row, frame_index=1)

    assert torch.isfinite(second.geometry.values).all()
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].id != first_id


def test_obb_motion_state_damps_angle_update_and_records_history() -> None:
    track = OBBKalmanBoxTracker(
        np.array([0, 0, 20, 8, 0.0, 0.9], dtype=np.float32),
        cls=0,
        det_ind=0,
        is_obb=True,
        id_allocator=TrackIdAllocator(),
    )
    track.predict()
    track.update(np.array([0, 0, 20, 8, 0.4, 0.9], dtype=np.float32), cls=0, det_ind=0)

    angle = float(track.get_state()[0, 4])
    assert 0.0 < angle < 0.4
    assert len(track.history_observations) >= 2
    assert track.history_observations[-1].shape == (8,)


def test_obb_observation_dictionary_is_bounded() -> None:
    track = OBBKalmanBoxTracker(
        np.array([0, 0, 20, 8, 0.0, 0.9], dtype=np.float32),
        cls=0,
        det_ind=0,
        max_obs=2,
        is_obb=True,
        id_allocator=TrackIdAllocator(),
    )

    for frame_index in range(1, 5):
        track.predict()
        track.update(
            np.array([frame_index, 0, 20, 8, 0.0, 0.9], dtype=np.float32),
            cls=0,
            det_ind=frame_index,
        )

    assert list(track.observations) == [3, 4]


def test_sfsort_obb_center_penalty_is_representation_invariant() -> None:
    active = np.array([[10, 20, 32, 8, 0.31]], dtype=np.float32)
    equivalent = active.copy()
    equivalent[:, 2:4] = equivalent[:, [3, 2]]
    equivalent[:, 4] += np.pi / 2
    candidate = np.array([[34, 27, 20, 6, -0.42]], dtype=np.float32)

    np.testing.assert_allclose(
        SFSORT._obb_center_penalty(active, candidate),
        SFSORT._obb_center_penalty(equivalent, candidate),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        SFSORT._calculate_cost_obb(active, candidate),
        SFSORT._calculate_cost_obb(equivalent, candidate),
        atol=1e-6,
    )


def test_obb_tracker_tuning_spaces_only_offer_supported_association_modes() -> None:
    supported = {"iou", "giou", "diou", "ciou", "hmiou", "centroid"}
    for tracker_name in ("ocsort", "deepocsort", "hybridsort"):
        options = set(load_yaml_config(tracker_name)["asso_func"]["options"])
        assert options == supported
