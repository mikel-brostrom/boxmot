"""Temporary observations preserve ByteTrack geometry and complete trajectories."""

from types import MethodType

import numpy as np
import pytest

from boxmot.trackers.bytetrack.config import ByteTrackConfig
from boxmot.trackers.bytetrack.track import STrack, _Detection
from boxmot.trackers.bytetrack.tracker import ByteTrack
from boxmot.trackers.common.motion.kalman_filters.config import KalmanConfig

KINDS = ("xyah", "xywh", "xysr", "xyhr", "xyscr")


def _full_detections(tracker: ByteTrack, rows: np.ndarray) -> list[STrack]:
    """Use the original full-track observation constructor as the reference."""
    detections = []
    for row in rows:
        track = STrack(
            row,
            max_obs=tracker.max_obs,
            id_allocator=tracker.id_allocator,
            is_obb=tracker.is_obb,
            motion_model=tracker.motion_model,
        )
        # The new birth path consumes this row; constructing a fresh STrack
        # there is equivalent because ID allocation happens only at activation.
        track.raw_row = row
        detections.append(track)
    return detections


def _rows(is_obb: bool, frame: int, observations: tuple[tuple[int, float], ...]) -> np.ndarray:
    """Translate several identities, including equivalent OBB representations."""
    rows = []
    for identity, confidence in observations:
        x = 25.123456789 + identity * 48 + frame * 0.1337
        y, width, height = 45.345678912, 18.123456789, 32.345678912
        if is_obb:
            angle = 0.234567891 + frame * 0.001
            if frame % 3 == 2:
                width, height, angle = height, width, angle + np.pi / 2
            box = [x, y, width, height, angle]
        else:
            box = [x, y, x + width, y + height]
        rows.append([*box, confidence, identity % 2])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7 if is_obb else 6)


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_detection_records_preserve_geometry_measurements_and_input_ownership(is_obb, kind, dtype) -> None:
    """Round input once to float32 before applying the scalar conversion order."""
    tracker = ByteTrack(kalman=KalmanConfig(type=kind), is_obb=is_obb)
    rows = _rows(is_obb, 2, ((0, 0.9123456789), (1, 0.3234567891))).astype(dtype)
    indexed = np.column_stack((rows, np.arange(len(rows), dtype=dtype)))
    original = indexed.copy()
    reference = _full_detections(tracker, indexed)
    records = tracker._make_detections(indexed)
    np.testing.assert_array_equal(indexed, original)
    np.testing.assert_array_equal(tracker._association_boxes(records), tracker._association_boxes(reference))
    for record, full in zip(records, reference, strict=True):
        np.testing.assert_array_equal(record.xywh, full.xywh)
        np.testing.assert_array_equal(record.xyxy, full.xyxy)
        np.testing.assert_array_equal(record.xywha, full.xywha)
        np.testing.assert_array_equal(full._measurement_for_update(record), full._measurement_for_update(full))
        assert (record.conf, record.cls, record.det_ind) == (full.conf, full.cls, full.det_ind)
        assert not hasattr(record, "history_observations")
        assert not hasattr(record, "mean")
        assert not np.shares_memory(record.raw_row, indexed)
    indexed[:] = 0
    np.testing.assert_array_equal(tracker._association_boxes(records), tracker._association_boxes(reference))
    geometry = tracker._association_boxes(records)
    geometry[:] = 0
    np.testing.assert_array_equal(tracker._association_boxes(records), tracker._association_boxes(reference))


@pytest.mark.parametrize("is_obb", [False, True])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fusion", ["off", "gate", "blend"])
@pytest.mark.parametrize("per_class", [False, True])
def test_detection_records_preserve_all_matching_stages_and_lifecycles(is_obb, kind, fusion, per_class) -> None:
    """Keep exact outputs, KF states, metadata, and histories across gaps."""
    options = dict(
        config=ByteTrackConfig(max_lost_frames=2, motion_fusion=fusion),
        kalman=KalmanConfig(type=kind),
        is_obb=is_obb,
        per_class=per_class,
    )
    actual, expected = ByteTrack(**options), ByteTrack(**options)
    expected._make_detections = MethodType(_full_detections, expected)
    observations = (
        (),
        ((0, 0.95), (1, 0.91)),
        ((0, 0.96), (1, 0.93), (2, 0.98)),
        ((0, 0.35), (1, 0.96)),
        (),
        ((0, 0.97), (1, 0.94)),
        ((0, 0.95),),
        (),
        (),
        (),
        (),
        ((3, 0.96),),
        ((3, 0.98),),
    )
    for frame, visible in enumerate(observations):
        rows = _rows(is_obb, frame, visible)
        np.testing.assert_array_equal(actual.update(rows), expected.update(rows))
        for group in ("active_tracks", "lost_stracks", "removed_stracks"):
            tracks, references = getattr(actual, group), getattr(expected, group)
            assert len(tracks) == len(references)
            for track, reference in zip(tracks, references, strict=True):
                assert type(track) is STrack
                assert track.meta == reference.meta
                assert (track.state, track.tracklet_len) == (reference.state, reference.tracklet_len)
                np.testing.assert_array_equal(track.mean, reference.mean)
                np.testing.assert_array_equal(track.covariance, reference.covariance)
                np.testing.assert_array_equal(track.history_observations, reference.history_observations)


def test_matched_and_low_confidence_observations_do_not_allocate_tracks(monkeypatch) -> None:
    """Persistent histories are allocated only when a new identity is born."""
    created = []
    original = STrack.__init__

    def record_construction(self, *args, **kwargs) -> None:
        original(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(STrack, "__init__", record_construction)
    tracker = ByteTrack()
    tracker.update(_rows(False, 0, ((0, 0.95), (1, 0.95))))
    assert len(created) == 2
    tracker.update(_rows(False, 1, ((0, 0.95), (1, 0.35), (2, 0.35))))
    assert len(created) == 2
    tracker.update(_rows(False, 2, ((0, 0.95), (1, 0.95), (2, 0.95))))
    assert len(created) == 3


@pytest.mark.parametrize("is_obb", [False, True])
def test_empty_detection_batch(is_obb: bool) -> None:
    """Empty frames retain the expected geometry width."""
    tracker = ByteTrack(is_obb=is_obb)
    records = _Detection.from_rows(np.empty((0, 8 if is_obb else 7)), is_obb=is_obb)
    assert records == []
    assert tracker._association_boxes(records).shape == (0, 5 if is_obb else 4)
