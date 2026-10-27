"""OC-family trajectory/output parity against scalar geometry and row packing."""

from importlib import import_module
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from boxmot import KalmanConfig
from boxmot.trackers.common.motion import batching
from boxmot.trackers.common.motion.cmc import batching as cmc_batching
from boxmot.trackers.common.motion.models import supported_motion_model_kinds
from tests.unit.trackers.box.test_searchable_kalman_state import _rows
from tests.unit.trackers.test_association_switcher import TRACKER_FACTORIES
from tests.unit.trackers.test_variable_frame_time import _update


def _scalar_boxes(tracks, *, is_obb: bool) -> np.ndarray:
    """Reproduce current-state collection before grouped box conversion."""
    return np.asarray([track.get_state()[0] for track in tracks]).reshape(-1, 5 if is_obb else 4)


def _scalar_output(self, tracks, dtype=np.float32, *, boxes=None) -> np.ndarray:
    """Reproduce observation selection and one-row-at-a-time packing."""
    rows = []
    for track in tracks:
        box = (
            track.get_state()[0]
            if track.last_observation[-1] < 0
            else track.last_observation[: self.detection_layout.box_cols]
        )
        rows.append(self.format_output_row(box, track.id, track.conf, track.cls, track.det_ind, dtype=dtype))
    return self.format_output_rows(rows, dtype=dtype)


def _scalar_history_warp(tracks, transform, *, model) -> None:
    """Retain the original ordered scalar prediction-history CMC loop."""
    for track in tracks:
        for box in track.history:
            values = np.asarray(box).reshape(1, -1)
            values[:] = model.transform_boxes(values, transform)


@pytest.mark.parametrize("name", ["ocsort", "deepocsort", "hybridsort"])
@pytest.mark.parametrize(
    "kind,is_obb", [(kind, obb) for obb in (False, True) for kind in supported_motion_model_kinds(is_obb=obb)]
)
@pytest.mark.parametrize("with_cmc", [False, True])
def test_geometry_and_output_batches_preserve_complete_trajectories(name, kind, is_obb, with_cmc, monkeypatch):
    """Births, missing frames, recovery and expiry preserve IDs and output order."""
    actual, expected = (
        TRACKER_FACTORIES[name](is_obb=is_obb, kalman=KalmanConfig(type=kind), max_age=3)
        for _ in range(2)
    )
    module = import_module(f"boxmot.trackers.{name}.tracker")
    expected.format_outputs = MethodType(_scalar_output, expected)
    if with_cmc:
        transform = np.array([[1.0, -0.001, 0.05], [0.001, 1.0, -0.02]])
        for tracker in (actual, expected):
            tracker.cmc = SimpleNamespace(apply=lambda image, boxes: transform, requires_frame_pixels=False)
    for frame in range(15):
        rows = _rows(frame, is_obb=is_obb)
        if frame in (4, 5) or frame >= 11:
            rows = rows[:0]
        elif frame == 7:
            rows = rows[::-1].copy()
        packed = frame % 2 == 0
        observed = _update(actual, rows, index=frame, packed=packed)
        with monkeypatch.context() as scalar:
            scalar.setattr(batching, "_batch_prediction_hook", lambda track: False)
            scalar.setattr(module, "current_track_boxes", _scalar_boxes)
            scalar.setattr(cmc_batching, "transform_prediction_histories", _scalar_history_warp)
            reference = _update(expected, rows, index=frame, packed=packed)
        if not packed:
            observed = (observed.to_obb_rows() if is_obb else observed.to_aabb_rows()).numpy()
            reference = (reference.to_obb_rows() if is_obb else reference.to_aabb_rows()).numpy()
        np.testing.assert_array_equal(observed, reference)
        assert len(actual.active_tracks) == len(expected.active_tracks)
        for track, original in zip(actual.active_tracks, expected.active_tracks, strict=True):
            np.testing.assert_array_equal(track.kf.x, original.kf.x)
            np.testing.assert_array_equal(track.kf.P, original.kf.P)
            np.testing.assert_allclose(track.history, original.history, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(track.history_observations, original.history_observations)
            for attribute in ("id", "cls", "det_ind", "age", "time_since_update", "hit_streak"):
                assert getattr(track, attribute) == getattr(original, attribute)
    assert not actual.active_tracks
