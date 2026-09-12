from __future__ import annotations

import numpy as np
import pytest

from boxmot.native.trackers import occluboost as native_binding
from boxmot.trackers.occluboost import native as native_module
from boxmot.trackers.occluboost.tracker import OccluBoost

from ._helpers import detections_from_rows, empty_native_batch, update_rows


class _FakeLibrary:
    def __init__(self) -> None:
        self.last_embeddings: np.ndarray | None = None

    def create(self, _cfg):
        return "handle"

    def reset(self, _handle):
        return None

    def update(
        self,
        _handle,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ):
        self.last_embeddings = embeddings
        return empty_native_batch(geometry.shape[1])

    def destroy(self, _handle):
        return None


def test_native_occluboost_requirements_are_configuration_derived():
    appearance = native_module.NativeOccluBoostTracker(
        {"use_embeddings": True, "use_cmc": False},
        library=_FakeLibrary(),
    )
    motion = native_module.NativeOccluBoostTracker(
        {"use_embeddings": False, "use_cmc": True},
        library=_FakeLibrary(),
    )
    assert appearance.requirements.embeddings is True
    assert appearance.requirements.frame is False
    assert motion.requirements.embeddings is False
    assert motion.requirements.frame is True
    appearance.close()
    motion.close()


def test_native_occluboost_consumes_supplied_embeddings():
    library = _FakeLibrary()
    tracker = native_module.NativeOccluBoostTracker(
        {"use_embeddings": True, "use_cmc": False},
        library=library,
    )
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    detections = detections_from_rows(
        np.array([[1, 1, 4, 5, 0.9, 0]], dtype=np.float32),
        embeddings=embeddings,
    )
    tracker.update(detections)
    np.testing.assert_array_equal(library.last_embeddings, detections.embeddings.numpy())
    tracker.close()


def test_native_occluboost_resolves_and_forwards_obb_operating_point():
    cfg = native_module._resolve_tracker_config(
        {
            "obb_det_thresh": 0.11,
            "obb_iou_threshold": 0.12,
            "obb_new_track_thresh": 0.13,
            "obb_instant_confirm_thresh": 0.14,
            "obb_max_age": 17,
            "obb_recovery_max_age": 9,
            "obb_second_iou_thresh": 0.16,
            "use_embeddings": False,
        }
    )
    c_cfg = native_binding._build_c_config(cfg)

    assert np.isclose(c_cfg.obb_det_thresh, 0.11)
    assert np.isclose(c_cfg.obb_iou_threshold, 0.12)
    assert np.isclose(c_cfg.obb_new_track_thresh, 0.13)
    assert np.isclose(c_cfg.obb_instant_confirm_thresh, 0.14)
    assert c_cfg.obb_max_age == 17
    assert c_cfg.obb_recovery_max_age == 9
    assert np.isclose(c_cfg.obb_second_iou_thresh, 0.16)


def _obb_tracker(**overrides):
    options = {
        "use_embeddings": False,
        "use_cmc": False,
        "det_thresh": 0.1,
        "iou_threshold": 0.01,
        "obb_det_thresh": 0.1,
        "obb_iou_threshold": 0.01,
        "new_track_thresh": 0.1,
        "obb_new_track_thresh": 0.1,
        "instant_confirm_thresh": 0.1,
        "obb_instant_confirm_thresh": 0.1,
        "confirm_hits": 1,
        "min_hits": 0,
        "aspect_ratio_thresh": 10.0,
        "min_box_area": 1,
    }
    options.update(overrides)
    library = native_binding.OccluBoostLibrary(native_binding.ensure_occluboost_cpp_library())
    return native_module.NativeOccluBoostTracker(
        options,
        geometry="obb",
        library=library,
    )


def test_native_occluboost_live_obb_damps_angular_velocity():
    tracker = _obb_tracker()

    def detection(angle: float) -> np.ndarray:
        return np.array([[80, 60, 80, 20, angle, 0.95, 0]], dtype=np.float32)

    try:
        first = update_rows(tracker, detection(0.2))
        second = update_rows(tracker, detection(0.7))
        third = update_rows(tracker, detection(0.7))
    finally:
        tracker.close()

    assert first.shape == second.shape == third.shape == (1, 9)
    assert first[0, 5] == second[0, 5] == third[0, 5]
    overshoot = float(third[0, 4] - 0.7)
    assert 0.0 < overshoot < (0.0009 * (0.7 - 0.2))


@pytest.mark.parametrize(
    "detection",
    (
        np.array([[40, 40, 2, 2, 0.2, 0.95, 0]], dtype=np.float32),
        np.array([[60, 40, 40, 4, 0.0, 0.95, 0]], dtype=np.float32),
    ),
)
def test_native_occluboost_obb_applies_geometry_filter(detection):
    tracker = _obb_tracker(
        use_dlo_boost=False,
        use_duo_boost=False,
        use_second_pass=False,
        min_box_area=10,
        aspect_ratio_thresh=5.0,
    )
    try:
        output = update_rows(tracker, detection)
    finally:
        tracker.close()
    assert output.shape == (0, 9)


def test_native_occluboost_obb_matches_python_lifecycle_with_external_embeddings():
    cfg = native_module._resolve_tracker_config(
        {
            "use_cmc": False,
            "use_embeddings": True,
            "min_box_area": 1,
            "aspect_ratio_thresh": 20.0,
            "second_pass_min_hits": 1,
        }
    )
    python_tracker = OccluBoost(**cfg, is_obb=True)
    library = native_binding.OccluBoostLibrary(native_binding.ensure_occluboost_cpp_library())
    native_tracker = native_module.NativeOccluBoostTracker(
        cfg,
        geometry="obb",
        library=library,
    )

    frames = []
    for frame_id in range(8):
        rows = []
        embeddings = []
        if frame_id != 5:
            rows.append(
                [45 + 8 * frame_id, 70 + 2 * frame_id, 42, 16, 0.15 + 0.05 * frame_id, 0.92, 0]
            )
            embeddings.append([1.0, 0.0, 0.0])
        if frame_id >= 2 and frame_id != 7:
            rows.append([190 - 7 * frame_id, 110 - frame_id, 28, 12, -1.45 + 0.04 * frame_id, 0.88, 1])
            embeddings.append([0.0, 1.0, 0.0])
        frames.append(
            (
                np.asarray(rows, dtype=np.float32).reshape(-1, 7),
                np.asarray(embeddings, dtype=np.float32).reshape(-1, 3),
            )
        )

    try:
        for rows, embeddings in frames:
            python_output = update_rows(python_tracker, rows, embeddings=embeddings)
            native_output = update_rows(native_tracker, rows, embeddings=embeddings)
            assert native_output.shape == python_output.shape
            np.testing.assert_allclose(native_output[:, :5], python_output[:, :5], atol=1e-4)
            np.testing.assert_allclose(native_output[:, 6:], python_output[:, 6:], atol=1e-6)
    finally:
        native_tracker.close()
