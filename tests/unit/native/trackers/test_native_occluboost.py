from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxmot import KalmanConfig, OccluBoostConfig
from boxmot.native.trackers import occluboost as native_binding
from boxmot.trackers.common.config import load_tracker_config, nest_tracker_options
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


def test_native_occluboost_resolves_and_forwards_shared_thresholds() -> None:
    """Forward all seven shared operating-point values across the C ABI."""
    cfg = native_module._resolve_tracker_config(
        {
            "det_thresh": 0.11,
            "iou_threshold": 0.12,
            "new_track_thresh": 0.13,
            "instant_confirm_thresh": 0.14,
            "max_age": 17,
            "recovery_max_age": 9,
            "second_iou_thresh": 0.16,
            "use_embeddings": False,
        }
    )
    c_cfg = native_binding._build_c_config(cfg)

    assert np.isclose(c_cfg.det_thresh, 0.11)
    assert np.isclose(c_cfg.iou_threshold, 0.12)
    assert np.isclose(c_cfg.new_track_thresh, 0.13)
    assert np.isclose(c_cfg.instant_confirm_thresh, 0.14)
    assert c_cfg.max_age == 17
    assert c_cfg.recovery_max_age == 9
    assert np.isclose(c_cfg.second_iou_thresh, 0.16)


def test_native_occluboost_rejects_stale_config_abi_before_creating_a_tracker(monkeypatch) -> None:
    """Reject old libraries before passing them the shortened configuration struct."""
    calls = []
    stale_library = SimpleNamespace(boxmot_occluboost_create=lambda _config: calls.append("unsafe create"))
    monkeypatch.setattr(native_binding.ctypes, "CDLL", lambda _path: stale_library)
    with pytest.raises(RuntimeError, match="incompatible configuration ABI; rebuild the native library"):
        native_binding.OccluBoostLibrary(Path("stale-native-library"))
    assert calls == []


def _native_tracker(*, geometry: str = "obb", **overrides) -> native_module.NativeOccluBoostTracker:
    """Build a live tracker with permissive shared thresholds for either geometry."""
    options = {
        "use_embeddings": False,
        "use_cmc": False,
        "det_thresh": 0.1,
        "iou_threshold": 0.01,
        "new_track_thresh": 0.1,
        "instant_confirm_thresh": 0.1,
        "confirm_hits": 1,
        "min_hits": 0,
        "aspect_ratio_thresh": 10.0,
        "min_box_area": 1,
    }
    options.update(overrides)
    library = native_binding.OccluBoostLibrary(native_binding.ensure_occluboost_cpp_library())
    return native_module.NativeOccluBoostTracker(
        options,
        geometry=geometry,
        library=library,
    )


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("threshold", ["det_thresh", "new_track_thresh", "instant_confirm_thresh"])
def test_native_occluboost_shared_confidence_thresholds_control_birth(geometry: str, threshold: str) -> None:
    """The same confidence settings gate track output in both native kernels."""
    tracker = _native_tracker(
        geometry=geometry,
        **{threshold: 0.6},
        confirm_hits=2,
        use_dlo_boost=False,
        use_duo_boost=False,
        use_second_pass=False,
    )
    box = [40, 50, 120, 70] if geometry == "aabb" else [80, 60, 80, 20, 0.0]
    try:
        below = update_rows(tracker, np.array([[*box, 0.4, 0]], dtype=np.float32))
        tracker.reset()
        above = update_rows(tracker, np.array([[*box, 0.8, 0]], dtype=np.float32))
    finally:
        tracker.close()
    assert len(below) == 0
    assert len(above) == 1


@pytest.mark.parametrize("geometry", ["aabb", "obb"])
def test_native_occluboost_shared_max_age_expires_tracks(geometry: str) -> None:
    """A short canonical max_age prevents reusing an expired track's identity."""
    tracker = _native_tracker(geometry=geometry, max_age=1)
    box = [40, 50, 120, 70] if geometry == "aabb" else [80, 60, 80, 20, 0.0]
    detection = np.array([[*box, 0.95, 0]], dtype=np.float32)
    try:
        first = update_rows(tracker, detection)
        update_rows(tracker, np.empty((0, detection.shape[1]), dtype=np.float32))
        update_rows(tracker, np.empty((0, detection.shape[1]), dtype=np.float32))
        reborn = update_rows(tracker, detection)
    finally:
        tracker.close()
    assert len(first) == len(reborn) == 1
    assert first[0, len(box)] != reborn[0, len(box)]


def test_native_occluboost_live_obb_damps_angular_velocity():
    tracker = _native_tracker()

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
    tracker = _native_tracker(
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
    options = {
        "use_cmc": False,
        "use_embeddings": True,
        "min_box_area": 1,
        "aspect_ratio_thresh": 20.0,
        "second_pass_min_hits": 1,
    }
    python_config = nest_tracker_options(load_tracker_config("occluboost", None, options))
    kalman = KalmanConfig.from_mapping(python_config.pop("kalman"))
    python_tracker = OccluBoost(
        config=OccluBoostConfig(**{key: value for key, value in python_config.items() if key != "edgetam"}),
        edgetam=python_config.get("edgetam"),
        kalman=kalman,
        is_obb=True,
    )
    library = native_binding.OccluBoostLibrary(native_binding.ensure_occluboost_cpp_library())
    native_tracker = native_module.NativeOccluBoostTracker(
        options,
        geometry="obb",
        library=library,
    )

    frames = []
    for frame_id in range(8):
        rows = []
        embeddings = []
        if frame_id != 5:
            rows.append([45 + 8 * frame_id, 70 + 2 * frame_id, 42, 16, 0.15 + 0.05 * frame_id, 0.92, 0])
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
