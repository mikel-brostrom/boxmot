from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from boxmot.native.trackers import sfsort as native_binding
from boxmot.structures import Tracks
from boxmot.trackers.common.protocols import TrackerRequirements
from boxmot.trackers.sfsort import native as native_module
from boxmot.trackers.sfsort.tracker import SFSORT

from ._helpers import detections_from_rows, empty_native_batch, frame_from_bgr, update_rows


class _FakeLibrary:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def create(self, cfg: dict[str, Any]) -> str:
        self.calls.append(("create", cfg["high_th"], cfg["dynamic_tuning"]))
        return "handle"

    def reset(self, handle: str) -> None:
        self.calls.append(("reset", handle))

    def update(
        self,
        handle: str,
        *,
        geometry,
        scores,
        class_ids,
        detection_indices,
        embeddings,
        image,
    ):
        image_shape = None if image is None else image.shape
        self.calls.append(("update", handle, len(geometry), image_shape, geometry.shape[1], embeddings))
        return empty_native_batch(geometry.shape[1])

    def destroy(self, handle: str) -> None:
        self.calls.append(("destroy", handle))


def test_native_sfsort_routes_canonical_structures_through_live_library() -> None:
    library = _FakeLibrary()
    tracker = native_module.NativeSFSORTTracker(
        {"high_th": 0.55, "dynamic_tuning": True},
        geometry="aabb",
        library=library,
    )
    detections = detections_from_rows(
        np.array(
            [[1, 1, 4, 5, 0.9, 0], [2, 2, 6, 7, 0.8, 0]],
            dtype=np.float32,
        )
    )
    frame = frame_from_bgr(np.zeros((8, 8, 3), dtype=np.uint8))

    output = tracker.update(detections, frame)
    tracker.reset()
    tracker.close()

    assert isinstance(output, Tracks)
    assert output.sample_id == detections.sample_id
    assert output.to_aabb_rows().shape == (0, 8)
    assert library.calls == [
        ("create", 0.55, True),
        ("update", "handle", 2, (8, 8, 3), 4, None),
        ("reset", "handle"),
        ("destroy", "handle"),
    ]


def test_native_sfsort_accepts_numpy_aabb6_when_frame_requirement_is_met() -> None:
    library = _FakeLibrary()
    tracker = native_module.NativeSFSORTTracker(geometry="aabb", library=library)
    rows = np.array([[1, 1, 4, 5, 0.9, 0]], dtype=np.float64)
    frame = frame_from_bgr(np.zeros((8, 8, 3), dtype=np.uint8), sample_id="camera-1:000042")

    try:
        with pytest.raises(ValueError, match="requires a frame"):
            tracker.update(rows)

        output = tracker.update(rows, frame)
        assert type(output) is np.ndarray
        assert output.shape == (0, 8)
        assert library.calls[1] == ("update", "handle", 1, (8, 8, 3), 4, None)
    finally:
        tracker.close()


@pytest.mark.parametrize(
    ("options", "requirements"),
    [
        (None, TrackerRequirements(frame=True, frame_dimensions_only=True)),
        ({"central_timeout": 5, "marginal_timeout": 1}, TrackerRequirements(frame=True, frame_dimensions_only=True)),
        ({"asso_func": "centroid"}, TrackerRequirements(frame=True, frame_dimensions_only=True)),
        ({"frame_width": 640, "frame_height": 480}, TrackerRequirements()),
        (
            {"central_timeout": 5, "marginal_timeout": 1, "frame_width": 640, "frame_height": 480},
            TrackerRequirements(),
        ),
        ({"asso_func": "centroid", "frame_width": 640, "frame_height": 480}, TrackerRequirements()),
    ],
)
def test_native_sfsort_requirements_are_frozen_from_configuration(
    options: dict[str, Any] | None,
    requirements: TrackerRequirements,
) -> None:
    tracker = native_module.NativeSFSORTTracker(options, library=_FakeLibrary())
    try:
        assert tracker.requirements == requirements
        assert tracker.supports_masks is False
        assert tracker.use_embeddings is False
    finally:
        tracker.close()


@pytest.mark.parametrize(
    "options",
    (
        {"frame_width": 640},
        {"frame_height": 480},
        {"frame_width": 0, "frame_height": 0},
        {"frame_width": -1, "frame_height": 480},
        {"frame_width": 640, "frame_height": -1},
    ),
)
def test_native_sfsort_rejects_partial_or_negative_frame_dimensions(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="frame_width and frame_height"):
        native_module.NativeSFSORTTracker(options, library=_FakeLibrary())


@pytest.mark.parametrize(
    "options",
    (
        {"frame_width": True, "frame_height": 480},
        {"frame_width": 640, "frame_height": "480"},
    ),
)
def test_native_sfsort_rejects_noninteger_frame_dimensions(options: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        native_module.NativeSFSORTTracker(options, library=_FakeLibrary())


def test_native_sfsort_geometry_mode_is_fixed_at_construction() -> None:
    tracker = native_module.NativeSFSORTTracker(geometry="aabb", library=_FakeLibrary())
    detections = detections_from_rows(np.array([[4, 5, 3, 2, 0.1, 0.9, 0]], dtype=np.float32))
    frame = frame_from_bgr(np.zeros((12, 12, 3), dtype=np.uint8))

    try:
        with pytest.raises(ValueError, match="fixed to AABB geometry"):
            tracker.update(detections, frame)
    finally:
        tracker.close()


def test_native_sfsort_accepts_numpy_obb7_with_frame() -> None:
    library = _FakeLibrary()
    tracker = native_module.NativeSFSORTTracker(geometry="obb", library=library)
    rows = np.array([[4, 5, 3, 2, 0.1, 0.9, 0]], dtype=np.float64)
    frame = frame_from_bgr(np.zeros((12, 12, 3), dtype=np.uint8))

    try:
        output = tracker.update(rows, frame)
    finally:
        tracker.close()

    assert type(output) is np.ndarray
    assert output.shape == (0, 9)
    assert library.calls[1] == ("update", "handle", 1, (12, 12, 3), 5, None)


def test_native_sfsort_live_obb_cost_is_equivalent_form_invariant() -> None:
    library = native_binding.SFSORTLibrary(native_binding.ensure_sfsort_cpp_library())
    tracker = native_module.NativeSFSORTTracker(
        {
            "high_th": 0.5,
            "new_track_th": 0.5,
            "low_th": 0.1,
            "match_th_first": 0.1,
            "dynamic_tuning": False,
            "frame_width": 160,
            "frame_height": 120,
        },
        geometry="obb",
        library=library,
    )
    first = np.array(
        [[80, 60, 80, 20, (4 * np.pi) + 0.2, 0.95, 0]],
        dtype=np.float32,
    )
    equivalent = np.array(
        [[80, 60, 20, 80, 0.2 + (np.pi / 2), 0.95, 0]],
        dtype=np.float32,
    )
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    try:
        first_output = update_rows(tracker, first, image)
        equivalent_output = update_rows(tracker, equivalent, image)
    finally:
        tracker.close()

    assert first_output.shape == (1, 9)
    assert equivalent_output.shape == (1, 9)
    # Canonical OBB angles intentionally remain finite and unwrapped.
    assert np.isfinite(first_output[0, 4])
    assert equivalent_output[0, 5] == first_output[0, 5]
    np.testing.assert_allclose(equivalent_output[0, :5], first_output[0, :5], atol=1e-4)


@pytest.mark.parametrize("common_rotation", [0.0, 0.73])
def test_native_sfsort_obb_directional_center_penalty_matches_python(common_rotation: float) -> None:
    cfg = {
        "high_th": 0.1,
        "new_track_th": 0.1,
        "low_th": 0.01,
        "match_th_first": 0.55,
        "dynamic_tuning": False,
        "frame_width": 320,
        "frame_height": 240,
    }
    center = np.array([150.0, 100.0])
    short_axis_move = np.array([-50.0 / np.sqrt(2.0), 50.0 / np.sqrt(2.0)])
    cosine = np.cos(common_rotation)
    sine = np.sin(common_rotation)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    moved_center = center + rotation @ short_axis_move
    angle = (np.pi / 4.0) + common_rotation
    first_detection = np.array([[*center, 100.0, 10.0, angle, 0.999, 0]], dtype=np.float32)
    moved_detection = np.array([[*moved_center, 100.0, 10.0, angle, 0.999, 0]], dtype=np.float32)

    python_tracker = SFSORT(is_obb=True, **cfg)
    library = native_binding.SFSORTLibrary(native_binding.ensure_sfsort_cpp_library())
    native_tracker = native_module.NativeSFSORTTracker(cfg, geometry="obb", library=library)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        python_first = update_rows(python_tracker, first_detection, image)
        native_first = update_rows(native_tracker, first_detection, image)
        python_moved = update_rows(python_tracker, moved_detection, image)
        native_moved = update_rows(native_tracker, moved_detection, image)
    finally:
        native_tracker.close()

    np.testing.assert_allclose(native_first, python_first, atol=1e-5)
    np.testing.assert_allclose(native_moved, python_moved, atol=1e-5)
    assert python_moved[0, 5] != python_first[0, 5]
    assert native_moved[0, 5] != native_first[0, 5]


def test_native_sfsort_tiny_obb_shape_cost_matches_python() -> None:
    cfg = {
        "high_th": 0.1,
        "new_track_th": 0.1,
        "low_th": 0.01,
        "match_th_first": 0.5,
        "dynamic_tuning": False,
        "frame_width": 320,
        "frame_height": 240,
    }
    first_detection = np.array([[0, 0, 4e-9, 2e-9, 0.3, 0.999, 0]], dtype=np.float32)
    candidate_detection = np.array([[0, 0, 3e-9, 1.5e-9, -0.2, 0.999, 0]], dtype=np.float32)
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    python_tracker = SFSORT(is_obb=True, **cfg)
    library = native_binding.SFSORTLibrary(native_binding.ensure_sfsort_cpp_library())
    native_tracker = native_module.NativeSFSORTTracker(cfg, geometry="obb", library=library)
    try:
        python_first = update_rows(python_tracker, first_detection, image)
        native_first = update_rows(native_tracker, first_detection, image)
        python_candidate = update_rows(python_tracker, candidate_detection, image)
        native_candidate = update_rows(native_tracker, candidate_detection, image)
    finally:
        native_tracker.close()

    assert python_first.shape == native_first.shape == (1, 9)
    assert python_candidate.shape == native_candidate.shape == (1, 9)
    assert python_candidate[0, 5] == python_first[0, 5]
    assert native_candidate[0, 5] == native_first[0, 5]


def test_native_sfsort_low_only_obb_frame_keeps_track() -> None:
    library = native_binding.SFSORTLibrary(native_binding.ensure_sfsort_cpp_library())
    tracker = native_module.NativeSFSORTTracker(
        {
            "high_th": 0.6,
            "new_track_th": 0.5,
            "low_th": 0.1,
            "match_th_second": 0.3,
            "dynamic_tuning": False,
            "frame_width": 160,
            "frame_height": 120,
        },
        geometry="obb",
        library=library,
    )
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    high = np.array([[80, 60, 40, 20, 0.2, 0.9, 0]], dtype=np.float32)
    low = high.copy()
    low[:, 5] = 0.3

    try:
        first = update_rows(tracker, high, image)
        second = update_rows(tracker, low, image)
    finally:
        tracker.close()

    assert first.shape == second.shape == (1, 9)
    assert second[0, 5] == first[0, 5]


@pytest.mark.parametrize(
    ("initial_detections", "ambiguous_detections", "match_threshold"),
    [
        (
            [[0, 0, 10, 10, 0.95, 0], [8, 0, 18, 10, 0.95, 0]],
            [[-4, 0, 2, 10, 0.95, 0], [1, 0, 13, 10, 0.95, 0]],
            0.2,
        ),
        (
            [[5, 5, 10, 10, 0, 0.95, 0], [13, 5, 10, 10, 0, 0.95, 0]],
            [[3, 5, 6, 10, 0, 0.95, 0], [6, 5, 10, 10, 0, 0.95, 0]],
            0.1,
        ),
    ],
    ids=["aabb", "obb"],
)
def test_native_sfsort_threshold_aware_assignment_matches_python(
    initial_detections: list[list[float]],
    ambiguous_detections: list[list[float]],
    match_threshold: float,
) -> None:
    """Keep the same valid identity selected by threshold-aware assignment."""
    cfg = {
        "high_th": 0.6,
        "new_track_th": 0.7,
        "low_th": 0.1,
        "match_th_first": match_threshold,
        "match_th_second": 0.3,
        "dynamic_tuning": False,
        "frame_width": 100,
        "frame_height": 100,
        "horizontal_margin": 0,
        "vertical_margin": 0,
    }
    is_obb = len(initial_detections[0]) == 7
    geometry = "obb" if is_obb else "aabb"
    python_tracker = SFSORT(is_obb=is_obb, **cfg)
    library = native_binding.SFSORTLibrary(native_binding.ensure_sfsort_cpp_library())
    native_tracker = native_module.NativeSFSORTTracker(cfg, geometry=geometry, library=library)
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        for rows in (initial_detections, ambiguous_detections):
            detections = np.asarray(rows, dtype=np.float32)
            python_output = update_rows(python_tracker, detections, image)
            native_output = update_rows(native_tracker, detections, image)
            np.testing.assert_allclose(native_output, python_output, atol=1e-5)
    finally:
        native_tracker.close()
