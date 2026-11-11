"""Choose a fixed tracker geometry from validated custom detection batches."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from boxmot import create_tracker
from boxmot.structures import Boxes, Detections, GeometryKind, OrientedBoxes, Tracks
from boxmot.trackers.common import factory
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.mask_guidance import MaskGuidanceConfig
from boxmot.trackers.common.motion.kalman_filters.profile import calibration_profile_signature
from boxmot.trackers.common.protocols import TrackerRequirements
from boxmot.trackers.common.specs import TrackerSpec


def _detections(geometry: str, *, packed: bool = False, empty: bool = False) -> Detections | np.ndarray:
    """Build equivalent typed and packed inputs, including empty shaped batches."""
    oriented = geometry == "obb"
    coordinates = [[20.0, 24.0, 12.0, 20.0, 0.2]] if oriented else [[14.0, 14.0, 26.0, 34.0]]
    values = torch.tensor(coordinates, dtype=torch.float32)
    count = 0 if empty else 1
    detections = Detections(
        geometry=(OrientedBoxes if oriented else Boxes)(values[:count]),
        scores=torch.tensor([0.95], dtype=torch.float32)[:count],
        class_ids=torch.tensor([0], dtype=torch.int64)[:count],
    )
    if packed:
        return (detections.to_obb_rows() if oriented else detections.to_aabb_rows()).numpy()
    return detections


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("packed", (False, True), ids=("typed", "packed"))
@pytest.mark.parametrize("empty", (False, True), ids=("populated", "empty"))
def test_detection_batch_selects_geometry_without_consuming_first_update(
    geometry: str, packed: bool, empty: bool
) -> None:
    """Construction fixes geometry even with no detections and never advances time."""
    detections = _detections(geometry, packed=packed, empty=empty)
    before = detections.copy() if packed else detections.geometry.values.clone()

    tracker = create_tracker("bytetrack", detections=detections)

    assert tracker.is_obb is (geometry == "obb")
    assert tracker.frame_count == 0
    assert not tracker.active_tracks
    if packed:
        np.testing.assert_array_equal(detections, before)
    else:
        torch.testing.assert_close(detections.geometry.values, before)

    tracks = tracker.update(detections)

    assert tracker.frame_count == 1
    assert len(tracks) == (0 if empty else 1)
    if packed:
        assert tracks.shape[1] == (9 if geometry == "obb" else 8)
    else:
        assert isinstance(tracks, Tracks)
        assert isinstance(tracks.geometry, OrientedBoxes if geometry == "obb" else Boxes)


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("packed", (False, True), ids=("typed", "packed"))
def test_geometry_changes_are_rejected_without_advancing_state(geometry: str, packed: bool) -> None:
    """Inference establishes a persistent state model rather than switching each frame."""
    first = _detections(geometry, packed=packed)
    tracker = create_tracker("bytetrack", detections=first)
    tracker.update(first)
    track_ids = [track.id for track in tracker.active_tracks]
    opposite = "aabb" if geometry == "obb" else "obb"

    with pytest.raises(ValueError):
        tracker.update(_detections(opposite, packed=packed))

    assert tracker.frame_count == 1
    assert [track.id for track in tracker.active_tracks] == track_ids
    assert tracker.is_obb is (geometry == "obb")


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("packed", (False, True), ids=("typed", "packed"))
def test_inferred_geometry_reaches_native_constructor(
    geometry: str, packed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise native dispatch without loading or compiling a native library."""
    calls: list[tuple[dict[str, Any], str]] = []

    def construct(options: dict[str, Any], *, geometry: str) -> SimpleNamespace:
        calls.append((options, geometry))
        return SimpleNamespace(requirements=TrackerRequirements())

    monkeypatch.setattr(factory, "_load_native_tracker_class", lambda _: construct)

    tracker = create_tracker("bytetrack", backend="cpp", detections=_detections(geometry, packed=packed))

    assert len(calls) == 1
    options, selected_geometry = calls[0]
    assert selected_geometry == geometry
    assert "detections" not in options
    assert GeometryKind(geometry) in tracker.capabilities.geometry_kinds


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("selection", ("keyword", "spec"))
def test_explicit_geometry_constrains_supplied_detections(geometry: str, selection: str) -> None:
    """Both public selection forms reject a conflicting detection schema."""
    opposite = "aabb" if geometry == "obb" else "obb"
    spec = TrackerSpec("bytetrack", geometry=geometry) if selection == "spec" else "bytetrack"
    overrides = {} if selection == "spec" else {"geometry": geometry}

    with pytest.raises(ValueError, match="conflicts with requested tracker geometry"):
        create_tracker(spec, detections=_detections(opposite), **overrides)

    tracker = create_tracker(spec, detections=_detections(geometry), **overrides)
    assert tracker.is_obb is (geometry == "obb")


def test_default_geometry_in_an_explicit_spec_remains_a_constraint() -> None:
    """A concrete AABB spec is never silently rewritten to match OBB observations."""
    with pytest.raises(ValueError, match="conflicts with requested tracker geometry"):
        create_tracker(TrackerSpec("bytetrack"), detections=_detections("obb"))


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_geometry_keyword_overrides_spec_before_checking_detections(geometry: str) -> None:
    """Existing keyword precedence is preserved without mutating the source spec."""
    original_geometry = "aabb" if geometry == "obb" else "obb"
    spec = TrackerSpec("bytetrack", geometry=original_geometry)

    tracker = create_tracker(spec, geometry=geometry, detections=_detections(geometry))

    assert tracker.is_obb is (geometry == "obb")
    assert spec.geometry == original_geometry
    with pytest.raises(ValueError, match="conflicts with requested tracker geometry"):
        create_tracker(spec, geometry=geometry, detections=_detections(original_geometry))


def test_unsupported_inferred_geometry_fails_before_tracker_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference must not bypass a tracker's supported geometry contract."""
    monkeypatch.setattr(factory, "_load_tracker_class", lambda _: pytest.fail("Unexpected tracker construction"))

    with pytest.raises(ValueError, match="does not support geometry kind 'obb'"):
        create_tracker("edgetam", detections=_detections("obb"))


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_calibration_uses_inferred_geometry(geometry: str) -> None:
    """A matching profile loads; a profile fitted for the other geometry fails."""
    defaults = load_tracker_defaults("bytetrack")
    matching = calibration_profile_signature("bytetrack", geometry, defaults)
    tracker = create_tracker("bytetrack", detections=_detections(geometry), options={"calibration": matching})
    assert tracker.is_obb is (geometry == "obb")

    opposite = "aabb" if geometry == "obb" else "obb"
    conflicting = calibration_profile_signature("bytetrack", opposite, defaults)
    with pytest.raises(ValueError, match="Calibrated tracker profile geometry=.*incompatible"):
        create_tracker("bytetrack", detections=_detections(geometry), options={"calibration": conflicting})


@pytest.mark.parametrize("shape", ((0,), (6,), (0, 5), (0, 8), (1, 1, 6)))
def test_malformed_shapes_cannot_choose_geometry(shape: tuple[int, ...]) -> None:
    """Neither unshaped empty input nor track-output columns are detection schemas."""
    with pytest.raises(ValueError, match="Cannot infer detection geometry"):
        create_tracker("bytetrack", detections=np.empty(shape, dtype=np.float32))


@pytest.mark.parametrize("dtype", (np.bool_, np.complex64, object, np.str_))
def test_non_real_numeric_dtypes_cannot_choose_geometry(dtype: Any) -> None:
    """Column count alone must not accept invalid packed input."""
    with pytest.raises(TypeError, match="real numeric dtype"):
        create_tracker("bytetrack", detections=np.empty((0, 6), dtype=dtype))


@pytest.mark.parametrize(
    ("geometry", "column", "value", "message"),
    (
        ("aabb", 0, float("nan"), "finite"),
        ("aabb", 2, 0.0, "x2 > x1"),
        ("obb", 2, 0.0, "positive width"),
        ("aabb", -2, 1.1, "scores"),
        ("obb", -1, 0.5, "class IDs"),
    ),
)
def test_invalid_detection_values_fail_before_tracker_construction(
    geometry: str, column: int, value: float, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate box coordinates, confidence, and classes before allocating state."""
    detections = _detections(geometry, packed=True)
    detections[0, column] = value
    monkeypatch.setattr(factory, "_load_tracker_class", lambda _: pytest.fail("Unexpected tracker construction"))

    with pytest.raises(ValueError, match=message):
        create_tracker("bytetrack", detections=detections)


def test_canonical_detections_are_revalidated_before_geometry_inference() -> None:
    """Canonical tensor contents can change after their initial construction."""
    detections = _detections("obb")
    detections.scores[0] = -1.0

    with pytest.raises(ValueError, match="scores"):
        create_tracker("bytetrack", detections=detections)


def test_array_subclasses_and_lists_are_not_accepted_as_detection_batches() -> None:
    """Keep factory inference aligned with the tracker's plain-NumPy boundary."""
    detections = _detections("aabb", packed=True)
    for invalid in (np.ma.array(detections), detections.tolist()):
        with pytest.raises(TypeError, match="plain numpy.ndarray"):
            create_tracker("bytetrack", detections=invalid)


@pytest.mark.parametrize("spec", ("bytetrack", TrackerSpec("bytetrack")))
def test_omitting_detections_keeps_aabb_construction_default(spec: str | TrackerSpec) -> None:
    """The optional input batch does not change existing calls without detections."""
    tracker = create_tracker(spec)
    assert tracker.is_obb is False
    assert tracker.frame_count == 0


@pytest.mark.parametrize("packed", (False, True), ids=("typed", "packed"))
def test_inferred_aabb_preserves_optional_mask_guidance(packed: bool) -> None:
    """Geometry inference retains the existing lazily initialized mask component."""
    guidance = MaskGuidanceConfig("unused.pt", max_objects=4)

    tracker = create_tracker("bytetrack", detections=_detections("aabb", packed=packed), mask_guidance=guidance)

    assert tracker.is_obb is False
    assert tracker.requirements.frame_pixels
    assert tracker._mask_guidance.config == guidance
    assert tracker._mask_guidance._propagator is None
    assert tracker.frame_count == 0


@pytest.mark.parametrize("packed", (False, True), ids=("typed", "packed"))
def test_inferred_obb_rejects_incompatible_mask_guidance(packed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference must still enforce the guidance component's AABB-only contract."""
    monkeypatch.setattr(factory, "_load_tracker_class", lambda _: pytest.fail("Unexpected tracker construction"))

    with pytest.raises(ValueError, match="Mask guidance requires AABB"):
        create_tracker(
            "bytetrack", detections=_detections("obb", packed=packed), mask_guidance=MaskGuidanceConfig("unused.pt")
        )
