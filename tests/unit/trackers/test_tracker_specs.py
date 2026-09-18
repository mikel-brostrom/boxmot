from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from boxmot import ByteTrack
from boxmot.structures import GeometryKind
from boxmot.trackers.common.registry import TRACKER_CLASS_SPECS
from boxmot.trackers.common.specs import (
    TrackerCapabilities,
    TrackerFamily,
    TrackerSpec,
    normalize_tracker_backend,
    parse_tracker_spec,
)


def test_tracker_capabilities_are_frozen_typed_and_use_shared_geometry_kinds() -> None:
    capabilities = TrackerCapabilities(
        family=TrackerFamily.BOX,
        geometry_kinds=frozenset({GeometryKind.AABB, GeometryKind.OBB}),
        requires_embeddings=True,
        accepts_embeddings=True,
    )

    assert str(TrackerFamily.MULTIMODAL) == "multimodal"
    assert str(GeometryKind.OBB) == "obb"
    assert hash(capabilities)
    with pytest.raises(FrozenInstanceError):
        capabilities.requires_embeddings = False


@pytest.mark.parametrize("input_name", ("embeddings", "masks", "frame", "detections_3d", "camera", "ego_motion"))
def test_tracker_capabilities_requirements_must_also_be_accepted(input_name: str) -> None:
    kwargs = {f"requires_{input_name}": True}
    with pytest.raises(ValueError, match=f"requiring {input_name}"):
        TrackerCapabilities(
            family=TrackerFamily.BOX,
            geometry_kinds=frozenset({GeometryKind.AABB}),
            **kwargs,
        )


def test_tracker_capabilities_reject_untyped_or_empty_geometry_sets() -> None:
    with pytest.raises(TypeError, match="must be a frozenset"):
        TrackerCapabilities(TrackerFamily.BOX, {GeometryKind.AABB})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        TrackerCapabilities(TrackerFamily.BOX, frozenset())
    with pytest.raises(TypeError, match="only GeometryKind"):
        TrackerCapabilities(TrackerFamily.BOX, frozenset({"aabb"}))  # type: ignore[arg-type]


def test_parse_tracker_spec_defaults_to_python_backend():
    parsed = parse_tracker_spec("bytetrack")

    assert parsed == TrackerSpec(name="bytetrack", backend="python")


@pytest.mark.parametrize("value", ["botsort:cpp", "cpp:botsort", "botsort@cpp"])
def test_parse_tracker_spec_rejects_inline_backend_syntax(value):
    with pytest.raises(ValueError, match="tracker_backend"):
        parse_tracker_spec(value)


@pytest.mark.parametrize("value", ["", "py", "native", "c++", "CPP", " cpp "])
def test_normalize_tracker_backend_rejects_aliases(value):
    with pytest.raises(ValueError, match="Unknown tracker backend"):
        normalize_tracker_backend(value)


def test_normalize_tracker_backend_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown tracker backend"):
        normalize_tracker_backend("rust")


@pytest.mark.parametrize("value", ["BoTSORT", "OCSORT", " botsort", "botsort "])
def test_parse_tracker_spec_rejects_noncanonical_tracker_names(value):
    with pytest.raises(ValueError, match="canonical lowercase identifier"):
        parse_tracker_spec(value)


def test_parse_tracker_spec_accepts_registered_class_and_instance():
    tracker = ByteTrack()

    assert parse_tracker_spec(ByteTrack, class_specs=TRACKER_CLASS_SPECS) == TrackerSpec("bytetrack")
    assert parse_tracker_spec(tracker, class_specs=TRACKER_CLASS_SPECS) == TrackerSpec("bytetrack")


def test_parse_tracker_spec_rejects_matching_unregistered_class_name():
    impostor = type("ByteTrack", (), {"__module__": "third_party"})

    with pytest.raises(ValueError, match="not registered"):
        parse_tracker_spec(impostor, class_specs=TRACKER_CLASS_SPECS)


def test_parse_tracker_spec_rejects_tracker_name_duck_typing():
    class ExternalTracker:
        tracker_name = "bytetrack"

    with pytest.raises(ValueError, match="not registered"):
        parse_tracker_spec(ExternalTracker(), class_specs=TRACKER_CLASS_SPECS)


def test_registered_class_backend_cannot_be_overridden_by_an_attribute():
    tracker = ByteTrack()
    tracker.tracker_backend = "cpp"

    assert parse_tracker_spec(tracker, class_specs=TRACKER_CLASS_SPECS) == TrackerSpec("bytetrack", "python")


def test_tracker_constructor_rejects_removed_private_name_override():
    with pytest.raises(TypeError, match="unexpected keyword argument '_tracker_name'"):
        ByteTrack(_tracker_name="BYTETracker")
