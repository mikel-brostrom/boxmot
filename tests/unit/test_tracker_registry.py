import pytest

from boxmot.trackers.registry import (
    REID_TRACKERS,
    TRACKER_CLASS_TO_NAME,
    TRACKER_DEFINITIONS,
    TRACKER_MAPPING,
    get_tracker_config,
    get_tracker_definition,
)
from boxmot.utils import TRACKER_CONFIGS


def test_tracker_public_mappings_are_derived_from_definitions():
    assert TRACKER_MAPPING == {
        name: definition.class_path
        for name, definition in TRACKER_DEFINITIONS.items()
    }
    assert REID_TRACKERS == [
        name
        for name, definition in TRACKER_DEFINITIONS.items()
        if definition.needs_reid
    ]
    assert TRACKER_CLASS_TO_NAME == {
        definition.class_name.lower(): name
        for name, definition in TRACKER_DEFINITIONS.items()
    }


def test_tracker_definition_captures_constructor_metadata():
    strongsort = get_tracker_definition("strongsort")
    bytetrack = get_tracker_definition("bytetrack")

    assert strongsort.needs_reid is True
    assert strongsort.accepts_per_class is False
    assert bytetrack.needs_reid is False
    assert bytetrack.accepts_per_class is True


def test_get_tracker_config_preserves_unknown_tracker_path_behavior():
    assert get_tracker_config("botsort") == TRACKER_CONFIGS / "botsort.yaml"
    assert get_tracker_config("unknown_tracker") == TRACKER_CONFIGS / "unknown_tracker.yaml"


def test_get_tracker_definition_rejects_unknown_tracker():
    with pytest.raises(ValueError, match="Unknown tracker type: 'unknown_tracker'"):
        get_tracker_definition("unknown_tracker")
