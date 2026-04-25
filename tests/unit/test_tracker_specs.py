from __future__ import annotations

import pytest

from boxmot.engine import workflow_support as support
from boxmot.trackers.specs import TrackerSpec, normalize_tracker_backend, parse_tracker_spec


def test_parse_tracker_spec_defaults_to_python_backend():
    parsed = parse_tracker_spec("bytetrack")

    assert parsed == TrackerSpec(name="bytetrack", backend="python")


@pytest.mark.parametrize(
    ("value", "expected_name", "expected_backend"),
    [
        ("botsort:cpp", "botsort", "cpp"),
        ("cpp:botsort", "botsort", "cpp"),
        ("botsort@cpp", "botsort", "cpp"),
        ("strongsort:native", "strongsort", "cpp"),
    ],
)
def test_parse_tracker_spec_supports_inline_backend_syntax(value, expected_name, expected_backend):
    parsed = parse_tracker_spec(value)

    assert parsed.name == expected_name
    assert parsed.backend == expected_backend


def test_normalize_tracker_backend_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown tracker backend"):
        normalize_tracker_backend("rust")


def test_workflow_support_extracts_name_and_backend_from_string_spec():
    assert support.tracker_name_from_spec("botsort:cpp") == "botsort"
    assert support.tracker_backend_from_spec("botsort:cpp") == "cpp"