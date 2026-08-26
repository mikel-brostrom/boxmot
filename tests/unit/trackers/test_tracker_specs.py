from __future__ import annotations

import pytest

from boxmot.engine.workflows import support
from boxmot.trackers.specs import TrackerSpec, normalize_tracker_backend, parse_tracker_spec


def test_parse_tracker_spec_defaults_to_python_backend():
    parsed = parse_tracker_spec("bytetrack")

    assert parsed == TrackerSpec(name="bytetrack", backend="python")


@pytest.mark.parametrize("value", ["botsort:cpp", "cpp:botsort", "botsort@cpp"])
def test_parse_tracker_spec_rejects_inline_backend_syntax(value):
    with pytest.raises(ValueError, match="tracker_backend"):
        parse_tracker_spec(value)


@pytest.mark.parametrize("value", ["py", "native", "c++"])
def test_normalize_tracker_backend_rejects_aliases(value):
    with pytest.raises(ValueError, match="Unknown tracker backend"):
        normalize_tracker_backend(value)


def test_normalize_tracker_backend_rejects_unknown_values():
    with pytest.raises(ValueError, match="Unknown tracker backend"):
        normalize_tracker_backend("rust")


def test_workflow_support_extracts_name_and_backend_from_string_spec():
    assert support.tracker_name_from_spec("botsort") == "botsort"
    assert support.tracker_backend_from_spec("botsort") == "python"
