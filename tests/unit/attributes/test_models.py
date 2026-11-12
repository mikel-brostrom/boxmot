import json
from dataclasses import FrozenInstanceError

import pytest

from boxmot.attributes.models import AttributeResult, AttributeSample


def test_result_roundtrip_preserves_owner_and_detaches_values():
    result = AttributeResult(
        run_id="run-a",
        sequence_id="camera-1",
        generation=2,
        track_id=7,
        class_id=0,
        model="vision",
        samples=(AttributeSample("sample-2", 2, 0.2), AttributeSample("sample-3", 3, 0.3)),
        status="success",
        values={"appearance": {"color": "red"}},
        display_fields=("appearance.color",),
    )
    serialized = result.to_dict()
    restored = AttributeResult.from_dict(json.loads(json.dumps(serialized)))
    assert restored == result
    serialized["values"]["appearance"]["color"] = "blue"
    assert result.values["appearance"]["color"] == "red"
    assert restored.values["appearance"]["color"] == "red"
    with pytest.raises(FrozenInstanceError):
        restored.track_id = 99


def test_error_result_roundtrip():
    result = AttributeResult("run", None, 0, 4, 2, "vision", (), "error", error="unavailable")
    assert AttributeResult.from_dict(result.to_dict()) == result
