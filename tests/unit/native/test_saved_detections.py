"""Saved KITTI boxes reach compiled trackers in the native CI suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from boxmot.engine.eval import saved_detections
from tests.unit.engine.eval.test_saved_detections import _args, _dataset, _Encoder, _stub_encoder


@pytest.mark.parametrize("tracker", ("botsort", "occluboost"))
def test_saved_boxes_support_native_trackers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tracker: str
) -> None:
    """Exercise real C++ tracking and built-in scoring without optional TrackEval."""
    monkeypatch.setitem(sys.modules, "trackeval", None)
    dataset = _dataset(tmp_path)
    _stub_encoder(monkeypatch, _Encoder())
    result = saved_detections.run_saved_detections(_args(tmp_path, dataset, tracker=tracker, tracker_backend="cpp"))
    assert result.raw["car"]["HOTA"] == result.raw["pedestrian"]["HOTA"] == 100
    metadata = json.loads((result.exp_dir / "run.json").read_text())
    assert metadata["tracker_backend"] == "cpp"
    assert metadata["per_class"] is False
