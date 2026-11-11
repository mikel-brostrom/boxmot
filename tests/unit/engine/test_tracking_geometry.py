"""Live tracker geometry follows the detector's resolved output contract."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from types import SimpleNamespace

import pytest
import torch
import yaml

from boxmot import create_tracker
from boxmot.detectors import DetectorCapabilities, DetectorSpec
from boxmot.engine.tracking import workflow
from boxmot.engine.tracking.sinks import NullSink
from boxmot.structures import Boxes, Detections, Frame, OrientedBoxes


class _Source:
    """Provide a small stream that starts with an empty detection frame."""

    def __init__(self, frames: int = 2) -> None:
        self.frames = frames
        self.closed = False

    def __iter__(self) -> Iterator[Frame]:
        for index in range(self.frames):
            yield Frame(
                torch.zeros((3, 48, 48), dtype=torch.uint8),
                sample_id=f"frame-{index}",
                sequence_id="sequence",
                frame_index=index,
            )

    def close(self) -> None:
        self.closed = True


class _Detector:
    """Emit canonical geometry even when no objects were detected."""

    def __init__(self, geometry: str) -> None:
        self.geometry = geometry
        self.capabilities = DetectorCapabilities(supports_aabb=geometry == "aabb", supports_obb=geometry == "obb")

    def predict(self, frames: Sequence[Frame]) -> list[Detections]:
        results = []
        for frame in frames:
            rows = [] if frame.frame_index == 0 else [[8.0, 8.0, 12.0, 14.0]]
            if self.geometry == "obb":
                rows = [row + [0.2] for row in rows]
                geometry = OrientedBoxes(torch.tensor(rows, dtype=torch.float32).reshape(-1, 5))
            else:
                geometry = Boxes(torch.tensor(rows, dtype=torch.float32).reshape(-1, 4))
            results.append(
                Detections(
                    geometry=geometry,
                    scores=torch.full((len(rows),), 0.9),
                    class_ids=torch.zeros(len(rows), dtype=torch.int64),
                    sample_id=frame.sample_id,
                )
            )
        return results


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_auto_geometry_uses_loaded_detector_before_tracker_configuration(monkeypatch, geometry: str) -> None:
    """Unknown checkpoint names and empty first frames retain the loaded geometry."""
    selected_specs = []
    detector_constraints = []
    tracker_geometry = []
    source = _Source()
    args = SimpleNamespace(detector="custom-checkpoint.pt", tracker="bytetrack", geometry="auto")

    def resolve_detector(reference, **kwargs):
        assert reference == "custom-checkpoint.pt"
        detector_constraints.append(kwargs["geometry"])
        return DetectorSpec("fixture"), {}

    def construct_detector(spec):
        selected_specs.append(spec)
        return _Detector(geometry)

    def construct_tracker(spec):
        assert args.geometry == geometry
        tracker_geometry.append(spec.geometry)
        return create_tracker(spec)

    monkeypatch.setattr(workflow, "resolve_detector_spec", resolve_detector)
    monkeypatch.setattr(workflow, "create_detector", construct_detector)
    monkeypatch.setattr(workflow, "create_tracker", construct_tracker)

    run = workflow.run_track(args, source=source, sinks=(NullSink(),))

    assert detector_constraints == ["auto"]
    assert selected_specs[0].geometry_mode == "auto"
    assert tracker_geometry == [geometry]
    assert run.summary.frames == 2
    assert run.summary.detections == 1
    assert source.closed


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_empty_source_still_resolves_detector_geometry(monkeypatch, geometry: str) -> None:
    """No first detection or warmup inference is required to select tracker state."""
    configured = []
    source = _Source(frames=0)

    def construct_tracker(spec):
        configured.append(spec.geometry)
        return create_tracker(spec)

    monkeypatch.setattr(workflow, "create_tracker", construct_tracker)
    run = workflow.run_track(tracker="bytetrack", detector=_Detector(geometry), source=source, sinks=(NullSink(),))

    assert configured == [geometry]
    assert run.summary.frames == 0
    assert source.closed


@pytest.mark.parametrize(("detected", "requested"), (("aabb", "obb"), ("obb", "aabb")))
def test_explicit_geometry_rejects_detector_mismatch(monkeypatch, detected: str, requested: str) -> None:
    monkeypatch.setattr(workflow, "create_tracker", lambda *_: pytest.fail("Mismatched tracker constructed"))

    with pytest.raises(ValueError, match="conflicts with requested geometry"):
        workflow.run_track(tracker="bytetrack", detector=_Detector(detected), geometry=requested)


@pytest.mark.parametrize(("detected", "configured"), (("aabb", "obb"), ("obb", "aabb")))
def test_injected_tracker_keeps_its_fixed_geometry(detected: str, configured: str) -> None:
    tracker = create_tracker("bytetrack", geometry=configured)

    with pytest.raises(ValueError, match="Injected tracker is configured"):
        workflow.run_track(detector=_Detector(detected), tracker=tracker)

    assert tracker.is_obb is (configured == "obb")
    assert tracker.frame_count == 0


def test_ambiguous_detector_requires_explicit_geometry() -> None:
    detector = _Detector("obb")
    detector.capabilities = DetectorCapabilities(supports_aabb=True, supports_obb=True)

    with pytest.raises(ValueError, match="Cannot infer tracking geometry"):
        workflow.run_track(tracker="bytetrack", detector=detector)


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_explicit_geometry_supports_ambiguous_detector(geometry: str) -> None:
    detector = _Detector(geometry)
    detector.capabilities = DetectorCapabilities(supports_aabb=True, supports_obb=True)

    run = workflow.run_track(
        tracker="bytetrack", detector=detector, geometry=geometry, source=_Source(), sinks=(NullSink(),)
    )

    assert run.summary.frames == 2
    assert run.summary.detections == 1


@pytest.mark.parametrize("tracker", ("does-not-exist", "eagermot"))
def test_invalid_tracker_is_rejected_before_detector_loading(monkeypatch, tracker: str) -> None:
    monkeypatch.setattr(workflow, "create_detector", lambda *_: pytest.fail("Unexpected detector loading"))

    with pytest.raises(ValueError):
        workflow.run_track(tracker=tracker, detector="custom-checkpoint.pt")


def test_invalid_tracker_timing_is_rejected_before_detector_loading(monkeypatch) -> None:
    monkeypatch.setattr(workflow, "_detector_spec", lambda *_: pytest.fail("Unexpected detector resolution"))

    with pytest.raises(TypeError, match="kalman.variable_dt must be bool"):
        workflow.run_track(tracker="bytetrack", variable_dt="yes")


@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("matching", (True, False))
def test_calibration_validation_uses_resolved_detector_geometry(tmp_path, geometry: str, matching: bool) -> None:
    """Auto resolution must neither reject matching OBB calibration nor bypass conflicts."""
    from boxmot.trackers.common.config import load_tracker_config
    from boxmot.trackers.common.motion.kalman_filters.profile import calibration_profile_signature

    profile_geometry = geometry if matching else ("obb" if geometry == "aabb" else "aabb")
    baseline = load_tracker_config("bytetrack", None)
    signature = calibration_profile_signature("bytetrack", profile_geometry, baseline)
    config = tmp_path / "calibrated.yaml"
    config.write_text(yaml.safe_dump({"calibration": signature}))
    options = {
        "tracker": "bytetrack",
        "tracker_config": config,
        "detector": _Detector(geometry),
        "source": _Source(frames=0),
        "sinks": (NullSink(),),
    }

    if matching:
        assert workflow.run_track(**options).summary.frames == 0
    else:
        with pytest.raises(ValueError, match="Calibrated tracker profile geometry=.*incompatible"):
            workflow.run_track(**options)


def test_prompted_tracking_auto_resolves_aabb(monkeypatch) -> None:
    from unittest.mock import Mock

    from boxmot.engine.tracking import prompted

    args = SimpleNamespace(geometry="auto")
    window = SimpleNamespace(open=Mock(), close=Mock(), controller=SimpleNamespace())
    tracker = SimpleNamespace(reset=Mock())
    source = _Source(frames=0)
    monkeypatch.setattr(prompted, "PromptWindow", lambda: window)

    run = prompted.run_prompted_track(args, tracker=tracker, source=source, sinks=())

    assert args.geometry == "aabb"
    assert run.summary.frames == 0
    assert source.closed
    window.open.assert_called_once()
    window.close.assert_called_once()


def test_auto_obb_rejects_mask_guidance_before_loading_its_weights(monkeypatch) -> None:
    """Detector-selected OBB geometry still enforces the AABB-only guidance contract."""
    from boxmot.segmentors.propagation import weights

    monkeypatch.setattr(
        weights, "resolve_edgetam_artifact", lambda *_: pytest.fail("Loaded incompatible guidance weights")
    )
    monkeypatch.setattr(workflow, "create_tracker", lambda *_: pytest.fail("Constructed incompatible tracker"))

    with pytest.raises(ValueError, match="requires AABB"):
        workflow.run_track(tracker="bytetrack", detector=_Detector("obb"), edgetam=True)
