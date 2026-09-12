"""Actual appearance trackers and native adapters honor their selected input paths."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, CameraModel, Detections, Frame, MaskBatch, OrientedBoxes, Tracks
from boxmot.trackers.botsort.native import NativeBotSortTracker
from boxmot.trackers.common.config import load_tracker_defaults
from boxmot.trackers.common.registry import get_tracker_class
from boxmot.trackers.occluboost.native import NativeOccluBoostTracker
from tests.unit.native.trackers._helpers import empty_native_batch

_PYTHON_TRACKERS = ("boosttrack", "botsort", "deepocsort", "occluboost")
_NATIVE_TRACKERS = {"botsort": NativeBotSortTracker, "occluboost": NativeOccluBoostTracker}


class _Encoder:
    """Record image fallback without model construction or downloads."""

    def __init__(self) -> None:
        self.calls = 0

    def get_features(self, boxes: np.ndarray, image: np.ndarray) -> np.ndarray:
        assert image.shape == (192, 192, 3)
        self.calls += 1
        return np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (len(boxes), 1))


class _Library:
    """Inspect native input forwarding while keeping ABI tests independent of compilation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, cfg: dict[str, Any]) -> object:
        return object()

    def update(self, handle: object, **inputs: Any) -> Any:
        self.calls.append(inputs)
        return empty_native_batch(inputs["geometry"].shape[1])

    def reset(self, handle: object) -> None:
        pass

    def destroy(self, handle: object) -> None:
        pass


def _detections(*, geometry: str = "aabb", embeddings: bool = False) -> Detections:
    values = torch.tensor([[20.0, 30.0, 20.0, 40.0, 0.1]] if geometry == "obb" else [[10.0, 10.0, 30.0, 50.0]])
    return Detections(
        geometry=OrientedBoxes(values) if geometry == "obb" else Boxes(values),
        scores=torch.tensor([0.99]),
        class_ids=torch.tensor([0]),
        sample_id="input-contract",
        embeddings=torch.tensor([[1.0, 0.0, 0.0]]) if embeddings else None,
    )


def _frame() -> Frame:
    pixels = np.random.default_rng(11).integers(0, 256, (3, 192, 192), dtype=np.uint8)
    return Frame(torch.from_numpy(pixels), "input-contract")


def _without_cmc(name: str) -> dict[str, bool]:
    return {"cmc_off": True} if name == "deepocsort" else {"use_cmc": False}


def _python_tracker(name: str, **options: Any) -> Any:
    defaults = load_tracker_defaults(name)
    defaults.update(options)
    return get_tracker_class(name)(**defaults)


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("supplied_embeddings", (False, True))
def test_default_python_appearance_inputs_drive_real_tracking(
    name: str, geometry: str, supplied_embeddings: bool
) -> None:
    encoder = _Encoder()
    tracker = _python_tracker(name, reid_model=encoder, is_obb=geometry == "obb")
    assert tracker.requirements.embeddings and tracker.requirements.frame_pixels
    detections, frame = _detections(geometry=geometry, embeddings=supplied_embeddings), _frame()

    for _ in range(2):
        result = tracker.update(detections, frame)
        assert isinstance(result, Tracks)
        assert len(result) == 1
    assert encoder.calls == (0 if supplied_embeddings else 2)


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
@pytest.mark.parametrize("use_embeddings", (False, True))
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_python_cmc_disabled_runs_without_image_pixels(name: str, use_embeddings: bool, geometry: str) -> None:
    tracker = _python_tracker(name, **_without_cmc(name), use_embeddings=use_embeddings, is_obb=geometry == "obb")
    assert tracker.requirements.embeddings is use_embeddings
    assert not tracker.requirements.frame
    for _ in range(2):
        result = tracker.update(_detections(geometry=geometry, embeddings=use_embeddings))
        assert len(result) == 1


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
@pytest.mark.parametrize("cmc", (False, True))
def test_python_centroid_uses_only_dimensions_unless_cmc_needs_pixels(name: str, cmc: bool) -> None:
    options = {} if cmc else _without_cmc(name)
    tracker = _python_tracker(name, **options, use_embeddings=False, asso_func="centroid")
    assert tracker.requirements.frame
    assert tracker.requirements.frame_dimensions_only is not cmc
    assert tracker.requirements.frame_pixels is cmc
    result = tracker.update(_detections(), _frame())
    assert len(result) == 1


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
def test_python_cmc_requires_images_even_with_supplied_embeddings(name: str) -> None:
    tracker = _python_tracker(name)
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(_detections(embeddings=True))
    assert tracker.frame_count == 0


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
def test_python_missing_appearance_inputs_fail_before_tracking(name: str) -> None:
    tracker = _python_tracker(name, **_without_cmc(name))
    with pytest.raises(ValueError, match="frame.*embeddings"):
        tracker.update(_detections())
    assert tracker.frame_count == 0


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
@pytest.mark.parametrize("field", ("masks", "embeddings", "camera"))
def test_python_does_not_silently_drop_disabled_or_unsupported_inputs(name: str, field: str) -> None:
    tracker = _python_tracker(name, **_without_cmc(name), use_embeddings=False)
    detections = _detections(embeddings=field == "embeddings")
    kwargs = {}
    if field == "masks":
        detections = replace(detections, masks=MaskBatch(torch.ones((1, 192, 192), dtype=torch.bool)))
    elif field == "camera":
        kwargs["camera"] = CameraModel(torch.eye(3, 4), (192, 192))
    with pytest.raises(ValueError, match=field):
        tracker.update(detections, **kwargs)
    assert tracker.frame_count == 0


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
def test_python_variable_timing_requires_and_consumes_capture_timestamps(name: str) -> None:
    tracker = _python_tracker(name, **_without_cmc(name), use_embeddings=False, variable_dt=True)
    with pytest.raises(ValueError, match="requires timestamp_s"):
        tracker.update(_detections())
    tracker.update(_detections(), timestamp_s=1.0)
    tracker.update(_detections(), timestamp_s=1.4)
    assert tracker._prediction_dt == pytest.approx(0.4)
    assert tracker._last_timestamp_s == pytest.approx(1.4)


@pytest.mark.parametrize("name", _PYTHON_TRACKERS)
@pytest.mark.parametrize("invalid", (None, "false", 0, 1))
def test_python_camera_motion_toggles_require_booleans(name: str, invalid: Any) -> None:
    option = "cmc_off" if name == "deepocsort" else "use_cmc"
    with pytest.raises(TypeError, match=f"{option} must be bool"):
        _python_tracker(name, **{option: invalid})


@pytest.mark.parametrize("name", _NATIVE_TRACKERS)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
@pytest.mark.parametrize("supplied_embeddings", (False, True))
def test_native_defaults_forward_images_and_resolved_embeddings(
    name: str, geometry: str, supplied_embeddings: bool
) -> None:
    library, encoder = _Library(), _Encoder()
    tracker = _NATIVE_TRACKERS[name](library=library, geometry=geometry, reid_model=encoder)
    try:
        assert tracker.requirements.frame_pixels and tracker.requirements.embeddings
        tracker.update(_detections(geometry=geometry, embeddings=supplied_embeddings), _frame())
        assert library.calls[0]["image"].shape == (192, 192, 3)
        np.testing.assert_array_equal(library.calls[0]["embeddings"], [[1.0, 0.0, 0.0]])
        assert encoder.calls == (0 if supplied_embeddings else 1)
    finally:
        tracker.close()


@pytest.mark.parametrize("name", _NATIVE_TRACKERS)
@pytest.mark.parametrize("use_embeddings", (False, True))
@pytest.mark.parametrize("centroid", (False, True))
def test_native_disabled_cmc_needs_no_pixels(name: str, use_embeddings: bool, centroid: bool) -> None:
    library = _Library()
    tracker = _NATIVE_TRACKERS[name](
        {"use_cmc": False, "use_embeddings": use_embeddings, "asso_func": "centroid" if centroid else "iou"},
        library=library,
    )
    try:
        assert tracker.requirements.frame is centroid
        assert tracker.requirements.frame_dimensions_only is centroid
        assert not tracker.requirements.frame_pixels
        tracker.update(_detections(embeddings=use_embeddings), _frame() if centroid else None)
        assert (library.calls[0]["image"] is not None) is centroid
        assert (library.calls[0]["embeddings"] is not None) is use_embeddings
    finally:
        tracker.close()


@pytest.mark.parametrize("name", _NATIVE_TRACKERS)
@pytest.mark.parametrize("field", ("masks", "embeddings"))
def test_native_does_not_silently_drop_disabled_or_unsupported_inputs(name: str, field: str) -> None:
    library = _Library()
    tracker = _NATIVE_TRACKERS[name]({"use_cmc": False, "use_embeddings": False}, library=library)
    detections = _detections(embeddings=field == "embeddings")
    if field == "masks":
        detections = replace(detections, masks=MaskBatch(torch.ones((1, 192, 192), dtype=torch.bool)))
    try:
        with pytest.raises(ValueError, match=field):
            tracker.update(detections)
        assert not library.calls
    finally:
        tracker.close()


@pytest.mark.parametrize("name", _NATIVE_TRACKERS)
@pytest.mark.parametrize("option", ("use_cmc", "use_embeddings"))
@pytest.mark.parametrize("invalid", (None, "false", 0, 1))
def test_native_input_toggles_require_booleans(name: str, option: str, invalid: Any) -> None:
    with pytest.raises(TypeError, match=f"{option} must be bool"):
        _NATIVE_TRACKERS[name]({option: invalid}, library=_Library())


@pytest.mark.parametrize("name", _NATIVE_TRACKERS)
@pytest.mark.parametrize("method", (None, "none", "orb", "sift"))
def test_native_enabled_cmc_rejects_unavailable_estimators(name: str, method: str | None) -> None:
    with pytest.raises(ValueError, match="cmc_method"):
        _NATIVE_TRACKERS[name]({"cmc_method": method}, library=_Library())


@pytest.mark.parametrize("name", _NATIVE_TRACKERS)
def test_native_trackers_do_not_claim_variable_frame_time(name: str) -> None:
    with pytest.raises(ValueError, match="variable_dt"):
        _NATIVE_TRACKERS[name]({"variable_dt": True}, library=_Library())
