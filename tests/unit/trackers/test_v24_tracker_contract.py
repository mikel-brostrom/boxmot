"""Focused tests for the v24 Python tracker boundary."""

from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_type_hints

import numpy as np
import pytest
import torch

import boxmot.trackers as public_trackers
import boxmot.trackers.common.factory as tracker_factory
import boxmot.trackers.common.registry as tracker_registry
from boxmot.reid import ReIDEncoderSpec
from boxmot.structures import Boxes, Detections, Frame, GeometryKind, MaskBatch, OrientedBoxes, Tracks
from boxmot.trackers import (
    ReIDConfigurableTracker,
    Tracker,
    TrackerCapabilities,
    TrackerFamily,
    TrackerRequirements,
    TrackerSpec,
    create_tracker,
)
from boxmot.trackers.common.base import BaseTracker
from boxmot.trackers.common.config import TRACKER_CONFIGS_DIR


def _frame(sample_id: str = "sequence/000001", *, height: int = 64, width: int = 64) -> Frame:
    red = torch.full((height, width), 10, dtype=torch.uint8)
    green = torch.full((height, width), 20, dtype=torch.uint8)
    blue = torch.full((height, width), 30, dtype=torch.uint8)
    return Frame(
        image=torch.stack((red, green, blue)),
        sample_id=sample_id,
        sequence_id="sequence",
        frame_index=0,
    )


def _detections(
    sample_id: str = "sequence/000001",
    *,
    is_obb: bool = False,
    embeddings: bool = True,
    masks: bool = True,
    height: int = 64,
    width: int = 64,
) -> Detections:
    if is_obb:
        geometry = OrientedBoxes(torch.tensor([[20.0, 28.0, 20.0, 32.0, 0.1]], dtype=torch.float32))
    else:
        geometry = Boxes(torch.tensor([[10.0, 12.0, 30.0, 44.0]], dtype=torch.float32))
    mask_values = torch.zeros((1, height, width), dtype=torch.bool)
    mask_values[:, 12:44, 10:30] = True
    return Detections(
        geometry=geometry,
        scores=torch.tensor([0.95], dtype=torch.float32),
        class_ids=torch.tensor([2], dtype=torch.int64),
        sample_id=sample_id,
        instance_ids=("instance-0",),
        masks=MaskBatch(mask_values) if masks else None,
        embeddings=torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32) if embeddings else None,
    )


class _RecordingTracker(BaseTracker):
    supports_obb = True

    def __init__(
        self,
        *,
        needs_embeddings: bool = False,
        needs_masks: bool = False,
        needs_frame: bool = False,
        frame_dimensions_only: bool = False,
        is_obb: bool = False,
    ) -> None:
        self.use_embeddings = needs_embeddings
        self._requires_masks = needs_masks
        self._requires_frame = needs_frame
        self._requires_frame_dimensions_only = frame_dimensions_only
        self.seen: dict[str, np.ndarray | None] = {}
        super().__init__(is_obb=is_obb, min_hits=1)

    def _track_detections(
        self,
        dets: np.ndarray,
        img: np.ndarray | None,
        embs: np.ndarray | None = None,
        masks: np.ndarray | None = None,
    ):
        self.seen = {"dets": dets, "img": img, "embs": embs, "masks": masks}
        if len(dets) == 0:
            rows = self._empty_output(dtype=np.float32)
            if self._requires_masks:
                assert masks is not None
                return rows, np.empty((0, *masks.shape[1:]), dtype=bool)
            return rows

        geometry = dets[0, : self.detection_layout.box_cols]
        row = np.concatenate(
            (
                geometry,
                np.array(
                    [
                        7,
                        dets[0, self.detection_layout.conf_idx],
                        dets[0, self.detection_layout.cls_idx],
                        0,
                    ],
                    dtype=np.float32,
                ),
            )
        )[None]
        if self._requires_masks:
            assert masks is not None
            return row, masks[:1]
        return row


def test_public_package_exports_only_contracts_and_factory() -> None:
    assert public_trackers.__all__ == (
        "GeometryKind",
        "ReIDConfigurableTracker",
        "Tracker",
        "TrackerCapabilities",
        "TrackerFamily",
        "TrackerRequirements",
        "TrackerSpec",
        "create_tracker",
    )
    assert public_trackers.GeometryKind is GeometryKind
    assert public_trackers.ReIDConfigurableTracker is ReIDConfigurableTracker
    assert public_trackers.Tracker is Tracker
    assert public_trackers.TrackerCapabilities is TrackerCapabilities
    assert public_trackers.TrackerFamily is TrackerFamily
    assert public_trackers.TrackerRequirements is TrackerRequirements
    assert public_trackers.TrackerSpec is TrackerSpec
    assert public_trackers.create_tracker is create_tracker
    assert tuple(inspect.signature(Tracker.update).parameters) == (
        "self",
        "detections",
        "frame",
        "timestamp_s",
        "detections_3d",
        "camera",
    )
    for implementation_name in ("ByteTrack", "BotSort", "StrongSort", "MafHda"):
        assert not hasattr(public_trackers, implementation_name)


def test_reid_configurable_tracker_is_an_optional_runtime_protocol() -> None:
    class _Tracker:
        name = "fixture"
        supports_variable_dt = False
        variable_dt = False
        capabilities = TrackerCapabilities(
            family=TrackerFamily.BOX,
            geometry_kinds=frozenset({GeometryKind.AABB}),
            accepts_embeddings=True,
            accepts_frame=True,
        )
        requirements = TrackerRequirements(embeddings=True)
        generates_embeddings = True

        def update(self, detections, frame=None, *, timestamp_s=None):
            raise AssertionError("Runtime protocol checks must not call update().")

        def validate_timing(self, frame=None, *, timestamp_s=None) -> float | None:
            raise AssertionError("Runtime protocol checks must not call validate_timing().")

        def reset(self) -> None:
            return None

    class _Configurable(_Tracker):
        def configure_reid(self, spec: ReIDEncoderSpec) -> None:
            self.spec = spec

    configurable = _Configurable()
    plain_tracker = _Tracker()
    assert isinstance(plain_tracker, Tracker)
    assert isinstance(configurable, ReIDConfigurableTracker)
    assert not isinstance(plain_tracker, ReIDConfigurableTracker)
    assert get_type_hints(ReIDConfigurableTracker.configure_reid) == {
        "spec": ReIDEncoderSpec,
        "return": type(None),
    }
    assert get_type_hints(BaseTracker.configure_reid) == {
        "spec": ReIDEncoderSpec,
        "return": type(None),
    }


def test_tracker_requirements_distinguish_frame_dimensions_from_pixels() -> None:
    assert TrackerRequirements().frame_pixels is False
    assert TrackerRequirements(frame=True).frame_pixels is True
    dimensions_only = TrackerRequirements(frame=True, frame_dimensions_only=True)
    assert dimensions_only.frame_pixels is False

    with pytest.raises(ValueError, match="requires frame=True"):
        TrackerRequirements(frame_dimensions_only=True)


def test_package_root_import_does_not_load_tracker_or_heavy_runtimes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = """
import sys
import boxmot

assert {name for name in sys.modules if name.startswith("boxmot.trackers")} == {
    "boxmot.trackers",
    "boxmot.trackers.common",
    "boxmot.trackers.common.manifest",
}
assert not any(name.startswith("boxmot.native") for name in sys.modules)
assert not any(name in sys.modules for name in ("cv2", "numpy", "torch", "yaml"))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("is_obb", "rows", "geometry_columns"),
    (
        (False, np.array([[10, 12, 30, 44, 0.95, 16_777_217]], dtype=np.float64), 4),
        (True, np.array([[20, 28, 20, 32, 0.1, 0.95, 16_777_217]], dtype=np.float64), 5),
    ),
)
def test_base_tracker_accepts_exact_packed_numpy_layout_for_configured_mode(
    is_obb: bool,
    rows: np.ndarray,
    geometry_columns: int,
) -> None:
    signature = inspect.signature(BaseTracker.update)
    assert tuple(signature.parameters) == ("self", "detections", "frame", "timestamp_s", "detections_3d", "camera")
    for name in ("timestamp_s", "detections_3d", "camera"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    tracker = _RecordingTracker(is_obb=is_obb)
    tracks = tracker.update(rows)

    assert type(tracks) is np.ndarray
    assert tracks.dtype == np.float64
    assert tracks.shape == (1, geometry_columns + 4)
    assert tracks.flags.c_contiguous
    np.testing.assert_allclose(tracks[0, :geometry_columns], rows[0, :geometry_columns].astype(np.float32))
    assert tracks[0, geometry_columns] == 7
    assert tracks[0, geometry_columns + 1] == pytest.approx(np.float32(0.95))
    assert tracks[0, geometry_columns + 2] == 16_777_217
    assert tracks[0, geometry_columns + 3] == 0
    assert tracker.seen["dets"].shape == (1, geometry_columns + 2)
    assert tracker.seen["dets"].dtype == np.float32


@pytest.mark.parametrize(("is_obb", "input_columns", "output_columns"), ((False, 6, 8), (True, 7, 9)))
def test_numpy_input_preserves_empty_output_schema_across_reset(
    is_obb: bool,
    input_columns: int,
    output_columns: int,
) -> None:
    tracker = _RecordingTracker(is_obb=is_obb)
    empty = np.empty((0, input_columns), dtype=np.float64)

    first = tracker.update(empty)
    tracker.reset()
    after_reset = tracker.update(empty)

    for tracks in (first, after_reset):
        assert type(tracks) is np.ndarray
        assert tracks.dtype == np.float64
        assert tracks.shape == (0, output_columns)
        assert tracks.flags.c_contiguous


def test_numpy_input_returns_numpy_with_or_without_frame() -> None:
    tracker = _RecordingTracker()
    rows = np.array([[10, 12, 30, 44, 0.95, 2]], dtype=np.float32)

    framed = tracker.update(rows, _frame("camera-1:000042"))
    unframed = tracker.update(rows)

    assert type(framed) is np.ndarray
    assert type(unframed) is np.ndarray
    np.testing.assert_array_equal(framed, unframed)


def test_numpy_output_rejects_integer_columns_outside_exact_float64_range() -> None:
    rows = np.array([[10, 12, 30, 44, 1.0, 2**53 + 2]], dtype=np.float64)
    tracker = _RecordingTracker()

    with pytest.raises(ValueError, match="exact float64 integer range"):
        tracker.update(rows)

    assert tracker.seen == {}


@pytest.mark.parametrize(
    ("is_obb", "rows", "message"),
    (
        (False, [[10, 12, 30, 44, 0.9, 0]], "plain numpy.ndarray"),
        (False, np.array([10, 12, 30, 44, 0.9, 0]), r"shape \[N, 6\]"),
        (False, np.empty((0, 7), dtype=np.float32), r"shape \[N, 6\]"),
        (True, np.empty((0, 6), dtype=np.float32), r"shape \[N, 7\]"),
        (False, np.array([["10", "12", "30", "44", "0.9", "0"]]), "real numeric dtype"),
        (False, np.array([[10, 12, 30, 44, 0.9, 0]], dtype=np.complex64), "real numeric dtype"),
        (False, np.array([[np.nan, 12, 30, 44, 0.9, 0]]), "only finite values"),
        (False, np.array([[10, 12, 30, 44, 1.1, 0]]), r"range \[0, 1\]"),
        (False, np.array([[10, 12, 30, 44, 1.0 + 1e-12, 0]]), r"range \[0, 1\]"),
        (False, np.array([[10, 12, 30, 44, -1e-50, 0]]), r"range \[0, 1\]"),
        (False, np.array([[10, 12, 30, 44, 0.9, 1.5]]), "non-negative integers"),
        (False, np.array([[10, 12, 30, 44, 0.9, -1]]), "non-negative integers"),
        (False, np.array([[10, 12, 10, 44, 0.9, 0]]), "x2 > x1 and y2 > y1"),
        (True, np.array([[20, 28, 0, 32, 0.1, 0.9, 0]]), "positive width and height"),
        (
            False,
            np.array([[10, 12, np.finfo(np.float64).max, 44, 0.9, 0]]),
            "representable as finite float32",
        ),
    ),
)
def test_numpy_input_validation_is_explicit(
    is_obb: bool,
    rows: object,
    message: str,
) -> None:
    tracker = _RecordingTracker(is_obb=is_obb)
    invalid_type = type(rows) is not np.ndarray or not np.issubdtype(rows.dtype, np.number)
    invalid_type = invalid_type or (isinstance(rows, np.ndarray) and np.issubdtype(rows.dtype, np.complexfloating))
    error = TypeError if invalid_type else ValueError
    with pytest.raises(error, match=message):
        tracker.update(rows)  # type: ignore[arg-type]


def test_numpy_input_rejects_ndarray_subclasses() -> None:
    class ArraySubclass(np.ndarray):
        pass

    rows = np.array([[10, 12, 30, 44, 0.9, 0]], dtype=np.float32).view(ArraySubclass)

    with pytest.raises(TypeError, match="plain numpy.ndarray"):
        _RecordingTracker().update(rows)


@pytest.mark.parametrize(
    ("tracker", "frame", "message"),
    (
        (_RecordingTracker(needs_embeddings=True), None, "requires detection embeddings"),
        (_RecordingTracker(needs_masks=True), None, "requires full-frame detection masks"),
        (_RecordingTracker(needs_frame=True), None, "requires a frame"),
    ),
)
def test_numpy_input_cannot_bypass_declared_requirements(
    tracker: _RecordingTracker,
    frame: Frame | None,
    message: str,
) -> None:
    rows = np.array([[10, 12, 30, 44, 0.9, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        tracker.update(rows, frame)


@pytest.mark.parametrize("strided", (False, True))
def test_numpy_image_preserves_bgr_pixels_and_normalizes_strides(strided: bool) -> None:
    tracker = _RecordingTracker(needs_frame=True)
    image = np.full((64, 64, 3), (30, 20, 10), dtype=np.uint8)
    image[0, 0] = (3, 2, 1)
    if strided:
        image = image[::-1, ::-1]
        assert not image.flags.c_contiguous
    original = image.copy()
    rows = np.array([[10, 12, 30, 44, 0.9, 2]], dtype=np.float32)

    result = tracker.update(rows, image)

    assert type(result) is np.ndarray
    assert result.shape == (1, 8)
    assert tracker.seen["img"].flags.c_contiguous
    assert tracker.seen["img"].dtype == np.uint8
    np.testing.assert_array_equal(tracker.seen["img"], original)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    ("image", "error", "message"),
    (
        (np.zeros((64, 64, 3), dtype=np.float32), TypeError, "uint8"),
        (np.zeros((64, 64), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((3, 64, 64), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((64, 64, 4), dtype=np.uint8), ValueError, "shape"),
        (np.zeros((0, 64, 3), dtype=np.uint8), ValueError, "positive"),
        (np.zeros((64, 0, 3), dtype=np.uint8), ValueError, "positive"),
    ),
)
def test_numpy_image_validation_precedes_kernel_update(
    image: np.ndarray,
    error: type[Exception],
    message: str,
) -> None:
    tracker = _RecordingTracker()

    with pytest.raises(error, match=message):
        tracker.update(np.empty((0, 6), dtype=np.float32), image)

    assert tracker.seen == {}


@pytest.mark.parametrize("frame_representation", ("canonical", "numpy"))
def test_private_adapter_preserves_frame_colors_and_wraps_tracks_and_masks(frame_representation: str) -> None:
    tracker = _RecordingTracker(needs_embeddings=True, needs_masks=True, needs_frame=True)
    detections = _detections()
    frame = _frame() if frame_representation == "canonical" else np.full((64, 64, 3), (30, 20, 10), dtype=np.uint8)
    result = tracker.update(detections, frame)

    assert isinstance(result, Tracks)
    assert result.sample_id == detections.sample_id
    assert result.track_ids.tolist() == [7]
    assert result.class_ids.tolist() == [2]
    assert result.detection_indices.tolist() == [0]
    assert result.masks is not None
    assert result.masks.values.dtype is torch.bool
    assert result.masks.image_size == (64, 64)
    assert tracker.seen["dets"].shape == (1, 6)
    assert tracker.seen["embs"].shape == (1, 4)
    assert tracker.seen["masks"].shape == (1, 64, 64)
    assert tracker.seen["img"].shape == (64, 64, 3)
    assert tracker.seen["img"][0, 0].tolist() == [30, 20, 10]


@pytest.mark.parametrize("frame_representation", ("canonical", "numpy"))
def test_private_adapter_initializes_dimensions_without_copying_frame_pixels(frame_representation: str) -> None:
    tracker = _RecordingTracker(needs_frame=True, frame_dimensions_only=True)
    frame = (
        _frame(height=48, width=80) if frame_representation == "canonical" else np.zeros((48, 80, 3), dtype=np.uint8)
    )

    tracker.update(_detections(height=48, width=80, embeddings=False, masks=False), frame)

    assert tracker.requirements == TrackerRequirements(frame=True, frame_dimensions_only=True)
    assert tracker.seen["img"] is None
    assert (tracker.w, tracker.h) == (80, 48)


def test_private_adapter_preserves_unwrapped_obb_angle_continuity() -> None:
    tracker = _RecordingTracker(is_obb=True)

    def at_angle(angle: float, sample_id: str) -> Detections:
        return Detections(
            geometry=OrientedBoxes(torch.tensor([[20.0, 28.0, 32.0, 20.0, angle]], dtype=torch.float32)),
            scores=torch.tensor([0.95], dtype=torch.float32),
            class_ids=torch.tensor([2], dtype=torch.int64),
            sample_id=sample_id,
        )

    first = tracker.update(at_angle(3.1, "sequence/000001"))
    second = tracker.update(at_angle(-3.1, "sequence/000002"))

    first_angle = float(first.geometry.values[0, 4])
    second_angle = float(second.geometry.values[0, 4])
    assert first_angle == pytest.approx(3.1)
    assert second_angle > np.pi
    assert second_angle - first_angle == pytest.approx((2 * np.pi) - 6.2, abs=1e-5)

    tracker.reset()
    after_reset = tracker.update(at_angle(-3.1, "sequence/000003"))
    assert float(after_reset.geometry.values[0, 4]) == pytest.approx(-3.1)


@pytest.mark.parametrize(
    ("detections", "frame", "message"),
    (
        (_detections(embeddings=False), _frame(), "requires detection embeddings"),
        (_detections(masks=False), _frame(), "requires full-frame detection masks"),
        (_detections(), None, "requires a frame"),
    ),
)
def test_declared_requirements_are_strict(
    detections: Detections,
    frame: Frame | None,
    message: str,
) -> None:
    tracker = _RecordingTracker(needs_embeddings=True, needs_masks=True, needs_frame=True)
    with pytest.raises(ValueError, match=message):
        tracker.update(detections, frame)


def test_frame_identity_and_mask_spatial_shape_are_strict() -> None:
    tracker = _RecordingTracker(needs_embeddings=True, needs_masks=True)
    with pytest.raises(ValueError, match="same sample"):
        tracker.update(_detections(), _frame("sequence/000002"))
    with pytest.raises(ValueError, match="must match the frame spatial size"):
        tracker.update(_detections(height=32, width=32), _frame())
    with pytest.raises(ValueError, match="must match the frame spatial size"):
        tracker.update(_detections(height=32, width=32), np.zeros((64, 64, 3), dtype=np.uint8))


def test_geometry_is_fixed_by_construction_and_preserved_by_reset() -> None:
    tracker = _RecordingTracker(is_obb=False)
    tracker.reset()
    with pytest.raises(ValueError, match="configured for AABB geometry, got OBB"):
        tracker.update(_detections(is_obb=True))


def test_tracker_spec_is_frozen_hashable_and_recursively_immutable() -> None:
    spec = TrackerSpec(
        "bytetrack",
        options=(
            ("nested", ("value", 3, (True, None))),
            ("track_buffer", 45),
        ),
    )
    assert hash(spec)
    assert spec.option_dict["nested"] == ("value", 3, (True, None))
    with pytest.raises(FrozenInstanceError):
        spec.name = "botsort"


@pytest.mark.parametrize(
    "options",
    (
        (("bad", []),),
        (("bad", {}),),
        (("bad", ("nested", [])),),
        (("bad", ("nested", {})),),
    ),
)
def test_tracker_spec_rejects_mutable_option_values(options) -> None:
    with pytest.raises(TypeError, match="immutable JSON values"):
        TrackerSpec("bytetrack", options=options)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_tracker_spec_rejects_non_finite_options(value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        TrackerSpec("bytetrack", options=(("value", value),))


def test_create_tracker_accepts_only_a_canonical_spec_and_dispatches_native(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tuple(inspect.signature(create_tracker).parameters) == ("spec",)
    with pytest.raises(TypeError, match="spec must be TrackerSpec"):
        create_tracker("bytetrack")

    native_tracker = object()
    monkeypatch.setattr(tracker_factory, "_create_native_tracker", lambda _spec, _definition, _kind: native_tracker)
    monkeypatch.setattr(tracker_factory, "_bind_and_validate_capabilities", lambda tracker, _capabilities: tracker)
    assert create_tracker(TrackerSpec("bytetrack", backend="cpp")) is native_tracker


def test_factory_merges_options_then_applies_canonical_spec_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _Tracker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.requirements = TrackerRequirements()

    monkeypatch.setattr(tracker_factory, "_load_tracker_class", lambda _definition: _Tracker)
    spec = TrackerSpec(
        "bytetrack",
        geometry="obb",
        per_class=True,
        class_ids=(0, 2),
        class_names=((0, "person"), (2, "car")),
        options=(("is_obb", False), ("match_thresh", 0.71), ("per_class", False)),
    )

    tracker_factory.create_tracker(spec)

    assert captured["match_thresh"] == 0.71
    assert captured["is_obb"] is True
    assert captured["per_class"] is True
    assert captured["class_ids"] == (0, 2)
    assert captured["class_names"] == {0: "person", 2: "car"}


@pytest.mark.parametrize(
    ("tracker_name", "geometry"),
    (
        (name, kind.value)
        for name, definition in tracker_registry.TRACKER_DEFINITIONS.items()
        if not definition.capabilities.requires_detections_3d
        for kind in sorted(definition.capabilities.geometry_kinds, key=lambda kind: kind.value)
    ),
)
def test_image_trackers_consume_canonical_inputs(tracker_name: str, geometry: str) -> None:
    tracker = create_tracker(TrackerSpec(tracker_name, geometry=geometry))
    assert isinstance(tracker, Tracker)
    assert isinstance(tracker.requirements, TrackerRequirements)
    requirements = tracker.requirements
    assert not hasattr(tracker, "model")
    assert not hasattr(tracker, "reid_model")

    detections = _detections(
        is_obb=geometry == "obb",
        embeddings=requirements.embeddings,
        masks=requirements.masks,
    )
    frame = _frame() if requirements.frame else None
    tracks = tracker.update(detections, frame)

    assert isinstance(tracks, Tracks)
    assert tracker.requirements == requirements
    assert tracks.sample_id == detections.sample_id
    assert tracks.is_obb is (geometry == "obb")
    if tracker.requirements.masks:
        assert tracks.masks is not None
        assert tracks.masks.image_size == (64, 64)
        assert len(tracks.masks) == len(tracks)


def test_embedding_tracker_can_be_resolved_as_geometry_only() -> None:
    tracker = create_tracker(
        TrackerSpec(
            "botsort",
            options=(("use_cmc", False), ("use_embeddings", False)),
        )
    )
    assert tracker.requirements == TrackerRequirements()
    assert isinstance(tracker.update(_detections(embeddings=False, masks=False)), Tracks)


def test_mask_tracker_requires_and_returns_full_frame_boolean_masks() -> None:
    tracker = create_tracker(TrackerSpec("maf_hda"))
    assert tracker.requirements == TrackerRequirements(masks=True, frame=True)
    with pytest.raises(ValueError, match="requires full-frame detection masks"):
        tracker.update(_detections(masks=False, embeddings=False), _frame())
    with pytest.raises(ValueError, match="requires a frame"):
        tracker.update(_detections(embeddings=False), None)

    tracks = tracker.update(_detections(embeddings=False), _frame())
    assert tracks.masks is not None
    assert tracks.masks.values.dtype is torch.bool
    assert tracks.masks.values.shape == (len(tracks), 64, 64)


def test_embedding_config_names_are_positive_and_legacy_names_are_absent() -> None:
    configurable = {"boosttrack", "botsort", "deepocsort", "hybridsort", "occluboost"}
    base_parameters = inspect.signature(BaseTracker.__init__).parameters
    assert {"reid_model", "reid_weights", "device", "half", "reid_preprocess"} <= set(base_parameters)
    embedding_trackers = (
        name
        for name, definition in tracker_registry.TRACKER_DEFINITIONS.items()
        if definition.capabilities.accepts_embeddings
    )
    for tracker_name in embedding_trackers:
        parameters = inspect.signature(tracker_registry.get_tracker_class(tracker_name).__init__).parameters
        assert "with_reid" not in parameters
        assert "embedding_off" not in parameters
        if tracker_name in configurable:
            assert "use_embeddings" in parameters

    for config_path in TRACKER_CONFIGS_DIR.rglob("*.yaml"):
        config_text = config_path.read_text(encoding="utf-8")
        assert "with_reid:" not in config_text
        assert "embedding_off:" not in config_text


@pytest.mark.parametrize(
    ("option", "value"),
    (
        ("reid_model", None),
        ("reid_weights", None),
        ("device", "cpu"),
        ("half", False),
        ("reid_preprocess", None),
        ("reid_weights", "model.pt"),
    ),
)
def test_non_reid_trackers_reject_reid_model_configuration(option: str, value: object) -> None:
    tracker_class = tracker_registry.get_tracker_class("bytetrack")
    with pytest.raises(TypeError, match=rf"does not accept ReID model options: {option}"):
        tracker_class(**{option: value})


def test_tracker_factory_rejects_reid_options_for_non_reid_trackers() -> None:
    with pytest.raises(ValueError, match="does not accept ReID model options"):
        create_tracker(TrackerSpec("bytetrack", options=(("reid_weights", "model.pt"),)))


def test_tracker_spec_rejects_reid_model_configuration_for_python_reid_trackers() -> None:
    with pytest.raises(ValueError, match="tracker-algorithm options only"):
        create_tracker(TrackerSpec("botsort", options=(("reid_weights", "model.pt"),)))
