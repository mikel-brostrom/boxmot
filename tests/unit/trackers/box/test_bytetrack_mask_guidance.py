"""Exercise optional McByte++ guidance through ByteTrack's public boundary."""

from __future__ import annotations

import sys
import weakref
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.structures import Boxes, Detections, Frame, Tracks
from boxmot.trackers import MaskGuidance, MaskGuidanceConfig, TrackerSpec, create_tracker
from boxmot.trackers.bytetrack.tracker import ByteTrack

FRAME_SHAPE = (96, 128)


@dataclass
class _PropagationCall:
    frame_index: int
    active_boxes: dict[int, np.ndarray]
    new_boxes: dict[int, np.ndarray]


@dataclass
class _Propagator:
    """Supply deterministic temporal masks while recording real prompt inputs."""

    frame_shape: tuple[int, int] = FRAME_SHAPE
    max_objects: int = 96
    prompt_overlap: float = 0.10
    device: torch.device = torch.device("cpu")
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    calls: list[_PropagationCall] = field(default_factory=list)
    retained: list[set[int]] = field(default_factory=list)
    resets: int = 0

    def propagate(
        self, frame_index: int, frame: np.ndarray, active_boxes: dict[int, np.ndarray], new_boxes: dict[int, np.ndarray]
    ) -> dict[int, np.ndarray]:
        if frame.shape[:2] != self.frame_shape:
            raise ValueError("EdgeTAM requires constant frame dimensions")
        self.calls.append(
            _PropagationCall(
                frame_index,
                {key: box.copy() for key, box in active_boxes.items()},
                {key: box.copy() for key, box in new_boxes.items()},
            )
        )
        return self.masks.copy()

    def retain_tracks(self, track_ids: set[int]) -> None:
        self.retained.append(set(track_ids))
        self.masks = {key: value for key, value in self.masks.items() if key in track_ids}

    def reset(self) -> None:
        self.resets += 1
        self.masks.clear()


def _config() -> MaskGuidanceConfig:
    return MaskGuidanceConfig(checkpoint="edgetam.pt", device="cpu")


def _guided_tracker(**kwargs: object) -> tuple[ByteTrack, _Propagator]:
    tracker = ByteTrack(mask_guidance=_config(), **kwargs)
    propagator = _Propagator()
    tracker._mask_guidance._propagator = propagator
    return tracker, propagator


def _rows(score: float = 0.95) -> np.ndarray:
    return np.array([[10, 10, 30, 40, score, 0], [14, 10, 34, 40, score, 0]], dtype=np.float32)


def _update(
    tracker: ByteTrack,
    rows: np.ndarray,
    index: int | None,
    *,
    canonical: bool,
    sequence_id: str | None = "sequence",
) -> Tracks | np.ndarray:
    if not canonical:
        return tracker.update(rows, np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
    sample_id = f"{sequence_id}/{index}"
    detections = Detections(
        geometry=Boxes(torch.from_numpy(np.ascontiguousarray(rows[:, :4]))),
        scores=torch.from_numpy(np.ascontiguousarray(rows[:, 4])),
        class_ids=torch.from_numpy(np.ascontiguousarray(rows[:, 5], dtype=np.int64)),
        sample_id=sample_id,
    )
    frame = Frame(
        image=torch.zeros((3, *FRAME_SHAPE), dtype=torch.uint8),
        sample_id=sample_id,
        sequence_id=sequence_id,
        frame_index=index,
    )
    return tracker.update(detections, frame)


def _result_rows(result: Tracks | np.ndarray) -> np.ndarray:
    return result.to_aabb_rows().numpy() if isinstance(result, Tracks) else result


@pytest.mark.parametrize("canonical", [False, True])
def test_disabling_guidance_preserves_outputs_and_avoids_temporal_model(
    monkeypatch: pytest.MonkeyPatch, canonical: bool
) -> None:
    def forbidden_model(*args: object, **kwargs: object) -> None:
        pytest.fail("Disabled mask guidance must not construct a temporal model")

    monkeypatch.setattr(MaskGuidance, "__init__", forbidden_model)
    default = ByteTrack()
    disabled = ByteTrack(mask_guidance=None)
    sequence = [_rows(), _rows(0.3), np.empty((0, 6), dtype=np.float32), _rows()]

    for index, rows in enumerate(sequence):
        expected = _update(default, rows, index, canonical=canonical)
        actual = _update(disabled, rows, index, canonical=canonical)
        np.testing.assert_array_equal(_result_rows(actual), _result_rows(expected))

    assert not default.requirements.frame
    assert not disabled.requirements.frame


@pytest.mark.parametrize("canonical", [False, True])
@pytest.mark.parametrize("score", [0.95, 0.30], ids=["high_confidence", "low_confidence"])
def test_temporal_masks_resolve_crossing_in_both_association_stages(score: float, canonical: bool) -> None:
    baseline = ByteTrack()
    guided, propagator = _guided_tracker()
    original = _rows()
    first = _result_rows(_update(guided, original, 0, canonical=canonical))
    _update(baseline, original, 0, canonical=canonical)
    identities = first[:, 4].astype(int)
    # Both boxes overlap enough for either stage. Temporal masks place each
    # identity in the opposite box; ordinary IoU would retain the old location.
    for track_id, box in zip(identities, original[::-1, :4].astype(int)):
        mask = np.zeros(FRAME_SHAPE, dtype=bool)
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = True
        propagator.masks[track_id] = mask

    expected = _result_rows(_update(baseline, _rows(score), 1, canonical=canonical))
    result = _update(guided, _rows(score), 1, canonical=canonical)
    actual = _result_rows(result)

    np.testing.assert_array_equal(expected[:, [4, 7]], np.column_stack((identities, [0, 1])))
    np.testing.assert_array_equal(actual[:, [4, 7]], np.column_stack((identities, [1, 0])))
    np.testing.assert_allclose(actual[:, 5], score)
    assert len(propagator.calls) == 2
    if canonical:
        assert isinstance(result, Tracks)
        assert result.sample_id == "sequence/1"
        assert isinstance(result.geometry, Boxes)
        assert result.masks is None
        result.validate()
    else:
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 8)
        assert result.dtype == np.float64
        assert result.flags.c_contiguous


def test_prompts_use_previous_detection_boxes_and_only_add_confirmed_identities() -> None:
    tracker, propagator = _guided_tracker()
    first_box = np.array([[10, 10, 30, 40, 0.95, 0]], dtype=np.float32)
    first = tracker.update(first_box, np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
    first_id = int(first[0, 4])
    moved = np.array([[14, 10, 34, 40, 0.95, 0], [70, 10, 90, 40, 0.95, 0]], dtype=np.float32)

    second = _result_rows(_update(tracker, moved, 1, canonical=False))

    assert len(second) == 1  # The late arrival has not been confirmed yet.
    np.testing.assert_array_equal(propagator.calls[1].active_boxes[first_id], first_box[0, :4])
    assert propagator.calls[1].new_boxes == {}
    assert not np.array_equal(second[0, :4], moved[0, :4])  # KF state differs from the raw box.
    # Mutation of caller-owned detections must not alter saved prompt geometry.
    saved_moved = moved.copy()
    moved[0, :4] += 1
    third = _result_rows(_update(tracker, saved_moved, 2, canonical=False))

    assert len(third) == 2
    np.testing.assert_array_equal(propagator.calls[2].active_boxes[first_id], saved_moved[0, :4])
    assert propagator.calls[2].new_boxes == {}
    second_id = int(third[third[:, 4] != first_id, 4].item())
    _update(tracker, saved_moved, 3, canonical=False)

    assert set(propagator.calls[3].active_boxes) == {first_id, second_id}
    assert set(propagator.calls[3].new_boxes) == {second_id}
    np.testing.assert_array_equal(propagator.calls[3].new_boxes[second_id], saved_moved[1, :4])
    _update(tracker, saved_moved, 4, canonical=False)
    assert propagator.calls[4].new_boxes == {}


def test_empty_frames_advance_temporal_memory_and_reset_clears_reused_ids() -> None:
    tracker, propagator = _guided_tracker()
    first = _result_rows(_update(tracker, _rows(), 0, canonical=False))
    identity = int(first[0, 4])
    propagator.masks[identity] = np.ones(FRAME_SHAPE, dtype=bool)
    empty = np.empty((0, 6), dtype=np.float32)

    for index in (1, 2):
        result = _update(tracker, empty, index, canonical=False)
        assert result.shape == (0, 8)
    assert [call.frame_index for call in propagator.calls] == [0, 1, 2]
    assert set(propagator.calls[1].active_boxes) == set(first[:, 4].astype(int))
    assert propagator.calls[2].active_boxes == {}

    tracker.reset()
    assert propagator.resets == 1
    assert propagator.masks == {}
    restarted = _result_rows(_update(tracker, _rows(), 0, canonical=False))

    np.testing.assert_array_equal(restarted, first)
    assert propagator.calls[-1].frame_index == 0
    assert propagator.calls[-1].active_boxes == {}
    assert propagator.calls[-1].new_boxes == {}


def test_factory_exposes_frame_requirement_only_when_guidance_is_enabled() -> None:
    enabled = create_tracker("bytetrack", mask_guidance=_config())
    disabled = create_tracker("bytetrack")

    assert enabled.requirements.frame
    assert not enabled.requirements.frame_dimensions_only
    assert not enabled.requirements.masks
    assert not disabled.requirements.frame
    assert len(disabled.update(_rows())) == 2
    with pytest.raises(ValueError, match="requires a frame"):
        enabled.update(_rows())


@pytest.mark.parametrize("match_thresh", [0.1, 0.8])
def test_guidance_preserves_configured_high_stage_threshold_without_masks(match_thresh: float) -> None:
    guided, _ = _guided_tracker(match_thresh=match_thresh)
    baseline = ByteTrack(match_thresh=match_thresh)
    initial = _rows()[:1]
    _update(guided, initial, 0, canonical=False)
    _update(baseline, initial, 0, canonical=False)
    moved = initial.copy()
    moved[:, [0, 2]] += 15  # Score-fused IoU cost is about .864: admitted only at .9.

    actual = _result_rows(_update(guided, moved, 1, canonical=False))
    unmatched = _result_rows(_update(baseline, moved, 1, canonical=False))

    np.testing.assert_array_equal(actual, unmatched)
    assert len(unmatched) == 0


@pytest.mark.parametrize("canonical", [False, True])
def test_temporal_mask_does_not_admit_isolated_track_after_abrupt_motion(canonical: bool) -> None:
    tracker, propagator = _guided_tracker(match_thresh=0.8)
    first = _result_rows(_update(tracker, _rows()[:1], 0, canonical=canonical))
    identity = int(first[0, 4])
    moved = np.array([[70, 10, 90, 40, 0.95, 0]], dtype=np.float32)
    mask = np.zeros(FRAME_SHAPE, dtype=bool)
    mask[10:40, 70:90] = True
    propagator.masks[identity] = mask

    unmatched = _result_rows(_update(tracker, moved, 1, canonical=canonical))

    assert len(unmatched) == 0
    assert identity in {track.id for track in tracker.lost_stracks}


@pytest.mark.parametrize("factory", [False, True], ids=["constructor", "factory"])
def test_prebuilt_guidance_runtime_uses_injected_propagator(factory: bool) -> None:
    propagator = _Propagator()
    guidance = MaskGuidance(_config(), propagator=propagator)
    tracker = create_tracker("bytetrack", mask_guidance=guidance) if factory else ByteTrack(mask_guidance=guidance)

    _update(tracker, _rows(), 0, canonical=False)

    assert tracker._mask_guidance is guidance
    assert len(propagator.calls) == 1
    tracker.reset()
    assert propagator.resets == 1


@pytest.mark.parametrize("field_name", ["max_objects", "device", "prompt_overlap"])
def test_injected_propagator_must_match_config_without_changing_state(field_name: str) -> None:
    mask = np.ones(FRAME_SHAPE, dtype=bool)
    propagator = _Propagator(masks={7: mask})
    if field_name == "max_objects":
        propagator.max_objects = 8
    elif field_name == "device":
        propagator.device = torch.device("cuda:0")
    else:
        propagator.prompt_overlap = 0.20

    with pytest.raises(ValueError, match=f"Injected propagator {field_name} must match"):
        MaskGuidance(_config(), propagator=propagator)

    assert propagator.resets == 0
    assert propagator.masks[7] is mask
    assert propagator.calls == []


def test_injected_propagator_accepts_equivalent_device_spellings() -> None:
    config = MaskGuidanceConfig(checkpoint="edgetam.pt", device="0")
    propagator = _Propagator(device=torch.device("cuda:0"))

    guidance = MaskGuidance(config, propagator=propagator)

    assert guidance._propagator is propagator


def test_guidance_releases_previous_mask_views_before_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    mask = np.ones(FRAME_SHAPE, dtype=bool)
    mask_reference = weakref.ref(mask)
    propagator = _Propagator(masks={7: mask})
    guidance = MaskGuidance(_config(), propagator=propagator)
    guidance.advance(0, np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
    del mask

    def release_previous_masks(*args: object) -> dict[int, np.ndarray]:
        propagator.masks.clear()
        assert mask_reference() is None, "Guidance must not pin masks the propagator has released"
        return {}

    monkeypatch.setattr(propagator, "propagate", release_previous_masks)
    guidance.advance(1, np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))

    assert guidance._masks == {}


@pytest.mark.parametrize("options", [{"is_obb": True}, {"per_class": True}, {"asso_func": "giou"}])
def test_guidance_rejects_unvalidated_tracker_modes(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="requires AABB"):
        ByteTrack(mask_guidance=_config(), **options)


@pytest.mark.parametrize(
    "spec", [TrackerSpec(name="maf_hda"), TrackerSpec(name="eagermot"), TrackerSpec(name="bytetrack", backend="cpp")]
)
def test_factory_rejects_guidance_for_multimodal_algorithms_and_native_backend(spec: TrackerSpec) -> None:
    with pytest.raises(ValueError, match="Python 2D box tracker"):
        create_tracker(spec, mask_guidance=_config())


@pytest.mark.parametrize("value", [True, {}, "edgetam.pt"])
def test_tracker_and_factory_reject_untyped_guidance(value: object) -> None:
    with pytest.raises(TypeError, match="MaskGuidanceConfig, MaskGuidance, or None"):
        ByteTrack(mask_guidance=value)
    with pytest.raises(TypeError, match="MaskGuidanceConfig, MaskGuidance, or None"):
        create_tracker("bytetrack", mask_guidance=value)


def test_mask_guidance_config_is_immutable_and_normalizes_paths() -> None:
    config = _config()

    assert config.checkpoint == Path("edgetam.pt")
    assert config.max_objects == 96
    with pytest.raises(FrozenInstanceError):
        config.device = "cuda"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "32", None])
def test_mask_guidance_config_rejects_invalid_object_cap(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="max_objects must be a positive integer"):
        MaskGuidanceConfig(checkpoint="edgetam.pt", max_objects=value)


def test_guidance_passes_object_cap_to_lazy_propagator(monkeypatch: pytest.MonkeyPatch) -> None:
    propagator = _Propagator()
    constructed: list[tuple[Path, str, int, float]] = []

    def construct(checkpoint: Path, *, device: str, max_objects: int, prompt_overlap: float) -> _Propagator:
        constructed.append((checkpoint, device, max_objects, prompt_overlap))
        return propagator

    monkeypatch.setitem(
        sys.modules, "boxmot.segmentors.propagation.edgetam", SimpleNamespace(EdgeTAMMaskPropagator=construct)
    )
    tracker = ByteTrack(
        mask_guidance=MaskGuidanceConfig(checkpoint="edgetam.pt", device="cpu", max_objects=4, prompt_overlap=0.2)
    )
    assert constructed == []

    _update(tracker, _rows(), 0, canonical=False)
    _update(tracker, _rows(), 1, canonical=False)

    assert constructed == [(Path("edgetam.pt"), "cpu", 4, 0.2)]
    assert len(propagator.calls) == 2


@pytest.mark.parametrize("field_name", ["checkpoint", "device"])
def test_mask_guidance_config_rejects_empty_required_values(field_name: str) -> None:
    values = {"checkpoint": "edgetam.pt", "device": "cpu"}
    values[field_name] = " "

    with pytest.raises(TypeError, match=field_name):
        MaskGuidanceConfig(**values)


def test_guidance_rejects_frames_with_different_source_dimensions() -> None:
    tracker, propagator = _guided_tracker()

    with pytest.raises(ValueError, match="constant frame dimensions"):
        tracker.update(_rows(), np.zeros((48, 64, 3), dtype=np.uint8))

    assert propagator.calls == []


@pytest.mark.parametrize("frame_index", [10, 9], ids=["repeated", "reversed"])
def test_guidance_rejects_misaligned_canonical_frames_before_advancing(frame_index: int) -> None:
    tracker, propagator = _guided_tracker()
    first = _update(tracker, _rows(), 10, canonical=True)

    with pytest.raises(ValueError, match="increasing source frame indices"):
        _update(tracker, _rows(), frame_index, canonical=True)

    assert tracker.frame_count == 1
    assert [call.frame_index for call in propagator.calls] == [0]
    continued = _update(tracker, _rows(), 11, canonical=True)
    torch.testing.assert_close(continued.track_ids, first.track_ids)


def test_guidance_accepts_source_offsets_and_gaps_with_contiguous_model_steps() -> None:
    tracker, propagator = _guided_tracker()
    for index in (100, 102, 105):
        _update(tracker, _rows(), index, canonical=True)
    assert tracker.frame_count == 3
    assert [call.frame_index for call in propagator.calls] == [0, 1, 2]


def test_guidance_requires_reset_before_switching_canonical_sequences() -> None:
    tracker, propagator = _guided_tracker()
    first = _update(tracker, _rows(), 0, canonical=True, sequence_id="first")

    with pytest.raises(ValueError, match="reset before processing sequence 'second'"):
        _update(tracker, _rows(), 1, canonical=True, sequence_id="second")

    assert tracker.frame_count == 1
    assert len(propagator.calls) == 1
    tracker.reset()
    restarted = _update(tracker, _rows(), 0, canonical=True, sequence_id="second")

    assert restarted.sample_id == "second/0"
    torch.testing.assert_close(restarted.track_ids, first.track_ids)
    assert propagator.resets == 1


def test_invalid_first_update_does_not_bind_guidance_to_its_sequence() -> None:
    tracker, propagator = _guided_tracker()
    invalid_rows = np.ones((2, 5), dtype=np.float32)
    frame = Frame(
        image=torch.zeros((3, *FRAME_SHAPE), dtype=torch.uint8),
        sample_id="invalid/0",
        sequence_id="invalid",
        frame_index=0,
    )

    with pytest.raises(ValueError):
        tracker.update(invalid_rows, frame)
    valid = _update(tracker, _rows(), 0, canonical=True, sequence_id="valid")

    assert len(valid) == 2
    assert len(propagator.calls) == 1


def test_guidance_accepts_optional_metadata_and_checks_it_when_available() -> None:
    tracker, propagator = _guided_tracker()

    _update(tracker, _rows(), None, canonical=True, sequence_id=None)
    _update(tracker, _rows(), 1, canonical=True, sequence_id="known")
    _update(tracker, _rows(), None, canonical=True, sequence_id=None)
    with pytest.raises(ValueError, match="increasing source frame indices"):
        _update(tracker, _rows(), 1, canonical=True, sequence_id="known")

    assert [call.frame_index for call in propagator.calls] == [0, 1, 2]


def test_guidance_keeps_lost_candidates_then_releases_retired_masks() -> None:
    tracker, propagator = _guided_tracker(track_buffer=2)
    result = _update(tracker, _rows()[:1], 0, canonical=True)
    identity = int(result.track_ids[0])
    propagator.masks[identity] = np.ones(FRAME_SHAPE, dtype=bool)
    empty = np.empty((0, 6), dtype=np.float32)
    _update(tracker, empty, 1, canonical=True)
    assert propagator.retained[-1] == {identity}
    assert identity in tracker._mask_guidance._masks
    for index in range(2, 6):
        _update(tracker, empty, index, canonical=True)
    assert propagator.retained[-1] == set()
    assert propagator.masks == {}
    assert tracker._mask_guidance._masks == {}


def test_disabled_guidance_preserves_existing_canonical_metadata_behavior() -> None:
    tracker = ByteTrack()

    _update(tracker, _rows(), 10, canonical=True, sequence_id="first")
    result = _update(tracker, _rows(), 20, canonical=True, sequence_id="second")

    assert len(result) == 2
    assert tracker.frame_count == 2
