"""Prepared transforms retain the live tracker motion and input contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot import create_tracker
from boxmot.structures import Boxes, Detections, OrientedBoxes
from boxmot.trackers.common.mask_guidance import MaskGuidanceConfig
from boxmot.trackers.common.motion.cmc.prepared import PreparedCMC
from boxmot.trackers.common.specs import TrackerSpec
from tests.unit.trackers._reid import RecordingEncoder


def _warps(count: int = 5) -> np.ndarray:
    """Keep a nonzero camera displacement after the first delivered frame."""
    warps = np.repeat(np.eye(2, 3, dtype=np.float32)[None], count, axis=0)
    warps[1:, :, 2] = [3.0, -1.0]
    return warps


def test_prepared_cmc_selection_is_explicit_repeatable_and_resettable() -> None:
    source = _warps(2)
    prepared = PreparedCMC(source)
    source[:] = 0
    with pytest.raises(RuntimeError, match="Select a delivered frame"):
        prepared.apply()
    with pytest.raises(ValueError, match="expected delivered frame 0"):
        prepared.select_frame(1)
    prepared.select_frame(0)
    with pytest.raises(RuntimeError, match="previous.*not applied"):
        prepared.select_frame(1)
    np.testing.assert_array_equal(prepared.apply(), np.eye(2, 3))
    np.testing.assert_array_equal(prepared.apply(), np.eye(2, 3))
    with pytest.raises(ValueError):
        prepared.apply()[0, 0] = 9
    with pytest.raises(RuntimeError, match="every delivered frame"):
        prepared.validate_complete()
    prepared.select_frame(np.int64(1))
    with pytest.raises(RuntimeError, match="every delivered frame"):
        prepared.validate_complete()
    np.testing.assert_array_equal(prepared.apply(), _warps(2)[1])
    prepared.validate_complete()
    with pytest.raises(ValueError, match="expected delivered frame 2"):
        prepared.select_frame(1)
    with pytest.raises(ValueError):
        prepared.select_frame(2)
    prepared.reset()
    prepared.select_frame(0)
    np.testing.assert_array_equal(prepared.apply(), np.eye(2, 3))


@pytest.mark.parametrize("frame_index", [True, 0.0, "0"])
def test_prepared_cmc_rejects_noninteger_frame_indices(frame_index: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        PreparedCMC(_warps()).select_frame(frame_index)


@pytest.mark.parametrize(
    "warps",
    [np.eye(2, 3), np.zeros((2, 3, 3)), np.zeros((2, 2, 3)), np.full((2, 2, 3), np.nan), _warps()[1:]],
)
def test_prepared_cmc_rejects_invalid_transforms(warps: np.ndarray) -> None:
    with pytest.raises(ValueError):
        PreparedCMC(warps)


@pytest.mark.parametrize(
    "name", ["bytetrack", "botsort", "ocsort", "deepocsort", "boosttrack", "hybridsort", "occluboost", "strongsort"]
)
@pytest.mark.parametrize("geometry", ["aabb", "obb"])
@pytest.mark.parametrize("per_class", [False, True])
def test_prepared_cmc_matches_live_motion_without_frame_pixels(name, geometry, per_class, monkeypatch) -> None:
    """Real kernels must keep output geometry, IDs and lifecycle through empty frames."""
    options = {"use_cmc": True, "cmc_method": "sof", "cmc_min_conf": 0.1}
    if name not in {"bytetrack", "ocsort", "strongsort"}:
        options["use_embeddings"] = False
    spec = TrackerSpec(name, geometry=geometry, per_class=per_class, options=tuple(sorted(options.items())))
    live, replayed = create_tracker(spec), create_tracker(spec)
    warps = _warps()
    consumed = []

    def estimate(image, boxes):
        assert image is not None
        consumed.append(boxes.copy())
        return warps[len(consumed) - 1].copy()

    live.cmc = SimpleNamespace(apply=estimate, reset=lambda: None)
    prepared = replayed.set_prepared_cmc(warps)
    assert not replayed.requirements.frame_pixels
    monkeypatch.setattr(replayed, "_frame_to_bgr", lambda frame: pytest.fail("Prepared CMC read image pixels"))
    for index in range(len(warps)):
        if index in {0, 3}:
            geometry_values = torch.empty((0, 5 if geometry == "obb" else 4))
        elif geometry == "obb":
            geometry_values = torch.tensor(
                [[30 + 3 * index, 40 - index, 10, 20, 0.2], [80 + 3 * index, 60 - index, 10, 20, -0.2]],
                dtype=torch.float32,
            )
        else:
            geometry_values = torch.tensor(
                [
                    [20 + 3 * index, 30 - index, 30 + 3 * index, 50 - index],
                    [70 + 3 * index, 50 - index, 80 + 3 * index, 70 - index],
                ],
                dtype=torch.float32,
            )
        count = len(geometry_values)
        detections = Detections(
            geometry=(OrientedBoxes if geometry == "obb" else Boxes)(geometry_values),
            scores=torch.full((count,), 0.95),
            class_ids=torch.arange(count, dtype=torch.int64),
            embeddings=torch.ones((count, 4)) if live.requirements.embeddings else None,
            sample_id=f"frame-{index}",
        )
        expected = live.update(detections, np.zeros((100, 120, 3), dtype=np.uint8))
        prepared.select_frame(index)
        actual = replayed.update(detections)
        np.testing.assert_allclose(actual.geometry.values.numpy(), expected.geometry.values.numpy(), atol=1e-6)
        np.testing.assert_array_equal(actual.track_ids.numpy(), expected.track_ids.numpy())
        np.testing.assert_array_equal(actual.class_ids.numpy(), expected.class_ids.numpy())
    prepared.validate_complete()
    assert len(consumed) == len(warps)
    replayed.reset()
    prepared.select_frame(0)
    prepared.apply()


def test_prepared_cmc_preserves_non_cmc_frame_requirements() -> None:
    options = (("cmc_method", "sof"), ("use_cmc", True))
    tracker = create_tracker(TrackerSpec("ocsort", options=(("asso_func", "centroid"), *options)))
    assert tracker.requirements.frame_pixels
    tracker.set_prepared_cmc(_warps())
    assert tracker.requirements.frame_dimensions_only

    guided = create_tracker(
        TrackerSpec("ocsort", options=options),
        mask_guidance=MaskGuidanceConfig(checkpoint="unused.pt"),
    )
    guided.set_prepared_cmc(_warps())
    assert guided.requirements.frame_pixels


def test_prepared_cmc_rejects_disabled_or_running_tracker() -> None:
    disabled = create_tracker(TrackerSpec("ocsort"))
    with pytest.raises(ValueError, match="enabled"):
        disabled.set_prepared_cmc(_warps())
    tracker = create_tracker(TrackerSpec("ocsort", options=(("use_cmc", True),)))
    tracker.update(np.empty((0, 6)), np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="before the first update"):
        tracker.set_prepared_cmc(_warps())


def test_prepared_cmc_preserves_live_embedding_frame_validation() -> None:
    """Removing estimator pixels must not bypass the live appearance input guard."""
    encoder = RecordingEncoder(np.ones((1, 4), dtype=np.float32))
    tracker = create_tracker("strongsort", use_cmc=True, reid=encoder)
    prepared = tracker.set_prepared_cmc(_warps(1))
    detections = np.array([[10, 20, 30, 50, 0.95, 0]], dtype=np.float32)
    prepared.select_frame(0)
    with pytest.raises(ValueError, match="requires a frame to generate detection embeddings"):
        tracker.update(detections)
    tracker.update(detections, np.zeros((64, 64, 3), dtype=np.uint8))
    prepared.validate_complete()
