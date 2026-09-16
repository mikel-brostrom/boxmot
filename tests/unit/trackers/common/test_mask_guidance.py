from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.trackers.common.association.masks import apply_mask_guidance
from boxmot.trackers.common.association.matching import linear_assignment


@pytest.fixture(params=["numpy", "tensor"], autouse=True)
def mask_input_kind(request, monkeypatch) -> None:
    """Exercise every existing association invariant with both mask representations."""
    if request.param == "tensor":
        original = apply_mask_guidance

        def with_tensor_masks(costs, boxes, masks, **kwargs):
            tensors = [None if mask is None else torch.from_numpy(mask) for mask in masks]
            return original(costs, boxes, tensors, **kwargs)

        monkeypatch.setattr(f"{__name__}.apply_mask_guidance", with_tensor_masks)


@pytest.fixture(params=["no_reid", "with_reid"])
def mcbyte_reference_assignment(request):
    """Execute the reference method without importing its detector/model stack."""
    reference = (
        Path(__file__).resolve().parents[4]
        / "McBytePlusPlus"
        / "yolox"
        / "tracker"
        / f"mcbyteplusplus_tracker__{request.param}.py"
    )
    if not reference.is_file():
        pytest.skip("Optional McBytePlusPlus reference checkout is unavailable")
    tree = ast.parse(reference.read_text())
    method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "conditioned_assignment"
    )
    namespace = {
        "np": np,
        "MIN_MM1": 0.90,
        "MIN_MM2": 0.05,
        "matching": SimpleNamespace(linear_assignment=linear_assignment),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(reference), "exec"), namespace)
    return namespace["conditioned_assignment"]


def test_mask_guidance_resolves_crossing_association_without_changing_solver() -> None:
    costs = np.array([[0.35, 0.25], [0.25, 0.35]])
    boxes = np.array([[0, 0, 4, 4], [2, 0, 6, 4]])
    masks = np.zeros((2, 4, 6), dtype=bool)
    masks[0, :2, :2] = True
    masks[1, :2, 4:6] = True

    guided = apply_mask_guidance(costs, boxes, masks, threshold=0.5)
    baseline_matches, _, _ = linear_assignment(costs, 0.5)
    guided_matches, _, _ = linear_assignment(guided, 0.5)

    np.testing.assert_array_equal(baseline_matches, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(guided_matches, [[0, 0], [1, 1]])
    np.testing.assert_allclose(guided, [[0.10, 0.25], [0.25, 0.10]])
    np.testing.assert_array_equal(costs, [[0.35, 0.25], [0.25, 0.35]])


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_mask_guidance_handles_row_and_column_ambiguity(shape: tuple[int, int]) -> None:
    costs = np.full(shape, 0.4)
    boxes = np.tile([0, 0, 4, 4], (shape[1], 1))
    masks = np.ones((shape[0], 4, 4), dtype=bool)

    guided = apply_mask_guidance(costs, boxes, masks, threshold=0.4)

    np.testing.assert_allclose(guided, np.full(shape, -0.6))


def test_mask_guidance_accepts_exact_paper_fill_and_coverage_thresholds() -> None:
    # Nine of ten mask pixels lie in a 180-pixel box: mc=0.90, mf=0.05.
    mask = np.zeros((11, 20), dtype=bool)
    mask[0, :9] = True
    mask[10, 0] = True

    guided = apply_mask_guidance(
        np.array([[0.4, 0.4]]),
        np.array([[0, 0, 20, 9], [0, 0, 20, 9]]),
        [mask],
        threshold=0.5,
    )

    np.testing.assert_allclose(guided, [[0.35, 0.35]])


@pytest.mark.parametrize("failed_gate", ["coverage", "fill", "invisible", "missing"])
@pytest.mark.parametrize("costs", [np.array([[0.3, 0.4]]), np.array([[0.95, 0.95]])], ids=["ambiguous", "isolated"])
def test_mask_guidance_preserves_costs_when_mask_gate_fails(failed_gate: str, costs: np.ndarray) -> None:
    mask = np.zeros((10, 10), dtype=bool)
    if failed_gate == "coverage":
        mask[:2, :5] = True
        boxes = np.tile([0, 0, 4, 4], (2, 1))  # mc=0.8, mf=0.5.
    else:
        boxes = np.tile([0, 0, 10, 10], (2, 1))
        if failed_gate == "fill":
            mask[:2, :2] = True  # mc=1.0, mf=0.04.
    guided = apply_mask_guidance(costs, boxes, [None if failed_gate == "missing" else mask], threshold=0.5)

    np.testing.assert_array_equal(guided, costs)


def test_mask_guidance_uses_positive_logits_for_visibility() -> None:
    logits = np.full((4, 4), -1.0)
    logits[:2, :2] = 0.0
    costs = np.array([[0.3, 0.4]])
    boxes = np.tile([0, 0, 4, 4], (2, 1))

    invisible = apply_mask_guidance(costs, boxes, [logits], threshold=0.5)
    logits[:2, :2] = 1.0
    visible = apply_mask_guidance(costs, boxes, [logits], threshold=0.5)

    np.testing.assert_array_equal(invisible, costs)
    np.testing.assert_allclose(visible, [[0.05, 0.15]])


@pytest.mark.parametrize("has_masks", [False, True])
def test_mask_guidance_preserves_clear_matches_and_penalizes_their_competitors(has_masks: bool) -> None:
    costs = np.array([[0.2, 0.95], [0.95, 0.4]])
    boxes = np.tile([0, 0, 4, 4], (2, 1))
    masks = np.ones((2, 4, 4), dtype=bool) if has_masks else [None, None]

    guided = apply_mask_guidance(costs, boxes, masks, threshold=0.5)

    np.testing.assert_allclose(guided, [[0.2, 20.95], [20.95, 0.4]])
    np.testing.assert_array_equal(costs, [[0.2, 0.95], [0.95, 0.4]])


def test_mask_guidance_does_not_recover_isolation_above_original_assignment_threshold() -> None:
    costs = np.array([[0.95]])

    guided = apply_mask_guidance(
        costs,
        np.array([[0, 0, 4, 4]]),
        [np.ones((4, 4), dtype=bool)],
        threshold=0.9,
    )

    np.testing.assert_array_equal(guided, costs)
    matches, unmatched_tracks, unmatched_detections = linear_assignment(guided, 0.9)
    assert matches.size == 0
    np.testing.assert_array_equal(unmatched_tracks, [0])
    np.testing.assert_array_equal(unmatched_detections, [0])


def test_mask_guidance_uses_original_costs_and_current_stage_threshold() -> None:
    costs = np.array([[0.4, 0.8]])
    boxes = np.tile([0, 0, 4, 4], (2, 1))
    masks = [np.ones((4, 4), dtype=bool)]

    low_threshold = apply_mask_guidance(costs, boxes, masks, threshold=0.5)
    high_threshold = apply_mask_guidance(costs, boxes, masks, threshold=0.9)

    np.testing.assert_allclose(low_threshold, [[0.4, 10.8]])
    np.testing.assert_allclose(high_threshold, [[-0.6, -0.2]])


@pytest.mark.parametrize("box", [[-2.9, -1.9, 2.9, 2.9], [-5, -5, -1, -1]])
def test_mask_guidance_clamps_tlwh_origin_before_clipping_extent(box: list[float]) -> None:
    boxes = np.tile(box, (2, 1))
    costs = np.array([[0.4, 0.4]])
    mask = np.ones((4, 4), dtype=bool)

    guided = apply_mask_guidance(costs, boxes, [mask], threshold=0.5)

    np.testing.assert_allclose(guided, [[-0.6, -0.6]])


def test_mask_guidance_truncates_fractional_tlwh_coordinates() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    insufficient_coverage = apply_mask_guidance(
        np.array([[0.4, 0.4]]), np.tile([1.2, 1.2, 2.2, 2.2], (2, 1)), [mask], threshold=0.5
    )
    np.testing.assert_array_equal(insufficient_coverage, [[0.4, 0.4]])
    mask[:] = False
    mask[1, 1] = True

    guided = apply_mask_guidance(np.array([[0.4, 0.4]]), np.tile([1.2, 1.2, 2.2, 2.2], (2, 1)), [mask], threshold=0.5)

    np.testing.assert_allclose(guided, [[-0.6, -0.6]])


@pytest.mark.parametrize("box", [[0, 0, 0, 4], [0, 0, 4, 0], [3, 3, 1, 1], [5, 5, 9, 9], [1.1, 1.1, 1.9, 1.9]])
def test_mask_guidance_skips_boxes_without_rasterized_pixels(box: list[float]) -> None:
    costs = np.array([[0.3, 0.4]])

    guided = apply_mask_guidance(costs, np.tile(box, (2, 1)), [np.ones((4, 4), dtype=bool)], threshold=0.5)

    np.testing.assert_array_equal(guided, costs)


@pytest.mark.parametrize("shape", [(0, 0), (0, 2), (2, 0)])
def test_mask_guidance_preserves_empty_matrix_shape(shape: tuple[int, int]) -> None:
    costs = np.empty(shape)

    guided = apply_mask_guidance(costs, np.zeros((shape[1], 4)), [None] * shape[0], threshold=0.5)

    assert guided.shape == shape
    assert guided is not costs


@pytest.mark.parametrize("transpose", [False, True], ids=["isolated_track", "isolated_detection"])
def test_mask_guidance_never_adjusts_inadmissible_pairs_beside_ambiguity(transpose: bool) -> None:
    costs = np.array([[0.95, 0.95], [0.2, 0.2]])
    if transpose:
        costs = costs.T

    guided = apply_mask_guidance(costs, np.tile([0, 0, 4, 4], (2, 1)), np.ones((2, 4, 4), dtype=bool), threshold=0.5)

    np.testing.assert_array_equal(guided[costs == 0.95], costs[costs == 0.95])


def test_mask_guidance_protects_clear_pairs_beside_ambiguity_and_isolation() -> None:
    costs = np.array(
        [[0.2, 0.95, 0.95, 0.95], [0.95, 0.3, 0.4, 0.95], [0.95, 0.4, 0.3, 0.95], [0.95, 0.95, 0.95, 0.95]]
    )

    guided = apply_mask_guidance(costs, np.tile([0, 0, 4, 4], (4, 1)), np.ones((4, 4, 4), dtype=bool), threshold=0.5)

    np.testing.assert_allclose(
        guided,
        [[0.2, 10.95, 10.95, 10.95], [10.95, -0.7, -0.6, 0.95], [10.95, -0.6, -0.7, 0.95], [10.95, 0.95, 0.95, 0.95]],
    )
    matches, unmatched_tracks, unmatched_detections = linear_assignment(guided, 0.5)
    np.testing.assert_array_equal(matches, [[0, 0], [1, 1], [2, 2]])
    np.testing.assert_array_equal(unmatched_tracks, [3])
    np.testing.assert_array_equal(unmatched_detections, [3])


@pytest.mark.parametrize("mask_state", ["visible", "partly_missing", "absent"])
def test_mask_guidance_matches_actual_mcbyte_reference_costs_and_assignments(
    mcbyte_reference_assignment, mask_state: str
) -> None:
    """Compare both reference variants over fractional boxes and varied cost graphs."""
    rng = np.random.default_rng(491)
    for iteration in range(24):
        rows = 1 + iteration % 5
        columns = 1 + iteration % 7
        threshold = (0.2, 0.5, 0.9)[iteration % 3]
        costs = rng.random((rows, columns))
        origin = rng.uniform(-3.5, 3.5, size=(columns, 2))
        extent = rng.uniform(1.0, 9.0, size=(columns, 2))
        boxes = np.column_stack((origin, origin + extent))
        # Some detections cover the full frame, ensuring coverage passes too.
        boxes[::2] = [0, 0, 8, 6]
        masks = [rng.normal(size=(6, 8)) for _ in range(rows)]
        if iteration % 4 == 0:
            masks[0][:] = -1  # A registered mask can be invisible this frame.
        if mask_state == "absent":
            masks = [None] * rows
        elif mask_state == "partly_missing":
            masks[::2] = [None] * len(masks[::2])
        ids = [index for index, mask in enumerate(masks) if mask is not None]
        prediction = (
            {"mask_ids": ids, "mask_logits": [masks[index][None] for index in ids]} if mask_state != "absent" else None
        )
        tracks = [SimpleNamespace(track_id=index) for index in range(rows)]
        detections = [SimpleNamespace(tlwh=np.array([x1, y1, x2 - x1, y2 - y1])) for x1, y1, x2, y2 in boxes]

        reference_matches, reference_unmatched_tracks, reference_unmatched_detections, reference_costs = (
            mcbyte_reference_assignment(
                None, costs, threshold, tracks, detections, prediction, {index: index for index in ids}, (6, 8)
            )
        )
        guided = apply_mask_guidance(costs, boxes, masks, threshold=threshold)
        matches, unmatched_tracks, unmatched_detections = linear_assignment(guided, threshold)

        np.testing.assert_allclose(guided, reference_costs, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(matches, reference_matches)
        np.testing.assert_array_equal(unmatched_tracks, reference_unmatched_tracks)
        np.testing.assert_array_equal(unmatched_detections, reference_unmatched_detections)
