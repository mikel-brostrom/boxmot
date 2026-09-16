"""Device masks transfer exact scalar counts while preserving CPU association gates."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot.trackers.common.association import masks as mask_association
from boxmot.trackers.common.association.masks import apply_mask_guidance


@pytest.fixture(params=["cpu", "mps", "cuda"])
def device(request) -> torch.device:
    """Run real device reductions where the current process has access to them."""
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable in this process")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this process")
    return torch.device(request.param)


def _count_transfers(monkeypatch) -> list[tuple[torch.device, torch.dtype, tuple[int, ...]]]:
    """Reject per-pair scalar synchronization and record actual CPU-transfer shapes."""
    calls = []
    original = torch.Tensor.cpu

    def cpu(tensor, *args, **kwargs):
        calls.append((tensor.device, tensor.dtype, tuple(tensor.shape)))
        return original(tensor, *args, **kwargs)

    def item(tensor, *args, **kwargs):
        pytest.fail("Association must transfer one count vector instead of synchronizing individual scalars")

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "item", item)
    return calls


def test_transfers_one_count_vector_for_multiple_masks(device, monkeypatch) -> None:
    masks = np.ones((3, 11, 20), dtype=bool)
    masks[0] = False
    masks[0, 0, :9] = True
    masks[0, 10, 0] = True
    masks[2] = False
    costs = np.full((3, 2), 0.4)
    boxes = np.tile([0, 0, 20, 9], (2, 1))
    expected = apply_mask_guidance(costs, boxes, masks, threshold=0.5)
    tensors = [torch.tensor(mask, device=device) for mask in masks]
    calls = _count_transfers(monkeypatch)

    actual = apply_mask_guidance(costs, boxes, tensors, threshold=0.5)

    np.testing.assert_array_equal(actual, expected)
    assert calls == [(tensors[0].device, torch.int32, (9,))]  # One area and two intersections per mask.


@pytest.mark.parametrize("case", ["clear", "isolated", "invalid_boxes", "missing"])
def test_no_counts_transfer_without_eligible_mask_pairs(device, monkeypatch, case) -> None:
    costs = np.array([[0.2, 0.95], [0.95, 0.4]]) if case == "clear" else np.full((2, 2), 0.4)
    if case == "isolated":
        costs[:] = 0.95
    boxes = np.tile([0, 0, 0, 4] if case == "invalid_boxes" else [0, 0, 4, 4], (2, 1))
    masks = [None, None] if case == "missing" else [torch.ones((4, 4), dtype=torch.bool, device=device)] * 2
    calls = _count_transfers(monkeypatch)

    actual = apply_mask_guidance(costs, boxes, masks, threshold=0.5)

    expected = np.array([[0.2, 20.95], [20.95, 0.4]]) if case == "clear" else costs
    np.testing.assert_array_equal(actual, expected)
    assert calls == []


@pytest.mark.parametrize("threshold", [0.2, 0.7])
def test_noncontiguous_logits_and_mixed_masks_match_numpy_exactly(device, threshold) -> None:
    rng = np.random.default_rng(246)
    raw = rng.normal(size=(4, 17, 15)).astype(np.float32)
    masks = [raw[index].T for index in range(4)]
    masks[2] = None
    costs = rng.random((4, 12))
    boxes = rng.uniform(-10, 25, size=(12, 4))
    boxes[:, 2:] += 12
    boxes[0] = [-1e100, -1e100, 1e100, 1e100]
    kwargs = dict(threshold=threshold, min_coverage=0.1, min_fill=0.1)
    expected = apply_mask_guidance(costs, boxes, masks, **kwargs)
    inputs = [torch.tensor(raw[0], device=device).T, masks[1], None, torch.tensor(raw[3], device=device).T]
    originals = [value.clone() if isinstance(value, torch.Tensor) else value for value in inputs]

    actual = apply_mask_guidance(costs, boxes, inputs, **kwargs)

    np.testing.assert_array_equal(actual, expected)
    for before, after in zip(originals, inputs):
        if isinstance(after, torch.Tensor):
            assert torch.equal(after, before)


@pytest.mark.parametrize("shape", [(3,), (1, 3, 4)])
def test_invalid_tensor_mask_dimensions_fail_before_counting(shape) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        apply_mask_guidance(np.array([[0.4, 0.4]]), np.tile([0, 0, 3, 4], (2, 1)), [torch.ones(shape)], threshold=0.5)


def test_masks_on_distinct_devices_transfer_one_vector_per_device(monkeypatch) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable in this process")
    costs = np.full((2, 2), 0.4)
    boxes = np.tile([0, 0, 4, 4], (2, 1))
    masks = [torch.ones((4, 4), dtype=torch.bool), torch.ones((4, 4), dtype=torch.bool, device="mps")]
    calls = _count_transfers(monkeypatch)

    actual = apply_mask_guidance(costs, boxes, masks, threshold=0.5)

    np.testing.assert_array_equal(actual, costs - 1)
    assert calls == [(mask.device, torch.int32, (3,)) for mask in masks]


@pytest.mark.parametrize("pixel_budget,crop_limit", [(48, 3), (4 * 1024 * 1024, 2)])
def test_crop_batches_preserve_mixed_shape_pair_order_with_bounded_workspace(
    monkeypatch, pixel_budget, crop_limit
) -> None:
    """Grouping across track rows must preserve indices and both workspace bounds."""
    rng = np.random.default_rng(331)
    masks = rng.random((4, 10, 10)) > 0.4
    boxes = np.array([[0, 0, 4, 4], [2, 1, 8, 5], [0, 0, 10, 10]])
    costs = np.full((4, 3), 0.4)
    expected = apply_mask_guidance(costs, boxes, masks, threshold=0.5, min_coverage=0, min_fill=0)
    monkeypatch.setattr(mask_association, "_MAX_CROP_BATCH_PIXELS", pixel_budget)
    monkeypatch.setattr(mask_association, "_MAX_CROP_BATCH_SIZE", crop_limit)
    stacked_shapes = []
    original_stack = torch.stack

    def stack(tensors, *args, **kwargs):
        if tensors[0].ndim == 2:
            shape = (len(tensors), *tensors[0].shape)
            stacked_shapes.append(shape)
            assert shape[0] <= crop_limit
            assert np.prod(shape) <= pixel_budget
        return original_stack(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "stack", stack)
    transfers = _count_transfers(monkeypatch)

    actual = apply_mask_guidance(
        costs, boxes, [torch.from_numpy(mask) for mask in masks], threshold=0.5, min_coverage=0, min_fill=0
    )

    np.testing.assert_array_equal(actual, expected)
    assert stacked_shapes
    assert all(shape[0] > 1 for shape in stacked_shapes)
    assert transfers == [(torch.device("cpu"), torch.int32, (16,))]
    if pixel_budget == 48:
        assert stacked_shapes == [(3, 4, 4), (2, 4, 6), (2, 4, 6)]
