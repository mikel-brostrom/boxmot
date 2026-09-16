"""Rendering requests CPU pixels lazily, independently of mask association."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boxmot.trackers.common.mask_guidance import MaskGuidance, MaskGuidanceConfig, _CPUMaskView


def test_view_metadata_never_materializes_pixels(monkeypatch) -> None:
    masks = {7: torch.ones((6, 8), dtype=torch.bool)}
    monkeypatch.setattr(torch.Tensor, "cpu", lambda *a, **kw: pytest.fail("Unexpected mask download"))
    monkeypatch.setattr(torch.Tensor, "numpy", lambda *a, **kw: pytest.fail("Unexpected pixel conversion"))
    view = _CPUMaskView(masks)
    assert len(view) == 1 and list(view) == [7] and set(view.keys()) == {7}
    assert 7 in view and 8 not in view
    with pytest.raises(KeyError):
        view[8]


def test_cpu_view_caches_read_only_arrays_and_preserves_native_numpy_masks() -> None:
    tensor = torch.ones((6, 8), dtype=torch.bool)
    native = np.zeros((6, 8), dtype=bool)
    view = _CPUMaskView({1: tensor, 2: native})
    first = view[1]
    assert first is view[1] and view[2] is native
    assert first.dtype == bool and not first.flags.writeable
    assert np.shares_memory(first, tensor.numpy())
    with pytest.raises(ValueError, match="read-only"):
        first[0, 0] = False
    with pytest.raises(TypeError):
        view[3] = native


def test_guidance_releases_its_old_view_when_advancing_retiring_and_resetting() -> None:
    masks = {0: torch.ones((6, 8), dtype=torch.bool)}
    propagator = SimpleNamespace(
        max_objects=96,
        device="cpu",
        prompt_overlap=0.1,
        propagate=lambda *args: masks.copy(),
        retain_tracks=lambda ids: None,
        reset=lambda: None,
    )
    guidance = MaskGuidance(MaskGuidanceConfig("unused.pt", device="cpu"), propagator=propagator)
    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    guidance.advance(0, frame)
    previous = weakref.ref(guidance.masks)
    assert guidance.masks is previous()
    guidance.advance(1, frame)
    assert previous() is None
    previous = weakref.ref(guidance.masks)
    guidance.observe({}, (), retained_track_ids=())
    assert previous() is None and not guidance.masks
    previous = weakref.ref(guidance.masks)
    guidance.reset()
    assert previous() is None and not guidance.masks


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable in this process")
def test_render_downloads_device_masks_once_and_reuses_the_arrays(monkeypatch) -> None:
    masks = {key: torch.full((6, 8), bool(key % 2), device="mps", dtype=torch.bool) for key in range(8)}
    original_cpu, transfers = torch.Tensor.cpu, []

    def cpu(tensor, *args, **kwargs):
        transfers.append((tuple(tensor.shape), tensor.dtype))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    view = _CPUMaskView(masks)
    assert set(view) == set(masks) and transfers == []
    for key, array in view.items():
        np.testing.assert_array_equal(array, np.full((6, 8), bool(key % 2)))
        assert array is view[key] and not array.flags.writeable
    assert transfers == [((8, 6, 8), torch.bool)]
    assert all(isinstance(value, np.ndarray) for value in view._masks.values())
