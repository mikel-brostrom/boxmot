"""Canonical tracker pixel conversion preserves colors, layout, and ownership."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot.structures import Frame
from boxmot.trackers.common.appearance.live import LiveReIDMixin


@pytest.mark.parametrize("image_size", [(1, 1), (1, 13), (19, 1), (17, 31), (240, 320)])
def test_canonical_frame_conversion_preserves_bgr_pixels_and_owns_storage(
    image_size: tuple[int, int],
) -> None:
    """Swapping channels must preserve every pixel without aliasing the input."""
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (3, *image_size), dtype=np.uint8)
    original = rgb.copy()
    frame = Frame(image=torch.from_numpy(rgb), sample_id="sequence:0")
    expected = frame.image.permute(1, 2, 0).flip(-1).contiguous().numpy()

    actual = LiveReIDMixin._frame_to_bgr(frame)

    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (*image_size, 3)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert not np.shares_memory(actual, rgb)
    actual.fill(0)
    np.testing.assert_array_equal(rgb, original)


def test_numpy_frame_conversion_preserves_existing_bgr_buffer() -> None:
    """Already normalized NumPy tracker pixels require no conversion or copy."""
    image = np.full((17, 31, 3), (11, 22, 33), dtype=np.uint8)

    assert LiveReIDMixin._frame_to_bgr(image) is image


def test_canonical_frame_conversion_resolves_lazy_negative_view() -> None:
    """A contiguous uint8 tensor's lazy negation remains a valid RGB frame."""
    pixels = torch.tensor([[[1, 2]], [[10, 20]], [[100, 200]]], dtype=torch.uint8)
    image = torch._neg_view(pixels)
    frame = Frame(image=image, sample_id="sequence:0")
    expected = image.permute(1, 2, 0).flip(-1).contiguous().numpy()

    actual = LiveReIDMixin._frame_to_bgr(frame)

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert image.is_neg()
    torch.testing.assert_close(pixels, torch.tensor([[[1, 2]], [[10, 20]], [[100, 200]]], dtype=torch.uint8))
