"""Compiled camera-motion input contracts run with the native build toolchain."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from boxmot.structures import Frame
from tests.unit.trackers.test_appearance_input_contracts import _NATIVE_TRACKERS, _detections


@pytest.mark.parametrize("name", _NATIVE_TRACKERS)
@pytest.mark.parametrize("geometry", ("aabb", "obb"))
def test_real_native_sof_accepts_small_textured_images(name: str, geometry: str) -> None:
    """The compiled camera-motion path must accept the same small frames as Python."""
    tracker = _NATIVE_TRACKERS[name](
        {"use_cmc": True, "cmc_method": "sof", "use_embeddings": False}, geometry=geometry
    )
    pixels = np.random.default_rng(17).integers(0, 256, (3, 96, 96), dtype=np.uint8)
    frame = Frame(torch.from_numpy(pixels), "input-contract")
    try:
        for _ in range(3):
            result = tracker.update(_detections(geometry=geometry), frame)
            assert len(result) == 1
            assert torch.isfinite(result.geometry.values).all()
    finally:
        tracker.close()
