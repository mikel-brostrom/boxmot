"""Segmentor model placement uses shared logical device selection."""

import os
from unittest.mock import Mock

import pytest
import torch

from boxmot.segmentors.backends.maskrcnn import MaskRCNNSegmentor
from boxmot.segmentors.specs import SegmentorSpec


@pytest.mark.parametrize("selector", ["1", "cuda:1"])
def test_maskrcnn_places_model_on_requested_logical_gpu(monkeypatch, selector: str) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    model = Mock()

    MaskRCNNSegmentor(SegmentorSpec(backend="maskrcnn", device=selector), model=model)

    model.to.assert_called_once_with(device=torch.device("cuda:1"), dtype=torch.float32)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,4"


def test_maskrcnn_rejects_unavailable_gpu_before_moving_model(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = Mock()

    with pytest.raises(RuntimeError, match="cuda:1 is unavailable"):
        MaskRCNNSegmentor(SegmentorSpec(backend="maskrcnn", device="1"), model=model)

    model.to.assert_not_called()
