import cv2
import torch
import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import ROOT

from boxmot.appearance.reid_auto_backend import ReidAutoBackend

REID_MODELS = [
    Path('mobilenetv2_x1_0_market1501.pt'),
]


@pytest.mark.parametrize("reid_model", REID_MODELS)
def test_reidbackend_device(reid_model):

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = ReidAutoBackend(
        weights=reid_model, device=device, half=False
    ).model

    if torch.cuda.is_available():
        assert next(model.model.parameters()).is_cuda
    else:
        assert next(model.model.parameters()).device.type == 'cpu'


@pytest.mark.parametrize("reid_model", REID_MODELS)
def test_reidbackend_half(reid_model):

    half = True if torch.cuda.is_available() else False
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ReidAutoBackend(
        weights=reid_model, device=device, half=False
    ).model

    if device == 'cpu':
        expected_dtype = torch.float32
    else:
        expected_dtype = torch.float16
    actual_dtype = next(model.model.parameters()).dtype
    assert actual_dtype == expected_dtype
