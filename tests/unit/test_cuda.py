import cv2
import torch
import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import ROOT

from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend

REID_MODELS = [
    Path('osnet_x0_25_msmt17.pt'),
]


@pytest.mark.parametrize("reid_model", REID_MODELS)
def test_reidbackend_device(reid_model):

    r = ReIDDetectMultiBackend(
        weights=reid_model,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        fp16=False  # not compatible with OSNet
    )

    if torch.cuda.is_available():
        assert next(r.model.parameters()).is_cuda
    else:
        assert next(r.model.parameters()).device.type == 'cpu'
