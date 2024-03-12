import cv2
import torch
import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import ROOT

from boxmot.appearance.reid_multibackend import ReIDDetectMultiBackend

REID_MODELS = [
    Path('osnet_x0_25_msmt17.pt'),
    Path('osnet_x1_0_dukemtmcreid.pt')
]


@pytest.mark.parametrize("reid_model", REID_MODELS)
def test_reidbackend_output(reid_model):

    r = ReIDDetectMultiBackend(
        weights=reid_model,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        fp16=False  # not compatible with OSNet
    )

    if torch.cuda.is_available():
        assert r.model.is_cuda
    else:
        assert r.model.device.type == 'cpu'
