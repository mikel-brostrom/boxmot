import cv2
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
        device='cuda:0',
        fp16=False  # not compatible with OSNet
    )

    assert r.model.is_cuda