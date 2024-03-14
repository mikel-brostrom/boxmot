import cv2
import pytest
import numpy as np
from pathlib import Path
from boxmot.utils import ROOT, WEIGHTS
from boxmot.appearance.backends.onnx_backend import ONNXBackend
from boxmot.appearance.backends.openvino_backend import OpenVinoBackend
from boxmot.appearance.backends.pytorch_backend import PyTorchBackend
from boxmot.appearance.backends.tensorrt_backend import TensorRTBackend
from boxmot.appearance.backends.tflite_backend import TFLiteBackend
from boxmot.appearance.backends.torchscript_backend import TorchscriptBackend

from boxmot.appearance.reid_auto_backend import ReidAutoBackend

# generated in previous job step
EXPORTED_REID_MODELS = [
    WEIGHTS / 'osnet_x0_25_msmt17.pt',
    WEIGHTS / 'osnet_x0_25_msmt17.torchscript',
    WEIGHTS / 'osnet_x0_25_msmt17.onnx',
    WEIGHTS / 'osnet_x0_25_msmt17_openvino_model'
]

ASSOCIATED_BACKEND = [
    PyTorchBackend,
    TorchscriptBackend,
    ONNXBackend,
    OpenVinoBackend
]


@pytest.mark.parametrize("reid_model", EXPORTED_REID_MODELS)
def test_reidbackend_output(reid_model):

    rab = ReidAutoBackend(
        weights=reid_model, device='cpu', half=False
    )
    b = rab.get_backend()

    img = cv2.imread(str(ROOT / 'assets/MOT17-mini/train/MOT17-04-FRCNN/img1/000001.jpg'))
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                    [425, 281, 576, 472, 0.56, 65]])

    embs = b.get_features(dets[:, 0:4], img)
    assert embs.shape[0] == 2   # two crops should give two embeddings
    assert embs.shape[1] == 512 # osnet embeddings are of size 512


@pytest.mark.parametrize("exported_reid_model, backend", zip(EXPORTED_REID_MODELS, ASSOCIATED_BACKEND))
def test_reidbackend_type(exported_reid_model, backend):

    rab = ReidAutoBackend(
        weights=exported_reid_model, device='cpu', half=False
    )
    b = rab.get_backend()

    assert isinstance(b, backend)