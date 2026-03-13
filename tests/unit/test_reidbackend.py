from pathlib import Path

import cv2
import numpy as np
import pytest

from boxmot.reid.backends.onnx_backend import ONNXBackend
from boxmot.reid.backends.openvino_backend import OpenVinoBackend
from boxmot.reid.backends.pytorch_backend import PyTorchBackend
from boxmot.reid.backends.torchscript_backend import TorchscriptBackend
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.utils import ROOT, WEIGHTS

# Exported artifacts are covered by the dedicated CI export job.
REID_MODEL_CASES = [
    (WEIGHTS / "osnet_x0_25_msmt17.pt", PyTorchBackend, False),
    (WEIGHTS / "osnet_x0_25_msmt17.torchscript", TorchscriptBackend, True),
    (WEIGHTS / "osnet_x0_25_msmt17.onnx", ONNXBackend, True),
    (WEIGHTS / "osnet_x0_25_msmt17_openvino_model", OpenVinoBackend, True),
]


def get_backend(weights: Path, requires_export: bool):
    """Return a backend instance, skipping exported formats when artifacts are absent."""
    if requires_export and not weights.exists():
        pytest.skip(f"Missing exported ReID artifact: {weights}")

    rab = ReidAutoBackend(weights=weights, device="cpu", half=False)
    return rab.get_backend()


@pytest.mark.parametrize("reid_model, _, requires_export", REID_MODEL_CASES)
def test_reidbackend_output(reid_model, _, requires_export):
    b = get_backend(reid_model, requires_export)

    img = cv2.imread(
        str(ROOT / "assets/MOT17-mini/train/MOT17-04-FRCNN/img1/000001.jpg")
    )
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                     [425, 281, 576, 472, 0.56, 65]])

    embs = b.get_features(dets[:, 0:4], img)
    assert embs.shape[0] == 2  # two crops should give two embeddings
    assert embs.shape[1] == 512  # osnet embeddings are of size 512


@pytest.mark.parametrize("reid_model, backend, requires_export", REID_MODEL_CASES)
def test_reidbackend_type(reid_model, backend, requires_export):
    b = get_backend(reid_model, requires_export)
    assert isinstance(b, backend)
