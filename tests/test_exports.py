import torch

from boxmot.appearance.backbones import build_model
from boxmot.appearance.reid_export import (export_onnx, export_openvino,
                                           export_tflite, export_torchscript)
from boxmot.appearance.reid_model_factory import (get_model_name,
                                                  load_pretrained_weights)
from boxmot.utils import WEIGHTS

PT_WEIGHTS = WEIGHTS / 'osnet_x0_25_msmt17.pt'
ONNX_WEIGHTS = WEIGHTS / 'osnet_x0_25_msmt17.onnx'


im = torch.zeros(1, 3, 256, 128)

model = build_model(
    get_model_name(PT_WEIGHTS),
    num_classes=1,
    pretrained=False,
    use_gpu=False,
).to('cpu')

load_pretrained_weights(model, PT_WEIGHTS)
model.eval()


def test_export_torchscript():
    f = export_torchscript(
        model,
        im,
        PT_WEIGHTS,
        True,
    )
    assert f is not None


def test_export_onnx():
    f = export_onnx(
        model=model,
        im=im,
        file=PT_WEIGHTS,
        opset=12,
        dynamic=True,
        fp16=False,  # export failure: "slow_conv2d_cpu" not implemented for 'Half', osnet
        simplify=True
    )
    assert f is not None


def test_export_openvino():
    f = export_openvino(
        file=ONNX_WEIGHTS,
        half=True
    )
    assert f is not None


def test_export_tflite(enabled=False):
    f = export_tflite(
        file=ONNX_WEIGHTS,
    )
    assert f is not None
