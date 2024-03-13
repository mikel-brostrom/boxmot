import torch

from boxmot.utils import logger as LOGGER
from boxmot.appearance.backends.onnx_backend import ONNXBackend
from boxmot.appearance.backends.openvino_backend import OpenVinoBackend
from boxmot.appearance.backends.pytorch_backend import PyTorchBackend
from boxmot.appearance.backends.tensorrt_backend import TensorRTBackend
from boxmot.appearance.backends.tflite_backend import TFLiteBackend



class ReidAutoBackend():
    def __init__(self, weights="osnet_x0_25_msmt17.pt", device=torch.device("cpu"), half=False):
        super().__init__()
        w = weights[0] if isinstance(weights, list) else weights
        (
            self.pt,
            self.jit,
            self.onnx,
            self.xml,
            self.engine,
            self.tflite,
        ) = self.model_type(w)  # get backend

        self.weights = weights
        self.device = device
        self.half = half

    def get_backend(self, weights, device, half):
        # Logic to determine which backend to use
        if self.pt:  # Condition for PyTorch
            return PyTorchBackend(weights, device, half)
        elif self.jit:  # Conditions for other backends
            return TorchscriptBackend(weights, device, half)
        elif self.onnx:
            return ONNXBackend(weights, device, half)
        elif self.engine:
            TensorRTBackend(weights, device, half)
        elif self.xml:  # OpenVINO
            OpenVinoBackend(weights, device, half)
        elif self.tflite:
            TFLiteBackend(weights, device, half)
        else:
            LOGGER.error("This model framework is not supported yet!")
            exit()

    def forward(self, im_batch):
        im_batch = self.backend.preprocess_input(im_batch)
        return self.backend.get_features(im_batch)