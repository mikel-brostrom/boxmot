import torch
from pathlib import Path

from boxmot.utils import logger as LOGGER
from boxmot.appearance.backends.onnx_backend import ONNXBackend
from boxmot.appearance.backends.openvino_backend import OpenVinoBackend
from boxmot.appearance.backends.pytorch_backend import PyTorchBackend
from boxmot.appearance.backends.tensorrt_backend import TensorRTBackend
from boxmot.appearance.backends.tflite_backend import TFLiteBackend
from boxmot.appearance.backends.base_backend import BaseModelBackend



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

    def get_backend(self):
        # Mapping of conditions to backend constructors
        backend_map = {
            self.pt: PyTorchBackend,
            self.jit: TorchscriptBackend,
            self.onnx: ONNXBackend,
            self.engine: TensorRTBackend,
            self.xml: OpenVinoBackend,
            self.tflite: TFLiteBackend
        }

        # Iterate through the mapping and return the first matching backend
        for condition, backend_class in backend_map.items():
            if condition:
                return backend_class(self.weights, self.device, self.half)

        # If no condition is met, log an error and exit
        LOGGER.error("This model framework is not supported yet!")
        exit()

    def forward(self, im_batch):
        im_batch = self.backend.preprocess_input(im_batch)
        return self.backend.get_features(im_batch)

    def check_suffix(self, file="osnet_x0_25_msmt17.pt", suffix=(".pt",), msg=""):
        # Check file(s) for acceptable suffix
        if file and suffix:
            if isinstance(suffix, str):
                suffix = [suffix]
            for f in file if isinstance(file, (list, tuple)) else [file]:
                s = Path(f).suffix.lower()  # file suffix
                if len(s):
                    try:
                        assert s in suffix
                    except AssertionError as err:
                        LOGGER.error(f"{err}{f} acceptable suffix is {suffix}")

    def model_type(self, p="path/to/model.pt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from boxmot.appearance import export_formats

        sf = list(export_formats().Suffix)  # export suffixes
        print(sf)
        print(p)
        self.check_suffix(p, sf)  # checks
        types = [s in Path(p).name for s in sf]
        return types