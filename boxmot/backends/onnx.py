from pathlib import Path

import numpy as np
import torch

from boxmot.backends.backend import Backend


class ONNXBackend(Backend):

    def __init__(self, weights: str | Path, device: str):
        super().__init__()
        self.weights = Path(weights)
        self.device = device
        self.model = self.load()

    def load(self):

        # Load the ONNX model using onnxruntime
        import onnxruntime

        # ONNXRuntime will attempt to use the first provider, and if it fails or is not
        # available for some reason, it will fall back to the next provider in the list
        if self.device == "mps":
            self.checker.check_packages(("onnxruntime-silicon==1.20.0",))
            providers = ["MPSExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "cuda":
            self.checker.check_packages(("onnxruntime-gpu==1.20.0",))
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            self.checker.check_packages(("onnxruntime==1.20.0",))
            providers = ["CPUExecutionProvider"]

        return onnxruntime.InferenceSession(str(self.weights), providers=providers)

    def preprocess(self, x: torch.Tensor):
        return x.cpu().numpy()

    def process(self, x: np.ndarray) -> np.ndarray:

        y = self.model.run(
            [self.model.get_outputs()[0].name],
            {self.model.get_inputs()[0].name: x},
        )[0]

        return y
