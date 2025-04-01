import numpy as np
from pathlib import Path

from boxmot.appearance.backends.base_backend import BaseModelBackend


class ONNXBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):

            # ONNXRuntime will attempt to use the first provider, and if it fails or is not
            # available for some reason, it will fall back to the next provider in the list
            if self.device == "mps":
                self.checker.check_packages(("onnxruntime-silicon==1.17.0",))
                providers = ["MPSExecutionProvider", "CPUExecutionProvider"]
            elif self.device == "cuda":
                self.checker.check_packages(("onnxruntime-gpu==1.17.0",))
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self.checker.check_packages(("onnxruntime==1.17.0",))
                providers = ["CPUExecutionProvider"]

            # Load the ONNX model using onnxruntime
            import onnxruntime
            self.session = onnxruntime.InferenceSession(str(w), providers=providers)

    def forward(self, im_batch):
        # Convert torch tensor to numpy (onnxruntime expects numpy arrays)
        im_batch = im_batch.cpu().numpy()

        # Run inference using ONNX session
        features = self.session.run(
            [self.session.get_outputs()[0].name],
            {self.session.get_inputs()[0].name: im_batch},
        )[0]

        return features