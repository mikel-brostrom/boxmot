import numpy as np
from pathlib import Path

from boxmot.appearance.backends.base_backend import BaseModelBackend


class ONNXBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):

        self.checker.check_packages(("onnxruntime-gpu==1.16.3" if self.cuda else "onnxruntime==1.16.3", ))
        import onnxruntime

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if self.cuda else ["CPUExecutionProvider"])
        self.session = onnxruntime.InferenceSession(str(w), providers=providers)

    def forward(self, im_batch):
        im_batch = im_batch.cpu().numpy()  # torch to numpy
        features = self.session.run(
            [self.session.get_outputs()[0].name],
            {self.session.get_inputs()[0].name: im_batch},
        )[0]
        return features
