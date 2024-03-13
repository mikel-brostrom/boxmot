import numpy as np
from pathlib import Path

from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.utils.checks import TestRequirements

tr = TestRequirements()


class ONNXBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):

        tr.check_packages(("onnxruntime-gpu==1.16.3" if self.cuda else "onnxruntime==1.16.3", ))
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


im = (np.random.rand(800,800,3) * 255).astype(np.uint8)
xyxy = np.array(
    [[ 345,  246,  794,  498],
     [ 400,  400, 500, 500]]
)

b = ONNXBackend(
    weights=Path("/home/mikel.brostrom/yolo_tracking/tracking/weights/osnet_x0_25_msmt17.onnx"),
    half=False,
    device="cpu"
)

f = b.get_features(xyxy, im)
print(f.shape)