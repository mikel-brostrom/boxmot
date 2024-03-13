import torch
import numpy as np
from pathlib import Path
from boxmot.utils import logger as LOGGER

from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.utils.checks import TestRequirements

tr = TestRequirements()


class TorchscriptBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):

        LOGGER.info(f"Loading {w} for TorchScript inference...")
        self.model = torch.jit.load(w)
        self.model.half() if self.half else self.model.float()

    def forward(self, im_batch):
        features = self.model(im_batch)
        return features


im = (np.random.rand(800, 800, 3) * 255).astype(np.uint8)
xyxy = np.array(
    [[ 345,  246,  794,  498],
     [ 400,  400, 500, 500]]
)

b = TorchscriptBackend(
    weights=Path("/home/mikel.brostrom/yolo_tracking/tracking/weights/osnet_x0_25_msmt17.torchscript"),
    half=False,
    device="cpu"
)

f = b.get_features(xyxy, im)
print(f.shape)