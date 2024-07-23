import torch
import numpy as np
from pathlib import Path
from boxmot.utils import logger as LOGGER

from boxmot.appearance.backends.base_backend import BaseModelBackend


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
