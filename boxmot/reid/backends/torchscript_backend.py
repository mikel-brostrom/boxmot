from pathlib import Path

import torch

from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.utils import logger as LOGGER


class TorchscriptBackend(BaseModelBackend):
    build_source_model = False

    def __init__(self, weights, device, half, preprocess=None):
        super().__init__(weights, device, half, preprocess=preprocess)
        self.nhwc = False
        self.half = half

    def load_model(self, w: str | Path) -> None:
        """Load serialized tensors on the selected inference device."""

        LOGGER.info(f"Loading {w} for TorchScript inference...")
        self.model = torch.jit.load(w, map_location=self.device).eval()
        self.model.half() if self.half else self.model.float()

    def forward(self, im_batch):
        features = self.model(im_batch)
        return features
