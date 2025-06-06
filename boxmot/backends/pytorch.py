from pathlib import Path

import torch

from boxmot.backends.backend import Backend


class PyTorchBackend(Backend):

    def __init__(self, weights: str | Path, device: str, half: bool):
        super().__init__(weights, device, half)
        self.weights = weights
        self.device = device
        self.half = half
        self.model = self.load()

    def load(self):

        # Determine target device
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please specify device='cpu' or install CUDA.")
        if self.device == "mps" and not getattr(torch, "has_mps", False):
            raise RuntimeError(
                "MPS is not available. Please specify device='cpu' or use a supported platform.")

        dev = torch.device(self.device)

        if self.weights.stem in ["pt", "pth"]:
            model = torch.load(self.weights, map_location=dev)
        if self.weights.stem == "jit":
            model = torch.jit.load(self.weights, map_location=dev)

        # Switch to evaluation mode
        model.eval()

        # If half precision is requested, convert model parameters to fp16
        if self.half:
            if dev.type not in ("cuda", "mps"):
                raise RuntimeError(
                    f"Half precision (fp16) is not supported on device '{dev}'.")
            model.half()

        # Move model to the target device

        model.to(dev)

        return model

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def process(self, x: torch.Tensor):
        y = self.model(x)
        return y
