from boxmot.appearance.backends.base_backend import BaseModelBackend
from boxmot.appearance.reid.registry import ReIDModelRegistry


class PyTorchBackend(BaseModelBackend):

    def __init__(self, weights, device, half):
        super().__init__(weights, device, half)
        self.nhwc = False
        self.half = half

    def load_model(self, w):
        # Load a PyTorch model
        if w and w.is_file():
            ReIDModelRegistry.load_pretrained_weights(self.model, w)
        self.model.to(self.device).eval()
        self.model.half() if self.half else self.model.float()

    def forward(self, im_batch):
        features = self.model(im_batch)
        return features
