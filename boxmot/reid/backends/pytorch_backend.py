from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.reid.core.registry import ReIDModelRegistry
from boxmot.utils import logger as LOGGER


class PyTorchBackend(BaseModelBackend):

    def __init__(self, weights, device, half, preprocess=None):
        super().__init__(weights, device, half, preprocess=preprocess)
        self.nhwc = False
        self.half = half

    def load_model(self, w):
        # Load a PyTorch model
        if w and w.is_file():
            # Warn if the checkpoint was trained with a different preprocessing
            ckpt_preprocess = ReIDModelRegistry.get_checkpoint_preprocess(w)
            if ckpt_preprocess is not None:
                from boxmot.reid.core.preprocessing import get_preprocess_fn
                current_name = getattr(self, "_preprocess_name", None)
                if current_name and ckpt_preprocess != current_name:
                    LOGGER.warning(
                        f"ReID weights '{w.name}' were trained with preprocess='{ckpt_preprocess}' "
                        f"but inference is configured with preprocess='{current_name}'. "
                        f"This mismatch will degrade embedding quality. "
                        f"Set preprocess='{ckpt_preprocess}' in your ReID config."
                    )
            ReIDModelRegistry.load_pretrained_weights(self.model, w)
        self.model.to(self.device).eval()
        self.model.half() if self.half else self.model.float()

    def forward(self, im_batch):
        features = self.model(im_batch)
        return features
