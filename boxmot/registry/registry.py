from abc import abstractmethod
from pathlib import Path
import gdown

import torch

from boxmot.utils import logger as LOGGER


class ModelRegistry:
    BACKBONES: dict[str, str] = {}
    URLS:      dict[str, str] = {}
    DOWNLOAD_DIR = Path("weights")

    def get_name(self, weights: Path):
        for name in self.BACKBONES:
            if name in weights.name:
                return name
        return None

    @classmethod
    def get_url(cls, weights: Path):
        return cls.URLS.get(weights.stem, None)

    @classmethod
    def show_available_models(cls):
        LOGGER.info("Available models:")
        LOGGER.info(list(cls.BACKBONES.keys()))

    @classmethod
    def show_downloadable_models(cls):
        LOGGER.info("Available .pt models for automatic download:")
        LOGGER.info(sorted(cls.URLS.keys()))

    @classmethod
    def download(cls, weights: Path):
        if weights.suffix in [".pt", ".pth"]:
            weights_url = cls.get_url(weights)
            if not weights.exists():
                if weights_url is not None:
                    download_path = cls.DOWNLOAD_DIR / weights.name
                    gdown.download(weights_url, str(
                        download_path), quiet=False)
                else:
                    LOGGER.error(
                        f"No URL associated with the chosen weights: '{weights}'. Choose between:"
                    )
                    cls.show_downloadable_models()
                    exit()

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs) -> torch.nn.Module:
        """
        Build a model instance.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    @classmethod
    @abstractmethod
    def load(cls, weights, **kwargs) -> torch.nn.Module:
        """
        Prepare model for inference.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.")
