# model_registry.py
import torch
from boxmot.utils import logger as LOGGER

from abc import abstractmethod


class ModelRegistry:
    BACKBONES: dict[str, str] = {}
    URLS:      dict[str, str] = {}

    def get_name(self, model: torch.nn.Module):
        for name in self.BACKBONES:
            if name in model.name:
                return name
        return None

    @classmethod
    def get_url(cls, model: torch.nn.Module):
        return cls.URLS.get(model.name, None)

    @classmethod
    def show_available_models(cls):
        LOGGER.info("Available models:")
        LOGGER.info(list(cls.BACKBONES.keys()))

    @classmethod
    @abstractmethod
    def build(cls, *args, **kwargs):
        """
        Build a model instance.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.")
