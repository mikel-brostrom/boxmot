# model_registry.py
import torch
from collections import OrderedDict
from boxmot.utils import logger as LOGGER

from boxmot.appearance.reid.config import MODEL_TYPES, TRAINED_URLS, NR_CLASSES_DICT
from boxmot.appearance.reid.factory import MODEL_FACTORY

class ReIDModelRegistry:
    """Encapsulates model registration and related utilities."""

    @staticmethod
    def show_downloadable_models():
        LOGGER.info("Available .pt ReID models for automatic download")
        LOGGER.info(list(TRAINED_URLS.keys()))

    @staticmethod
    def get_model_name(model):
        for name in MODEL_TYPES:
            if name in model.name:
                return name
        return None

    @staticmethod
    def get_model_url(model):
        return TRAINED_URLS.get(model.name, None)

    @staticmethod
    def load_pretrained_weights(model, weight_path):
        """
        Loads pretrained weights into a model.
        Chooses the proper map_location based on CUDA availability.
        """
        device = "cpu" if not torch.cuda.is_available() else None
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu") if device == "cpu" else None)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_dict = model.state_dict()

        if "lmbn" in weight_path.parts:
            model.load_state_dict(model_dict, strict=True)
        else:
            new_state_dict = OrderedDict()
            matched_layers, discarded_layers = [], []
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present
                key = k[7:] if k.startswith("module.") else k
                if key in model_dict and model_dict[key].size() == v.size():
                    new_state_dict[key] = v
                    matched_layers.append(key)
                else:
                    discarded_layers.append(key)
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict)
            
            if not matched_layers:
                LOGGER.debug(f"Pretrained weights from {weight_path} cannot be loaded. Check key names manually.")
            else:
                LOGGER.success(f"Loaded pretrained weights from {weight_path}")

            if discarded_layers:
                LOGGER.debug(f"Discarded layers due to unmatched keys or size: {discarded_layers}")

    @staticmethod
    def show_available_models():
        LOGGER.info("Available models:")
        LOGGER.info(list(MODEL_FACTORY.keys()))

    @staticmethod
    def get_nr_classes(weights):
        # Extract dataset name from weights name, then look up in the class dictionary
        dataset_key = weights.name.split('_')[1]
        return NR_CLASSES_DICT.get(dataset_key, 1)

    @staticmethod
    def build_model(name, num_classes, loss="softmax", pretrained=True, use_gpu=True):
        if name not in MODEL_FACTORY:
            available = list(MODEL_FACTORY.keys())
            raise KeyError(f"Unknown model '{name}'. Must be one of {available}")

        # Special case handling for clip model
        if 'clip' in name:
            from boxmot.appearance.backbones.clip.config.defaults import _C as cfg
            return MODEL_FACTORY[name](cfg, num_class=num_classes, camera_num=2, view_num=1)
        
        return MODEL_FACTORY[name](
            num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu
        )
