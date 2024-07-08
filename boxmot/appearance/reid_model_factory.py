import sys
import time
from collections import OrderedDict

import torch
from torch import nn
from boxmot.utils import logger as LOGGER

# Model Factory and Construction
from boxmot.appearance.backbones.clip.make_model import make_model
from boxmot.appearance.backbones.hacnn import HACNN
from boxmot.appearance.backbones.lmbn.lmbn_n import LMBN_n
from boxmot.appearance.backbones.mlfn import mlfn
from boxmot.appearance.backbones.mobilenetv2 import mobilenetv2_x1_0, mobilenetv2_x1_4
from boxmot.appearance.backbones.osnet import (
    osnet_ibn_x1_0,
    osnet_x0_5,
    osnet_x0_25,
    osnet_x0_75,
    osnet_x1_0,
)
from boxmot.appearance.backbones.osnet_ain import (
    osnet_ain_x0_5,
    osnet_ain_x0_25,
    osnet_ain_x0_75,
    osnet_ain_x1_0,
)
from boxmot.appearance.backbones.resnet import resnet50, resnet101

# Constants
__model_types = [
    "resnet50",
    "resnet101",
    "mlfn",
    "hacnn",
    "mobilenetv2_x1_0",
    "mobilenetv2_x1_4",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
    "lmbn_n",
    "clip",
]

__trained_urls = {
    # Example URLs for pretrained models (partial list for brevity)
    "resnet50_market1501.pt": "https://example.com/resnet50_market1501.pt",
    "mlfn_market1501.pt": "https://example.com/mlfn_market1501.pt",
    # Add more URLs as needed
}

NR_CLASSES_DICT = {
    'market1501': 751,
    'duke': 702,
    'veri': 576,
    'vehicleid': 576
}

__model_factory = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "mobilenetv2_x1_0": mobilenetv2_x1_0,
    "mobilenetv2_x1_4": mobilenetv2_x1_4,
    "hacnn": HACNN,
    "mlfn": mlfn,
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
    "osnet_ain_x1_0": osnet_ain_x1_0,
    "osnet_ain_x0_75": osnet_ain_x0_75,
    "osnet_ain_x0_5": osnet_ain_x0_5,
    "osnet_ain_x0_25": osnet_ain_x0_25,
    "lmbn_n": LMBN_n,
    "clip": make_model,
}


# Utility functions
def show_downloadable_models():
    LOGGER.info("Available .pt ReID models for automatic download")
    LOGGER.info(list(__trained_urls.keys()))
    
    
def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None

def get_model_url(model):
    if model.name in __trained_urls:
        return __trained_urls[model.name]
    else:
        None


def load_pretrained_weights(model, weight_path):
    """Loads pretrained weights to a model."""
    if not torch.cuda.is_available():
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(weight_path)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()

    if "lmbn" in str(weight_path):
        model.load_state_dict(model_dict, strict=True)
    else:
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]  # remove 'module.' prefix if present

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if len(matched_layers) == 0:
            LOGGER.debug(
                f"Pretrained weights from {weight_path} cannot be loaded. Check key names manually."
            )
        else:
            LOGGER.success(f"Loaded pretrained weights from {weight_path}")

        if len(discarded_layers) > 0:
            LOGGER.debug(
                f"Discarded layers due to unmatched keys or layer size: {discarded_layers}"
            )


def show_available_models():
    """Displays available models."""
    LOGGER.info("Available models:")
    LOGGER.info(list(__model_factory.keys()))


def get_nr_classes(weights):
    """Returns the number of classes based on weights."""
    num_classes = NR_CLASSES_DICT.get(weights.name.split('_')[1], 1)
    return num_classes


def build_model(name, num_classes, loss="softmax", pretrained=True, use_gpu=True):
    """Builds a model based on specified parameters."""
    available_models = list(__model_factory.keys())

    if name not in available_models:
        raise KeyError(f"Unknown model '{name}'. Must be one of {available_models}")

    if 'clip' in name:
        # Assuming clip requires special configuration, adjust as needed
        from boxmot.appearance.backbones.clip.config.defaults import _C as cfg
        return __model_factory[name](cfg, num_class=num_classes, camera_num=2, view_num=1)

    return __model_factory[name](
        num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu
    )
