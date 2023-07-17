from __future__ import absolute_import

from boxmot.appearance.backbones.hacnn import HACNN
from boxmot.appearance.backbones.lmbn_n import LMBN_n
from boxmot.appearance.backbones.mlfn import mlfn
from boxmot.appearance.backbones.mobilenetv2 import (mobilenetv2_x1_0,
                                                     mobilenetv2_x1_4)
from boxmot.appearance.backbones.osnet import (osnet_ibn_x1_0, osnet_x0_5,
                                               osnet_x0_25, osnet_x0_75,
                                               osnet_x1_0)
from boxmot.appearance.backbones.osnet_ain import (osnet_ain_x0_5,
                                                   osnet_ain_x0_25,
                                                   osnet_ain_x0_75,
                                                   osnet_ain_x1_0)
from boxmot.appearance.backbones.resnet import resnet50, resnet101

__model_factory = {
    # image classification models
    "resnet50": resnet50,
    "resnet101": resnet101,
    "mobilenetv2_x1_0": mobilenetv2_x1_0,
    "mobilenetv2_x1_4": mobilenetv2_x1_4,
    # reid-specific models
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
}


def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(name, num_classes, loss="softmax", pretrained=True, use_gpu=True):
    """A function wrapper for building a model.

    Args:
        name (str): model name.
        num_classes (int): number of training identities.
        loss (str, optional): loss function to optimize the model. Currently
            supports "softmax" and "triplet". Default is "softmax".
        pretrained (bool, optional): whether to load ImageNet-pretrained weights.
            Default is True.
        use_gpu (bool, optional): whether to use gpu. Default is True.

    Returns:
        nn.Module

    Examples::
        >>> from torchreid import models
        >>> model = models.build_model('resnet50', 751, loss='softmax')
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError("Unknown model: {}. Must be one of {}".format(name, avai_models))
    return __model_factory[name](
        num_classes=num_classes, loss=loss, pretrained=pretrained, use_gpu=use_gpu
    )
