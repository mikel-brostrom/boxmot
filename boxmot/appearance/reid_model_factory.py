# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import sys
import time
from collections import OrderedDict
from enum import Enum
from dataclasses import dataclass


import torch

from boxmot.utils import logger as LOGGER

class ModelType(Enum):
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    MLFN = "mlfn"
    HACNN = "hacnn"
    MOBILENETV2_X1_0 = "mobilenetv2_x1_0"
    MOBILENETV2_X1_4 = "mobilenetv2_x1_4"
    OSNET_X1_0 = "osnet_x1_0"
    OSNET_X0_75 = "osnet_x0_75"
    OSNET_X0_5 = "osnet_x0_5"
    OSNET_X0_25 = "osnet_x0_25"
    OSNET_IBN_X1_0 = "osnet_ibn_x1_0"
    OSNET_AIN_X1_0 = "osnet_ain_x1_0"
    LMBN_N = "lmbn_n"
    CLIP = "clip"


@dataclass
class ModelInfo:
    name: ModelType
    url: str


trained_urls = {
    ModelType.RESNET50: [
        ModelInfo(name=ModelType.RESNET50, url="https://drive.google.com/uc?id=1dUUZ4rHDWohmsQXCRe2C_HbYkzz94iBV"),
        ModelInfo(name=ModelType.RESNET50, url="https://drive.google.com/uc?id=17ymnLglnc64NRvGOitY3BqMRS9UWd1wg"),
        ModelInfo(name=ModelType.RESNET50, url="https://drive.google.com/uc?id=1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj"),
        ModelInfo(name=ModelType.RESNET50, url="https://drive.google.com/uc?id=1kv8l5laX_YCdIGVCetjlNdzKIA3NvsSt"),
        ModelInfo(name=ModelType.RESNET50, url="https://drive.google.com/uc?id=13QN8Mp3XH81GK4BPGXobKHKyTGH50Rtx"),
        ModelInfo(name=ModelType.RESNET50, url="https://drive.google.com/uc?id=1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud"),
    ],
    ModelType.MLFN: [
        ModelInfo(name=ModelType.MLFN, url="https://drive.google.com/uc?id=1wXcvhA_b1kpDfrt9s2Pma-MHxtj9pmvS"),
        ModelInfo(name=ModelType.MLFN, url="https://drive.google.com/uc?id=1rExgrTNb0VCIcOnXfMsbwSUW1h2L1Bum"),
        ModelInfo(name=ModelType.MLFN, url="https://drive.google.com/uc?id=18JzsZlJb3Wm7irCbZbZ07TN4IFKvR6p-"),
    ],
    ModelType.HACNN: [
        ModelInfo(name=ModelType.HACNN, url="https://drive.google.com/uc?id=1LRKIQduThwGxMDQMiVkTScBwR7WidmYF"),
        ModelInfo(name=ModelType.HACNN, url="https://drive.google.com/uc?id=1zNm6tP4ozFUCUQ7Sv1Z98EAJWXJEhtYH"),
        ModelInfo(name=ModelType.HACNN, url="https://drive.google.com/uc?id=1MsKRtPM5WJ3_Tk2xC0aGOO7pM3VaFDNZ"),
    ],
    ModelType.MOBILENETV2_X1_0: [
        ModelInfo(name=ModelType.MOBILENETV2_X1_0, url="https://drive.google.com/uc?id=18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp"),
        ModelInfo(name=ModelType.MOBILENETV2_X1_0, url="https://drive.google.com/uc?id=1q1WU2FETRJ3BXcpVtfJUuqq4z3psetds"),
        ModelInfo(name=ModelType.MOBILENETV2_X1_0, url="https://drive.google.com/uc?id=1j50Hv14NOUAg7ZeB3frzfX-WYLi7SrhZ"),
    ],
    ModelType.MOBILENETV2_X1_4: [
        ModelInfo(name=ModelType.MOBILENETV2_X1_4, url="https://drive.google.com/uc?id=1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5"),
        ModelInfo(name=ModelType.MOBILENETV2_X1_4, url="https://drive.google.com/uc?id=12uD5FeVqLg9-AFDju2L7SQxjmPb4zpBN"),
        ModelInfo(name=ModelType.MOBILENETV2_X1_4, url="https://drive.google.com/uc?id=1ZY5P2Zgm-3RbDpbXM0kIBMPvspeNIbXz"),
    ],
    ModelType.OSNET_X1_0: [
        ModelInfo(name=ModelType.OSNET_X1_0, url="https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA"),
        ModelInfo(name=ModelType.OSNET_X1_0, url="https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq"),
        ModelInfo(name=ModelType.OSNET_X1_0, url="https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M"),
    ],
    ModelType.OSNET_X0_75: [
        ModelInfo(name=ModelType.OSNET_X0_75, url="https://drive.google.com/uc?id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer"),
        ModelInfo(name=ModelType.OSNET_X0_75, url="https://drive.google.com/uc?id=1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or"),
        ModelInfo(name=ModelType.OSNET_X0_75, url="https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc"),
    ],
    ModelType.OSNET_X0_5: [
        ModelInfo(name=ModelType.OSNET_X0_5, url="https://drive.google.com/uc?id=1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT"),
        ModelInfo(name=ModelType.OSNET_X0_5, url="https://drive.google.com/uc?id=1KoUVqmiST175hnkALg9XuTi1oYpqcyTu"),
        ModelInfo(name=ModelType.OSNET_X0_5, url="https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv"),
    ],
    ModelType.OSNET_X0_25: [
        ModelInfo(name=ModelType.OSNET_X0_25, url="https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj"),
        ModelInfo(name=ModelType.OSNET_X0_25, url="https://drive.google.com/uc?id=1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l"),
        ModelInfo(name=ModelType.OSNET_X0_25, url="https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"),
    ],
    ModelType.OSNET_IBN_X1_0: [
        ModelInfo(name=ModelType.OSNET_IBN_X1_0, url="https://drive.google.com/uc?id=1q3Sj2ii34NlfxA4LvmHdWO_75NDRmECJ")
    ],
    ModelType.OSNET_AIN_X1_0: [
        ModelInfo(name=ModelType.OSNET_AIN_X1_0, url="https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal")
    ],
    ModelType.LMBN_N: [
        ModelInfo(name=ModelType.LMBN_N, url="https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth"),
        ModelInfo(name=ModelType.LMBN_N, url="https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_market.pth"),
        ModelInfo(name=ModelType.LMBN_N, url="https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_cuhk03_d.pth"),
    ],
    ModelType.CLIP: [
        ModelInfo(name=ModelType.CLIP, url="https://drive.google.com/uc?id=1GnyAVeNOg3Yug1KBBWMKKbT2x43O5Ch7"),
        ModelInfo(name=ModelType.CLIP, url="https://drive.google.com/uc?id=1ldjSkj-7pXAWmx8on5x0EftlCaolU4dY"),
        ModelInfo(name=ModelType.CLIP, url="https://drive.google.com/uc?id=1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e"),
        ModelInfo(name=ModelType.CLIP, url="https://drive.google.com/uc?id=168BLegHHxNqatW5wx1YyL2REaThWoof5"),
    ],
}


def show_downloadable_models():
    LOGGER.info("\nAvailable .pt ReID models for automatic download")
    for model_type, model_infos in trained_urls.items():
        for model_info in model_infos:
            LOGGER.info(f"{model_type.value} - {model_info.url}")


def get_model_urls(model_name: ModelType):
    return trained_urls.get(model_name, [])


def get_model_name(model):
    return next((x for x in __model_types if x in model.name), None)


def load_pretrained_weights(model, weight_path):
    """Loads pretrained weights to model.

    Features:
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples:
        >>> from boxmot.appearance.backbones import build_model
        >>> from boxmot.appearance.reid_model_factory import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> model = build_model()
        >>> load_pretrained_weights(model, weight_path)
    """

    checkpoint = torch.load(weight_path, map_location=torch.device("cpu") if not torch.cuda.is_available() else None)

    state_dict = checkpoint.get("state_dict", checkpoint)
    model_dict = model.state_dict()

    if "lmbn" in weight_path:
        model.load_state_dict(model_dict, strict=True)
    elif "clip" in weight_path:
        def forward_override(self, x: torch.Tensor, cv_emb=None, old_forward=None):
            _, image_features, image_features_proj = old_forward(x, cv_emb)
            return torch.cat([image_features[:, 0], image_features_proj[:, 0]], dim=1)

        model.load_param(str(weight_path))
        model = model.image_encoder

        LOGGER.success(f'Successfully loaded pretrained weights from "{weight_path}"')
    else:
        new_state_dict = OrderedDict()
        matched_layers, discarded_layers = [], []

        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]

            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict)

        if not matched_layers:
            LOGGER.debug(
                f'The pretrained weights "{weight_path}" cannot be loaded, '
                "please check the key names manually (** ignored and continue **)"
            )
        else:
            LOGGER.success(f'Successfully loaded pretrained weights from "{weight_path}"')
            if discarded_layers:
                LOGGER.debug(
                    "The following layers are discarded due to unmatched keys or layer size: "
                    f"{', '.join(discarded_layers)}"
                )
                
                
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
    if name not in __model_factory:
        raise KeyError(f"Unknown model: {name}. Must be one of {list(__model_factory.keys())}")

    if 'clip' in name:
        from boxmot.appearance.backbones.clip.config.defaults import _C as cfg
        return __model_factory[name](cfg, num_class=num_classes, camera_num=2, view_num=1)

    return __model_factory[name](
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        use_gpu=use_gpu
    )