# BoxMOT AGPL-3.0 license

from torch import nn

from boxmot.reid.backbones.common.pretrained import (
    load_gdrive_checkpoint,
    load_partial_state_dict,
    log_pretrained_result,
)
from boxmot.utils import logger as LOGGER

__all__ = ["load_osnet_pretrained", "pretrained_urls"]

pretrained_urls = {
    "osnet_x1_0": "https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY",
    "osnet_x0_75": "https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq",
    "osnet_x0_5": "https://drive.google.com/uc?id=16DGLbZukvVYgINws8u8deSaOqjybZ83i",
    "osnet_x0_25": "https://drive.google.com/uc?id=1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs",
    "osnet_ibn_x1_0": "https://drive.google.com/uc?id=1sr90V6irlYYDd4_4ISU2iruoRG8J__6l",
    "osnet_ain_x1_0": "https://drive.google.com/uc?id=1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo",
    "osnet_ain_x0_75": "https://drive.google.com/uc?id=1apy0hpsMypqstfencdH-jKIUEFOW4xoM",
    "osnet_ain_x0_5": "https://drive.google.com/uc?id=1KusKvEYyKGDTUBVRxRiz55G31wkihB6l",
    "osnet_ain_x0_25": "https://drive.google.com/uc?id=1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt",
}


def load_osnet_pretrained(model: nn.Module, key: str) -> None:
    """Initialize OSNet variants with ImageNet pretrained weights."""
    cached_filename = f"{key}_imagenet.pth"
    state_dict = load_gdrive_checkpoint(
        pretrained_urls[key],
        filename=cached_filename,
        logger=LOGGER,
        quiet=False,
        weights_only=False,
    )
    matched_layers, discarded_layers = load_partial_state_dict(model, state_dict)
    log_pretrained_result(
        f'imagenet pretrained weights from "{cached_filename}"',
        matched_layers,
        discarded_layers,
        logger=LOGGER,
    )
