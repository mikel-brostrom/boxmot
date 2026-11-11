# BoxMOT AGPL-3.0 license

from torch import nn

from boxmot.reid.backbones.common.pretrained import (
    load_gdrive_checkpoint,
    load_partial_state_dict,
)
from boxmot.utils import logger as LOGGER

__all__ = ["load_osnet_pretrained", "pretrained_urls"]

_MIN_PRETRAINED_COVERAGE = 0.9

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
    """Initialize OSNet variants with a substantially matching ImageNet checkpoint."""
    cached_filename = f"{key}_imagenet.pth"
    state_dict = load_gdrive_checkpoint(
        pretrained_urls[key],
        filename=cached_filename,
        logger=LOGGER,
        quiet=False,
        weights_only=False,
    )
    matched_layers, discarded_layers = load_partial_state_dict(model, state_dict)
    matched_count = len(matched_layers)
    checkpoint_total = len(state_dict)
    model_total = len(model.state_dict())
    checkpoint_coverage = matched_count / checkpoint_total if checkpoint_total else 0.0
    model_coverage = matched_count / model_total if model_total else 0.0

    if (
        matched_count == 0
        or checkpoint_coverage < _MIN_PRETRAINED_COVERAGE
        or model_coverage < _MIN_PRETRAINED_COVERAGE
    ):
        raise RuntimeError(
            f'OSNet pretrained checkpoint "{cached_filename}" has insufficient coverage for {key}: '
            f"matched {matched_count}/{checkpoint_total} checkpoint tensors "
            f"({checkpoint_coverage:.1%}) and {matched_count}/{model_total} model tensors "
            f"({model_coverage:.1%}); at least {_MIN_PRETRAINED_COVERAGE:.0%} of both is required"
        )

    LOGGER.info(
        f'Successfully loaded ImageNet pretrained weights from "{cached_filename}": '
        f"matched {matched_count}/{checkpoint_total} checkpoint tensors "
        f"({checkpoint_coverage:.1%}) and {matched_count}/{model_total} model tensors "
        f"({model_coverage:.1%})"
    )
    if discarded_layers:
        LOGGER.debug(f"Skipped pretrained layers: {discarded_layers}")
