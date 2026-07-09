# BoxMOT AGPL-3.0 license

"""Lazy shared ReID backbone utility exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BatchFeatureErase_Top",
    "init_kaiming_reid",
    "load_gdrive_checkpoint",
    "load_gdrive_url",
    "load_hub_checkpoint",
    "load_partial_state_dict",
    "load_torch_url",
    "load_url_pretrained",
    "log_pretrained_result",
    "resolve_torch_cache_path",
    "warn_manual_pretrained_download",
]

_EXPORTS = {
    "BatchFeatureErase_Top": ("boxmot.reid.backbones.common.attention", "BatchFeatureErase_Top"),
    "init_kaiming_reid": ("boxmot.reid.backbones.common.init", "init_kaiming_reid"),
    "load_gdrive_checkpoint": ("boxmot.reid.backbones.common.pretrained", "load_gdrive_checkpoint"),
    "load_gdrive_url": ("boxmot.reid.backbones.common.pretrained", "load_gdrive_url"),
    "load_hub_checkpoint": ("boxmot.reid.backbones.common.pretrained", "load_hub_checkpoint"),
    "load_partial_state_dict": ("boxmot.reid.backbones.common.pretrained", "load_partial_state_dict"),
    "load_torch_url": ("boxmot.reid.backbones.common.pretrained", "load_torch_url"),
    "load_url_pretrained": ("boxmot.reid.backbones.common.pretrained", "load_url_pretrained"),
    "log_pretrained_result": ("boxmot.reid.backbones.common.pretrained", "log_pretrained_result"),
    "resolve_torch_cache_path": ("boxmot.reid.backbones.common.pretrained", "resolve_torch_cache_path"),
    "warn_manual_pretrained_download": ("boxmot.reid.backbones.common.pretrained", "warn_manual_pretrained_download"),
}

if TYPE_CHECKING:
    from boxmot.reid.backbones.common.attention import BatchFeatureErase_Top
    from boxmot.reid.backbones.common.init import init_kaiming_reid
    from boxmot.reid.backbones.common.pretrained import (
        load_gdrive_checkpoint,
        load_gdrive_url,
        load_hub_checkpoint,
        load_partial_state_dict,
        load_torch_url,
        load_url_pretrained,
        log_pretrained_result,
        resolve_torch_cache_path,
        warn_manual_pretrained_download,
    )


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'boxmot.reid.backbones.common' has no attribute {name!r}") from exc
    return getattr(import_module(module_name), attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
