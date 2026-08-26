"""Coverage checks for the OSNet ImageNet checkpoint loader."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from boxmot.reid.backbones.families.osnet import pretrained as osnet_pretrained


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def debug(self, message: str) -> None:
        self.debug_messages.append(message)


def _patch_checkpoint(monkeypatch, state_dict: dict[str, torch.Tensor]) -> _RecordingLogger:
    logger = _RecordingLogger()
    monkeypatch.setattr(osnet_pretrained, "LOGGER", logger)
    monkeypatch.setattr(
        osnet_pretrained,
        "load_gdrive_checkpoint",
        lambda *_args, **_kwargs: state_dict,
    )
    return logger


def test_osnet_pretrained_loader_reports_checkpoint_and_model_coverage(monkeypatch):
    model = nn.Linear(2, 2)
    logger = _patch_checkpoint(
        monkeypatch,
        {
            "module.weight": torch.ones_like(model.weight),
            "module.bias": torch.ones_like(model.bias),
        },
    )

    osnet_pretrained.load_osnet_pretrained(model, key="osnet_x0_25")

    assert torch.equal(model.weight, torch.ones_like(model.weight))
    assert len(logger.info_messages) == 1
    assert "matched 2/2 checkpoint tensors (100.0%)" in logger.info_messages[0]
    assert "2/2 model tensors (100.0%)" in logger.info_messages[0]


@pytest.mark.parametrize("checkpoint_kind", ("empty", "target_partial", "source_partial"))
def test_osnet_pretrained_loader_rejects_insufficient_coverage(monkeypatch, checkpoint_kind):
    model = nn.Linear(2, 2)
    checkpoints = {
        "empty": {},
        "target_partial": {"module.weight": torch.ones_like(model.weight)},
        "source_partial": {
            "module.weight": torch.ones_like(model.weight),
            "module.bias": torch.ones_like(model.bias),
            "module.unexpected": torch.ones(1),
        },
    }
    _patch_checkpoint(monkeypatch, checkpoints[checkpoint_kind])

    with pytest.raises(RuntimeError, match="insufficient coverage"):
        osnet_pretrained.load_osnet_pretrained(model, key="osnet_x0_25")
